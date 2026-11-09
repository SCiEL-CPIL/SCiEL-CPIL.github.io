"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _makespan(perm, pt):
    """Compute makespan for a given permutation of job indices."""
    n_jobs, n_machines = pt.shape
    if n_machines == 0 or n_jobs == 0:
        return 0.0
    
    C_prev = np.zeros(n_machines)
    for i in range(n_jobs):
        job = perm[i]
        C_curr = np.zeros(n_machines)
        C_curr[0] = C_prev[0] + pt[job, 0]
        for m in range(1, n_machines):
            C_curr[m] = max(C_curr[m-1], C_prev[m]) + pt[job, m]
        C_prev = C_curr
    return C_prev[-1]


def _completion_matrix(perm, pt):
    """Compute full completion matrix C[i][m] for permutation perm."""
    n_jobs, n_machines = pt.shape
    C = np.zeros((n_jobs, n_machines))
    for i in range(n_jobs):
        job = perm[i]
        C[i, 0] = (C[i-1, 0] if i > 0 else 0.0) + pt[job, 0]
        for m in range(1, n_machines):
            C[i, m] = max(C[i, m-1], C[i-1, m] if i > 0 else 0.0) + pt[job, m]
    return C


def _makespan_from_matrix(C):
    """Get makespan from completion matrix."""
    if C.size == 0:
        return 0.0
    return C[-1, -1]


def _neh_heuristic(pt):
    """NEH heuristic: sort by total processing time, insert each job optimally."""
    n_jobs, n_machines = pt.shape
    sums = pt.sum(axis=1)
    order = np.argsort(-sums)
    
    perm = [int(order[0])]
    for idx in range(1, n_jobs):
        job = int(order[idx])
        best_pos = 0
        best_makespan = float('inf')
        for pos in range(len(perm) + 1):
            new_perm = perm[:pos] + [job] + perm[pos:]
            ms = _makespan(new_perm, pt)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        perm.insert(best_pos, job)
    
    return perm


def _or_opt_move(perm, i, length, new_pos):
    """Perform an Or-opt move: remove 'length' jobs starting at position i, insert at new_pos.
    Returns new permutation and the inverse move info."""
    if i < 0 or i + length > len(perm):
        return None, None
    
    # Determine insertion position in the shortened list
    removed = perm[i:i+length]
    shortened = perm[:i] + perm[i+length:]
    
    # Adjust new_pos for the shortened list
    if new_pos > i:
        adjusted_pos = new_pos - length
    else:
        adjusted_pos = new_pos
    
    if adjusted_pos < 0 or adjusted_pos > len(shortened):
        return None, None
    
    new_perm = shortened[:adjusted_pos] + removed + shortened[adjusted_pos:]
    
    # Store inverse move: to undo, remove 'length' jobs at adjusted_pos, insert at i
    inverse_info = (adjusted_pos, i, length)
    
    return new_perm, inverse_info


def _tabu_search(perm, pt, time_budget):
    """Tabu search with Or-opt moves."""
    n_jobs, n_machines = pt.shape
    if n_jobs <= 1:
        return perm, _makespan(perm, pt)
    
    best_perm = perm[:]
    best_ms = _makespan(best_perm, pt)
    current_perm = perm[:]
    current_ms = best_ms
    
    # Tabu list: stores (position_from, position_to, length) for recent moves
    tabu_list = []
    tabu_tenure = max(10, n_jobs // 2)
    
    # Track best
    global_best_perm = best_perm[:]
    global_best_ms = best_ms
    
    start_time = time.time()
    no_improve_count = 0
    max_no_improve = 500
    
    while (time.time() - start_time) < time_budget:
        # Generate neighborhood: Or-opt moves with length 1, 2, 3
        best_neighbor_perm = None
        best_neighbor_ms = float('inf')
        best_neighbor_move = None
        
        for length in [1, 2, 3]:
            for i in range(n_jobs - length + 1):
                if (time.time() - start_time) > time_budget:
                    break
                for new_pos in range(n_jobs - length + 1):
                    if new_pos == i:
                        continue
                    
                    new_perm, inverse_info = _or_opt_move(current_perm, i, length, new_pos)
                    if new_perm is None:
                        continue
                    
                    # Check if move is tabu
                    move_key = (i, new_pos, length)
                    is_tabu = False
                    for t_move in tabu_list:
                        if t_move == move_key:
                            is_tabu = True
                            break
                    
                    ms = _makespan(new_perm, pt)
                    
                    # Aspiration criterion: accept if it improves global best even if tabu
                    if ms < best_neighbor_ms and (not is_tabu or ms < global_best_ms):
                        best_neighbor_ms = ms
                        best_neighbor_perm = new_perm
                        best_neighbor_move = move_key
        
        if best_neighbor_perm is None:
            # No valid moves, break
            break
        
        # Update tabu list
        tabu_list.append(best_neighbor_move)
        if len(tabu_list) > tabu_tenure:
            tabu_list.pop(0)
        
        # Move to best neighbor
        current_perm = best_neighbor_perm
        current_ms = best_neighbor_ms
        
        # Update global best
        if current_ms < global_best_ms:
            global_best_ms = current_ms
            global_best_perm = current_perm[:]
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # Diversification: if stuck, perturb the solution
        if no_improve_count >= max_no_improve:
            no_improve_count = 0
            # Random perturbation: do 3-5 random Or-opt moves
            n_perturb = random.randint(3, 5)
            for _ in range(n_perturb):
                length = random.randint(1, 3)
                i = random.randint(0, n_jobs - length)
                new_pos = random.randint(0, n_jobs - length)
                if new_pos != i:
                    new_perm, _ = _or_opt_move(current_perm, i, length, new_pos)
                    if new_perm is not None:
                        current_perm = new_perm
                        current_ms = _makespan(current_perm, pt)
        
        # Check time
        if (time.time() - start_time) > time_budget:
            break
    
    return global_best_perm, global_best_ms


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on local synthetic flow-shop instances.
    """
    try:
        start_time = time.time()
        deadline = 145.0  # Safety margin before 150s limit
        
        pt = np.asarray(processing_times, dtype=np.float64)
        if pt.ndim != 2:
            if pt.size == 0:
                return []
            pt = pt.reshape(-1, 1)
            
        n_jobs, n_machines = pt.shape
        
        if n_jobs == 0:
            return []
        if n_jobs == 1:
            return [0]
        
        # 1. Initial heuristic: NEH
        perm = _neh_heuristic(pt)
        
        # 2. Tabu search with Or-opt moves
        # Allocate time: 80% for tabu search, 20% for additional local improvement
        tabu_time = 0.8 * (deadline - (time.time() - start_time))
        
        best_perm, best_ms = _tabu_search(perm, pt, tabu_time)
        
        # 3. Final local improvement pass with insertion moves
        remaining_time = deadline - (time.time() - start_time)
        if remaining_time > 1.0:
            improved = True
            while improved and (time.time() - start_time) < deadline:
                improved = False
                for i in range(n_jobs):
                    if (time.time() - start_time) > deadline:
                        break
                    job = best_perm[i]
                    temp_perm = best_perm[:i] + best_perm[i+1:]
                    for j in range(len(temp_perm) + 1):
                        if j == i:
                            continue
                        new_perm = temp_perm[:j] + [job] + temp_perm[j:]
                        ms = _makespan(new_perm, pt)
                        if ms < best_ms:
                            best_ms = ms
                            best_perm = new_perm
                            improved = True
                            break
                    if improved:
                        break
        
        return list(best_perm)
    except Exception:
        # Fallback: return jobs sorted by total processing time
        try:
            pt = np.asarray(processing_times, dtype=np.float64)
            if pt.ndim != 2:
                if pt.size == 0:
                    return []
                pt = pt.reshape(-1, 1)
            sums = pt.sum(axis=1)
            return list(np.argsort(-sums))
        except Exception:
            return []
# EVOLVE-BLOCK-END
