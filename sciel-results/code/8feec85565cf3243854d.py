"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _makespan(perm, P):
    """Compute makespan for a given permutation."""
    n_machines = P.shape[1]
    completion = np.zeros(n_machines)
    for job in perm:
        for m in range(n_machines):
            completion[m] += P[job, m]
    return completion[-1]

def _makespan_fast(perm, P, early_stop_best=None):
    """Compute makespan with early termination if possible."""
    n_machines = P.shape[1]
    completion = np.zeros(n_machines)
    for job in perm:
        for m in range(n_machines):
            completion[m] = (completion[m] if m == 0 else max(completion[m], completion[m-1])) + P[job, m]
            if early_stop_best is not None and completion[m] >= early_stop_best:
                return completion[m]
    return completion[-1]

def _neh_order(P, randomize=False):
    """
    NEH heuristic:
    1. Sort jobs by total processing time (descending).
    2. Insert each job into the position that minimizes the makespan.
    """
    n_jobs = P.shape[0]
    n_machines = P.shape[1]
    
    # Step 1: Sort by total processing time
    totals = P.sum(axis=1)
    if randomize:
        # Add small random perturbation
        totals = totals + np.random.uniform(0, 0.1 * totals.max(), n_jobs)
    sorted_jobs = np.argsort(-totals)
    
    # Step 2: Insertive construction
    perm = []
    for job in sorted_jobs:
        best_pos = 0
        best_mk = float('inf')
        
        # Try inserting at each position
        for pos in range(len(perm) + 1):
            new_perm = perm[:pos] + [job] + perm[pos:]
            mk = _makespan(new_perm, P)
            if mk < best_mk:
                best_mk = mk
                best_pos = pos
        
        perm.insert(best_pos, job)
    
    return perm

def _tabu_search(perm, P, time_limit=10.0):
    """
    Tabu Search using block-adjacent insertion neighborhood.
    Moves: relocate a single job or a block of up to 3 jobs to any other position.
    Dynamic tabu list to prevent cycling.
    """
    n_jobs = len(perm)
    n_machines = P.shape[1]
    current_perm = list(perm)
    
    start_time = time.time()
    
    # Tabu list: stores (from_pos, to_pos, block_len) with expiration
    tabu_list = []
    tabu_tenure = 10  # base tenure
    
    def calc_mk(p):
        return _makespan_fast(p, P)
    
    current_mk = calc_mk(current_perm)
    best_perm = list(current_perm)
    best_mk = current_mk
    
    no_improve_count = 0
    iteration = 0
    
    while True:
        if time.time() - start_time > time_limit:
            break
        
        iteration += 1
        
        # Dynamically adjust tabu tenure
        if no_improve_count > 50:
            tabu_tenure = max(5, 15 - no_improve_count // 20)
        else:
            tabu_tenure = 10
        
        # Clean expired tabu entries
        tabu_list = [(f, t, l, exp) for f, t, l, exp in tabu_list if exp > iteration]
        
        # Generate neighbors: try moving blocks of length 1-3
        best_neighbor_perm = None
        best_neighbor_mk = current_mk
        best_move = None
        best_is_tabu = False
        
        max_block = min(3, n_jobs)
        
        for block_len in range(1, max_block + 1):
            for i in range(n_jobs - block_len + 1):
                # Extract block
                block = current_perm[i:i + block_len]
                
                # Remove block
                remaining = current_perm[:i] + current_perm[i + block_len:]
                n_remaining = len(remaining)
                
                # Try inserting at each position
                for pos in range(n_remaining + 1):
                    if pos == i:
                        continue  # same position
                    
                    new_perm = remaining[:pos] + block + remaining[pos:]
                    mk = calc_mk(new_perm)
                    
                    # Check if this move is tabu
                    is_tabu = any(f == i and t == pos and l == block_len for f, t, l, exp in tabu_list)
                    
                    # Aspiration: accept tabu move if it improves best
                    if not is_tabu and mk < best_neighbor_mk:
                        best_neighbor_mk = mk
                        best_neighbor_perm = new_perm
                        best_move = (i, pos, block_len)
                        best_is_tabu = False
                    elif is_tabu and mk < best_mk:
                        # Aspiration criterion
                        if best_neighbor_perm is None or mk < best_neighbor_mk:
                            best_neighbor_mk = mk
                            best_neighbor_perm = new_perm
                            best_move = (i, pos, block_len)
                            best_is_tabu = True
        
        if best_neighbor_perm is None:
            # No improving move found; accept the best non-tabu move even if worse
            # Or restart with a perturbation
            if no_improve_count > 200:
                # Perturb: random swap of 2-4 jobs
                n_swap = random.randint(2, min(4, n_jobs))
                idx = random.sample(range(n_jobs), n_swap)
                for a in range(len(idx)):
                    for b in range(a + 1, len(idx)):
                        current_perm[idx[a]], current_perm[idx[b]] = current_perm[idx[b]], current_perm[idx[a]]
                current_mk = calc_mk(current_perm)
                no_improve_count = 0
                continue
            break
        
        # Apply best move
        current_perm = best_neighbor_perm
        current_mk = best_neighbor_mk
        
        # Add to tabu list
        if best_move is not None:
            i, pos, block_len = best_move
            # Reverse move: from pos to i (approximately)
            tabu_list.append((pos, i, block_len, iteration + tabu_tenure))
            # Also tabu the forward move to prevent immediate undo
            tabu_list.append((i, pos, block_len, iteration + tabu_tenure // 2))
        
        if current_mk < best_mk:
            best_mk = current_mk
            best_perm = list(current_perm)
            no_improve_count = 0
        else:
            no_improve_count += 1
    
    return best_perm, best_mk

def order_jobs(processing_times):
    P = np.asarray(processing_times)
    n_jobs = P.shape[0]
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    # Deadline guard
    total_time_limit = 140.0  # Leave 10s margin for 150s limit
    start_global = time.time()
    
    # 1. Generate initial solution using NEH
    best_perm = _neh_order(P)
    best_mk = _makespan(best_perm, P)
    
    # 2. Improve with Tabu Search
    elapsed = time.time() - start_global
    remaining_time = total_time_limit - elapsed - 5.0  # leave 5s margin
    
    if remaining_time > 1.0:
        best_perm, best_mk = _tabu_search(best_perm, P, time_limit=remaining_time)
    
    # 3. If time remains, do a final polish with insertion local search
    elapsed = time.time() - start_global
    remaining_time = total_time_limit - elapsed - 2.0
    
    if remaining_time > 1.0:
        # Simple first-improvement insertion local search
        current_perm = list(best_perm)
        current_mk = best_mk
        improved = True
        while improved and (time.time() - start_global) < (total_time_limit - 2.0):
            improved = False
            for i in range(n_jobs):
                job = current_perm[i]
                current_perm.pop(i)
                for pos in range(len(current_perm) + 1):
                    if pos == i:
                        continue
                    new_perm = current_perm[:pos] + [job] + current_perm[pos:]
                    mk = _makespan_fast(new_perm, P, early_stop_best=current_mk)
                    if mk < current_mk:
                        current_perm = new_perm
                        current_mk = mk
                        improved = True
                        break
                if not improved:
                    current_perm.insert(i, job)
        
        if current_mk < best_mk:
            best_perm = current_perm
            best_mk = current_mk
    
    return list(best_perm)

# EVOLVE-BLOCK-END
