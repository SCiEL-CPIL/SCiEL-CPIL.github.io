"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random


def _makespan(perm, processing_times):
    """Compute makespan for a given permutation using vectorized numpy."""
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    if n_jobs == 0:
        return 0.0
    if n_jobs == 1:
        return float(processing_times[perm[0]].sum())
    
    # Vectorized computation using numpy
    # For each job in order, compute completion times
    # C[j][m] = max(C[j-1][m], C[j][m-1]) + p[j][m]
    # We can compute this row by row
    prev = np.zeros(n_machines)
    for j in range(n_jobs):
        pt = processing_times[perm[j]]
        cur = np.empty(n_machines)
        cur[0] = prev[0] + pt[0]
        if n_machines > 1:
            # cur[m] = max(prev[m], cur[m-1]) + pt[m]
            # This is a sequential dependency, but we can use numpy for the max
            for m in range(1, n_machines):
                cur[m] = max(prev[m], cur[m-1]) + pt[m]
        prev = cur
    return float(prev[-1])


def _makespan_fast(perm, processing_times):
    """Fast makespan using pre-extracted processing times."""
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    if n_jobs == 0:
        return 0.0
    if n_jobs == 1:
        return float(processing_times[perm[0]].sum())
    
    prev = np.zeros(n_machines)
    for j in range(n_jobs):
        pt = processing_times[perm[j]]
        cur = np.empty(n_machines)
        cur[0] = prev[0] + pt[0]
        if n_machines > 1:
            for m in range(1, n_machines):
                cur[m] = max(prev[m], cur[m-1]) + pt[m]
        prev = cur
    return float(prev[-1])


def _insert_best_position(job, partial, processing_times):
    """Insert job into partial list at the position minimizing makespan."""
    n_partial = len(partial)
    best_pos = 0
    best_ms = float('inf')
    
    for pos in range(n_partial + 1):
        candidate = partial[:pos] + [job] + partial[pos:]
        ms = _makespan(candidate, processing_times)
        if ms < best_ms:
            best_ms = ms
            best_pos = pos
    
    return partial[:best_pos] + [job] + partial[best_pos:], best_ms


def _neh_order(processing_times, rng=None):
    """Generate NEH heuristic ordering with optional randomization."""
    if rng is None:
        rng = random.Random()
    
    n_jobs, n_machines = processing_times.shape
    totals = processing_times.sum(axis=1)
    
    # Sort by total processing time descending
    order = list(np.argsort(-totals))
    
    # Add slight randomization to break ties and diversify
    if n_jobs > 2:
        for _ in range(min(5, n_jobs // 3)):
            i = rng.randint(0, n_jobs - 2)
            if rng.random() < 0.4:
                order[i], order[i+1] = order[i+1], order[i]
    
    partial = [order[0]]
    for i in range(1, n_jobs):
        job = order[i]
        partial, _ = _insert_best_position(job, partial, processing_times)
    
    return partial


def _neh_order_forward_backward(processing_times, rng=None):
    """NEH with forward and backward passes."""
    if rng is None:
        rng = random.Random()
    
    n_jobs, n_machines = processing_times.shape
    totals = processing_times.sum(axis=1)
    order = list(np.argsort(-totals))
    
    # Forward pass
    partial = [order[0]]
    for i in range(1, n_jobs):
        job = order[i]
        partial, _ = _insert_best_position(job, partial, processing_times)
    
    # Backward pass: remove each job and reinsert at best position
    result = list(partial)
    for i in range(n_jobs - 1, 0, -1):
        job = result[i]
        remaining = result[:i] + result[i+1:]
        result, _ = _insert_best_position(job, remaining, processing_times)
    
    return result


def _swap_adjacent(perm, i):
    """Create new permutation with adjacent swap at position i."""
    new_perm = list(perm)
    new_perm[i], new_perm[i+1] = new_perm[i+1], new_perm[i]
    return new_perm


def _move_job(perm, job, new_pos):
    """Move job to new position."""
    perm = [j for j in perm if j != job]
    return perm[:new_pos] + [job] + perm[new_pos:]


def _local_search(perm, processing_times, max_iterations=200, deadline=None):
    """
    Local search with adjacent swap and insert moves.
    Uses best-improvement for swaps and first-improvement for inserts.
    """
    n_jobs = len(perm)
    if n_jobs <= 1:
        return list(perm)
    
    best_perm = list(perm)
    best_ms = _makespan(best_perm, processing_times)
    
    for it in range(max_iterations):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        
        # 1. Adjacent Swaps (Best Improvement)
        best_swap_pos = -1
        best_swap_ms = best_ms
        for i in range(n_jobs - 1):
            new_perm = _swap_adjacent(best_perm, i)
            ms = _makespan(new_perm, processing_times)
            if ms < best_swap_ms - 1e-10:
                best_swap_ms = ms
                best_swap_pos = i
        
        if best_swap_pos != -1:
            best_perm = _swap_adjacent(best_perm, best_swap_pos)
            best_ms = best_swap_ms
            improved = True
        
        # 2. Insert Moves (First Improvement)
        found_improvement = False
        for i in range(n_jobs):
            if deadline is not None and time.time() > deadline:
                break
            job = best_perm[i]
            for pos in range(n_jobs):
                if pos == i or pos == i + 1:
                    continue
                new_perm = _move_job(best_perm, job, pos)
                ms = _makespan(new_perm, processing_times)
                if ms < best_ms - 1e-10:
                    best_perm = new_perm
                    best_ms = ms
                    improved = True
                    found_improvement = True
                    break
            if found_improvement:
                break
        
        if not improved:
            break
    
    return best_perm


def _tabu_search(perm, processing_times, start_time, deadline, rng=None):
    """
    Tabu search with adjacent swap and insert moves.
    Uses a tabu list to prevent cycling and intensifies around best solution.
    """
    if rng is None:
        rng = random.Random()
    
    n_jobs = len(perm)
    if n_jobs <= 1:
        return list(perm)
    
    current_perm = list(perm)
    current_ms = _makespan(current_perm, processing_times)
    best_perm = list(current_perm)
    best_ms = current_ms
    
    # Tabu list: store (job, from_pos, to_pos) or (swap_pos)
    tabu_list = []
    tabu_tenure = max(10, n_jobs // 2)
    
    # Diversification: occasionally jump to a random neighborhood
    diversify_counter = 0
    diversify_interval = max(50, n_jobs * 2)
    
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        if iteration % 1000 == 0:
            if time.time() > deadline:
                break
        
        # Generate neighbor
        move_type = rng.random()
        if move_type < 0.5:
            # Adjacent swap
            i = rng.randint(0, n_jobs - 2)
            new_perm = _swap_adjacent(current_perm, i)
            move_key = ('swap', i)
        else:
            # Insert move
            i = rng.randint(0, n_jobs - 1)
            j = rng.randint(0, n_jobs - 1)
            if j == i or j == i + 1:
                continue
            job = current_perm[i]
            new_perm = _move_job(current_perm, job, j)
            move_key = ('insert', job, i, j)
        
        new_ms = _makespan(new_perm, processing_times)
        
        # Check tabu status
        is_tabu = False
        if tabu_list:
            # Check if the reverse move is in tabu list
            if move_type < 0.5:
                # For swap, the reverse is the same swap
                for (t_type, *args) in tabu_list:
                    if t_type == 'swap' and args[0] == i:
                        is_tabu = True
                        break
            else:
                # For insert, the reverse move is moving the job back
                for (t_type, *args) in tabu_list:
                    if t_type == 'insert' and args[0] == job and args[2] == j and args[3] == i:
                        is_tabu = True
                        break
        
        # Aspiration: accept tabu move if it improves best
        if new_ms < best_ms - 1e-10:
            is_tabu = False
        
        if not is_tabu and (new_ms < current_ms or rng.random() < 0.1):
            # Accept move
            current_perm = new_perm
            current_ms = new_ms
            
            if current_ms < best_ms:
                best_perm = list(current_perm)
                best_ms = current_ms
                # Clear tabu list on improvement
                tabu_list = []
            
            # Add move to tabu list
            tabu_list.append(move_key)
            if len(tabu_list) > tabu_tenure:
                tabu_list.pop(0)
            
            diversify_counter += 1
        else:
            # Reject move, add to tabu to avoid trying again soon
            tabu_list.append(move_key)
            if len(tabu_list) > tabu_tenure:
                tabu_list.pop(0)
        
        # Diversification: periodically perturb the current solution
        if diversify_counter % diversify_interval == 0 and diversify_counter > 0:
            # Perform a random 2-opt reversal
            if n_jobs >= 3:
                i = rng.randint(0, n_jobs - 3)
                j = rng.randint(i + 1, n_jobs - 1)
                current_perm = current_perm[:i] + current_perm[i:j+1][::-1] + current_perm[j+1:]
                current_ms = _makespan(current_perm, processing_times)
                diversify_counter = 0
    
    return best_perm


def _simulated_annealing(perm, processing_times, start_time, deadline, rng=None):
    """
    Simulated annealing with adjacent swap and insert moves.
    Uses adaptive cooling based on acceptance rate.
    """
    if rng is None:
        rng = random.Random()
    
    n_jobs = len(perm)
    if n_jobs <= 1:
        return list(perm)
    
    current_perm = list(perm)
    current_ms = _makespan(current_perm, processing_times)
    best_perm = list(current_perm)
    best_ms = current_ms
    
    # Adaptive temperature
    initial_temp = max(current_ms * 0.05, 1.0)
    temp = initial_temp
    min_temp = 1e-6
    
    # Track acceptance rate for adaptive cooling
    accept_count = 0
    total_count = 0
    window_size = 100
    
    while time.time() < deadline:
        if temp < min_temp:
            break
        
        # Generate neighbor
        move_type = rng.random()
        if move_type < 0.5:
            # Adjacent swap
            i = rng.randint(0, n_jobs - 2)
            new_perm = _swap_adjacent(current_perm, i)
        else:
            # Insert move
            i = rng.randint(0, n_jobs - 1)
            j = rng.randint(0, n_jobs - 1)
            if j == i or j == i + 1:
                continue
            job = current_perm[i]
            new_perm = _move_job(current_perm, job, j)
        
        new_ms = _makespan(new_perm, processing_times)
        delta = new_ms - current_ms
        
        total_count += 1
        if delta < 0 or rng.random() < math.exp(-delta / temp):
            accept_count += 1
            current_perm = new_perm
            current_ms = new_ms
            
            if current_ms < best_ms:
                best_perm = list(current_perm)
                best_ms = current_ms
        
        # Adaptive cooling based on acceptance rate
        if total_count >= window_size:
            accept_rate = accept_count / total_count
            if accept_rate > 0.8:
                # Too many acceptances, cool faster
                temp *= 0.95
            elif accept_rate < 0.2:
                # Too few acceptances, reheat slightly
                temp = max(temp * 1.1, min_temp)
            else:
                # Good balance, standard cooling
                temp *= 0.99
            accept_count = 0
            total_count = 0
    
    return best_perm


def _final_polish(perm, processing_times, deadline):
    """
    High-precision polish: multiple passes of local search with tighter tolerance.
    """
    best_perm = list(perm)
    best_ms = _makespan(best_perm, processing_times)
    
    # Multiple passes with decreasing step sizes
    for pass_num in range(3):
        if time.time() > deadline:
            break
        
        # Pass 1: Fine-tune with adjacent swaps only
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(best_perm) - 1):
                new_perm = _swap_adjacent(best_perm, i)
                ms = _makespan(new_perm, processing_times)
                if ms < best_ms - 1e-12:
                    best_perm = new_perm
                    best_ms = ms
                    improved = True
        
        # Pass 2: Insert moves with tighter tolerance
        if time.time() < deadline:
            improved = True
            while improved and time.time() < deadline:
                improved = False
                for i in range(len(best_perm)):
                    if time.time() > deadline:
                        break
                    job = best_perm[i]
                    for pos in range(len(best_perm)):
                        if pos == i or pos == i + 1:
                            continue
                        new_perm = _move_job(best_perm, job, pos)
                        ms = _makespan(new_perm, processing_times)
                        if ms < best_ms - 1e-12:
                            best_perm = new_perm
                            best_ms = ms
                            improved = True
                            break
                    if improved:
                        break
    
    return best_perm


def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin: 10s before 150s limit
    
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    rng = random.Random(42)
    
    candidates = []
    
    # Generate multiple diverse initial solutions
    lpt_perm = list(np.argsort(-processing_times.sum(axis=1)))
    candidates.append(lpt_perm)
    
    if time.time() < deadline:
        neh_perm = _neh_order(processing_times, rng)
        candidates.append(neh_perm)
    
    if time.time() < deadline:
        neh2_perm = _neh_order(processing_times, random.Random(rng.randint(0, 1000000)))
        candidates.append(neh2_perm)
    
    if time.time() < deadline:
        neh3_perm = _neh_order(processing_times, random.Random(rng.randint(0, 1000000)))
        candidates.append(neh3_perm)
    
    if time.time() < deadline:
        neh_fb_perm = _neh_order_forward_backward(processing_times, rng)
        candidates.append(neh_fb_perm)
    
    # Select best initial solution
    best_perm = None
    best_ms = float('inf')
    for perm in candidates:
        ms = _makespan(perm, processing_times)
        if ms < best_ms:
            best_ms = ms
            best_perm = list(perm)
    
    # Phase 1: Local search polish
    if time.time() < deadline and n_jobs > 1:
        remaining = deadline - time.time()
        if n_jobs <= 30:
            max_iter = 500
        elif n_jobs <= 50:
            max_iter = 300
        elif n_jobs <= 100:
            max_iter = 150
        else:
            max_iter = 100
        
        polished = _local_search(best_perm, processing_times, max_iterations=max_iter, deadline=deadline)
        polished_ms = _makespan(polished, processing_times)
        if polished_ms < best_ms:
            best_perm = polished
            best_ms = polished_ms
    
    # Phase 2: Tabu search for global exploration
    if time.time() < deadline - 20.0 and n_jobs > 1:
        tabu_deadline = deadline - 20.0
        tabu_perm = _tabu_search(best_perm, processing_times, start_time, tabu_deadline, rng)
        tabu_ms = _makespan(tabu_perm, processing_times)
        if tabu_ms < best_ms:
            best_perm = tabu_perm
            best_ms = tabu_ms
    
    # Phase 3: Simulated annealing for further refinement
    if time.time() < deadline - 10.0 and n_jobs > 1:
        sa_deadline = deadline - 10.0
        sa_perm = _simulated_annealing(best_perm, processing_times, start_time, sa_deadline, rng)
        sa_ms = _makespan(sa_perm, processing_times)
        if sa_ms < best_ms:
            best_perm = sa_perm
            best_ms = sa_ms
    
    # Phase 4: Final high-precision polish
    if time.time() < deadline - 2.0 and n_jobs > 1:
        final_perm = _final_polish(best_perm, processing_times, deadline - 2.0)
        final_ms = _makespan(final_perm, processing_times)
        if final_ms < best_ms:
            best_perm = final_perm
            best_ms = final_ms
    
    return list(best_perm)
# EVOLVE-BLOCK-END
