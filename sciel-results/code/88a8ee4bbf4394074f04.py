"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _calc_makespan(processing_times, perm):
    """
    Calculate makespan for a given permutation using a fast list-based loop.
    This is typically faster than vectorized approaches for small M due to
    lower overhead and better cache locality for the small completion array.
    
    Args:
        processing_times: np.ndarray of shape (n_jobs, n_machines)
        perm: list of job indices in order
    
    Returns:
        makespan (float)
    """
    if len(perm) == 0:
        return 0.0
        
    n_machines = processing_times.shape[1]
    if n_machines == 1:
        return float(np.sum(processing_times[perm]))
    
    # Local variables for speed
    P = processing_times
    completion = [0.0] * n_machines
    
    for job in perm:
        row = P[job]
        prev_c = 0.0
        for m in range(n_machines):
            c_curr = completion[m]
            if c_curr < prev_c:
                c_curr = prev_c
            c_curr += row[m]
            completion[m] = c_curr
            prev_c = c_curr
            
    return completion[-1]

def _neh_heuristic(processing_times):
    """
    NEH heuristic: Sort jobs by total processing time descending,
    then insert each job into the position that minimizes makespan.
    """
    n_jobs, n_machines = processing_times.shape
    totals = processing_times.sum(axis=1)
    order = np.argsort(-totals)
    
    sequence = [order[0]]
    
    for i in range(1, n_jobs):
        job = order[i]
        best_pos = 0
        best_makespan = float('inf')
        
        for pos in range(len(sequence) + 1):
            new_seq = sequence[:pos] + [job] + sequence[pos:]
            ms = _calc_makespan(processing_times, new_seq)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        
        sequence.insert(best_pos, job)
    
    return sequence

def _v_shape_heuristic(processing_times):
    """
    V-shaped heuristic: 
    1. Sort jobs by total processing time descending.
    2. Split into two lists: first half (ascending) and second half (descending).
    3. Construct sequence by picking the best next job from the heads of these lists.
    This often yields good initial solutions for flow shop problems.
    """
    n_jobs, n_machines = processing_times.shape
    totals = processing_times.sum(axis=1)
    order = np.argsort(-totals)
    
    # Split into two halves
    mid = n_jobs // 2
    list1 = list(order[:mid])
    list2 = list(order[mid:][::-1])
    
    sequence = []
    while list1 or list2:
        best_job = -1
        best_ms = float('inf')
        
        # Try head of list1
        if list1:
            job = list1[0]
            new_seq = sequence + [job]
            ms = _calc_makespan(processing_times, new_seq)
            if ms < best_ms:
                best_ms = ms
                best_job = job
                best_source = 1
        
        # Try head of list2
        if list2:
            job = list2[0]
            new_seq = sequence + [job]
            ms = _calc_makespan(processing_times, new_seq)
            if ms < best_ms:
                best_ms = ms
                best_job = job
                best_source = 2
        
        if best_job == -1:
            break
            
        if best_source == 1:
            list1.pop(0)
        else:
            list2.pop(0)
            
        sequence.append(best_job)
        
    return sequence

def _tabu_search(processing_times, initial_seq, deadline):
    """
    Tabu Search with dynamic tabu list and diversification.
    """
    n_jobs = len(initial_seq)
    if n_jobs < 2:
        return list(initial_seq), _calc_makespan(processing_times, initial_seq)
    
    current_seq = list(initial_seq)
    current_ms = _calc_makespan(processing_times, current_seq)
    
    best_seq = list(current_seq)
    best_ms = current_ms
    
    # Dynamic tabu tenure
    base_tenure = max(1, n_jobs // 5)
    tabu_tenure = base_tenure
    tabu_list = {}  # key: (job, from_pos, to_pos) -> iteration when it becomes non-tabu
    
    # Aspiration criterion: accept tabu move if it improves global best
    
    no_improve_count = 0
    max_no_improve = 200
    
    while time.time() < deadline:
        # Generate neighbors: move a job to a different position
        # To speed up, we can limit the search or use random sampling if n_jobs is large
        # But for n_jobs <= 100, full neighborhood is O(N^2) which is 10k moves.
        # Each move evaluation is O(N*M). Total O(N^3*M).
        # For N=100, M=5, that's 500k ops per iteration. Python can do ~10M ops/s.
        # So ~20-50 iterations/sec. In 100s, we can do ~2000-5000 iterations.
        # This is reasonable.
        
        best_neighbor_seq = None
        best_neighbor_ms = float('inf')
        best_neighbor_key = None
        
        # Randomly sample a subset of moves if n_jobs is large to save time?
        # Let's stick to full neighborhood for now but optimize the loop.
        
        for i in range(n_jobs):
            if time.time() > deadline:
                break
            job = current_seq[i]
            for j in range(n_jobs):
                if j == i:
                    continue
                
                # Check tabu status
                key = (job, i, j)
                # Note: The key depends on the current state. 
                # A better tabu key is often (job, new_position) or (job, removed_from, inserted_to).
                # Since the permutation changes, (job, i, j) is unique to this specific move in this specific state.
                # However, to prevent cycling back, we need to remember that we *did* this move.
                # Standard approach: Tabu list stores moves that are forbidden.
                # If we just did move (job, i, j), we forbid the reverse move (job, j, i) in the next iterations?
                # Or we forbid the exact same move (job, i, j) for T iterations.
                # Let's use (job, i, j) as the key. If it's in tabu_list and not expired, skip.
                
                if key in tabu_list:
                    if tabu_list[key] > 0:
                        # Decrement counter
                        tabu_list[key] -= 1
                        # Check aspiration: if this move leads to a better solution than best_ms, accept anyway
                        # We don't know the ms yet, so we can't check aspiration here without evaluating.
                        # To save time, we skip tabu moves unless we are stuck?
                        # Or we evaluate and then check aspiration.
                        # Let's evaluate first, then check tabu/aspiration.
                        pass # Fall through to evaluation
                    
                # Generate new sequence
                new_seq = list(current_seq)
                new_seq.pop(i)
                # Adjust j if i < j
                insert_idx = j - 1 if j > i else j
                new_seq.insert(insert_idx, job)
                
                new_ms = _calc_makespan(processing_times, new_seq)
                
                # Check if this move is tabu
                is_tabu = False
                if key in tabu_list:
                    if tabu_list[key] > 0:
                        is_tabu = True
                        tabu_list[key] -= 1 # Decrement for next time? No, we should decrement at the start of the iteration for all keys?
                        # Actually, better to manage tabu list by iteration count.
                        # Let's use a simpler mechanism:
                        # tabu_list[key] = current_iteration + tenure
                        # If current_iteration < tabu_list[key], it's tabu.
                        pass # We need a better tabu implementation.
        
        # Let's re-implement tabu search with a cleaner structure.
        break # Exit to prevent infinite loop in this draft
    
    # Re-implementation of Tabu Search core loop
    current_seq = list(initial_seq)
    current_ms = _calc_makespan(processing_times, current_seq)
    best_seq = list(current_seq)
    best_ms = current_ms
    
    tabu_list = {}
    current_iter = 0
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline:
        best_neighbor_seq = None
        best_neighbor_ms = float('inf')
        best_neighbor_key = None
        best_neighbor_is_tabu = False
        
        for i in range(n_jobs):
            if time.time() > deadline:
                break
            job = current_seq[i]
            for j in range(n_jobs):
                if j == i:
                    continue
                
                key = (job, i, j)
                
                # Generate new sequence
                new_seq = list(current_seq)
                new_seq.pop(i)
                insert_idx = j - 1 if j > i else j
                new_seq.insert(insert_idx, job)
                
                new_ms = _calc_makespan(processing_times, new_seq)
                
                # Check tabu
                is_tabu = False
                if key in tabu_list:
                    if current_iter < tabu_list[key]:
                        is_tabu = True
                
                # Aspiration criterion
                if is_tabu and new_ms < best_ms:
                    is_tabu = False
                
                if not is_tabu and new_ms < best_neighbor_ms:
                    best_neighbor_ms = new_ms
                    best_neighbor_seq = new_seq
                    best_neighbor_key = key
                    best_neighbor_is_tabu = False
        
        if best_neighbor_seq is None:
            break
            
        # Accept best neighbor
        current_seq = best_neighbor_seq
        current_ms = best_neighbor_ms
        
        # Update tabu list
        if best_neighbor_key:
            # Dynamic tenure
            if current_ms < best_ms:
                # Improvement, keep tenure same or slightly reduce?
                tabu_list[best_neighbor_key] = current_iter + base_tenure
            else:
                # Non-improvement, increase tenure to encourage exploration
                tabu_list[best_neighbor_key] = current_iter + base_tenure + 2
            
        if current_ms < best_ms:
            best_ms = current_ms
            best_seq = list(current_seq)
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        # Diversification: if stuck, perturb
        if no_improve_count > max_no_improve:
            # Randomly swap 2 jobs
            i, j = random.sample(range(n_jobs), 2)
            current_seq[i], current_seq[j] = current_seq[j], current_seq[i]
            current_ms = _calc_makespan(processing_times, current_seq)
            no_improve_count = 0
            # Clear tabu list to escape deep local minima
            tabu_list.clear()
            
        current_iter += 1
        
    return best_seq, best_ms

def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # 1. Generate diverse initial candidates
    candidates = []
    
    # NEH
    if time.time() < deadline - 5:
        candidates.append(_neh_heuristic(processing_times))
        
    # V-shaped
    if time.time() < deadline - 5:
        candidates.append(_v_shape_heuristic(processing_times))
        
    # Random
    if time.time() < deadline - 5:
        rand_perm = list(range(n_jobs))
        random.shuffle(rand_perm)
        candidates.append(rand_perm)
        
    # Reversed NEH
    if time.time() < deadline - 5:
        neh = _neh_heuristic(processing_times)
        candidates.append(neh[::-1])
        
    # Evaluate candidates
    best_seq = None
    best_ms = float('inf')
    
    for cand in candidates:
        ms = _calc_makespan(processing_times, cand)
        if ms < best_ms:
            best_ms = ms
            best_seq = cand
            
    if best_seq is None:
        best_seq = list(range(n_jobs))
        best_ms = _calc_makespan(processing_times, best_seq)
        
    # 2. Tabu Search
    if time.time() < deadline - 10:
        ts_deadline = min(deadline - 5, time.time() + 100)
        ts_seq, ts_ms = _tabu_search(processing_times, best_seq, ts_deadline)
        if ts_ms < best_ms:
            best_seq = ts_seq
            best_ms = ts_ms
            
    return best_seq
# EVOLVE-BLOCK-END
