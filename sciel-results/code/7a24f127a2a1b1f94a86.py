"""Flow-shop makespan minimizer: NEH + CIG constructive, multi-start with insertion local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(processing_times, order):
    """Compute makespan for a given job order efficiently."""
    if len(order) == 0:
        return 0.0
    n_jobs = len(order)
    n_machines = processing_times.shape[1]
    
    prev_C = np.zeros(n_machines)
    for job_idx in order:
        curr_C = np.zeros(n_machines)
        for j in range(n_machines):
            if j == 0:
                curr_C[j] = prev_C[j] + processing_times[job_idx, j]
            else:
                curr_C[j] = max(curr_C[j-1], prev_C[j]) + processing_times[job_idx, j]
        prev_C = curr_C
    return prev_C[-1]


def _makespan_fast(processing_times, order):
    """Faster makespan using numpy vectorization where possible."""
    if len(order) == 0:
        return 0.0
    n_machines = processing_times.shape[1]
    
    prev_C = 0.0
    for job_idx in order:
        p = processing_times[job_idx]
        curr_C = np.maximum.accumulate(p + prev_C)
        prev_C = curr_C[-1]
    return float(prev_C)


def _makespan_row(perm, pt):
    """Compute makespan using row-wise accumulation (alternative fast method)."""
    n = len(perm)
    m = pt.shape[1]
    if m == 0 or n == 0:
        return 0.0
    
    prev = np.zeros(n)
    s = 0.0
    for i in range(n):
        s += pt[perm[i], 0]
        prev[i] = s
    
    for k in range(1, m):
        curr = np.zeros(n)
        curr[0] = prev[0] + pt[perm[0], k]
        for i in range(1, n):
            v1 = prev[i]
            v2 = curr[i-1]
            curr[i] = (v1 if v1 > v2 else v2) + pt[perm[i], k]
        prev = curr
    
    return prev[-1]


def _neh(processing_times):
    """NEH heuristic: insert jobs one by one in order of decreasing total processing time."""
    n_jobs, n_machines = processing_times.shape
    totals = processing_times.sum(axis=1)
    order_sorted = np.argsort(-totals).tolist()
    
    current_order = [order_sorted[0]]
    
    for k in range(1, n_jobs):
        job = order_sorted[k]
        best_pos = 0
        best_ms = _makespan(processing_times, current_order + [job])
        
        for pos in range(1, len(current_order) + 1):
            candidate = current_order[:pos] + [job] + current_order[pos:]
            ms = _makespan(processing_times, candidate)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        
        current_order = current_order[:best_pos] + [job] + current_order[best_pos:]
    
    return current_order


def _neh_variant(processing_times, sort_key='sum'):
    """NEH with different sorting keys."""
    n_jobs, n_machines = processing_times.shape
    
    if sort_key == 'sum':
        totals = processing_times.sum(axis=1)
    elif sort_key == 'mean':
        totals = processing_times.mean(axis=1)
    elif sort_key == 'min':
        totals = processing_times.min(axis=1)
    elif sort_key == 'max':
        totals = processing_times.max(axis=1)
    else:
        totals = processing_times.sum(axis=1)
    
    order_sorted = np.argsort(-totals).tolist()
    
    current_order = [order_sorted[0]]
    
    for k in range(1, n_jobs):
        job = order_sorted[k]
        best_pos = 0
        best_ms = _makespan(processing_times, current_order + [job])
        
        for pos in range(1, len(current_order) + 1):
            candidate = current_order[:pos] + [job] + current_order[pos:]
            ms = _makespan(processing_times, candidate)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        
        current_order = current_order[:best_pos] + [job] + current_order[best_pos:]
    
    return current_order


def _local_search_insertion(processing_times, order, max_iters=100, time_limit=None, start_time=None):
    """Local search with insertion moves: remove a job and reinsert at best position."""
    n_jobs = len(order)
    if n_jobs <= 1:
        return order
    
    best_order = list(order)
    best_ms = _makespan(processing_times, best_order)
    no_improve_count = 0
    
    for _ in range(max_iters):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit - 5:
                break
        
        improved = False
        for i in range(n_jobs):
            # Remove job at position i
            job = best_order[i]
            remaining = best_order[:i] + best_order[i+1:]
            
            # Try inserting at each position in remaining
            for pos in range(len(remaining) + 1):
                if pos == i:
                    continue  # Same position, skip
                candidate = remaining[:pos] + [job] + remaining[pos:]
                ms = _makespan(processing_times, candidate)
                if ms < best_ms:
                    best_ms = ms
                    best_order = candidate
                    improved = True
                    no_improve_count = 0
                    break
            if improved:
                break
        
        if not improved:
            no_improve_count += 1
            if no_improve_count >= 3:
                break
    
    return best_order


def _local_search_2opt(processing_times, order, max_iters=50, time_limit=None, start_time=None):
    """2-opt local search: reverse a segment of the permutation."""
    n_jobs = len(order)
    if n_jobs <= 2:
        return order
    
    best_order = list(order)
    best_ms = _makespan(processing_times, best_order)
    no_improve_count = 0
    
    for _ in range(max_iters):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit - 5:
                break
        
        improved = False
        for i in range(n_jobs - 1):
            for j in range(i + 1, n_jobs):
                new_order = best_order[:]
                new_order[i:j+1] = new_order[i:j+1][::-1]
                ms = _makespan(processing_times, new_order)
                if ms < best_ms:
                    best_ms = ms
                    best_order = new_order
                    improved = True
                    no_improve_count = 0
                    break
            if improved:
                break
        
        if not improved:
            no_improve_count += 1
            if no_improve_count >= 3:
                break
    
    return best_order


def _tabu_search(processing_times, order, deadline, tabu_tenure=15, max_iters=2000):
    """Tabu search with insertion moves for flow shop."""
    n_jobs = len(order)
    if n_jobs <= 1:
        return order
    
    best_order = list(order)
    best_ms = _makespan(processing_times, best_order)
    current_order = list(order)
    current_ms = best_ms
    
    # Tabu list: store (i, j) moves
    tabu_list = {}
    tabu_iter = 0
    
    for it in range(max_iters):
        if time.time() >= deadline:
            break
        
        if tabu_iter % 500 == 0 and tabu_iter > 0:
            tabu_list = {k: v for k, v in tabu_list.items() if v > tabu_iter - tabu_tenure}
        
        best_move = None
        best_move_ms = float('inf')
        
        # Generate all insertion moves
        for i in range(n_jobs):
            if time.time() >= deadline:
                return best_order
            job = current_order[i]
            remaining = current_order[:i] + current_order[i+1:]
            
            for pos in range(len(remaining) + 1):
                if pos == i:
                    continue
                new_order = remaining[:pos] + [job] + remaining[pos:]
                ms = _makespan(processing_times, new_order)
                
                move_key = (i, pos)
                is_tabu = tabu_list.get(move_key, -1) >= tabu_iter - tabu_tenure
                
                # Aspiration: accept tabu move if it improves global best
                if is_tabu and ms < best_ms - 1e-9:
                    is_tabu = False
                
                if not is_tabu:
                    if ms < best_move_ms:
                        best_move_ms = ms
                        best_move = (i, pos)
        
        if best_move is None:
            break
        
        # Apply best move
        i, pos = best_move
        job = current_order[i]
        remaining = current_order[:i] + current_order[i+1:]
        current_order = remaining[:pos] + [job] + remaining[pos:]
        current_ms = best_move_ms
        
        # Update tabu list
        tabu_list[(i, pos)] = tabu_iter
        reverse_pos = pos if pos > i else pos
        tabu_list[(pos, i)] = tabu_iter
        
        tabu_iter += 1
        
        if current_ms < best_ms - 1e-9:
            best_ms = current_ms
            best_order = current_order[:]
    
    return best_order


def _simulated_annealing(processing_times, order, deadline, n_restarts=3):
    """Simulated annealing for flow shop to escape local optima."""
    n_jobs = len(order)
    if n_jobs <= 1:
        return order
    
    best_order = list(order)
    best_ms = _makespan(processing_times, best_order)
    
    for restart in range(n_restarts):
        if time.time() >= deadline:
            break
        
        if restart == 0:
            current_order = best_order[:]
        else:
            # Perturb from best solution
            current_order = best_order[:]
            for _ in range(min(5, n_jobs // 2)):
                i, j = np.random.choice(n_jobs, 2, replace=False)
                current_order[i], current_order[j] = current_order[j], current_order[i]
        
        current_ms = _makespan(processing_times, current_order)
        
        # SA parameters
        T = current_ms * 0.1
        cooling_rate = 0.95
        max_steps = 500
        
        for step in range(max_steps):
            if time.time() >= deadline:
                return best_order
            
            # Generate neighbor by random insertion move
            i = np.random.randint(n_jobs)
            job = current_order[i]
            remaining = current_order[:i] + current_order[i+1:]
            pos = np.random.randint(len(remaining) + 1)
            if pos == i:
                continue
            new_order = remaining[:pos] + [job] + remaining[pos:]
            new_ms = _makespan(processing_times, new_order)
            
            delta = new_ms - current_ms
            if delta < 0 or np.random.random() < np.exp(-delta / (T + 1e-9)):
                current_order = new_order
                current_ms = new_ms
                if current_ms < best_ms:
                    best_ms = current_ms
                    best_order = current_order[:]
            
            T *= cooling_rate
    
    return best_order


def _multi_start(processing_times, time_limit, start_time):
    """Multi-start with NEH variants, LPT, SPT, and random initializations."""
    n_jobs, n_machines = processing_times.shape
    
    best_order = None
    best_ms = float('inf')
    
    # Try multiple initial heuristics
    heuristics = []
    
    # NEH with sum
    h1 = _neh(processing_times)
    heuristics.append(h1)
    
    # NEH with mean
    h2 = _neh_variant(processing_times, 'mean')
    heuristics.append(h2)
    
    # NEH with max
    h3 = _neh_variant(processing_times, 'max')
    heuristics.append(h3)
    
    # LPT (Longest Processing Time)
    totals = processing_times.sum(axis=1)
    lpt_order = np.argsort(-totals).tolist()
    heuristics.append(lpt_order)
    
    # SPT (Shortest Processing Time)
    spt_order = np.argsort(totals).tolist()
    heuristics.append(spt_order)
    
    # Random
    random_order = list(np.random.permutation(n_jobs))
    heuristics.append(random_order)
    
    # For each heuristic, run local search and keep the best
    for h in heuristics:
        elapsed = time.time() - start_time
        if elapsed > time_limit - 10:
            break
        
        # Run insertion local search
        order = _local_search_insertion(processing_times, h, max_iters=50, time_limit=time_limit, start_time=start_time)
        ms = _makespan(processing_times, order)
        
        if ms < best_ms:
            best_ms = ms
            best_order = list(order)
    
    return best_order, best_ms


def order_jobs(processing_times):
    """Construct a job permutation minimizing makespan on flow-shop instances."""
    start_time = time.time()
    time_limit = 115.0  # Safety margin before 120s limit
    
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Multi-start with local search refinement
    best_order, best_ms = _multi_start(processing_times, time_limit, start_time)
    
    # If we still have time, do another round of insertion local search
    elapsed = time.time() - start_time
    if elapsed < time_limit - 5:
        order = _local_search_insertion(processing_times, best_order, max_iters=50, time_limit=time_limit, start_time=start_time)
        ms = _makespan(processing_times, order)
        if ms < best_ms:
            best_ms = ms
            best_order = list(order)
    
    # 2-opt refinement if time allows
    elapsed = time.time() - start_time
    if elapsed < time_limit - 5:
        order = _local_search_2opt(processing_times, best_order, max_iters=30, time_limit=time_limit, start_time=start_time)
        ms = _makespan(processing_times, order)
        if ms < best_ms:
            best_ms = ms
            best_order = list(order)
    
    # Tabu search if time allows
    elapsed = time.time() - start_time
    if elapsed < time_limit - 10:
        deadline = start_time + time_limit - 5
        order = _tabu_search(processing_times, best_order, deadline, tabu_tenure=15, max_iters=1500)
        ms = _makespan(processing_times, order)
        if ms < best_ms:
            best_ms = ms
            best_order = list(order)
    
    # Simulated annealing if time allows
    elapsed = time.time() - start_time
    if elapsed < time_limit - 10:
        deadline = start_time + time_limit - 5
        order = _simulated_annealing(processing_times, best_order, deadline, n_restarts=3)
        ms = _makespan(processing_times, order)
        if ms < best_ms:
            best_ms = ms
            best_order = list(order)
    
    return best_order
# EVOLVE-BLOCK-END
