"""Flow-shop permutation via Iterated Local Search with Variable Neighborhood Descent."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _makespan_fast(pt_list, perm):
    """Fast makespan calculation using pure Python lists."""
    n_machines = len(pt_list[0]) if pt_list else 0
    n = len(perm)
    if n == 0:
        return 0.0
    if n_machines == 0:
        return 0.0

    prev = [0.0] * n_machines

    for job in perm:
        row = pt_list[job]
        curr = [0.0] * n_machines
        curr[0] = prev[0] + row[0]
        for m in range(1, n_machines):
            val = prev[m] if prev[m] > curr[m-1] else curr[m-1]
            curr[m] = val + row[m]
        prev = curr
    return prev[-1]


def _neh_initial(pt):
    """NEH heuristic: Non-Equidistant Heuristic."""
    n_jobs, n_machines = pt.shape
    totals = pt.sum(axis=1)
    order = list(np.argsort(-totals))
    perm = [order[0]]
    pt_list = pt.tolist()
    
    for i in range(1, n_jobs):
        job = order[i]
        best_pos = 0
        best_ms = float('inf')
        # Try inserting job at all positions in current perm
        for pos in range(len(perm) + 1):
            test = perm[:pos] + [job] + perm[pos:]
            ms = _makespan_fast(pt_list, test)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm.insert(best_pos, job)
    return perm


def _palmer_initial(pt):
    """Palmer's dispatching rule."""
    n_jobs, n_machines = pt.shape
    n = n_machines
    scores = np.zeros(n_jobs)
    for j in range(n_jobs):
        s = 0.0
        for m in range(n):
            if m < n // 2:
                s += (n - 1 - m) * pt[j, m]
            else:
                s -= m * pt[j, m]
        scores[j] = s
    return list(np.argsort(-scores))


def _lpt_initial(pt):
    """Longest Processing Time."""
    totals = pt.sum(axis=1)
    return list(np.argsort(-totals))


def _spt_initial(pt):
    """Shortest Processing Time."""
    totals = pt.sum(axis=1)
    return list(np.argsort(totals))


def _bottleneck_initial(pt):
    """Prioritize jobs with long processing time on the bottleneck machine."""
    n_jobs, n_machines = pt.shape
    machine_loads = pt.sum(axis=0)
    bottleneck = int(np.argmax(machine_loads))
    return list(np.argsort(-pt[:, bottleneck]))


def _random_initial(n_jobs, seed):
    rng = random.Random(seed)
    perm = list(range(n_jobs))
    rng.shuffle(perm)
    return perm


def _vnd_adjacent_swap(pt_list, perm, deadline):
    """N1: Adjacent swap neighborhood."""
    n = len(perm)
    if n <= 1:
        return perm, _makespan_fast(pt_list, perm)
    
    current = list(perm)
    current_ms = _makespan_fast(pt_list, current)
    best = list(current)
    best_ms = current_ms
    
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        for i in range(n - 1):
            if time.time() > deadline:
                break
            # Swap i and i+1
            current[i], current[i+1] = current[i+1], current[i]
            ms = _makespan_fast(pt_list, current)
            if ms < current_ms - 1e-9:
                current_ms = ms
                improved = True
                if current_ms < best_ms - 1e-9:
                    best_ms = current_ms
                    best = list(current)
            else:
                # Revert swap
                current[i], current[i+1] = current[i+1], current[i]
    return best, best_ms


def _vnd_or_opt(pt_list, perm, deadline):
    """N2: Or-opt block insertion (blocks of size 1, 2, 3)."""
    n = len(perm)
    if n <= 3:
        return perm, _makespan_fast(pt_list, perm)
    
    current = list(perm)
    current_ms = _makespan_fast(pt_list, current)
    best = list(current)
    best_ms = current_ms
    
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        for block_size in [1, 2, 3]:
            if time.time() > deadline:
                break
            for i in range(n - block_size + 1):
                if time.time() > deadline:
                    break
                block = current[i:i+block_size]
                rest = current[:i] + current[i+block_size:]
                
                best_pos_in_rest = -1
                best_ms_try = float('inf')
                
                for j in range(len(rest) + 1):
                    if j == i:
                        continue
                    if j == i + 1:
                        continue
                    test = rest[:j] + block + rest[j:]
                    ms = _makespan_fast(pt_list, test)
                    if ms < best_ms_try:
                        best_ms_try = ms
                        best_pos_in_rest = j
                
                if best_pos_in_rest != -1 and best_ms_try < current_ms - 1e-9:
                    # Apply the best move found for this block start
                    current = rest[:best_pos_in_rest] + block + rest[best_pos_in_rest:]
                    current_ms = best_ms_try
                    improved = True
                    if current_ms < best_ms - 1e-9:
                        best_ms = current_ms
                        best = list(current)
    return best, best_ms


def _vnd_reversal(pt_list, perm, deadline):
    """N3: Segment reversal (2-opt)."""
    n = len(perm)
    if n <= 2:
        return perm, _makespan_fast(pt_list, perm)
    
    current = list(perm)
    current_ms = _makespan_fast(pt_list, current)
    best = list(current)
    best_ms = current_ms
    
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        for i in range(n - 1):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                if time.time() > deadline:
                    break
                # Reverse segment i..j
                segment = current[i:j+1]
                segment.reverse()
                test = current[:i] + segment + current[j+1:]
                ms = _makespan_fast(pt_list, test)
                if ms < current_ms - 1e-9:
                    current = test
                    current_ms = ms
                    improved = True
                    if current_ms < best_ms - 1e-9:
                        best_ms = current_ms
                        best = list(current)
                    break # Restart inner loop with new current
            if improved:
                break # Restart outer loop with new current
    return best, best_ms


def _vnd_insertion(pt_list, perm, deadline):
    """N4: Insertion neighborhood (remove and re-insert)."""
    n = len(perm)
    if n <= 1:
        return perm, _makespan_fast(pt_list, perm)
    
    current = list(perm)
    current_ms = _makespan_fast(pt_list, current)
    best = list(current)
    best_ms = current_ms
    
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            job = current[i]
            rest = current[:i] + current[i+1:]
            
            best_pos = -1
            best_ms_try = float('inf')
            
            for j in range(len(rest) + 1):
                if j == i:
                    continue
                if j == i + 1:
                    continue
                test = rest[:j] + [job] + rest[j:]
                ms = _makespan_fast(pt_list, test)
                if ms < best_ms_try:
                    best_ms_try = ms
                    best_pos = j
            
            if best_pos != -1 and best_ms_try < current_ms - 1e-9:
                current = rest[:best_pos] + [job] + rest[best_pos:]
                current_ms = best_ms_try
                improved = True
                if current_ms < best_ms - 1e-9:
                    best_ms = current_ms
                    best = list(current)
    return best, best_ms


def _local_search(pt_list, perm, deadline, use_insertion=True):
    """Variable Neighborhood Descent."""
    current = list(perm)
    current_ms = _makespan_fast(pt_list, current)
    best = list(current)
    best_ms = current_ms
    
    # Order of neighborhoods: Fast to Slow/Strong
    neighborhoods = [
        _vnd_adjacent_swap,
        _vnd_or_opt,
        _vnd_reversal
    ]
    if use_insertion:
        neighborhoods.append(_vnd_insertion)
        
    # We cycle through them. If any improves, we restart the cycle.
    # To prevent getting stuck in a specific neighborhood, we can limit iterations or just rely on the 'improved' flag.
    
    max_cycles = 20
    for cycle in range(max_cycles):
        if time.time() > deadline:
            break
        improved_in_cycle = False
        for neighborhood in neighborhoods:
            if time.time() > deadline:
                break
            perm_new, ms_new = neighborhood(pt_list, current, deadline)
            if ms_new < current_ms - 1e-9:
                current = perm_new
                current_ms = ms_new
                improved_in_cycle = True
                if current_ms < best_ms - 1e-9:
                    best_ms = current_ms
                    best = list(current)
        if not improved_in_cycle:
            break
            
    return best, best_ms


def _perturb(pt_list, perm, strength):
    """Apply random perturbations to escape local optima."""
    n = len(perm)
    if n <= 1:
        return list(perm)
    
    current = list(perm)
    rng = random.Random(int(time.time() * 1000) % 100000)
    
    # Strength is number of moves
    for _ in range(strength):
        move_type = rng.choice(['swap', 'insert', 'reverse'])
        
        if move_type == 'swap':
            i = rng.randint(0, n - 1)
            j = rng.randint(0, n - 1)
            if i != j:
                current[i], current[j] = current[j], current[i]
                
        elif move_type == 'insert':
            i = rng.randint(0, n - 1)
            job = current.pop(i)
            j = rng.randint(0, n - 1)
            current.insert(j, job)
            
        elif move_type == 'reverse':
            i = rng.randint(0, n - 1)
            j = rng.randint(0, n - 1)
            if i > j:
                i, j = j, i
            if j - i >= 2: # At least length 3 segment
                current[i:j+1] = current[i:j+1][::-1]
                
    return current


def order_jobs(processing_times):
    pt = np.asarray(processing_times, dtype=np.float64)
    n_jobs, n_machines = pt.shape
    start = time.time()
    # Leave a safety margin of 5 seconds
    deadline = start + 115.0 

    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    if n_machines == 0:
        return list(range(n_jobs))

    pt_list = pt.tolist()

    # 1. Generate diverse initial solutions
    candidates = [
        _neh_initial(pt),
        _palmer_initial(pt),
        _lpt_initial(pt),
        _spt_initial(pt),
        _bottleneck_initial(pt),
    ]
    # Add a few random starts
    for s in range(5):
        candidates.append(_random_initial(n_jobs, 100 + s))

    # Evaluate all and pick the best
    best_ms = float('inf')
    best_perm = candidates[0]
    
    # Sort candidates by makespan to start ILS from the best ones
    evaluated_candidates = []
    for cand in candidates:
        ms = _makespan_fast(pt_list, cand)
        evaluated_candidates.append((ms, cand))
        if ms < best_ms:
            best_ms = ms
            best_perm = cand
            
    evaluated_candidates.sort(key=lambda x: x[0])
    
    # 2. Iterated Local Search
    # We will try to improve the global best using ILS
    # Start with the best initial solution
    current_global_best = list(best_perm)
    current_global_ms = best_ms
    
    # Run ILS from the top few initial solutions if time permits
    # But usually, one strong ILS run is better than many weak ones.
    # Let's run ILS on the best initial solution.
    
    current = list(best_perm)
    current_ms = best_ms
    
    # Phase 1: Intensive Local Search on the best initial solution
    # Use a tighter deadline for the first LS to ensure we get a good base
    ls_deadline = deadline - 10.0
    current, current_ms = _local_search(pt_list, current, ls_deadline, use_insertion=True)
    
    if current_ms < current_global_ms:
        current_global_ms = current_ms
        current_global_best = list(current)
        
    # Phase 2: Iterated Perturbation and Local Search
    # Perturbation strength increases slightly over time or stays constant
    # We use a simulated annealing style acceptance to escape deep minima
    
    T = current_global_ms * 0.01
    T_min = current_global_ms * 0.0001
    cooling = 0.999
    
    while time.time() < deadline - 5.0:
        # Determine perturbation strength
        # Start with small perturbations, increase if stuck
        # Simple strategy: random strength between 1 and 5
        strength = random.randint(1, 4)
        
        perturbed = _perturb(pt_list, current_global_best, strength)
        
        # Local search on perturbed solution
        # Use remaining time minus a buffer
        ls_deadline = deadline - 5.0
        new_solution, new_ms = _local_search(pt_list, perturbed, ls_deadline, use_insertion=True)
        
        # Acceptance criterion
        delta = new_ms - current_global_ms
        
        if delta < 0:
            # Better solution found
            current_global_best = new_solution
            current_global_ms = new_ms
            # Reset temperature or lower it
            T = max(T_min, T * 0.9)
        else:
            # Worse solution, accept with probability exp(-delta/T)
            if T > T_min:
                prob = np.exp(-delta / T)
                if random.random() < prob:
                    current_global_best = new_solution
                    current_global_ms = new_ms
                    T = max(T_min, T * 0.9)
            else:
                # If T is too low, we might be stuck. Force a larger perturbation?
                # Or just stop.
                pass
        
        T = max(T_min, T * cooling)

    # Final polish on the global best with all neighborhoods
    if time.time() < deadline - 1.0:
        final_perm, final_ms = _local_search(pt_list, current_global_best, deadline - 0.5, use_insertion=True)
        if final_ms < current_global_ms:
            current_global_best = final_perm
            current_global_ms = final_ms

    return current_global_best

# EVOLVE-BLOCK-END
