"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan_fast(perm, pt):
    """Optimized makespan computation using 2D array for speed."""
    n_machines = pt.shape[1]
    n_jobs = len(perm)
    
    if n_machines == 0 or n_jobs == 0:
        return 0.0
    
    # Build completion matrix C[j][m] = completion time of job j on machine m
    C = np.zeros((n_jobs, n_machines), dtype=np.float64)
    pt_perm = pt[np.asarray(perm, dtype=np.int64)]
    
    # First machine: cumulative sum
    C[:, 0] = np.cumsum(pt_perm[:, 0])
    
    if n_machines > 1:
        for m in range(1, n_machines):
            # C[0, m] = C[0, m-1] + pt_perm[0, m]
            C[0, m] = C[0, m-1] + pt_perm[0, m]
            for j in range(1, n_jobs):
                c_prev_m = C[j-1, m]
                c_curr_m1 = C[j, m-1]
                C[j, m] = (c_curr_m1 if c_curr_m1 > c_prev_m else c_prev_m) + pt_perm[j, m]
    
    return C[n_jobs-1, n_machines-1]


def _neh_heuristic(pt, sort_key='total'):
    """NEH (Nawaz, Enscore, Ham) heuristic for flow shop."""
    n_jobs = pt.shape[0]
    n_machines = pt.shape[1]
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    if sort_key == 'total':
        keys = pt.sum(axis=1)
    elif sort_key == 'first_last':
        keys = pt[:, 0] + pt[:, -1]
    elif sort_key == 'avg':
        keys = pt.mean(axis=1)
    elif sort_key == 'max':
        keys = pt.max(axis=1)
    else:
        keys = pt.sum(axis=1)
    
    order = list(np.argsort(-keys))
    
    # Insert each job into best position
    perm = [order[0]]
    for i in range(1, n_jobs):
        job = order[i]
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(perm) + 1):
            candidate = perm[:pos] + [job] + perm[pos:]
            ms = _makespan_fast(candidate, pt)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm = perm[:best_pos] + [job] + perm[best_pos:]
    
    return perm


def _johnsons_rule(pt):
    """Johnson's rule for 2-machine flow shop, extended to multi-machine via splitting."""
    n_jobs = pt.shape[0]
    n_machines = pt.shape[1]
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    if n_machines == 2:
        a = pt[:, 0]
        b = pt[:, 1]
    else:
        mid = n_machines // 2
        a = pt[:, :mid].sum(axis=1)
        b = pt[:, mid:].sum(axis=1)
    
    left = []
    right = []
    for j in range(n_jobs):
        if a[j] <= b[j]:
            left.append((a[j], j))
        else:
            right.append((b[j], j))
    left.sort(key=lambda x: x[0])
    right.sort(key=lambda x: -x[0])
    
    return [x[1] for x in left] + [x[1] for x in right]


def _local_search_insertion(perm, pt, deadline=None):
    """Insertion-based local search with swap moves."""
    n_jobs = len(perm)
    best_perm = list(perm)
    best_ms = _makespan_fast(best_perm, pt)
    
    while True:
        improved = False
        
        # Phase 1: insertion moves
        for i in range(n_jobs):
            if deadline is not None and time.time() > deadline:
                return best_perm, best_ms
            job = best_perm[i]
            for j in range(n_jobs):
                if i == j:
                    continue
                candidate = best_perm[:i] + best_perm[i+1:]
                insert_pos = j if j < i else j - 1
                candidate = candidate[:insert_pos] + [job] + candidate[insert_pos:]
                ms = _makespan_fast(candidate, pt)
                if ms < best_ms - 1e-12:
                    best_ms = ms
                    best_perm = candidate
                    improved = True
                    break
            if deadline is not None and time.time() > deadline:
                return best_perm, best_ms
        
        # Phase 2: swap moves
        if not improved and n_jobs > 1:
            for i in range(n_jobs):
                if deadline is not None and time.time() > deadline:
                    return best_perm, best_ms
                for j in range(i + 1, n_jobs):
                    if deadline is not None and time.time() > deadline:
                        return best_perm, best_ms
                    candidate = list(best_perm)
                    candidate[i], candidate[j] = candidate[j], candidate[i]
                    ms = _makespan_fast(candidate, pt)
                    if ms < best_ms - 1e-12:
                        best_ms = ms
                        best_perm = candidate
                        improved = True
                        break
                if improved:
                    break
        
        if not improved:
            break
    
    return best_perm, best_ms


def _or_opt(processing_times, perm, max_iterations=50):
    """Or-opt style local search: remove a segment of 1-3 jobs and reinsert."""
    n_jobs = len(perm)
    if n_jobs <= 1:
        return perm
    
    current_perm = list(perm)
    current_ms = _makespan_fast(current_perm, processing_times)
    
    for _ in range(max_iterations):
        improved = False
        for seg_len in [1, 2, 3]:
            for i in range(n_jobs - seg_len + 1):
                segment = current_perm[i:i+seg_len]
                remaining = current_perm[:i] + current_perm[i+seg_len:]
                
                for j in range(len(remaining) - seg_len + 1):
                    if i == j and seg_len == 1:
                        continue
                    if j >= i and j < i + seg_len:
                        continue
                    
                    trial = remaining[:j] + segment + remaining[j:]
                    ms = _makespan_fast(trial, processing_times)
                    if ms < current_ms - 1e-9:
                        current_perm = trial
                        current_ms = ms
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return current_perm


def _order_crossover(p1, p2):
    """Order Crossover (OX1) for permutations."""
    n = len(p1)
    if n <= 1:
        return list(p1)
    
    # Choose two random cut points
    i, j = sorted(np.random.choice(n, 2, replace=False))
    
    # Copy the segment from p1
    child = [None] * n
    child[i:j+1] = p1[i:j+1]
    
    # Fill the rest with elements from p2 in order, skipping those already in child
    fill = [x for x in p2 if x not in child[i:j+1]]
    
    # Fill positions after j first, then before i (wrapping around)
    pos = j + 1
    for val in fill:
        if pos >= n:
            pos = 0
        if child[pos] is None:
            child[pos] = val
            pos += 1
            if pos >= n:
                pos = 0
    
    return child


def _ga_search(pt, initial_perms, deadline):
    """Genetic Algorithm with Order Crossover and mutation."""
    n_jobs = pt.shape[0]
    if n_jobs <= 1:
        return initial_perms[0], _makespan_fast(initial_perms[0], pt)
    
    pop_size = min(30, max(10, n_jobs))
    
    # Initialize population
    population = [list(p) for p in initial_perms[:pop_size]]
    # Fill remaining slots with random permutations
    while len(population) < pop_size:
        random_perm = list(range(n_jobs))
        np.random.shuffle(random_perm)
        population.append(random_perm)
    
    # Evaluate initial population
    fitness = [_makespan_fast(p, pt) for p in population]
    
    best_idx = int(np.argmin(fitness))
    best_perm = list(population[best_idx])
    best_ms = fitness[best_idx]
    
    generation = 0
    max_generations = 500
    
    while time.time() < deadline - 2.0 and generation < max_generations:
        generation += 1
        
        # Select parents using tournament selection
        def tournament_select():
            candidates = np.random.choice(pop_size, size=3, replace=False)
            best_cand = candidates[np.argmin([fitness[c] for c in candidates])]
            return population[best_cand]
        
        new_population = []
        
        # Elitism: keep best individual
        new_population.append(list(best_perm))
        
        # Generate offspring
        while len(new_population) < pop_size:
            if time.time() > deadline - 2.0:
                break
            
            p1 = tournament_select()
            p2 = tournament_select()
            
            # Crossover
            if np.random.random() < 0.8:
                child = _order_crossover(p1, p2)
            else:
                child = list(p1)
            
            # Mutation: swap two random positions
            if np.random.random() < 0.3:
                i, j = np.random.choice(n_jobs, 2, replace=False)
                child[i], child[j] = child[j], child[i]
            
            # Or-opt mutation: move a segment
            elif np.random.random() < 0.3:
                seg_len = np.random.choice([1, 2, 3])
                if n_jobs > seg_len:
                    i = np.random.randint(0, n_jobs - seg_len + 1)
                    segment = child[i:i+seg_len]
                    remaining = child[:i] + child[i+seg_len:]
                    j = np.random.randint(0, len(remaining) - seg_len + 1)
                    child = remaining[:j] + segment + remaining[j:]
            
            new_population.append(child)
        
        # Evaluate new population
        fitness = [_makespan_fast(p, pt) for p in new_population]
        
        # Update best
        best_idx = int(np.argmin(fitness))
        if fitness[best_idx] < best_ms - 1e-12:
            best_ms = fitness[best_idx]
            best_perm = list(new_population[best_idx])
        
        population = new_population
    
    return best_perm, best_ms


def order_jobs(processing_times):
    """Construct a job permutation minimizing makespan on flow-shop instances."""
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    pt = np.asarray(processing_times)
    n_jobs, n_machines = pt.shape
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    # Generate multiple diverse initial solutions
    variants = ['total', 'first_last', 'avg', 'max']
    
    best_perm = None
    best_ms = float('inf')
    initial_perms = []
    
    for variant in variants:
        if time.time() > deadline - 10.0:
            break
        perm = _neh_heuristic(pt, variant)
        ms = _makespan_fast(perm, pt)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm
        initial_perms.append(perm)
    
    # Also try Johnson's rule
    if time.time() < deadline - 10.0:
        johnson_perm = _johnsons_rule(pt)
        johnson_ms = _makespan_fast(johnson_perm, pt)
        if johnson_ms < best_ms:
            best_ms = johnson_ms
            best_perm = johnson_perm
        initial_perms.append(johnson_perm)
    
    if best_perm is None:
        return list(range(n_jobs))
    
    # Local search refinement on the best initial solution
    if time.time() < deadline - 5.0:
        best_perm, best_ms = _local_search_insertion(best_perm, pt, deadline - 5.0)
    
    # Then Or-opt if time permits
    if time.time() < deadline - 2.0:
        or_opt_perm = _or_opt(pt, best_perm, max_iterations=30)
        or_opt_ms = _makespan_fast(or_opt_perm, pt)
        if or_opt_ms < best_ms - 1e-9:
            best_perm = or_opt_perm
            best_ms = or_opt_ms
    
    # Genetic Algorithm to explore the space further
    if time.time() < deadline - 5.0:
        # Add the locally optimized solution to initial permutations
        initial_perms.append(best_perm)
        
        ga_perm, ga_ms = _ga_search(pt, initial_perms, deadline)
        if ga_ms < best_ms - 1e-12:
            best_perm = ga_perm
            best_ms = ga_ms
    
    # Final local search refinement if time permits
    if time.time() < deadline - 1.0:
        best_perm, best_ms = _local_search_insertion(best_perm, pt, deadline - 1.0)
    
    return best_perm
# EVOLVE-BLOCK-END
