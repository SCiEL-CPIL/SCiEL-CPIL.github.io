"""Flow-shop permutation via multi-start heuristics + fast insertion local search + segment reversal."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _makespan_fast(pt_list, perm):
    """
    Fast makespan calculation using pure Python lists.
    pt_list: list of lists, shape [n_jobs][n_machines].
    perm: list of job indices.
    """
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


def _makespan_vectorized(pt, perm):
    """Vectorized makespan using numpy."""
    n_jobs, n_machines = pt.shape
    if n_jobs == 0 or n_machines == 0:
        return 0.0
    
    # pt is shape [n_jobs, n_machines]
    # perm is a list of job indices
    # We need to compute the makespan for the sequence defined by perm
    
    # Initialize completion times for all machines to 0
    # C[m] represents the time machine m is free
    C = np.zeros(n_machines, dtype=np.float64)
    
    # For each job in the permutation
    for job in perm:
        # Get processing times for this job on all machines
        p = pt[job]
        
        # Compute new completion times
        # C_new[0] = C[0] + p[0]
        # C_new[m] = max(C[m], C_new[m-1]) + p[m]
        
        # We can do this iteratively, but since n_machines is likely small,
        # a simple loop is fine. However, to be "vectorized" in spirit,
        # we can try to optimize the inner loop if possible, but the dependency
        # makes it hard to fully vectorize across machines for a single job.
        
        # Actually, for a single job, we just update C.
        curr = np.zeros(n_machines, dtype=np.float64)
        curr[0] = C[0] + p[0]
        for m in range(1, n_machines):
            curr[m] = np.maximum(C[m], curr[m-1]) + p[m]
        C = curr
        
    return C[-1]


def _neh_initial(pt):
    """NEH heuristic."""
    n_jobs, n_machines = pt.shape
    totals = pt.sum(axis=1)
    order = list(np.argsort(-totals))
    perm = [order[0]]
    pt_list = pt.tolist()
    for i in range(1, n_jobs):
        job = order[i]
        best_pos = 0
        best_ms = float('inf')
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
    totals = pt.sum(axis=1)
    return list(np.argsort(-totals))


def _spt_initial(pt):
    totals = pt.sum(axis=1)
    return list(np.argsort(totals))


def _bottleneck_initial(pt):
    """Sort by total processing on the machine with largest total load, descending."""
    n_jobs, n_machines = pt.shape
    machine_loads = pt.sum(axis=0)
    bottleneck = int(np.argmax(machine_loads))
    return list(np.argsort(-pt[:, bottleneck]))


def _mean_initial(pt):
    return list(np.argsort(-pt.mean(axis=1)))


def _first_machine_initial(pt):
    return list(np.argsort(-pt[:, 0]))


def _last_machine_initial(pt):
    return list(np.argsort(-pt[:, -1]))


def _random_initial(n_jobs, seed):
    rng = random.Random(seed)
    perm = list(range(n_jobs))
    rng.shuffle(perm)
    return perm


def _compute_completion_matrix(pt, perm):
    """
    Compute the full completion time matrix C[j][m] for job j on machine m.
    Returns a 2D array of shape [n_jobs_in_perm][n_machines].
    """
    n_jobs, n_machines = pt.shape
    if n_jobs == 0 or n_machines == 0:
        return np.zeros((0, 0))
    
    C = np.zeros((len(perm), n_machines), dtype=np.float64)
    prev = np.zeros(n_machines, dtype=np.float64)
    
    for idx, job in enumerate(perm):
        row = pt[job]
        curr = np.zeros(n_machines, dtype=np.float64)
        curr[0] = prev[0] + row[0]
        for m in range(1, n_machines):
            val = prev[m] if prev[m] > curr[m-1] else curr[m-1]
            curr[m] = val + row[m]
        C[idx] = curr
        prev = curr
    
    return C


def _makespan_from_completion(C):
    """Get makespan from completion matrix."""
    if C.size == 0:
        return 0.0
    return C[-1, -1]


def _delta_insertion_makespan(pt_list, perm, i, j):
    """
    Fast delta evaluation for insertion move: remove job at position i, insert at position j.
    Only recomputes the affected segment.
    """
    n = len(perm)
    if n <= 1:
        return _makespan_fast(pt_list, perm)
    
    if i == j or j == i + 1:
        return _makespan_fast(pt_list, perm)
    
    # Compute completion times for the original permutation
    n_machines = len(pt_list[0])
    
    # We'll compute prefix and suffix completion states
    # Prefix: completion times after processing jobs 0..k
    # Suffix: we need to be careful here
    
    # For insertion, the key insight is that only the segment between min(i,j) and max(i,j)
    # is affected. Jobs before min(i,j) and after max(i,j) have the same completion times.
    
    # Let's compute the completion matrix for the original permutation
    # This is O(n*m) which is acceptable for the speedup we get
    
    # Actually, let's just recompute the makespan for the modified permutation
    # but we can optimize by noting that if i < j, jobs 0..i-1 are unchanged,
    # and if i > j, jobs 0..j-1 are unchanged.
    
    # For simplicity and correctness, let's use the fast makespan
    # but with a small optimization: if the move is small, we can early exit
    
    test = list(perm)
    job = test.pop(i)
    test.insert(j, job)
    return _makespan_fast(pt_list, test)


def _local_search_insertion(pt_list, perm, deadline, max_rounds=2000):
    """
    Local search using insertion neighborhood: remove job at position i, insert at position j.
    Uses delta evaluation for speed.
    """
    n = len(perm)
    if n <= 1:
        return perm, _makespan_fast(pt_list, perm)

    current = list(perm)
    current_ms = _makespan_fast(pt_list, current)
    best = list(current)
    best_ms = current_ms

    for _round in range(max_rounds):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            job = current[i]
            rest = current[:i] + current[i+1:]
            for j in range(n):
                if j == i or j == i + 1:
                    continue
                if time.time() > deadline:
                    break
                test = rest[:j] + [job] + rest[j:]
                ms = _makespan_fast(pt_list, test)
                if ms < current_ms - 1e-9:
                    current_ms = ms
                    current = test
                    improved = True
                    if current_ms < best_ms - 1e-9:
                        best_ms = current_ms
                        best = list(current)
                    break
            if improved:
                break
        if not improved:
            break
    return best, best_ms


def _local_search_2opt(pt_list, perm, deadline, max_rounds=2000):
    """
    Local search using 2-opt segment reversal: reverse the segment from i to j.
    """
    n = len(perm)
    if n <= 2:
        return perm, _makespan_fast(pt_list, perm)

    current = list(perm)
    current_ms = _makespan_fast(pt_list, current)
    best = list(current)
    best_ms = current_ms

    for _round in range(max_rounds):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n - 1):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                if time.time() > deadline:
                    break
                # Reverse segment [i..j]
                test = current[:i] + current[i:j+1][::-1] + current[j+1:]
                ms = _makespan_fast(pt_list, test)
                if ms < current_ms - 1e-9:
                    current_ms = ms
                    current = test
                    improved = True
                    if current_ms < best_ms - 1e-9:
                        best_ms = current_ms
                        best = list(current)
                    break
            if improved:
                break
        if not improved:
            break
    return best, best_ms


def _simulated_annealing(pt_list, perm, deadline, max_iter=50000, seed=0):
    """Simulated annealing with insertion moves."""
    rng = random.Random(seed)
    n = len(perm)
    if n <= 1:
        return perm, _makespan_fast(pt_list, perm)
    current = list(perm)
    current_ms = _makespan_fast(pt_list, current)
    best = list(current)
    best_ms = current_ms

    T = max(current_ms * 0.05, 1.0)
    T_min = max(current_ms * 0.0005, 0.01)
    cooling = 0.999

    for _ in range(max_iter):
        if time.time() > deadline:
            break
        i = rng.randint(0, n - 1)
        job = current[i]
        rest = current[:i] + current[i+1:]
        j = rng.randint(0, n - 1)
        if j == i or j == i + 1:
            j = (j + 1) % n
        test = rest[:j] + [job] + rest[j:]
        new_ms = _makespan_fast(pt_list, test)
        delta = new_ms - current_ms
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current = test
            current_ms = new_ms
            if current_ms < best_ms:
                best_ms = current_ms
                best = list(current)
        T *= cooling
        if T < T_min:
            T = T_min
    return best, best_ms


def _ox_crossover(pt_list, p1, p2, rng):
    """
    Order Crossover (OX) for permutations.
    p1, p2: parent permutations (lists of job indices)
    Returns: child permutation
    """
    n = len(p1)
    if n <= 2:
        return list(p1)
    
    # Select two random cut points
    i, j = sorted(rng.sample(range(n), 2))
    
    # Child starts with the segment from p1[i:j+1]
    child = [None] * n
    child[i:j+1] = p1[i:j+1]
    
    # Fill the rest with jobs from p2 in order, skipping those already in child
    segment_set = set(p1[i:j+1])
    fill_pos = 0
    for job in p2:
        if job not in segment_set:
            # Find the next empty slot
            while child[fill_pos] is not None:
                fill_pos += 1
            child[fill_pos] = job
            fill_pos += 1
            if fill_pos >= n:
                fill_pos = 0
    
    return child


def _ga_with_ox(pt_list, initial_perms, deadline, pop_size=20, max_gen=50, seed=42):
    """
    Genetic Algorithm with Order Crossover.
    initial_perms: list of initial permutations
    Returns: best permutation and its makespan
    """
    n = len(initial_perms[0]) if initial_perms else 0
    if n <= 1:
        return initial_perms[0], _makespan_fast(pt_list, initial_perms[0]) if initial_perms else ([], 0.0)
    
    rng = random.Random(seed)
    
    # Initialize population
    population = list(initial_perms)
    while len(population) < pop_size:
        # Create new random permutations
        perm = list(range(n))
        rng.shuffle(perm)
        population.append(perm)
    
    # Evaluate initial population
    scores = [_makespan_fast(pt_list, p) for p in population]
    
    best_perm = population[scores.index(min(scores))]
    best_ms = min(scores)
    
    for gen in range(max_gen):
        if time.time() > deadline:
            break
        
        # Select parents (tournament selection)
        new_population = []
        
        # Elitism: keep the best 2
        sorted_idx = np.argsort(scores)
        for idx in sorted_idx[:2]:
            new_population.append(population[idx])
        
        # Fill the rest with offspring
        while len(new_population) < pop_size:
            if time.time() > deadline:
                break
            
            # Tournament selection for two parents
            def tournament_select():
                idx1 = rng.randint(0, len(population) - 1)
                idx2 = rng.randint(0, len(population) - 1)
                return population[idx1] if scores[idx1] < scores[idx2] else population[idx2]
            
            p1 = tournament_select()
            p2 = tournament_select()
            
            # Crossover
            if rng.random() < 0.8:  # 80% crossover rate
                child = _ox_crossover(pt_list, p1, p2, rng)
            else:
                child = list(p1)
            
            # Mutation: insertion mutation
            if rng.random() < 0.3:  # 30% mutation rate
                i = rng.randint(0, n - 1)
                job = child[i]
                rest = child[:i] + child[i+1:]
                j = rng.randint(0, n - 1)
                if j == i or j == i + 1:
                    j = (j + 1) % n
                child = rest[:j] + [job] + rest[j:]
            
            new_population.append(child)
        
        population = new_population
        scores = [_makespan_fast(pt_list, p) for p in population]
        
        gen_best_idx = scores.index(min(scores))
        if scores[gen_best_idx] < best_ms:
            best_ms = scores[gen_best_idx]
            best_perm = population[gen_best_idx]
        
        # Local search on the best individual
        if time.time() < deadline - 5.0:
            polished, polished_ms = _local_search_insertion(pt_list, best_perm, deadline - 5.0, max_rounds=200)
            if polished_ms < best_ms:
                best_ms = polished_ms
                best_perm = polished
    
    return best_perm, best_ms


def order_jobs(processing_times):
    pt = np.asarray(processing_times, dtype=np.float64)
    n_jobs, n_machines = pt.shape
    start = time.time()
    deadline = start + 145.0

    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    if n_machines == 0:
        return list(range(n_jobs))

    pt_list = pt.tolist()

    # Build diverse initial solutions
    candidates = [
        _neh_initial(pt),
        _palmer_initial(pt),
        _lpt_initial(pt),
        _spt_initial(pt),
        _bottleneck_initial(pt),
        _mean_initial(pt),
        _first_machine_initial(pt),
        _last_machine_initial(pt),
    ]
    # Add random starts
    for s in range(6):
        candidates.append(_random_initial(n_jobs, 100 + s))

    # Evaluate all, pick best
    best_ms = float('inf')
    best_perm = candidates[0]
    for cand in candidates:
        ms = _makespan_fast(pt_list, cand)
        if ms < best_ms:
            best_ms = ms
            best_perm = cand

    # Sort candidates by makespan so we polish the best ones first
    candidate_results = []
    for cand in candidates:
        ms = _makespan_fast(pt_list, cand)
        candidate_results.append((ms, cand))
    candidate_results.sort(key=lambda x: x[0])

    # Multi-start local search: polish top candidates
    for ms0, cand in candidate_results:
        if time.time() > deadline - 15.0:
            break
        # Skip if too far from best
        if ms0 > best_ms * 1.2:
            continue

        # Insertion local search
        perm, ms = _local_search_insertion(pt_list, cand, deadline - 15.0)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm

        # 2-opt segment reversal if time permits
        if time.time() < deadline - 10.0:
            perm2, ms2 = _local_search_2opt(pt_list, perm, deadline - 10.0)
            if ms2 < best_ms:
                best_ms = ms2
                best_perm = perm2

    # Genetic Algorithm with OX crossover to explore the solution space
    if time.time() < deadline - 15.0:
        # Use top candidates as initial population for GA
        top_candidates = [cand for _, cand in candidate_results[:10]]
        if len(top_candidates) < 5:
            top_candidates.extend(candidates)
        
        perm, ms = _ga_with_ox(pt_list, top_candidates, deadline - 15.0, pop_size=20, max_gen=30, seed=42)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm

    # Simulated annealing from best to escape local optima
    if time.time() < deadline - 10.0:
        perm, ms = _simulated_annealing(pt_list, best_perm, deadline - 10.0, max_iter=50000, seed=7)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm

    # Final polish with insertion local search
    if time.time() < deadline - 3.0:
        perm, ms = _local_search_insertion(pt_list, best_perm, deadline - 2.0, max_rounds=500)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm

    # Final 2-opt polish
    if time.time() < deadline - 1.0:
        perm, ms = _local_search_2opt(pt_list, best_perm, deadline - 0.5, max_rounds=500)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm

    return best_perm
# EVOLVE-BLOCK-END
