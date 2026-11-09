# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _makespan_vec(processing_times, perm):
    """Compute makespan using vectorized NumPy."""
    n = len(perm)
    m = processing_times.shape[1]
    if n == 0 or m == 0:
        return 0.0
    # C[j][k] = max(C[j][k-1], C[j-1][k]) + p[perm[j]][k]
    # Vectorize over machines for each job
    P = processing_times[perm]  # (n, m)
    C = np.zeros((n, m))
    C[0] = np.cumsum(P[0])
    for j in range(1, n):
        prev = C[j-1]
        c = np.empty(m)
        c[0] = max(prev[0], 0.0) + P[j, 0]
        for k in range(1, m):
            c[k] = max(c[k-1], prev[k]) + P[j, k]
        C[j] = c
    return C[-1, -1]

def _makespan_fast(processing_times, perm):
    """Fast makespan computation using list-based approach for small m."""
    n = len(perm)
    m = processing_times.shape[1]
    if n == 0 or m == 0:
        return 0.0
    P = processing_times
    # Use list of floats for speed
    prev_row = [0.0] * m
    curr_row = [0.0] * m
    for j in range(n):
        job = perm[j]
        if j == 0:
            s = 0.0
            for k in range(m):
                s += P[job, k]
                curr_row[k] = s
        else:
            c0 = prev_row[0]
            for k in range(m):
                val = P[job, k]
                if k == 0:
                    curr_row[0] = c0 + val
                else:
                    cm1 = curr_row[k-1]
                    pk = prev_row[k]
                    curr_row[k] = (cm1 if cm1 > pk else pk) + val
        prev_row, curr_row = curr_row, prev_row
    return prev_row[-1]

def _insertion_cost(processing_times, perm, job, pos):
    """Compute makespan of inserting `job` at position `pos` in `perm`."""
    n = len(perm)
    m = processing_times.shape[1]
    if m == 0:
        return 0.0
    P = processing_times
    # Build the new permutation implicitly and compute makespan
    # C[j][k] = max(C[j][k-1], C[j-1][k]) + P[perm[j]][k]
    prev_row = [0.0] * m
    curr_row = [0.0] * m
    for j in range(n + 1):
        if j < pos:
            job_j = perm[j]
        else:
            job_j = job if j == pos else perm[j - 1]
        if j == 0:
            s = 0.0
            for k in range(m):
                s += P[job_j, k]
                curr_row[k] = s
        else:
            c0 = prev_row[0]
            for k in range(m):
                val = P[job_j, k]
                if k == 0:
                    curr_row[0] = c0 + val
                else:
                    cm1 = curr_row[k-1]
                    pk = prev_row[k]
                    curr_row[k] = (cm1 if cm1 > pk else pk) + val
        prev_row, curr_row = curr_row, prev_row
    return prev_row[-1]

def _neh_standard(processing_times, rng=None):
    """Standard NEH: sort by total processing time descending, insert at best position."""
    n, m = processing_times.shape
    if n == 0:
        return []
    totals = processing_times.sum(axis=1)
    if rng is not None:
        noise = rng.uniform(0, 1e-6, size=n)
        order = np.argsort(-totals + noise)
    else:
        order = np.argsort(-totals)
    perm = []
    for job in order:
        best_pos = 0
        best_ms = np.inf
        for pos in range(len(perm) + 1):
            ms = _insertion_cost(processing_times, perm, job, pos)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm = perm[:best_pos] + [job] + perm[best_pos:]
    return perm

def _neh_first_last(processing_times, rng=None):
    """NEH variant: first job is the one with max total, then insert best."""
    n, m = processing_times.shape
    if n == 0:
        return []
    totals = processing_times.sum(axis=1)
    if rng is not None:
        noise = rng.uniform(0, 1e-6, size=n)
        totals_noisy = totals + noise
    else:
        totals_noisy = totals
    # Start with the job having max total
    first_job = int(np.argmax(totals_noisy))
    perm = [first_job]
    remaining = [j for j in range(n) if j != first_job]
    # Sort remaining by total descending
    if rng is not None:
        noise = rng.uniform(0, 1e-6, size=len(remaining))
        rem_order = sorted(remaining, key=lambda j: -(totals_noisy[j] + noise[remaining.index(j)]))
    else:
        rem_order = sorted(remaining, key=lambda j: -totals[j])
    for job in rem_order:
        best_pos = 0
        best_ms = np.inf
        for pos in range(len(perm) + 1):
            ms = _insertion_cost(processing_times, perm, job, pos)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm = perm[:best_pos] + [job] + perm[best_pos:]
    return perm

def _neh_last_machine(processing_times, rng=None):
    """NEH variant: sort by last machine processing time descending."""
    n, m = processing_times.shape
    if n == 0:
        return []
    if m == 0:
        return list(range(n))
    last_col = processing_times[:, m - 1]
    if rng is not None:
        noise = rng.uniform(0, 1e-6, size=n)
        order = np.argsort(-last_col + noise)
    else:
        order = np.argsort(-last_col)
    perm = []
    for job in order:
        best_pos = 0
        best_ms = np.inf
        for pos in range(len(perm) + 1):
            ms = _insertion_cost(processing_times, perm, job, pos)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm = perm[:best_pos] + [job] + perm[best_pos:]
    return perm

def _random_perm(processing_times, rng):
    """Random permutation."""
    n = processing_times.shape[0]
    return list(rng.permutation(n))

def _tabu_search(processing_times, perm, deadline, rng=None):
    """Tabu search with insertion moves."""
    n = len(perm)
    if n < 2:
        return list(perm), _makespan_fast(processing_times, perm)
    
    if rng is None:
        rng = np.random.default_rng(42)
    
    current_perm = list(perm)
    current_ms = _makespan_fast(processing_times, current_perm)
    best_perm = list(current_perm)
    best_ms = current_ms
    
    tabu_tenure = max(5, n // 2)
    tabu_list = {}  # (job, pos) -> iteration when it becomes non-tabu
    iteration = 0
    
    while time.time() < deadline:
        improved = False
        best_move = None
        best_move_ms = current_ms
        best_move_i = -1
        best_move_j = -1
        best_move_is_tabu = False
        
        for i in range(n):
            if time.time() >= deadline:
                break
            job = current_perm[i]
            remaining = current_perm[:i] + current_perm[i+1:]
            for j in range(len(remaining) + 1):
                if j == i:
                    continue
                # Check if this move is tabu
                move_key = (job, j)
                if move_key in tabu_list and tabu_list[move_key] > iteration:
                    is_tabu = True
                else:
                    is_tabu = False
                
                ms = _insertion_cost(processing_times, remaining, job, j)
                
                # Aspiration: accept tabu move if it improves best
                if is_tabu and ms >= best_ms:
                    continue
                
                if ms < best_move_ms:
                    best_move_ms = ms
                    best_move_i = i
                    best_move_j = j
                    best_move_is_tabu = is_tabu
        
        if best_move_i == -1:
            break
        
        # Make the reverse move tabu
        job_moved = current_perm[best_move_i]
        tabu_list[(job_moved, best_move_j)] = iteration + tabu_tenure
        
        # Apply the move
        remaining = current_perm[:best_move_i] + current_perm[best_move_i+1:]
        current_perm = remaining[:best_move_j] + [job_moved] + remaining[best_move_j:]
        current_ms = best_move_ms
        
        if current_ms < best_ms:
            best_ms = current_ms
            best_perm = list(current_perm)
        
        iteration += 1
    
    return best_perm, best_ms

def _or_opt_local_search(processing_times, perm, deadline):
    """Local search with Or-opt moves (relocate blocks of 1-3)."""
    n = len(perm)
    if n < 2:
        return list(perm), _makespan_fast(processing_times, perm)
    
    current_perm = list(perm)
    current_ms = _makespan_fast(processing_times, current_perm)
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() >= deadline:
                break
            for block_len in [1, 2, 3]:
                if i + block_len > n:
                    continue
                block = current_perm[i:i+block_len]
                remaining = current_perm[:i] + current_perm[i+block_len:]
                best_pos = i
                best_ms = current_ms
                for j in range(len(remaining) + 1):
                    if j == i:
                        continue
                    if j == i + block_len:
                        continue
                    new_perm = remaining[:j] + block + remaining[j:]
                    ms = _makespan_fast(processing_times, new_perm)
                    if ms < best_ms:
                        best_ms = ms
                        best_pos = j
                if best_ms < current_ms:
                    current_perm = remaining[:best_pos] + block + remaining[best_pos:]
                    current_ms = best_ms
                    improved = True
                    break
            if improved:
                break
    
    return current_perm, current_ms

def _genetic_algorithm(processing_times, population, deadline, rng):
    """
    Genetic Algorithm with Order Crossover (OX) and mutation operators.
    """
    n = len(population[0])
    if n < 2:
        return population[0], _makespan_fast(processing_times, population[0])
    
    # Evaluate initial population
    pop_ms = [_makespan_fast(processing_times, p) for p in population]
    
    # Keep track of best
    best_idx = int(np.argmin(pop_ms))
    best_perm = list(population[best_idx])
    best_ms = pop_ms[best_idx]
    
    pop_size = len(population)
    tournament_size = 3
    mutation_rate = 0.1
    crossover_rate = 0.8
    
    generation = 0
    stagnation = 0
    
    while time.time() < deadline:
        new_population = []
        new_pop_ms = []
        
        # Elitism: keep the best individual
        new_population.append(list(best_perm))
        new_pop_ms.append(best_ms)
        
        # Generate offspring
        while len(new_population) < pop_size:
            if time.time() >= deadline:
                break
            
            # Selection via tournament
            parent1_idx = _tournament_select(population, pop_ms, tournament_size, rng)
            parent2_idx = _tournament_select(population, pop_ms, tournament_size, rng)
            
            parent1 = population[parent1_idx]
            parent2 = population[parent2_idx]
            
            # Crossover
            if rng.random() < crossover_rate:
                child = _order_crossover(parent1, parent2, rng)
            else:
                child = list(parent1)
            
            # Mutation
            if rng.random() < mutation_rate:
                child = _insertion_mutation(child, rng)
            
            # Ensure validity
            if len(child) != n or set(child) != set(range(n)):
                child = list(parent1)
            
            ms = _makespan_fast(processing_times, child)
            new_population.append(child)
            new_pop_ms.append(ms)
        
        # Update population
        population = new_population
        pop_ms = new_pop_ms
        
        # Update best
        gen_best_idx = int(np.argmin(pop_ms))
        if pop_ms[gen_best_idx] < best_ms:
            best_ms = pop_ms[gen_best_idx]
            best_perm = list(population[gen_best_idx])
            stagnation = 0
        else:
            stagnation += 1
        
        # Adaptive mutation rate
        if stagnation > 5:
            mutation_rate = min(0.5, mutation_rate + 0.05)
        elif stagnation > 15:
            # Restart with fresh diversity
            for i in range(pop_size):
                if time.time() >= deadline:
                    break
                population[i] = _random_perm(processing_times, rng)
                pop_ms[i] = _makespan_fast(processing_times, population[i])
            gen_best_idx = int(np.argmin(pop_ms))
            if pop_ms[gen_best_idx] < best_ms:
                best_ms = pop_ms[gen_best_idx]
                best_perm = list(population[gen_best_idx])
            stagnation = 0
            mutation_rate = 0.1
        
        generation += 1
    
    return best_perm, best_ms

def _tournament_select(population, pop_ms, tournament_size, rng):
    """Tournament selection."""
    n = len(population)
    if n <= tournament_size:
        return int(np.argmin(pop_ms))
    
    indices = rng.choice(n, size=tournament_size, replace=False)
    best_idx = indices[0]
    best_val = pop_ms[indices[0]]
    for idx in indices[1:]:
        if pop_ms[idx] < best_val:
            best_val = pop_ms[idx]
            best_idx = idx
    return int(best_idx)

def _order_crossover(parent1, parent2, rng):
    """Order Crossover (OX1)."""
    n = len(parent1)
    if n < 2:
        return list(parent1)
    
    # Choose two cut points
    i = rng.integers(0, n)
    j = rng.integers(0, n)
    if i > j:
        i, j = j, i
    if i == j:
        i = (i + 1) % n
    
    # Extract segment from parent1
    segment = parent1[i:j+1]
    segment_set = set(segment)
    
    # Fill remaining positions from parent2
    result = [None] * n
    for gene in segment:
        result[0] = None  # placeholder
    
    # Copy segment
    for k in range(i, j+1):
        result[k] = segment[k - i]
    
    # Fill rest from parent2
    pos = 0
    for gene in parent2:
        if gene not in segment_set:
            while result[pos] is not None:
                pos += 1
            result[pos] = gene
            pos += 1
    
    return result

def _insertion_mutation(perm, rng):
    """Insertion mutation: remove a random job and reinsert it at a random position."""
    n = len(perm)
    if n < 2:
        return list(perm)
    
    i = int(rng.integers(0, n))
    job = perm[i]
    remaining = perm[:i] + perm[i+1:]
    j = int(rng.integers(0, len(remaining) + 1))
    return remaining[:j] + [job] + remaining[j:]

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on a flow-shop instance.
    
    Args:
        processing_times: np.ndarray of shape (jobs, machines)
        
    Returns:
        List of job indices in the order they should be processed.
    """
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s limit
    
    n, m = processing_times.shape
    if n == 0:
        return []
    if n == 1:
        return [0]
    if m == 0:
        return list(range(n))
    
    rng = np.random.default_rng(12345)
    
    # Generate multiple candidate permutations from different heuristics
    candidates = []
    
    # Standard NEH (deterministic)
    candidates.append(_neh_standard(processing_times))
    
    # First-last NEH (deterministic)
    candidates.append(_neh_first_last(processing_times))
    
    # Last-machine NEH (deterministic)
    candidates.append(_neh_last_machine(processing_times))
    
    # Randomized NEH variants
    for i in range(5):
        if time.time() > deadline:
            break
        candidates.append(_neh_standard(processing_times, rng))
    
    for i in range(5):
        if time.time() > deadline:
            break
        candidates.append(_neh_first_last(processing_times, rng))
    
    for i in range(5):
        if time.time() > deadline:
            break
        candidates.append(_neh_last_machine(processing_times, rng))
    
    # A few random starts
    for i in range(3):
        if time.time() > deadline:
            break
        candidates.append(_random_perm(processing_times, rng))
    
    # Evaluate all candidates and find the best
    best_perm = candidates[0]
    best_ms = _makespan_fast(processing_times, best_perm)
    
    for perm in candidates[1:]:
        if time.time() > deadline:
            break
        ms = _makespan_fast(processing_times, perm)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm
    
    # Use Genetic Algorithm with the best candidates as initial population
    if time.time() < deadline - 5:
        # Create initial population for GA
        ga_population = [list(best_perm)]
        for perm in candidates[1:]:
            if len(ga_population) < 10:
                ga_population.append(list(perm))
        
        # Add some random individuals for diversity
        while len(ga_population) < 10:
            ga_population.append(_random_perm(processing_times, rng))
        
        ga_deadline = deadline - 5
        ga_perm, ga_ms = _genetic_algorithm(processing_times, ga_population, ga_deadline, rng)
        
        if ga_ms < best_ms:
            best_perm = ga_perm
            best_ms = ga_ms
    
    # Final local search polish
    if time.time() < deadline - 1:
        improved_perm, improved_ms = _or_opt_local_search(processing_times, best_perm, deadline)
        if improved_ms < best_ms:
            best_perm = improved_perm
            best_ms = improved_ms
    
    return list(best_perm)
# EVOLVE-BLOCK-END
