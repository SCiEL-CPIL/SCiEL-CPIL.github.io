"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _makespan(perm, pt):
    """Compute makespan for a given permutation. pt is (n_jobs, n_machines)."""
    n_jobs = len(perm)
    n_machines = pt.shape[1]
    # completion times per machine
    completion = np.zeros(n_machines)
    for j in perm:
        for m in range(n_machines):
            completion[m] = completion[m] + pt[j, m]
    return completion[-1]


def _makespan_fast(perm, pt):
    """Vectorized-ish makespan computation."""
    n_jobs = len(perm)
    n_machines = pt.shape[1]
    # For each job in order, completion on machine m = max(completion[m-1], completion[m]) + pt[j,m]
    # We can do this iteratively but it's inherently sequential per job.
    # Let's use a simple loop which is fast enough for moderate sizes.
    comp = np.zeros(n_machines)
    for j in perm:
        # comp[m] = max(comp[m-1], comp[m]) + pt[j, m]
        for m in range(1, n_machines):
            if comp[m-1] > comp[m]:
                comp[m] = comp[m-1]
        comp += pt[j]
    return comp[-1]


def _neh_order(pt):
    """NEH heuristic for flow shop."""
    n_jobs = pt.shape[0]
    n_machines = pt.shape[1]
    
    # Sort jobs by total processing time in descending order
    totals = pt.sum(axis=1)
    order = list(np.argsort(-totals))
    
    # Insert each job into the best position in the current sequence
    seq = [order[0]]
    for j in order[1:]:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(seq) + 1):
            new_seq = seq[:pos] + [j] + seq[pos:]
            ms = _makespan_fast(new_seq, pt)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        seq = seq[:best_pos] + [j] + seq[best_pos:]
    return seq


def _local_search(perm, pt, max_iters=100, time_limit=None, start_time=None):
    """Iterative improvement with 2-opt style moves (reversal) and insertion."""
    n_jobs = len(perm)
    if n_jobs <= 1:
        return perm
    
    best_perm = list(perm)
    best_ms = _makespan_fast(best_perm, pt)
    
    for it in range(max_iters):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        improved = False
        
        # Try insertion moves: remove job i, insert at position j
        for i in range(n_jobs):
            job = best_perm[i]
            base = best_perm[:i] + best_perm[i+1:]
            for j in range(len(base) + 1):
                if j == i or j == i + 1:
                    continue
                new_perm = base[:j] + [job] + base[j:]
                ms = _makespan_fast(new_perm, pt)
                if ms < best_ms:
                    best_ms = ms
                    best_perm = new_perm
                    improved = True
        
        # Try 2-opt reversals
        for i in range(n_jobs - 1):
            for j in range(i + 1, n_jobs):
                new_perm = best_perm[:i] + best_perm[i:j+1][::-1] + best_perm[j+1:]
                ms = _makespan_fast(new_perm, pt)
                if ms < best_ms:
                    best_ms = ms
                    best_perm = new_perm
                    improved = True
        
        if not improved:
            break
    
    return best_perm


def _ga_crossover(parent1, parent2):
    """Order crossover (OX) for permutations."""
    n = len(parent1)
    child = [None] * n
    # Randomly select a slice from parent1
    i, j = sorted(random.sample(range(n), 2))
    child[i:j+1] = parent1[i:j+1]
    
    # Fill remaining with jobs from parent2 in order
    remaining = [x for x in parent2 if x not in child[i:j+1]]
    pos = 0
    for k in range(n):
        if child[k] is None:
            child[k] = remaining[pos]
            pos += 1
    return child


def _ga_mutation(perm, rate=0.1):
    """Swap mutation."""
    perm = list(perm)
    n = len(perm)
    if n < 2:
        return perm
    for i in range(n):
        if random.random() < rate:
            j = random.randint(0, n-1)
            perm[i], perm[j] = perm[j], perm[i]
    return perm


def _ga(perm_init, pt, pop_size=50, generations=50, time_limit=None, start_time=None):
    """Simple genetic algorithm."""
    n_jobs = pt.shape[0]
    if n_jobs <= 1:
        return list(perm_init)
    
    # Initialize population
    pop = [list(perm_init)]
    for _ in range(pop_size - 1):
        # Random permutation
        p = list(range(n_jobs))
        random.shuffle(p)
        pop.append(p)
    
    best_overall = list(perm_init)
    best_overall_ms = _makespan_fast(best_overall, pt)
    
    for gen in range(generations):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        # Evaluate fitness
        fitness = [_makespan_fast(p, pt) for p in pop]
        
        # Update best
        min_idx = np.argmin(fitness)
        if fitness[min_idx] < best_overall_ms:
            best_overall_ms = fitness[min_idx]
            best_overall = list(pop[min_idx])
        
        # Selection: tournament
        new_pop = []
        for _ in range(pop_size):
            # Tournament of 3
            candidates = random.sample(range(pop_size), min(3, pop_size))
            best_cand = min(candidates, key=lambda x: fitness[x])
            new_pop.append(list(pop[best_cand]))
        
        # Crossover and mutation
        next_pop = []
        for i in range(0, pop_size - 1, 2):
            p1, p2 = new_pop[i], new_pop[i+1]
            if random.random() < 0.9:  # crossover rate
                c1 = _ga_crossover(p1, p2)
                c2 = _ga_crossover(p2, p1)
            else:
                c1, c2 = list(p1), list(p2)
            c1 = _ga_mutation(c1)
            c2 = _ga_mutation(c2)
            next_pop.extend([c1, c2])
        
        # Elitism: keep best individual
        next_pop[0] = list(best_overall)
        pop = next_pop
    
    return best_overall


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on flow-shop instances.
    Uses NEH heuristic, local search, and genetic algorithm with time management.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    pt = np.asarray(processing_times, dtype=float)
    n_jobs = pt.shape[0]
    n_machines = pt.shape[1]
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    # Step 1: NEH heuristic
    neh_perm = _neh_order(pt)
    neh_ms = _makespan_fast(neh_perm, pt)
    
    best_perm = list(neh_perm)
    best_ms = neh_ms
    
    # Step 2: Local search on NEH result
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        ls_perm = _local_search(neh_perm, pt, max_iters=200, 
                                 time_limit=remaining_time - 2, 
                                 start_time=start_time)
        ls_ms = _makespan_fast(ls_perm, pt)
        if ls_ms < best_ms:
            best_ms = ls_ms
            best_perm = list(ls_perm)
    
    # Step 3: Genetic algorithm if time remains
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        # Use NEH-based initialization for GA
        ga_perm = _ga(best_perm, pt, pop_size=40, generations=30,
                      time_limit=remaining_time - 5,
                      start_time=start_time)
        ga_ms = _makespan_fast(ga_perm, pt)
        if ga_ms < best_ms:
            best_ms = ga_ms
            best_perm = list(ga_perm)
        
        # Step 4: Final local search on GA result
        remaining_time = deadline - time.time()
        if remaining_time > 5:
            final_perm = _local_search(best_perm, pt, max_iters=100,
                                       time_limit=remaining_time - 2,
                                       start_time=start_time)
            final_ms = _makespan_fast(final_perm, pt)
            if final_ms < best_ms:
                best_ms = final_ms
                best_perm = list(final_perm)
    
    return best_perm
# EVOLVE-BLOCK-END
