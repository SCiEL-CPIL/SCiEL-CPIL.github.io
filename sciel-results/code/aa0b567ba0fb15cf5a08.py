"""
Genetic Algorithm approach for Flow-Shop Problem.
Strategy:
1. Vectorized makespan evaluation using NumPy for speed.
2. Population-based search with Crossover (Order Crossover) and Mutation (Swap/Insert).
3. Elitism to preserve best solutions.
4. Time-budgeted loop to maximize search depth within 120s limit.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _makespan_vectorized(perm, processing_times):
    """
    Compute makespan for a single permutation efficiently.
    perm: array of job indices
    processing_times: 2D array (n_jobs, n_machines)
    """
    # Get processing times for the jobs in the order of perm
    # shape: (n_jobs, n_machines)
    job_times = processing_times[perm]
    
    # For each machine, the completion time is the cumulative sum of processing times
    # for that machine, but constrained by the previous machine's completion time for the same job.
    # Actually, the standard flow shop completion time C[j][m] is:
    # C[j][0] = sum_{i=0}^{j} p[i][0]
    # C[j][m] = max(C[j-1][m], C[j][m-1]) + p[j][m]
    
    # We can compute this iteratively, but we want to be fast.
    # Let's use the recurrence relation.
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    
    # C[j][m] is the completion time of job j on machine m.
    # We only need the last row (last job) to get the makespan.
    # But we can compute it row by row.
    
    # Initialize completion times for the first job
    # C[0][m] = sum_{k=0}^{m} p[0][k]
    c_prev = np.cumsum(job_times[0])
    
    for j in range(1, n_jobs):
        # C[j][0] = C[j-1][0] + p[j][0]
        c_curr = np.empty(n_machines)
        c_curr[0] = c_prev[0] + job_times[j, 0]
        
        if n_machines > 1:
            # C[j][m] = max(C[j-1][m], C[j][m-1]) + p[j][m]
            # Vectorized:
            # max_term = np.maximum(c_prev[1:], c_curr[:-1])
            # c_curr[1:] = max_term + job_times[j, 1:]
            c_curr[1:] = np.maximum(c_prev[1:], c_curr[:-1]) + job_times[j, 1:]
            
        c_prev = c_curr
        
    return c_prev[-1]

def _order_crossover(p1, p2):
    """
    Order Crossover (OX1).
    """
    n = len(p1)
    # Select two random cut points
    i, j = sorted(np.random.choice(n, 2, replace=False))
    
    child = [None] * n
    # Copy segment from p1
    child[i:j+1] = p1[i:j+1]
    
    # Fill remaining positions with elements from p2 in order, skipping those already in child
    fill = [x for x in p2 if x not in child[i:j+1]]
    
    pos = i
    # Fill backwards from i to 0, then forwards from j+1 to n-1
    # Actually, standard OX1 fills the rest of the array in the order they appear in p2.
    # Let's implement the standard "fill the rest" logic.
    
    # Indices to fill
    indices_to_fill = list(range(0, i)) + list(range(j+1, n))
    
    for idx, val in zip(indices_to_fill, fill):
        child[idx] = val
        
    return np.array(child)

def _swap_mutation(ind, rate=0.1):
    """
    Swap mutation: swap two random jobs.
    """
    ind = ind.copy()
    n = len(ind)
    if n < 2:
        return ind
    # Number of swaps
    n_swaps = max(1, int(n * rate))
    for _ in range(n_swaps):
        i, j = np.random.choice(n, 2, replace=False)
        ind[i], ind[j] = ind[j], ind[i]
    return ind

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan using a Genetic Algorithm.
    """
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs <= 1:
        return list(range(n_jobs))
        
    # Population size
    pop_size = min(100, max(10, n_jobs * 2))
    
    # Initialize population
    # Mix of random and sorted (LPT/SPT)
    population = []
    
    # LPT
    lpt_order = np.argsort(-processing_times.sum(axis=1))
    population.append(lpt_order)
    
    # SPT
    spt_order = np.argsort(processing_times.sum(axis=1))
    population.append(spt_order)
    
    # Random
    while len(population) < pop_size:
        population.append(np.random.permutation(n_jobs))
        
    # Evaluate initial population
    makespans = np.array([_makespan_vectorized(ind, processing_times) for ind in population])
    
    best_perm = population[np.argmin(makespans)]
    best_makespan = np.min(makespans)
    
    # GA Loop
    generation = 0
    while time.time() < deadline:
        generation += 1
        
        # Selection: Tournament Selection
        new_population = []
        
        # Elitism: Keep the best individual
        best_idx = np.argmin(makespans)
        new_population.append(population[best_idx].copy())
        
        while len(new_population) < pop_size:
            # Select two parents
            idx1 = np.random.randint(pop_size)
            idx2 = np.random.randint(pop_size)
            
            p1 = population[idx1]
            p2 = population[idx2]
            
            # Crossover
            if np.random.rand() < 0.8:
                child = _order_crossover(p1, p2)
            else:
                child = p1.copy()
                
            # Mutation
            child = _swap_mutation(child)
            
            new_population.append(child)
            
        # Evaluate new population
        makespans = np.array([_makespan_vectorized(ind, processing_times) for ind in new_population])
        
        # Update best
        current_best_idx = np.argmin(makespans)
        if makespans[current_best_idx] < best_makespan:
            best_makespan = makespans[current_best_idx]
            best_perm = new_population[current_best_idx].copy()
            
        # Replace population
        population = new_population
        
    return list(best_perm)
# EVOLVE-BLOCK-END
