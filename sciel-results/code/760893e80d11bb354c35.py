"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _vectorized_makespan(perms, pt):
    """
    Compute makespan for a batch of permutations.
    perms: np.ndarray of shape [batch_size, n_jobs] containing job indices
    pt: np.ndarray of shape [n_jobs, n_machines]
    Returns: np.ndarray of shape [batch_size] with makespans
    """
    batch_size, n_jobs = perms.shape
    n_machines = pt.shape[1]
    
    # Get processing times for each job in each permutation
    # shape: [batch_size, n_jobs, n_machines]
    p_times = pt[perms]
    
    # We need to compute completion times C[m, j] for each batch element
    # C[m, j] = max(C[m-1, j], C[m, j-1]) + p_times[j, m]
    # This is a dynamic programming recurrence that is hard to fully vectorize across the job dimension
    # because C[m, j] depends on C[m, j-1].
    # However, we can vectorize across the batch dimension (axis 0).
    
    # Initialize C for the first machine
    # C[0, j] = sum(p_times[0..j, 0])
    # We can use cumsum along axis 1 for the first machine
    C = np.zeros((batch_size, n_machines, n_jobs))
    
    # Machine 0
    C[:, 0, :] = np.cumsum(p_times[:, :, 0], axis=1)
    
    # For machines m > 0
    for m in range(1, n_machines):
        # C[m, 0] = C[m-1, 0] + p_times[0, m]
        C[:, m, 0] = C[:, m-1, 0] + p_times[:, 0, m]
        
        # For j > 0: C[m, j] = max(C[m-1, j], C[m, j-1]) + p_times[j, m]
        # We must iterate j sequentially because of the dependency on C[m, j-1]
        # But we can vectorize across the batch (axis 0)
        for j in range(1, n_jobs):
            C[:, m, j] = np.maximum(C[:, m-1, j], C[:, m, j-1]) + p_times[:, j, m]
            
    # Makespan is C[:, -1, -1]
    return C[:, -1, -1]

def _de_mutation(pop, best_idx, r1, r2, n_jobs):
    """
    Discrete DE mutation for permutations.
    v = x_best + F * (x_r1 - x_r2)
    For permutations, we interpret the 'difference' as a set of positions to swap or insert.
    A common approach: 
    1. Take x_best.
    2. For each position i, with probability F, swap with position i in x_r1 or x_r2?
    
    Alternative robust approach for DE on permutations:
    1. Identify the jobs that are in different positions in x_r1 and x_r2 compared to x_best.
    2. Construct a new permutation by starting with x_best and applying moves derived from the difference.
    
    Simpler and effective heuristic for DE-like mutation:
    1. Start with x_best.
    2. Pick a random segment from x_r1 and insert it into x_best? No, that's crossover.
    
    Let's use a "Swap-based" DE mutation:
    1. Create a mask of length n_jobs.
    2. For each position i, if random() < F, we "take" the job from x_r1[i] and put it in position i, 
       and shift the others? This is complex.
    
    Let's use a simpler "Directed Swap" mutation:
    1. Start with x_best.
    2. Generate a random permutation p of indices.
    3. For i in p:
       With probability F, swap position i in the current solution with position i in x_r1.
       This creates a solution that is a blend of best and r1.
    """
    x_best = pop[best_idx]
    x_r1 = pop[r1]
    x_r2 = pop[r2]
    
    # We will create a new permutation by starting with x_best
    # and randomly swapping elements with x_r1 and x_r2 based on a binomial mask
    # This is a form of "Binomial Crossover" which is standard in DE.
    
    mask = np.random.rand(n_jobs) < 0.5 # 50% probability of taking from r1/r2 mix
    # Actually, standard DE binomial crossover:
    # v[i] = x_best[i] if rand() > F else (x_r1[i] if rand() > 0.5 else x_r2[i])
    # But this results in a multiset, not a permutation, unless we fix duplicates.
    
    # A better "Permutation DE" mutation:
    # 1. Take x_best.
    # 2. Select a random subset of positions (controlled by F, e.g., 20% of positions).
    # 3. For these positions, swap the job with the job at the same position in x_r1.
    # This preserves the permutation property and moves towards x_r1.
    
    new_perm = x_best.copy()
    # Number of swaps proportional to F * n_jobs
    n_swaps = int(0.2 * n_jobs) # Fixed crossover rate for simplicity in mutation step
    if n_swaps == 0:
        return new_perm
    
    # Choose random positions to swap
    swap_pos = np.random.choice(n_jobs, size=n_swaps, replace=False)
    
    for pos in swap_pos:
        # Swap new_perm[pos] with x_r1[pos]
        # Find where x_r1[pos] is in new_perm
        target_job = x_r1[pos]
        current_job = new_perm[pos]
        
        # If they are the same, skip
        if target_job == current_job:
            continue
            
        # Find index of target_job in new_perm
        idx_target = np.where(new_perm == target_job)[0][0]
        
        # Swap
        new_perm[pos] = target_job
        new_perm[idx_target] = current_job
        
    return new_perm

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan using a Genetic Algorithm 
    with Discrete DE-style mutation and vectorized evaluation.
    """
    start_time = time.time()
    deadline = start_time + 115.0
    
    n_jobs, n_machines = processing_times.shape
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    # 1. Initial Population
    # Size 50-100 depending on n_jobs
    pop_size = min(100, max(50, n_jobs * 2))
    
    # NEH Heuristic for the first individual
    totals = processing_times.sum(axis=1)
    neh_order = list(np.argsort(-totals))
    
    # SPT for the second
    spt_order = list(np.argsort(totals))
    
    # Random for the rest
    random.seed(42)
    population = []
    population.append(np.array(neh_order))
    population.append(np.array(spt_order))
    
    for _ in range(pop_size - 2):
        perm = list(range(n_jobs))
        random.shuffle(perm)
        population.append(np.array(perm))
    
    population = np.array(population, dtype=int)
    
    # Evaluate initial population
    fitness = _vectorized_makespan(population, processing_times)
    
    best_perm = population[np.argmin(fitness)]
    best_mk = np.min(fitness)
    
    # 2. Evolution Loop
    # We will run generations until time is up.
    # Each generation:
    # 1. Select best, r1, r2
    # 2. Mutate
    # 3. Evaluate new offspring
    # 4. Replace worst if improved
    
    # To maximize throughput, we evaluate in batches.
    # Let's process one offspring per iteration but vectorize the evaluation of the *current* population
    # if we were to do selection. But here we do individual replacement.
    
    # Optimization: We can evaluate a batch of offspring at once.
    # Let's try to generate a batch of offspring per iteration.
    batch_size = 10
    
    while time.time() < deadline:
        # Generate batch of offspring
        offspring = np.zeros((batch_size, n_jobs), dtype=int)
        
        for i in range(batch_size):
            # Select best
            best_idx = np.argmin(fitness)
            
            # Select r1, r2 randomly (distinct from best and each other)
            r1 = np.random.randint(0, pop_size)
            while r1 == best_idx:
                r1 = np.random.randint(0, pop_size)
                
            r2 = np.random.randint(0, pop_size)
            while r2 == best_idx or r2 == r1:
                r2 = np.random.randint(0, pop_size)
                
            # Mutate
            offspring[i] = _de_mutation(population, best_idx, r1, r2, n_jobs)
            
        # Evaluate offspring
        off_fitness = _vectorized_makespan(offspring, processing_times)
        
        # Selection: Replace the worst individual in the population if an offspring is better
        # We can do this efficiently by finding the global worst in population
        # and comparing with the best offspring.
        # For simplicity and speed, we replace the individual at the index of the worst fitness 
        # if the offspring is better, or just keep the top K.
        
        # Let's do a simple tournament:
        # Combine population and offspring, keep top pop_size
        # This is O(pop_size log pop_size)
        
        combined = np.vstack([population, offspring])
        combined_fit = np.concatenate([fitness, off_fitness])
        
        # Get indices of the best pop_size individuals
        # argsort is O(N log N). With N=110, this is fast.
        sorted_indices = np.argsort(combined_fit)
        keep_indices = sorted_indices[:pop_size]
        
        population = combined[keep_indices]
        fitness = combined_fit[keep_indices]
        
        # Update global best
        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < best_mk:
            best_mk = fitness[current_best_idx]
            best_perm = population[current_best_idx]
            
        # Time check
        if time.time() > deadline - 5.0:
            break
            
    # Final Polish: Local Search on the best solution found
    # Use a fast 2-opt or Insert local search on the single best permutation
    # This leverages the global search to find a good basin, then local search to refine.
    
    if time.time() < deadline - 2.0:
        current_perm = list(best_perm)
        current_mk = best_mk
        improved = True
        n = n_jobs
        
        # Simple Insert Local Search
        # Try moving each job to every other position
        while improved and time.time() < deadline:
            improved = False
            for i in range(n):
                if time.time() > deadline:
                    break
                job = current_perm[i]
                for j in range(n):
                    if i == j or i == j + 1:
                        continue
                    # Move job from i to j
                    new_perm = current_perm[:i] + current_perm[i+1:]
                    # Adjust j if j > i
                    insert_j = j if j > i else j
                    if insert_j >= len(new_perm):
                        insert_j = len(new_perm) - 1
                    new_perm = new_perm[:insert_j] + [job] + new_perm[insert_j:]
                    
                    # Evaluate
                    # Use vectorized makespan for a single perm (batch size 1)
                    mk = _vectorized_makespan(np.array([new_perm]), processing_times)[0]
                    
                    if mk < current_mk - 1e-9:
                        current_perm = new_perm
                        current_mk = mk
                        improved = True
                        break # Restart loop for this job? Or just continue?
                        # Breaking inner loop to restart outer is safer to avoid index issues
                if improved:
                    break
                    
        if current_mk < best_mk:
            best_mk = current_mk
            best_perm = np.array(current_perm)
            
    return list(best_perm)
# EVOLVE-BLOCK-END
