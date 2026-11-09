"""
Flow-shop permutation construction using a Permutation Genetic Algorithm (GA).
Strategy:
1. Initialize with a diverse population of NEH-based and random permutations.
2. Use a vectorized makespan evaluation for speed.
3. Employ a time-budgeted evolutionary loop with:
   - Tournament selection
   - Order Crossover (OX) to recombine parents (the 'cross' operator in question)
   - Insertion and Swap mutations for local refinement and diversity
   - Elitism to preserve the best solutions
4. The crossover operator allows the algorithm to combine good subsequences from
   different parents, potentially escaping local minima more effectively than
   pure mutation-based local search.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _makespan_vec(order, P):
    """
    Compute makespan for a given job order using a vectorized approach.
    P: (n_jobs, n_machines)
    order: list/array of job indices
    """
    if len(order) == 0:
        return 0.0
    
    n_machines = P.shape[1]
    seq_P = P[order]
    
    # c[m] is the completion time of the last job on machine m
    # We simulate the flow shop
    # For each job j in order:
    #   start_time = 0
    #   for m in range(n_machines):
    #       start_time = max(start_time, c[m])
    #       end_time = start_time + seq_P[j][m]
    #       c[m] = end_time
    #   makespan = c[-1] (last machine's completion time for the last job)
    
    # Vectorized simulation is tricky due to sequential dependency on c[m].
    # However, we can use a loop over jobs which is fast for small n_jobs.
    # For larger n_jobs, we might need a different approach, but flow-shop
    # instances in these tasks are usually small enough (n_jobs < 100).
    
    c = np.zeros(n_machines)
    makespan = 0.0
    for j in range(len(order)):
        row = seq_P[j]
        # For each machine, the start time is max(prev_end, c[m])
        # end[m] = max(start[m], c[m]) + P[j][m]
        # start[m] = end[m-1] (for m>0), 0 for m=0
        # This is a sequential process.
        # Let's use a simple loop, it's usually fast enough.
        current_time = 0.0
        for m in range(n_machines):
            current_time = max(current_time, c[m]) + row[m]
            c[m] = current_time
        makespan = current_time
        
    return makespan

def _neh_heuristic(P):
    """
    NEH heuristic for initial solution.
    """
    n_jobs, n_machines = P.shape
    totals = P.sum(axis=1)
    order_indices = np.argsort(-totals)
    
    current_seq = []
    for idx in order_indices:
        best_pos = 0
        best_makespan = float('inf')
        
        for pos in range(len(current_seq) + 1):
            new_seq = current_seq[:pos] + [idx] + current_seq[pos:]
            ms = _makespan_vec(new_seq, P)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        
        current_seq.insert(best_pos, idx)
        
    return current_seq

def _order_crossover(parent1, parent2):
    """
    Order Crossover (OX) for permutations.
    """
    n = len(parent1)
    if n <= 2:
        return parent1.copy()
    
    # Pick two cut points
    i, j = sorted(random.sample(range(n), 2))
    
    child = [None] * n
    # Copy segment from parent1
    for k in range(i, j + 1):
        child[k] = parent1[k]
    
    # Fill the rest with elements from parent2 in order, skipping those in child
    fill_pos = j + 1
    for val in parent2:
        if val not in child:
            if fill_pos >= n:
                fill_pos = 0
            child[fill_pos] = val
            fill_pos += 1
            if fill_pos >= n:
                fill_pos = 0
            if all(x is not None for x in child):
                break
                
    return child

def _insertion_mutation(individual, P, rate=0.5):
    """
    Perform insertion mutation: remove a job and re-insert it at a random position.
    """
    n = len(individual)
    if n <= 1:
        return individual
    
    if random.random() < rate:
        new_ind = individual.copy()
        pos1 = random.randint(0, n - 1)
        job = new_ind.pop(pos1)
        pos2 = random.randint(0, n - 1)
        new_ind.insert(pos2, job)
        return new_ind
    return individual

def _swap_mutation(individual, rate=0.5):
    """
    Perform swap mutation: swap two random jobs.
    """
    n = len(individual)
    if n <= 1:
        return individual
    
    if random.random() < rate:
        new_ind = individual.copy()
        i, j = random.sample(range(n), 2)
        new_ind[i], new_ind[j] = new_ind[j], new_ind[i]
        return new_ind
    return individual

def order_jobs(processing_times):
    """
    Order jobs to minimize makespan using a Genetic Algorithm.
    """
    P = np.asarray(processing_times)
    if P.size == 0:
        return []
        
    n_jobs, n_machines = P.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
        
    start_time = time.time()
    # 150s limit, use 145s to be safe
    time_limit = 145.0
    
    # Population size
    pop_size = min(50, max(10, n_jobs * 2))
    
    # Initialize population
    # 50% NEH-based, 50% random
    population = []
    for i in range(pop_size):
        if i < pop_size // 2:
            # NEH with some randomization
            if random.random() < 0.5:
                seq = _neh_heuristic(P)
            else:
                seq = list(range(n_jobs))
                random.shuffle(seq)
        else:
            seq = list(range(n_jobs))
            random.shuffle(seq)
            
        # Apply some local improvement
        # Simple 2-opt / insertion local search for initialization
        improved = True
        while improved:
            improved = False
            for i in range(n_jobs):
                for j in range(i + 1, n_jobs):
                    if abs(i - j) == 1:
                        continue
                    new_seq = seq.copy()
                    new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
                    ms_new = _makespan_vec(new_seq, P)
                    ms_old = _makespan_vec(seq, P)
                    if ms_new < ms_old:
                        seq = new_seq
                        improved = True
                        break
                if improved:
                    break
                    
        population.append(seq)
        
    # Evaluate initial population
    fitness = [_makespan_vec(ind, P) for ind in population]
    
    best_seq = population[fitness.index(min(fitness))].copy()
    best_fitness = min(fitness)
    
    # GA Loop
    generation = 0
    while time.time() - start_time < time_limit:
        generation += 1
        
        # Check time
        if time.time() - start_time > time_limit - 5:
            break
            
        # Create new population
        new_population = []
        
        # Elitism: keep top 10%
        elite_size = max(1, pop_size // 10)
        sorted_indices = np.argsort(fitness)
        for i in range(elite_size):
            new_population.append(population[sorted_indices[i]].copy())
            
        # Fill the rest with offspring
        while len(new_population) < pop_size:
            # Tournament selection
            def tournament_select():
                candidates = random.sample(range(pop_size), min(3, pop_size))
                best_candidate = min(candidates, key=lambda x: fitness[x])
                return population[best_candidate]
                
            parent1 = tournament_select()
            parent2 = tournament_select()
            
            # Crossover (OX) with probability 0.8
            if random.random() < 0.8:
                child = _order_crossover(parent1, parent2)
            else:
                child = parent1.copy()
                
            # Mutation
            child = _insertion_mutation(child, P, rate=0.3)
            child = _swap_mutation(child, rate=0.3)
            
            new_population.append(child)
            
        # Evaluate new population
        fitness = [_makespan_vec(ind, P) for ind in new_population]
        
        # Update best
        current_best_idx = fitness.index(min(fitness))
        if fitness[current_best_idx] < best_fitness:
            best_fitness = fitness[current_best_idx]
            best_seq = new_population[current_best_idx].copy()
            
        population = new_population
        
    return best_seq
# EVOLVE-BLOCK-END
