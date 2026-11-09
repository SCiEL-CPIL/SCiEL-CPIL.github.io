"""
Flow-shop permutation construction using a Genetic Algorithm with Order Crossover
(OX), multiple mutation operators, and diverse NEH/CDS seeding.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _makespan(order, P):
    """Compute makespan for a given job order on processing_times matrix P."""
    if len(order) == 0:
        return 0.0
    n_machines = P.shape[1]
    c_machines = np.zeros(n_machines)
    c_curr = 0.0
    
    for j in order:
        row = P[j]
        for m in range(n_machines):
            c_curr = max(c_machines[m], c_curr) + row[m]
            c_machines[m] = c_curr
            
    return c_curr

def _johnsons_rule(P2):
    """Apply Johnson's Rule to a 2-machine matrix P2 (N x 2)."""
    n_jobs = P2.shape[0]
    set_A = []
    set_B = []
    for j in range(n_jobs):
        if P2[j, 0] < P2[j, 1]:
            set_A.append((P2[j, 0], j))
        else:
            set_B.append((P2[j, 1], j))
            
    set_A.sort(key=lambda x: x[0])
    set_B.sort(key=lambda x: -x[0])
    
    sequence = [j for _, j in set_A] + [j for _, j in set_B]
    return sequence

def _cds_heuristic(P):
    """Campbell, Duke, and Sarma (CDS) heuristic."""
    n_jobs, n_machines = P.shape
    if n_machines < 2:
        return list(range(n_jobs))
        
    best_seq = None
    best_ms = float('inf')
    
    for i in range(1, n_machines):
        P2 = np.zeros((n_jobs, 2))
        P2[:, 0] = P[:, :i].sum(axis=1)
        P2[:, 1] = P[:, n_machines-i:].sum(axis=1)
        
        seq = _johnsons_rule(P2)
        ms = _makespan(seq, P)
        
        if ms < best_ms:
            best_ms = ms
            best_seq = seq
            
    return best_seq if best_seq else list(range(n_jobs))

def _neh_heuristic(P, variant=0):
    """
    NEH (Nawaz-Ensecure-Ham) heuristic with variants for diversity.
    variant=0: standard NEH (descending total)
    variant=1: ascending total
    variant=2: random order
    variant=3: descending average
    variant=4: ascending average
    """
    n_jobs, n_machines = P.shape
    totals = P.sum(axis=1)
    averages = P.mean(axis=1)
    
    if variant == 0:
        order_indices = np.argsort(-totals)
    elif variant == 1:
        order_indices = np.argsort(totals)
    elif variant == 2:
        order_indices = np.random.permutation(n_jobs)
    elif variant == 3:
        order_indices = np.argsort(-averages)
    elif variant == 4:
        order_indices = np.argsort(averages)
    else:
        order_indices = np.argsort(-totals)
    
    current_seq = []
    for idx in order_indices:
        best_pos = 0
        best_makespan = float('inf')
        
        for pos in range(len(current_seq) + 1):
            new_seq = current_seq[:pos] + [idx] + current_seq[pos:]
            ms = _makespan(new_seq, P)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        
        current_seq.insert(best_pos, idx)
        
    return current_seq

def _order_crossover(parent1, parent2):
    """
    Order Crossover (OX1) for permutations.
    Selects a random substring from parent1, places it in the child at the same position,
    then fills remaining positions with genes from parent2 in order.
    """
    n = len(parent1)
    if n < 2:
        return parent1[:]
    
    # Choose two cut points
    i, j = sorted(random.sample(range(n), 2))
    
    # Create child with None placeholders
    child = [None] * n
    
    # Copy the substring from parent1
    for k in range(i, j + 1):
        child[k] = parent1[k]
    
    # Get the set of genes already in the child
    filled = set(child[i:j+1])
    
    # Fill remaining positions from parent2 in order
    parent2_idx = 0
    for pos in range(n):
        if child[pos] is None:
            while parent2_idx < n and parent2[parent2_idx] in filled:
                parent2_idx += 1
            if parent2_idx >= n:
                # Should not happen if parents are valid permutations
                break
            child[pos] = parent2[parent2_idx]
            parent2_idx += 1
    
    return child

def _swap_mutation(individual, rate=0.1):
    """Randomly swap two elements."""
    n = len(individual)
    if n < 2:
        return individual[:]
    
    child = individual[:]
    n_swaps = max(1, int(rate * n))
    for _ in range(n_swaps):
        i, j = random.sample(range(n), 2)
        child[i], child[j] = child[j], child[i]
    
    return child

def _insertion_mutation(individual, rate=0.1):
    """Randomly move an element to a new position."""
    n = len(individual)
    if n < 2:
        return individual[:]
    
    child = individual[:]
    n_moves = max(1, int(rate * n))
    for _ in range(n_moves):
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i == j:
            continue
        elem = child[i]
        del child[i]
        # Adjust j if needed
        if j > i:
            j -= 1
        child.insert(j, elem)
    
    return child

def _reversal_mutation(individual, rate=0.05):
    """Randomly reverse a substring."""
    n = len(individual)
    if n < 2:
        return individual[:]
    
    child = individual[:]
    n_revs = max(1, int(rate * n))
    for _ in range(n_revs):
        i, j = sorted(random.sample(range(n), 2))
        child[i:j+1] = child[i:j+1][::-1]
    
    return child

def _local_search_insertion(order, P, max_iterations=50):
    """
    Fast insertion local search: try moving each job to its best position.
    Returns improved order and makespan.
    """
    n = len(order)
    if n < 2:
        return order, _makespan(order, P)
    
    current_order = list(order)
    current_ms = _makespan(current_order, P)
    
    for _ in range(max_iterations):
        improved = False
        
        for i in range(n):
            job_i = current_order[i]
            remaining = current_order[:i] + current_order[i+1:]
            
            best_pos = i
            best_ms = current_ms
            
            for pos in range(len(remaining) + 1):
                # Skip if inserting back to same position
                if pos == i:
                    continue
                new_order = remaining[:pos] + [job_i] + remaining[pos:]
                ms = _makespan(new_order, P)
                if ms < best_ms:
                    best_ms = ms
                    best_pos = pos
            
            if best_pos != i:
                new_order = remaining[:best_pos] + [job_i] + remaining[best_pos:]
                current_order = new_order
                current_ms = best_ms
                improved = True
        
        if not improved:
            break
    
    return current_order, current_ms

def _genetic_algorithm(P, time_limit, pop_size=None):
    """
    Genetic Algorithm with OX crossover and multiple mutation operators.
    Seeded with diverse NEH/CDS solutions.
    """
    n_jobs, n_machines = P.shape
    if n_jobs < 2:
        return list(range(n_jobs)), _makespan(list(range(n_jobs)), P)
    
    # Determine population size based on problem size
    if pop_size is None:
        pop_size = max(20, min(50, n_jobs))
    
    start_time = time.time()
    
    # Generate diverse initial population
    population = []
    
    # Add multiple NEH variants
    for variant in range(5):
        seq = _neh_heuristic(P, variant=variant)
        population.append(seq)
    
    # Add CDS
    seq_cds = _cds_heuristic(P)
    population.append(seq_cds)
    
    # Add random permutations to fill population
    while len(population) < pop_size:
        population.append(list(np.random.permutation(n_jobs)))
    
    # Evaluate initial population
    fitness = [_makespan(seq, P) for seq in population]
    
    # Track best
    best_idx = int(np.argmin(fitness))
    best_order = population[best_idx][:]
    best_ms = fitness[best_idx]
    
    # GA main loop
    generation = 0
    stagnation_counter = 0
    improvement_threshold = 1e-6
    
    while time.time() < time_limit - 2.0:
        generation += 1
        
        # Check time per generation
        if generation % 50 == 0 and time.time() > time_limit - 5.0:
            break
        
        # Create new population
        new_population = []
        
        # Elitism: keep top 2
        sorted_indices = np.argsort(fitness)
        for idx in sorted_indices[:2]:
            new_population.append(population[idx][:])
        
        # Generate offspring until population is filled
        while len(new_population) < pop_size:
            # Tournament selection
            def tournament_select(pop, fit, k=3):
                indices = random.sample(range(len(pop)), k)
                best_i = min(indices, key=lambda x: fit[x])
                return pop[best_i][:]
            
            parent1 = tournament_select(population, fitness)
            parent2 = tournament_select(population, fitness)
            
            # Crossover
            if random.random() < 0.7:  # Crossover rate
                child = _order_crossover(parent1, parent2)
            else:
                child = parent1[:]
            
            # Mutation
            r = random.random()
            if r < 0.4:
                child = _swap_mutation(child, rate=0.1)
            elif r < 0.7:
                child = _insertion_mutation(child, rate=0.1)
            elif r < 0.9:
                child = _reversal_mutation(child, rate=0.05)
            else:
                child = _swap_mutation(child, rate=0.15)
            
            new_population.append(child)
        
        # Evaluate new population
        fitness = [_makespan(seq, P) for seq in new_population]
        
        # Update best
        gen_best_idx = int(np.argmin(fitness))
        if fitness[gen_best_idx] < best_ms - improvement_threshold:
            best_ms = fitness[gen_best_idx]
            best_order = new_population[gen_best_idx][:]
            stagnation_counter = 0
        else:
            stagnation_counter += 1
        
        # If stagnant for many generations, inject some random solutions
        if stagnation_counter > 20:
            for _ in range(pop_size // 10):
                idx = random.randint(0, len(new_population) - 1)
                new_population[idx] = list(np.random.permutation(n_jobs))
                fitness[idx] = _makespan(new_population[idx], P)
            # Re-evaluate best
            gen_best_idx = int(np.argmin(fitness))
            if fitness[gen_best_idx] < best_ms - improvement_threshold:
                best_ms = fitness[gen_best_idx]
                best_order = new_population[gen_best_idx][:]
            stagnation_counter = 0
        
        population = new_population
        
        # Check if we should stop early
        if generation > 200 and stagnation_counter > 50:
            break
    
    # Final local search polish on the best solution
    if time.time() < time_limit - 1.0:
        best_order, best_ms = _local_search_insertion(best_order, P, max_iterations=100)
    
    return best_order, best_ms

def order_jobs(processing_times):
    """
    Order jobs to minimize makespan using a Genetic Algorithm with
    Order Crossover, diverse NEH/CDS seeding, and local search polish.
    """
    P = np.asarray(processing_times, dtype=np.float64)
    if P.size == 0:
        return []
        
    n_jobs, n_machines = P.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
        
    start_time = time.time()
    time_limit = 145.0
    
    # Run GA
    best_order, best_makespan = _genetic_algorithm(P, time_limit)
    
    # Final check: if we have time, do another quick local search
    if time.time() < time_limit - 0.5:
        best_order, best_makespan = _local_search_insertion(best_order, P, max_iterations=50)
    
    return best_order
# EVOLVE-BLOCK-END
