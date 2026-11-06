"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _calc_makespan(processing_times, perm):
    """
    Calculate makespan for a given permutation using the standard flow-shop recurrence.
    
    Args:
        processing_times: np.ndarray of shape (n_jobs, n_machines)
        perm: list or array of job indices in order
    
    Returns:
        makespan (float)
    """
    if len(perm) == 0:
        return 0.0
    
    n_machines = processing_times.shape[1]
    if n_machines == 1:
        return float(np.sum(processing_times[perm]))
    
    # completion[m] = completion time of current job on machine m
    completion = np.zeros(n_machines, dtype=np.float64)
    prev_c = 0.0
    
    for job in perm:
        row = processing_times[job]
        for m in range(n_machines):
            if m == 0:
                new_c = completion[0] + row[0]
            else:
                # max(completion[m] (prev job on same machine), prev_c (same job on prev machine))
                if completion[m] > prev_c:
                    new_c = completion[m] + row[m]
                else:
                    new_c = prev_c + row[m]
            completion[m] = new_c
            prev_c = new_c
    
    return float(completion[-1])


def _neh_heuristic(processing_times):
    """NEH heuristic for initial solution."""
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    totals = processing_times.sum(axis=1)
    sorted_jobs = np.argsort(-totals)
    
    current_perm = [int(sorted_jobs[0])]
    
    for i in range(1, n_jobs):
        job = int(sorted_jobs[i])
        best_pos = 0
        best_makespan = float('inf')
        
        for pos in range(len(current_perm) + 1):
            candidate_perm = current_perm[:pos] + [job] + current_perm[pos:]
            ms = _calc_makespan(processing_times, candidate_perm)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
                
        current_perm = current_perm[:best_pos] + [job] + current_perm[best_pos:]
        
    return current_perm


def _order_crossover(parent1, parent2):
    """Order-based crossover (OX1) for permutations."""
    n = len(parent1)
    if n < 2:
        return list(parent1)
    
    # Choose two cut points
    i, j = sorted(random.sample(range(n), 2))
    
    child = [-1] * n
    # Copy segment from parent1
    child[i:j+1] = parent1[i:j+1]
    
    # Fill remaining positions with values from parent2 in order
    segment_set = set(parent1[i:j+1])
    remaining = [x for x in parent2 if x not in segment_set]
    
    # Fill starting from position j+1, wrapping around
    pos = (j + 1) % n
    for val in remaining:
        while child[pos] != -1:
            pos = (pos + 1) % n
        child[pos] = val
        pos = (pos + 1) % n
    
    return child


def _swap_mutation(perm, rate=0.1):
    """Swap mutation: randomly swap two elements with probability rate."""
    n = len(perm)
    if n < 2:
        return list(perm)
    
    child = list(perm)
    for _ in range(int(rate * n)):
        i, j = random.sample(range(n), 2)
        child[i], child[j] = child[j], child[i]
    
    return child


def _insertion_mutation(perm):
    """Insertion mutation: remove a random element and insert it at another position."""
    n = len(perm)
    if n < 2:
        return list(perm)
    
    child = list(perm)
    i = random.randint(0, n - 1)
    val = child.pop(i)
    j = random.randint(0, n - 1)
    child.insert(j, val)
    
    return child


def _genetic_algorithm(processing_times, initial_perm, time_limit):
    """
    Genetic Algorithm for flow-shop scheduling.
    
    Args:
        processing_times: np.ndarray of shape (n_jobs, n_machines)
        initial_perm: initial permutation
        time_limit: absolute time limit (epoch seconds)
    
    Returns:
        (best_perm, best_makespan)
    """
    n_jobs = len(initial_perm)
    if n_jobs == 0:
        return [], 0.0
    if n_jobs == 1:
        return list(initial_perm), _calc_makespan(processing_times, initial_perm)
    
    # Population size
    pop_size = min(40, max(10, n_jobs // 2))
    
    # Initialize population
    population = [list(initial_perm)]
    # Add diverse individuals
    for _ in range(pop_size - 1):
        if random.random() < 0.3:
            # Reverse
            pop = list(initial_perm)
            pop.reverse()
        elif random.random() < 0.5:
            # Random shuffle
            pop = list(range(n_jobs))
            random.shuffle(pop)
        else:
            # NEH-like with random perturbation
            pop = _neh_heuristic(processing_times)
            # Perturb
            for _ in range(n_jobs // 10):
                i = random.randint(0, n_jobs - 1)
                val = pop.pop(i)
                j = random.randint(0, n_jobs - 1)
                pop.insert(j, val)
        population.append(pop)
    
    # Evaluate initial population
    fitnesses = [_calc_makespan(processing_times, ind) for ind in population]
    best_idx = int(np.argmin(fitnesses))
    best_perm = list(population[best_idx])
    best_ms = fitnesses[best_idx]
    
    # GA parameters
    tournament_size = 5
    crossover_rate = 0.8
    mutation_rate = 0.3
    
    generation = 0
    while time.time() < time_limit - 2.0:
        generation += 1
        
        # Selection: tournament selection
        new_population = []
        for _ in range(pop_size):
            # Tournament
            candidates = random.sample(range(pop_size), min(tournament_size, pop_size))
            winner = min(candidates, key=lambda i: fitnesses[i])
            parent1 = population[winner]
            
            candidates = random.sample(range(pop_size), min(tournament_size, pop_size))
            winner = min(candidates, key=lambda i: fitnesses[i])
            parent2 = population[winner]
            
            # Crossover
            if random.random() < crossover_rate and len(parent1) > 1:
                child = _order_crossover(parent1, parent2)
            else:
                child = list(parent1)
            
            # Mutation
            if random.random() < mutation_rate:
                if random.random() < 0.5:
                    child = _swap_mutation(child, rate=mutation_rate)
                else:
                    child = _insertion_mutation(child)
            
            new_population.append(child)
        
        # Evaluate new population
        new_fitnesses = [_calc_makespan(processing_times, ind) for ind in new_population]
        
        # Elitism: keep best individual
        best_idx = int(np.argmin(new_fitnesses))
        if new_fitnesses[best_idx] < best_ms:
            best_ms = new_fitnesses[best_idx]
            best_perm = list(new_population[best_idx])
        
        # Replace population
        population = new_population
        fitnesses = new_fitnesses
        
        # Occasional random restart to maintain diversity
        if generation % 50 == 0 and random.random() < 0.3:
            idx = random.randint(0, pop_size - 1)
            if random.random() < 0.5:
                population[idx] = list(range(n_jobs))
                random.shuffle(population[idx])
            else:
                population[idx] = _neh_heuristic(processing_times)
            fitnesses[idx] = _calc_makespan(processing_times, population[idx])
    
    return best_perm, best_ms


def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # 1. Initial solution: NEH heuristic
    initial_perm = _neh_heuristic(processing_times)
    initial_ms = _calc_makespan(processing_times, initial_perm)
    
    best_perm = list(initial_perm)
    best_ms = initial_ms
    
    # 2. Genetic Algorithm with time budget
    # Use 60% of time for first GA run
    ga_time_limit = start_time + (deadline - start_time) * 0.6
    
    ga_best, ga_ms = _genetic_algorithm(processing_times, initial_perm, ga_time_limit)
    if ga_ms < best_ms:
        best_ms = ga_ms
        best_perm = ga_best
    
    # 3. Second GA run with different initial solution if time remains
    remaining_time = deadline - time.time()
    if remaining_time > 15.0:
        # Create a different initial solution
        if random.random() < 0.5:
            alt_initial = list(initial_perm)
            alt_initial.reverse()
        else:
            alt_initial = list(range(n_jobs))
            random.shuffle(alt_initial)
        
        alt_ms = _calc_makespan(processing_times, alt_initial)
        
        ga2_best, ga2_ms = _genetic_algorithm(processing_times, alt_initial, deadline - 5.0)
        if ga2_ms < best_ms:
            best_ms = ga2_ms
            best_perm = ga2_best
    
    return best_perm
# EVOLVE-BLOCK-END
