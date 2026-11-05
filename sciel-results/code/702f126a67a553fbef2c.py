"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import random
import time

def _makespan(perm, processing_times):
    """
    Efficiently calculate makespan for a given permutation.
    perm: list of job indices
    processing_times: np.ndarray [n_jobs, n_machines]
    """
    n_machines = processing_times.shape[1]
    last_machine_time = [0] * n_machines
    max_completion = 0
    
    for job in perm:
        current_job_time = 0
        for m in range(n_machines):
            start_time = max(current_job_time, last_machine_time[m])
            end_time = start_time + processing_times[job, m]
            current_job_time = end_time
            last_machine_time[m] = end_time
            if end_time > max_completion:
                max_completion = end_time
    return max_completion

def _local_search(perm, processing_times, max_iters=10):
    """
    Perform local search using insertion moves to improve a single permutation.
    """
    n_jobs = len(perm)
    current_perm = perm[:]
    current_ms = _makespan(current_perm, processing_times)
    
    for _ in range(max_iters):
        improved = False
        # Try all insertion moves, pick the best one
        best_move = None
        best_ms = current_ms
        
        # Optimization: Limit search to a subset if n_jobs is large, but for small n_jobs (typical in flow shop benchmarks) full search is fine.
        # For safety with time limits, we can stop if we find a significant improvement early, but standard LS checks all.
        # To speed up, we can use a "steepest descent" approach.
        
        # Simple optimization: Only check moves that might be beneficial? 
        # Actually, for small N (e.g., < 50), full O(N^2) check per iteration is fast.
        for i in range(n_jobs):
            job = current_perm[i]
            for j in range(n_jobs):
                if i == j or abs(i-j) == 1:
                    continue
                # Create new perm
                # Remove job at i
                new_perm = current_perm[:i] + current_perm[i+1:]
                # Adjust j index if i < j
                if i < j:
                    j -= 1
                # Insert job at j
                new_perm = new_perm[:j] + [job] + new_perm[j:]
                
                ms = _makespan(new_perm, processing_times)
                if ms < best_ms:
                    best_ms = ms
                    best_move = (i, j, job)
        
        if best_move is not None:
            i, j, job = best_move
            new_perm = current_perm[:i] + current_perm[i+1:]
            if i < j:
                j -= 1
            new_perm = new_perm[:j] + [job] + new_perm[j:]
            current_perm = new_perm
            current_ms = best_ms
            improved = True
        else:
            break
            
    return current_perm, current_ms

def order_jobs(processing_times):
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
        
    # Deterministic random seed for reproducibility
    random.seed(42)
    
    start_time = time.time()
    time_limit = 115.0
    
    # 1. Initial Population Generation
    pop_size = min(100, max(20, n_jobs * 2)) # Adaptive population size
    
    # Heuristic 1: SPT (Shortest Processing Time)
    totals = processing_times.sum(axis=1)
    spt_perm = list(np.argsort(totals))
    
    # Heuristic 2: LPT (Longest Processing Time)
    lpt_perm = list(np.argsort(totals)[::-1])
    
    # Heuristic 3: NEH-like (approximate)
    # Sort by total processing time descending, then insert each job in best position
    neh_perm = lpt_perm[:]
    current_neh = []
    for job in neh_perm:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(current_neh) + 1):
            trial = current_neh[:pos] + [job] + current_neh[pos:]
            ms = _makespan(trial, processing_times)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        current_neh = current_neh[:best_pos] + [job] + current_neh[best_pos:]
    neh_perm = current_neh
    
    # Heuristic 4: Random
    rand_perm = list(range(n_jobs))
    random.shuffle(rand_perm)
    
    population = [spt_perm, lpt_perm, neh_perm, rand_perm]
    
    # Fill rest with random permutations
    while len(population) < pop_size:
        p = list(range(n_jobs))
        random.shuffle(p)
        population.append(p)
    
    # Evaluate initial population
    scores = [_makespan(p, processing_times) for p in population]
    
    best_perm = population[np.argmin(scores)]
    best_ms = min(scores)
    
    # 2. Genetic Algorithm Loop
    # Parameters
    mutation_rate = 0.3
    crossover_rate = 0.8
    tournament_size = 5
    
    while (time.time() - start_time) < time_limit:
        new_population = []
        
        # Elitism: Keep the best 10%
        elite_count = max(1, pop_size // 10)
        sorted_indices = np.argsort(scores)
        elites = [population[i] for i in sorted_indices[:elite_count]]
        new_population.extend(elites)
        
        # Generate offspring
        while len(new_population) < pop_size:
            # Selection
            def tournament():
                contenders = random.sample(range(pop_size), min(tournament_size, pop_size))
                best_idx = min(contenders, key=lambda i: scores[i])
                return population[best_idx]
            
            parent1 = tournament()
            parent2 = tournament()
            
            # Crossover (Position-based / Order Crossover)
            if random.random() < crossover_rate:
                # Order Crossover
                c1 = list(range(n_jobs))
                c2 = list(range(n_jobs))
                random.shuffle(c1)
                c2 = [x for x in c2 if x not in c1]
                
                child = [0] * n_jobs
                # Copy segment from parent1
                for i in c1:
                    child[i] = parent1[i]
                # Fill rest from parent2
                ptr = 0
                for i in c2:
                    if child[i] == 0:
                        child[i] = parent2[ptr]
                        ptr += 1
            else:
                child = parent1[:]
            
            # Mutation (Insertion)
            if random.random() < mutation_rate:
                i = random.randint(0, n_jobs - 1)
                j = random.randint(0, n_jobs - 1)
                if i != j:
                    job = child[i]
                    new_child = child[:i] + child[i+1:]
                    if i < j:
                        j -= 1
                    new_child = new_child[:j] + [job] + new_child[j:]
                    child = new_child
            
            # Local Improvement on offspring (Hybrid GA)
            # To save time, only do local search on a subset of offspring or if time permits
            # For small problems, do it on all. For larger, maybe just best few.
            # Let's do it on all for now, as LS is fast for small N.
            child, child_ms = _local_search(child, processing_times, max_iters=5)
            
            new_population.append(child)
            
        # Update population and scores
        population = new_population
        scores = [_makespan(p, processing_times) for p in population]
        
        current_best_ms = min(scores)
        if current_best_ms < best_ms:
            best_ms = current_best_ms
            best_idx = np.argmin(scores)
            best_perm = population[best_idx][:]
            
        # Adaptive Population Size? 
        # If no improvement in last X generations, reduce population or increase mutation?
        # Keep simple for now.
        
    return best_perm
# EVOLVE-BLOCK-END
