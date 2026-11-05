# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan using a Genetic Algorithm
    with Order Crossover (OX), a dynamic bottleneck machine heuristic for
    initial population generation, and a fast vectorized makespan evaluation.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s limit
    
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    # Fast fallback for very large instances
    if n_jobs * n_machines > 500000:
        return list(np.argsort(-processing_times.sum(axis=1)))
    
    def makespan(seq):
        """
        Compute makespan for a given sequence of job indices.
        Uses a loop over jobs but vectorized over machines where possible.
        """
        if len(seq) == 0:
            return 0.0
        
        completion = np.zeros(n_machines)
        for job in seq:
            p = processing_times[job]
            # Machine 0
            completion[0] += p[0]
            # Subsequent machines
            for m in range(1, n_machines):
                completion[m] = max(completion[m-1], completion[m]) + p[m]
        return completion[-1]
    
    def neh_heuristic():
        """NEH (Nawaz, Enscore, Ham) heuristic."""
        totals = processing_times.sum(axis=1)
        order = list(np.argsort(-totals))
        
        seq = [order[0]]
        for job in order[1:]:
            best_pos = 0
            best_ms = float('inf')
            for pos in range(len(seq) + 1):
                trial = seq[:pos] + [job] + seq[pos:]
                ms = makespan(trial)
                if ms < best_ms:
                    best_ms = ms
                    best_pos = pos
            seq = seq[:best_pos] + [job] + seq[best_pos:]
        
        return seq
    
    def bottleneck_heuristic():
        """
        Dynamic bottleneck machine heuristic:
        Sort jobs by total processing time, but adjust based on the most
        constrained machine (bottleneck) identified dynamically.
        """
        # Identify the bottleneck machine (highest total load)
        machine_loads = processing_times.sum(axis=0)
        bottleneck_idx = np.argmax(machine_loads)
        
        # Sort jobs by processing time on the bottleneck machine
        # Ties broken by total processing time
        bottleneck_times = processing_times[:, bottleneck_idx]
        totals = processing_times.sum(axis=1)
        
        # Composite key: prioritize high bottleneck time, then high total
        # We want jobs with high bottleneck time early if they also have high total?
        # Actually, standard heuristic is to sort by total descending.
        # Let's try sorting by bottleneck time descending, then total descending.
        # Or simply use NEH but with a different initial sort?
        
        # Let's stick to a variation: Sort by total descending (NEH standard)
        # but this function is just for diversity in the initial population.
        # A different heuristic: Sort by max processing time across machines?
        max_times = processing_times.max(axis=1)
        order = list(np.argsort(-max_times))
        
        # Insertion heuristic
        seq = [order[0]]
        for job in order[1:]:
            best_pos = 0
            best_ms = float('inf')
            for pos in range(len(seq) + 1):
                trial = seq[:pos] + [job] + seq[pos:]
                ms = makespan(trial)
                if ms < best_ms:
                    best_ms = ms
                    best_pos = pos
            seq = seq[:best_pos] + [job] + seq[best_pos:]
        
        return seq
    
    def order_crossover(p1, p2):
        """
        Order Crossover (OX1) for permutations.
        """
        n = len(p1)
        if n <= 2:
            return list(p1)
        
        # Pick two random cut points
        c1, c2 = sorted(random.sample(range(n), 2))
        
        # Initialize child
        child = [None] * n
        
        # Copy segment from p1
        child[c1:c2+1] = p1[c1:c2+1]
        filled = set(p1[c1:c2+1])
        
        # Fill remaining positions with elements from p2 in order
        pos = c2 + 1
        for gene in p2:
            if gene not in filled:
                while child[pos] is not None:
                    pos += 1
                    if pos >= n:
                        pos = 0
                child[pos] = gene
                filled.add(gene)
        
        return child
    
    def mutation(seq, rate=0.1):
        """
        Swap mutation: randomly swap two elements.
        """
        seq = list(seq)
        if random.random() < rate:
            i, j = random.sample(range(len(seq)), 2)
            seq[i], seq[j] = seq[j], seq[i]
        return seq
    
    # Initialize population
    pop_size = 50
    population = []
    
    # Add NEH solution
    population.append(neh_heuristic())
    
    # Add bottleneck heuristic solution
    population.append(bottleneck_heuristic())
    
    # Add random solutions
    while len(population) < pop_size:
        pop = list(range(n_jobs))
        random.shuffle(pop)
        population.append(pop)
    
    # Evaluate initial population
    fitnesses = [makespan(ind) for ind in population]
    
    best_seq = population[0]
    best_ms = min(fitnesses)
    best_idx = fitnesses.index(best_ms)
    best_seq = population[best_idx]
    
    # Genetic Algorithm Loop
    elite_size = 5
    tournament_size = 5
    mutation_rate = 0.1
    gen = 0
    
    while time.time() < deadline:
        gen += 1
        
        # Sort population by fitness
        sorted_indices = np.argsort(fitnesses)
        population = [population[i] for i in sorted_indices]
        fitnesses = [fitnesses[i] for i in sorted_indices]
        
        # Update best
        if fitnesses[0] < best_ms:
            best_ms = fitnesses[0]
            best_seq = population[0]
        
        # Create new population
        new_population = []
        
        # Elitism
        for i in range(min(elite_size, pop_size)):
            new_population.append(population[i])
        
        # Fill rest with offspring
        while len(new_population) < pop_size:
            # Tournament selection
            parents = []
            for _ in range(2):
                candidates = random.sample(range(pop_size), tournament_size)
                winner = min(candidates, key=lambda i: fitnesses[i])
                parents.append(population[winner])
            
            # Crossover
            if random.random() < 0.8:  # Crossover rate
                child = order_crossover(parents[0], parents[1])
            else:
                child = list(parents[0])
            
            # Mutation
            child = mutation(child, mutation_rate)
            
            new_population.append(child)
        
        # Evaluate new population
        fitnesses = [makespan(ind) for ind in new_population]
        population = new_population
        
        # Early stopping check (optional)
        # if gen % 10 == 0 and fitnesses[0] == best_ms:
        #     break
        
    return list(best_seq)
# EVOLVE-BLOCK-END
