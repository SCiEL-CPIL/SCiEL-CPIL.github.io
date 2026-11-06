import numpy as np
import random
import time

def construct():
    """
    Constructs a set of 32 distinct integers in 1..512 maximizing the number
    of distinct differences d > 0 such that a, a+d, a+2d are in the set.
    Uses a Genetic Algorithm with vectorized scoring.
    """
    
    def vectorized_score(mask):
        """
        mask: boolean array of size 513 (indices 0..512), where mask[i] is True if i is in the set.
        Returns the number of distinct differences d > 0 forming a 3-AP.
        """
        found_d = 0
        for d in range(1, 256):
            max_a = 512 - 2 * d
            if max_a < 1:
                break
            # Check if there exists a such that a, a+d, a+2d are in the set
            # mask[1:max_a+1] corresponds to a
            # mask[1+d:max_a+1+d] corresponds to a+d
            # mask[1+2*d:max_a+1+2*d] corresponds to a+2d
            if np.any(mask[1:max_a+1] & mask[1+d:max_a+1+d] & mask[1+2*d:max_a+1+2*d]):
                found_d += 1
        return found_d

    def evaluate(chrom):
        """
        chrom: numpy array of 32 integers.
        Returns the score.
        """
        mask = np.zeros(513, dtype=bool)
        mask[chrom] = True
        return vectorized_score(mask)

    def crossover(parent1, parent2):
        """
        Uniform crossover.
        """
        child = np.empty(32, dtype=int)
        for i in range(32):
            if random.random() < 0.5:
                child[i] = parent1[i]
            else:
                child[i] = parent2[i]
        # Ensure uniqueness
        # If duplicates exist, replace them with random values not in the child
        unique_vals = set(child)
        if len(unique_vals) < 32:
            missing_count = 32 - len(unique_vals)
            # Find values to replace
            seen = {}
            to_replace = []
            for v in child:
                if v in seen:
                    to_replace.append(v)
                else:
                    seen[v] = True
            # We need to replace 'missing_count' duplicates
            # Pick random numbers from 1..512 not in unique_vals
            candidates = [x for x in range(1, 513) if x not in unique_vals]
            random.shuffle(candidates)
            for i in range(missing_count):
                # Find index of one of the duplicates
                # This is a bit inefficient, but population size is small
                # Better: just rebuild unique set
                pass
            
            # Simpler repair: 
            # Create a set of the child
            child_set = set(child)
            while len(child_set) < 32:
                child_set.add(random.randint(1, 512))
            child = np.array(list(child_set))
            random.shuffle(child)
            
        return child

    def mutate(chrom, mutation_rate=0.1):
        """
        Randomly replace some elements.
        """
        new_chrom = chrom.copy()
        for i in range(32):
            if random.random() < mutation_rate:
                new_val = random.randint(1, 512)
                # Ensure uniqueness
                if new_val in new_chrom:
                    # Replace a random other element if collision? Or just try again?
                    # For simplicity, we'll just pick a new one until unique or force unique set
                    pass
                new_chrom[i] = new_val
        
        # Repair uniqueness
        child_set = set(new_chrom)
        while len(child_set) < 32:
            child_set.add(random.randint(1, 512))
        new_chrom = np.array(list(child_set))
        random.shuffle(new_chrom)
        
        return new_chrom

    random.seed(42)
    
    population_size = 20
    generation_limit = 50
    start_time = time.time()
    time_limit = 110.0
    
    # Initialize population
    population = []
    for _ in range(population_size):
        chrom = np.array(random.sample(range(1, 513), 32))
        population.append(chrom)
    
    # Evaluate initial population
    scores = [evaluate(chrom) for chrom in population]
    best_idx = np.argmax(scores)
    best_score = scores[best_idx]
    best_chrom = population[best_idx].copy()
    
    while time.time() - start_time < time_limit:
        # Sort by score
        sorted_indices = np.argsort(scores)[::-1]
        sorted_population = [population[i] for i in sorted_indices]
        sorted_scores = [scores[i] for i in sorted_indices]
        
        # Elitism: keep top 2
        new_population = []
        for i in range(2):
            new_population.append(sorted_population[i].copy())
        
        # Generate rest of population
        while len(new_population) < population_size:
            # Tournament selection
            p1_idx = random.randint(0, population_size - 1)
            p2_idx = random.randint(0, population_size - 1)
            if scores[p1_idx] > scores[p2_idx]:
                p1 = population[p1_idx]
            else:
                p1 = population[p2_idx]
            
            p3_idx = random.randint(0, population_size - 1)
            p4_idx = random.randint(0, population_size - 1)
            if scores[p3_idx] > scores[p4_idx]:
                p2 = population[p3_idx]
            else:
                p2 = population[p4_idx]
                
            child = crossover(p1, p2)
            child = mutate(child, mutation_rate=0.05)
            new_population.append(child)
            
        population = new_population
        scores = [evaluate(chrom) for chrom in population]
        
        current_best_idx = np.argmax(scores)
        if scores[current_best_idx] > best_score:
            best_score = scores[current_best_idx]
            best_chrom = population[current_best_idx].copy()
            
    return sorted(best_chrom)
