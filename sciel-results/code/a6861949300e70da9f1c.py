import numpy as np
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 118.0

    if q != 3 or n != 6 or radius != 1:
        # Fallback: return all words (valid but large)
        all_words = []
        for i in range(q ** n):
            word = [(i // (q ** p)) % q for p in range(n)]
            all_words.append(word)
        return all_words

    total = 3 ** 6
    powers = [3 ** p for p in range(6)]
    
    # Precompute neighbors as numpy array
    neighbors = np.empty((total, 13), dtype=np.int32)
    for i in range(total):
        d = [(i // powers[p]) % 3 for p in range(6)]
        neighbors[i, 0] = i
        k = 1
        for p in range(6):
            for alt in range(3):
                if alt != d[p]:
                    new_i = i + (alt - d[p]) * powers[p]
                    neighbors[i, k] = new_i
                    k += 1
    
    # Helper to convert index to list
    def idx_to_list(idx):
        return [(idx // powers[p]) % 3 for p in range(6)]

    # Helper to reduce a code (list of indices) to minimal size
    # Uses vectorized checks for redundancy
    def reduce_code(code_list):
        code_set = set(code_list)
        code = list(code_set)
        
        # Calculate coverage
        coverage_counts = np.zeros(total, dtype=np.int32)
        for c in code:
            coverage_counts[neighbors[c]] += 1
            
        # Check validity
        if np.any(coverage_counts == 0):
            return code, coverage_counts
            
        # Iteratively remove redundant codewords
        improved = True
        while improved:
            improved = False
            
            # Vectorized check for all 729 words
            min_cov = np.min(coverage_counts[neighbors], axis=1)
            is_redundant = min_cov >= 2
            weights = np.sum(coverage_counts[neighbors], axis=1)
            
            # Find best candidate in current code
            best_idx = -1
            best_weight = -1
            best_c_val = -1
            
            for i, c in enumerate(code):
                if is_redundant[c]:
                    w = weights[c]
                    if w > best_weight:
                        best_weight = w
                        best_idx = i
                        best_c_val = c
                        
            if best_idx != -1:
                code.pop(best_idx)
                coverage_counts[neighbors[best_c_val]] -= 1
                improved = True
                
        return code, coverage_counts

    # Initial Solution: Greedy Set Cover
    uncovered = np.ones(total, dtype=np.int8)
    codewords = []
    
    while uncovered.sum() > 0:
        if time.time() > deadline - 80:
            break
        counts = np.sum(uncovered[neighbors], axis=1)
        best_c = int(np.argmax(counts))
        if counts[best_c] == 0:
            break
        codewords.append(best_c)
        uncovered[neighbors[best_c]] = 0
        
    if uncovered.sum() != 0:
        while uncovered.sum() > 0:
            c = int(np.random.randint(total))
            codewords.append(c)
            uncovered[neighbors[c]] = 0
            
    # Reduce initial solution
    initial_code, _ = reduce_code(codewords)
    
    # Initialize Population
    population = [initial_code]
    
    # Add some variations to population
    for _ in range(10):
        cand = list(initial_code)
        for _ in range(random.randint(0, 5)):
            cand.append(int(np.random.randint(total)))
        reduced_cand, _ = reduce_code(cand)
        population.append(reduced_cand)
        
    # Deduplicate population
    unique_population = []
    seen = set()
    for p in population:
        key = tuple(sorted(p))
        if key not in seen:
            seen.add(key)
            unique_population.append(p)
    population = unique_population
    
    best_code = min(population, key=len)
    best_size = len(best_code)
    
    random.seed(42)
    np.random.seed(42)
    
    pop_size = 50
    
    while time.time() < deadline - 2:
        # Selection: Keep best 2
        population.sort(key=len)
        elite = population[:2]
        
        new_population = list(elite)
        
        # Generate offspring
        while len(new_population) < pop_size:
            if time.time() > deadline - 5:
                break
                
            # Tournament selection
            p1_idx = random.randint(0, len(population) - 1)
            p2_idx = random.randint(0, len(population) - 1)
            p3_idx = random.randint(0, len(population) - 1)
            
            p1, p2, p3 = population[p1_idx], population[p2_idx], population[p3_idx]
            
            # Choose parent 1 (best of 3)
            if len(p1) < len(p2) and len(p1) < len(p3):
                parent1 = p1
            elif len(p2) < len(p1) and len(p2) < len(p3):
                parent1 = p2
            else:
                parent1 = p3
                
            # Choose parent 2 (second best or random)
            if random.random() < 0.5:
                # Pick the other best
                if parent1 is p1:
                    if len(p2) < len(p3):
                        parent2 = p2
                    else:
                        parent2 = p3
                elif parent1 is p2:
                    if len(p1) < len(p3):
                        parent2 = p1
                    else:
                        parent2 = p3
                else:
                    if len(p1) < len(p2):
                        parent2 = p1
                    else:
                        parent2 = p2
            else:
                # Pick random from population
                parent2 = population[random.randint(0, len(population) - 1)]
            
            # Crossover: Union
            child_set = set(parent1) | set(parent2)
            child_code = list(child_set)
            
            # Mutation: Add random word
            if random.random() < 0.5:
                child_code.append(int(np.random.randint(total)))
            
            # Reduce
            reduced_child, cov_counts = reduce_code(child_code)
            
            # Check validity
            if np.any(cov_counts == 0):
                continue
            
            new_population.append(reduced_child)
            
            if len(reduced_child) < best_size:
                best_size = len(reduced_child)
                best_code = reduced_child
                
        # Update population
        unique_new_pop = []
        seen_new = set()
        for p in new_population:
            key = tuple(sorted(p))
            if key not in seen_new:
                seen_new.add(key)
                unique_new_pop.append(p)
        
        if not unique_new_pop:
            break
            
        while len(unique_new_pop) < pop_size:
            unique_new_pop.append(list(best_code))
            
        population = unique_new_pop[:pop_size]
        
        # Inject global best
        if best_code not in population:
            population[random.randint(0, len(population)-1)] = list(best_code)

    return [idx_to_list(idx) for idx in best_code]

# EVOLVE-BLOCK-END
