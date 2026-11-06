import numpy as np
import random
import time

def construct():
    # Try Paley tournaments first as they are fast to construct and check
    # Paley tournament exists for p = 3, 7, 11, 19, 23, 31, 43, 47, 59, 67, ...
    # p must be prime and p % 4 == 3
    
    def is_paley_valid(p):
        if p < 5:
            return False, None
        # Check if p is prime
        if p < 2:
            return False, None
        for i in range(2, int(p**0.5) + 1):
            if p % i == 0:
                return False, None
        if p % 4 != 3:
            return False, None
            
        squares = {x * x % p for x in range(1, p)}
        # in_masks[v]: bit u is 1 if u -> v
        in_masks = [0] * p
        orientation = []
        for i in range(p):
            for j in range(i + 1, p):
                if (j - i) % p in squares:
                    # i -> j
                    orientation.append(0)
                    in_masks[j] |= (1 << i)
                else:
                    # j -> i
                    orientation.append(1)
                    in_masks[i] |= (1 << j)
        
        # Check validity: every 4-set has a common dominator
        # A 4-set S is dominated by v if v is in the intersection of in-neighbors of all u in S
        # i.e., in_masks[u] has bit v set for all u in S.
        
        # To speed up, we can just check all 4-sets
        # For p up to 67, C(67,4) is ~ 700k. This is manageable.
        
        # Optimization: If max out-degree < 4, it's impossible.
        # In Paley, out-degree is (p-1)/2. For p=31, d=15. OK.
        
        for a in range(p):
            for b in range(a+1, p):
                # Early termination: if intersection of in-neighbors of a and b is empty, 
                # then any set containing a,b is not dominated? 
                # No, the dominator must be outside a,b? 
                # "vertex with arcs to all four". The vertex v must have arcs to a,b,c,d.
                # So v must be in in_neigh(a) AND in_neigh(b) AND in_neigh(c) AND in_neigh(d).
                # If in_neigh(a) & in_neigh(b) is empty, then no v can dominate {a,b,c,d}.
                # So we can prune.
                
                mask_ab = in_masks[a] & in_masks[b]
                if mask_ab == 0:
                    return False, None
                for c in range(b+1, p):
                    mask_abc = mask_ab & in_masks[c]
                    if mask_abc == 0:
                        return False, None
                    for d in range(c+1, p):
                        if (mask_abc & in_masks[d]) == 0:
                            return False, None
        
        return True, orientation

    # Try small primes first
    primes_to_try = [31, 43, 59, 67]
    
    for p in primes_to_try:
        valid, orient = is_paley_valid(p)
        if valid:
            return {'n': p, 'orientation': orient}
            
    # If no Paley tournament works, use Genetic Algorithm
    # We will try to find a solution for n=31, then 32, etc.
    # Given time constraints, we'll focus on n=31 and n=32.
    
    def run_ga(n, time_limit=110.0):
        start_time = time.time()
        num_edges = n * (n - 1) // 2
        
        # Precompute 4-sets indices
        # To save memory and time, we generate them on the fly in fitness?
        # No, precomputing list of tuples is better for iteration.
        # But for n=31, 31k sets.
        
        def get_fitness(orient):
            # orient is a list of 0/1
            # Build in_masks
            in_masks = [0] * n
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    if orient[idx] == 0:
                        # i -> j
                        in_masks[j] |= (1 << i)
                    else:
                        # j -> i
                        in_masks[i] |= (1 << j)
                    idx += 1
            
            uncovered = 0
            # Iterate 4-sets
            for a in range(n):
                ma = in_masks[a]
                for b in range(a+1, n):
                    mab = ma & in_masks[b]
                    if mab == 0:
                        # Count remaining sets containing a,b
                        # C(n-2, 2)
                        uncovered += (n-2)*(n-3)//2
                        continue
                    for c in range(b+1, n):
                        mabc = mab & in_masks[c]
                        if mabc == 0:
                            # Count remaining sets containing a,b,c
                            # n-3
                            uncovered += (n-3)
                            continue
                        for d in range(c+1, n):
                            if (mabc & in_masks[d]) == 0:
                                uncovered += 1
            return uncovered

        # Initialize population
        pop_size = 50
        population = []
        for _ in range(pop_size):
            # Random initialization
            ind = [random.randint(0, 1) for _ in range(num_edges)]
            population.append(ind)
        
        # Local search function
        def local_search(ind, steps=20):
            current_fit = get_fitness(ind)
            if current_fit == 0:
                return ind, current_fit
            for _ in range(steps):
                best_ind = ind
                best_fit = current_fit
                # Try flipping each bit? No, too slow.
                # Try random flips
                for _ in range(10):
                    new_ind = ind[:]
                    flip_idx = random.randint(0, num_edges - 1)
                    new_ind[flip_idx] ^= 1
                    new_fit = get_fitness(new_ind)
                    if new_fit < best_fit:
                        best_fit = new_fit
                        best_ind = new_ind
                if best_fit == current_fit:
                    break
                ind = best_ind
                current_fit = best_fit
            return ind, current_fit

        best_overall_ind = population[0]
        best_overall_fit = get_fitness(best_overall_ind)
        
        generation = 0
        while time.time() - start_time < time_limit:
            generation += 1
            
            # Evaluate fitness for all (cache if possible, but changes with mutation)
            fitnesses = [get_fitness(ind) for ind in population]
            
            # Update best
            min_fit_idx = min(range(pop_size), key=lambda i: fitnesses[i])
            if fitnesses[min_fit_idx] < best_overall_fit:
                best_overall_fit = fitnesses[min_fit_idx]
                best_overall_ind = population[min_fit_idx]
                if best_overall_fit == 0:
                    break
            
            # Apply local search to best individual
            improved_ind, improved_fit = local_search(population[min_fit_idx], steps=5)
            if improved_fit < best_overall_fit:
                best_overall_fit = improved_fit
                best_overall_ind = improved_ind
                if best_overall_fit == 0:
                    break
            population[min_fit_idx] = improved_ind
            fitnesses[min_fit_idx] = improved_fit
            
            # Selection and Crossover
            new_population = []
            # Elitism: keep best 2
            sorted_indices = sorted(range(pop_size), key=lambda i: fitnesses[i])
            for i in sorted_indices[:2]:
                new_population.append(population[i][:])
            
            while len(new_population) < pop_size:
                # Tournament selection
                p1 = population[random.randint(0, pop_size-1)]
                p2 = population[random.randint(0, pop_size-1)]
                f1 = fitnesses[population.index(p1)] # This is O(N), bad. Use indices.
                
                # Let's use indices for selection to avoid O(N) search
                # Re-implementing selection part properly
            
            # Reset new_population logic for efficiency
            new_population = [population[i][:] for i in sorted_indices[:2]]
            
            while len(new_population) < pop_size:
                # Tournament selection
                idx1 = random.randint(0, pop_size-1)
                idx2 = random.randint(0, pop_size-1)
                if fitnesses[idx1] < fitnesses[idx2]:
                    parent1 = population[idx1]
                else:
                    parent1 = population[idx2]
                
                idx3 = random.randint(0, pop_size-1)
                idx4 = random.randint(0, pop_size-1)
                if fitnesses[idx3] < fitnesses[idx4]:
                    parent2 = population[idx3]
                else:
                    parent2 = population[idx4]
                
                # Uniform Crossover
                child = []
                for i in range(num_edges):
                    if random.random() < 0.5:
                        child.append(parent1[i])
                    else:
                        child.append(parent2[i])
                
                # Mutation
                # Flip 1-3 bits
                num_mutations = random.randint(1, 3)
                for _ in range(num_mutations):
                    m_idx = random.randint(0, num_edges - 1)
                    child[m_idx] ^= 1
                
                new_population.append(child)
            
            population = new_population
            
        return best_overall_ind, best_overall_fit

    # Try n=31
    ind, fit = run_ga(31, time_limit=100.0)
    if fit == 0:
        return {'n': 31, 'orientation': ind}
        
    # Try n=32
    ind, fit = run_ga(32, time_limit=100.0)
    if fit == 0:
        return {'n': 32, 'orientation': ind}
        
    # Fallback to current best known if GA fails (should not happen if time is enough, but just in case)
    # Return the best found so far or a default
    # Since we must return a valid tournament if possible, and we don't have a guarantee, 
    # we should probably return the Paley 83 if GA fails? 
    # But the task says "Minimize n". 
    # If GA fails to find 31 or 32, we might still have found a better one than 83?
    # The run_ga function returns the best individual found.
    # We should check if that best individual is valid.
    # If not, we are stuck.
    # However, the Paley 83 is a valid fallback.
    
    # Let's assume the GA finds something or we fall back to 83.
    # But wait, if we fall back to 83, the score is 1.0.
    # If we find 31, score is 83/31 > 2.
    
    # To be safe, let's just return the best GA result if it's valid, else 83.
    # But run_ga only returns one result. 
    # We should store the best valid tournament found during the process.
    
    # Re-structure: run_ga should return the best VALID tournament if found, else the best INVALID one.
    # Actually, if fit > 0, it's invalid.
    
    # Let's modify the logic:
    # We already tried Paley. If that failed, we run GA.
    # If GA finds fit==0, return it.
    # If GA doesn't find fit==0, we can't return an invalid tournament.
    # The evaluator checks validity. An invalid result scores 0.
    # So we MUST return a valid tournament.
    # The only guaranteed valid tournament we have is the Paley 83.
    # So if GA fails, we return Paley 83.
    
    # But we can try to see if the "best" GA result is close to valid? No, it must be exactly 0.
    
    # Let's add a fallback to Paley 83.
    p = 83
    squares = {x * x % p for x in range(1, p)}
    return {'n': p, 'orientation': [0 if (j - i) % p in squares else 1 for i in range(p) for j in range(i + 1, p)]}

# EVOLVE-BLOCK-END
