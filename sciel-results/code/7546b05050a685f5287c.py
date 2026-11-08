import time
import random
import numpy as np
from math import log

def score_set(U):
    """
    Calculate the score for a set U.
    Score = 1 + log(|U-U|/|U+U|)/log(2 max U + 1)
    """
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    
    U_arr = np.array(U, dtype=np.int64)
    # Sumset: all pairwise sums (with repetition)
    sums = U_arr[:, None] + U_arr[None, :]
    sums_unique = np.unique(sums)
    # Difference set: all pairwise differences (with sign)
    diffs = U_arr[:, None] - U_arr[None, :]
    diffs_unique = np.unique(diffs)
    
    sum_count = len(sums_unique)
    diff_count = len(diffs_unique)
    
    if sum_count == 0:
        return -1
    ratio = diff_count / sum_count
    if ratio <= 1:
        return -1
    max_u = int(U_arr.max())
    D = 2 * max_u + 1
    if D <= 1:
        return -1
    return 1 + log(ratio) / log(D)

def generate_random_seed(min_n=5, max_n=20, max_val=10000):
    """Generate a random sparse set containing 0."""
    n = random.randint(min_n, max_n)
    U = [0]
    last = 0
    for _ in range(n-1):
        # Geometric-ish growth
        step = random.randint(1, max(1, int(last * random.uniform(0.5, 2.0)) + 1))
        last += step
        if last > max_val:
            last = max_val
            break
        U.append(last)
    return sorted(set(U))

def generate_geometric_seed(base, n, start=1):
    """Generate a geometric sequence set."""
    U = [0]
    val = float(start)
    seen = {0}
    for _ in range(n):
        val *= base
        iv = int(round(val))
        if iv not in seen and iv > 0:
            U.append(iv)
            seen.add(iv)
    return sorted(U)

def mutate(U, mutation_rate=0.1):
    """
    Mutate a set U.
    Returns a new set.
    """
    U = list(U)
    if not U:
        return [0]
    
    # Ensure 0 is present
    if 0 not in U:
        U.append(0)
    
    n = len(U)
    max_val = max(U) if U else 10
    
    # 1. Perturb values
    for i in range(len(U)):
        if random.random() < mutation_rate:
            if i == 0 and U[i] == 0:
                continue # Keep 0 fixed
            # Small relative change
            delta = random.randint(-max(1, int(U[i]*0.1)), max(1, int(U[i]*0.1)))
            new_val = U[i] + delta
            if new_val >= 0:
                U[i] = new_val
    
    # 2. Insert new element
    if random.random() < mutation_rate * 0.5 and n < 500:
        # Insert near existing elements or at the end
        if U:
            base_idx = random.randint(0, len(U)-1)
            base_val = U[base_idx]
            new_val = base_val + random.randint(-max(1, int(base_val*0.2)), max(1, int(base_val*0.2)))
            if new_val >= 0:
                U.append(new_val)
        else:
            U.append(random.randint(1, 100))
            
    # 3. Remove element (not 0)
    if random.random() < mutation_rate * 0.5 and n > 2:
        idx = random.randint(1, len(U)-1)
        U.pop(idx)
        
    # 4. Scale
    if random.random() < mutation_rate * 0.2:
        factor = random.uniform(0.8, 1.2)
        U = [0] + [int(v * factor) for v in U if v > 0]
        
    # Deduplicate and sort
    U = sorted(set(U))
    if 0 not in U:
        U.insert(0, 0)
    return U

def crossover(parent1, parent2):
    """
    Blend crossover for sorted lists.
    """
    # Simple: take random elements from both
    child = []
    for x in parent1:
        if random.random() < 0.5:
            child.append(x)
    for x in parent2:
        if random.random() < 0.5:
            child.append(x)
    
    # Ensure 0 is present
    if 0 not in child:
        child.append(0)
        
    child = sorted(set(child))
    return child

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    # Initialize population
    pop_size = 50
    population = []
    
    # Diverse initial seeds
    for _ in range(pop_size):
        if random.random() < 0.3:
            # Geometric
            base = random.uniform(1.1, 3.0)
            n = random.randint(10, 30)
            pop = generate_geometric_seed(base, n)
        else:
            # Random sparse
            pop = generate_random_seed(min_n=5, max_n=30, max_val=10000)
        population.append(pop)
        
    # Score initial population
    scored_pop = [(score_set(p), p) for p in population]
    scored_pop.sort(key=lambda x: x[0], reverse=True)
    
    best_score = scored_pop[0][0]
    best_U = scored_pop[0][1]
    
    # Evolutionary loop
    generation = 0
    while time.time() < deadline - 10:
        generation += 1
        
        # Create new population
        new_pop = []
        
        # Elitism: keep top 10%
        elite_count = max(1, pop_size // 10)
        for i in range(elite_count):
            new_pop.append(list(scored_pop[i][1]))
            
        # Fill rest with offspring
        while len(new_pop) < pop_size:
            # Tournament selection
            p1_idx = random.randint(0, len(scored_pop)-1)
            p2_idx = random.randint(0, len(scored_pop)-1)
            # Better score wins
            if scored_pop[p1_idx][0] > scored_pop[p2_idx][0]:
                p1 = scored_pop[p1_idx][1]
            else:
                p1 = scored_pop[p2_idx][1]
                
            if random.random() < 0.7:
                p2_idx = random.randint(0, len(scored_pop)-1)
                if scored_pop[p2_idx][0] > scored_pop[p1_idx][0]: # Note: p1_idx is from prev, need careful logic
                    pass # Just pick random other
                p2 = scored_pop[p2_idx][1]
                child = crossover(p1, p2)
            else:
                child = list(p1)
                
            child = mutate(child, mutation_rate=0.2)
            new_pop.append(child)
            
        # Score new population
        scored_new = []
        for p in new_pop:
            s = score_set(p)
            scored_new.append((s, p))
            
        # Update best
        for s, p in scored_new:
            if s > best_score:
                best_score = s
                best_U = p
                
        # Combine and select for next generation
        # Keep top pop_size from combined old and new? 
        # Simpler: just replace population with new_pop if it's better on average, 
        # or just use new_pop as the next population.
        # Let's use a mix: keep top 50% from new, and top 50% from old?
        # Standard GA: new_pop becomes the population.
        population = new_pop
        scored_pop = sorted(scored_new, key=lambda x: x[0], reverse=True)
        
        # Adaptive mutation rate? 
        # If stuck, increase mutation.
        if generation % 10 == 0:
             # Increase mutation rate for next generation implicitly by not changing rate, 
             # but we can adjust population size or other params.
             pass
             
    return best_U

if __name__ == "__main__":
    U = construct()
    print("Result:", U)
    print("Score:", score_set(U))
