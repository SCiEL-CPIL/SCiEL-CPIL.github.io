"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

N = 256
DEADLINE = 115.0  # Safety margin before 150s kill

def _count_mono4(c):
    """Count monochromatic 4-term APs in a 0/1 numpy array of length N (0-indexed)."""
    count = 0
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        if max_a <= 0:
            break
        c0 = c[0:max_a]
        c1 = c[d:d + max_a]
        c2 = c[2 * d:2 * d + max_a]
        c3 = c[3 * d:3 * d + max_a]
        mask = (c0 == c1) & (c0 == c2) & (c0 == c3)
        count += np.count_nonzero(mask)
    return count

def _thue_morse():
    """Generate Thue-Morse sequence for 0..255."""
    c = np.zeros(N, dtype=np.int8)
    for i in range(N):
        c[i] = bin(i).count('1') % 2
    return c

def _random_coloring():
    """Generate a random 0/1 coloring."""
    return np.random.randint(0, 2, N, dtype=np.int8)

def _local_search_delta(c, max_iters=500):
    """
    Local search using incremental delta computation.
    Returns improved coloring and count.
    """
    c = c.copy()
    # Precompute current count
    count = _count_mono4(c)
    
    # Precompute for each position i, the list of (d, a) tuples where i participates
    # i can be at position a, a+d, a+2d, a+3d
    # We need to compute delta if we flip i.
    # delta = new_count - old_count
    # new_count = count - (number of mono APs involving i that are currently mono) 
    #            + (number of mono APs involving i that become mono after flip)
    # This is complex to compute directly without iterating all d.
    # Given time constraints, a simple random restart local search with full recount
    # or a very limited delta search is safer. 
    # Let's use a simple greedy flip with full recount for the first few steps, 
    # but that's slow. 
    # Actually, the inspiration says delta is O(N). Let's implement a simplified delta.
    # For each flip at i, we check all d and a such that i is in {a, a+d, a+2d, a+3d}.
    # Number of such (d,a) pairs is roughly sum over d of (number of positions in AP).
    # For a fixed i, d can range 1 to N/3. For each d, there are at most 4 values of a.
    # So it's O(N) per flip.
    
    for _ in range(max_iters):
        if time.time() > DEADLINE:
            break
        best_delta = 0
        best_idx = -1
        # Sample a subset of indices to flip to save time
        # Or iterate all? N=256, O(N) per flip -> O(N^2) per pass. 
        # 256 * 256 = 65536 ops per pass. Very fast in Python if optimized? 
        # But Python loop overhead is high. 
        # Let's try iterating all indices but break early if time is tight.
        
        # To speed up, we can just pick a random index to flip and check if it improves.
        # Hill climbing with random restarts.
        
        i = random.randint(0, N - 1)
        old_val = c[i]
        c[i] = 1 - old_val
        new_count = _count_mono4(c)
        delta = new_count - count
        if delta < 0:
            count = new_count
            best_idx = i
            # Keep the flip
        else:
            # Revert
            c[i] = old_val
            
    return c, count

def construct():
    start_time = time.time()
    rng = np.random.RandomState(42)
    random.seed(42)
    
    # Initialize population
    pop_size = 20
    population = []
    
    # Add Thue-Morse
    tm = _thue_morse()
    population.append(tm)
    
    # Add random colorings
    for _ in range(pop_size - 1):
        population.append(_random_coloring())
        
    # Evaluate initial population
    scores = [_count_mono4(c) for c in population]
    best_idx = np.argmin(scores)
    best_c = population[best_idx].copy()
    best_score = scores[best_idx]
    
    # Genetic Algorithm Loop
    gen = 0
    while time.time() < DEADLINE:
        gen += 1
        new_population = []
        
        # Elitism: keep the best
        new_population.append(best_c.copy())
        
        # Generate offspring
        while len(new_population) < pop_size:
            if time.time() > DEADLINE:
                break
            
            # Selection: Tournament selection
            i1 = random.randint(0, len(population) - 1)
            i2 = random.randint(0, len(population) - 1)
            if scores[i1] < scores[i2]:
                p1 = population[i1]
            else:
                p1 = population[i2]
                
            i3 = random.randint(0, len(population) - 1)
            i4 = random.randint(0, len(population) - 1)
            if scores[i3] < scores[i4]:
                p2 = population[i3]
            else:
                p2 = population[i4]
                
            # Crossover: Uniform crossover
            mask = np.random.randint(0, 2, N, dtype=np.int8)
            child = np.where(mask == 1, p1, p2).astype(np.int8)
            
            # Mutation: Bit flip with probability 0.05
            if np.random.random() < 0.9: # Sometimes mutate
                mutation_mask = np.random.randint(0, 2, N, dtype=np.int8)
                # Flip bits where mutation_mask is 1 AND random prob is high? 
                # Simpler: flip each bit with prob 0.05
                flip_mask = (np.random.random(N) < 0.05).astype(np.int8)
                child = (child + flip_mask) % 2
                child = child.astype(np.int8)
                
            new_population.append(child)
            
        # Evaluate new population
        population = new_population
        scores = [_count_mono4(c) for c in population]
        
        # Update best
        curr_best_idx = np.argmin(scores)
        if scores[curr_best_idx] < best_score:
            best_score = scores[curr_best_idx]
            best_c = population[curr_best_idx].copy()
            
        # Local search on the best individual occasionally
        if gen % 5 == 0:
            improved_c, improved_score = _local_search_delta(best_c, max_iters=100)
            if improved_score < best_score:
                best_score = improved_score
                best_c = improved_c
                # Replace worst in population with improved one
                worst_idx = np.argmax(scores)
                population[worst_idx] = best_c
                scores[worst_idx] = best_score

    # Return the best coloring found
    return list(best_c)
# EVOLVE-BLOCK-END
