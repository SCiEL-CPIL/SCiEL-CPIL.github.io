"""Two-colourings of [1,256] with few monochromatic 4-term progressions."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

N = 256
K = 4

def count_mono_4ap_fast(coloring):
    """Count monochromatic 4-term APs in a 0/1 coloring of 1..256."""
    c = np.array(coloring, dtype=np.int8)
    count = 0
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        if max_a < 1:
            break
        seq1 = c[:max_a]
        seq2 = c[d:d + max_a]
        seq3 = c[2 * d:2 * d + max_a]
        seq4 = c[3 * d:3 * d + max_a]
        mono = (seq1 == seq2) & (seq2 == seq3) & (seq3 == seq4)
        count += np.sum(mono)
    return count

def flip_delta(c, pos):
    """
    Compute the change in mono-4AP count if we flip c[pos].
    c is a numpy array of int8.
    Returns the delta (new_score - old_score).
    """
    old_val = c[pos]
    new_val = 1 - old_val
    delta = 0
    
    # A 4-AP a, a+d, a+2d, a+3d contains pos.
    # We check all d where pos can be one of the 4 terms.
    
    # Case 1: pos = a (first element)
    # a = pos, so indices are pos-1, pos-1+d, pos-1+2d, pos-1+3d
    # Constraints: pos + 3d <= N => d <= (N - pos) // 3
    # Note: c is 0-indexed, so a=1 corresponds to index 0.
    # If pos is the 1-based index, 0-based index is pos-1.
    # Let's stick to 0-based indices for c.
    # If we flip index i (0-based), it corresponds to number i+1.
    # AP: a, a+d, a+2d, a+3d.
    # i = a-1 => a = i+1. a+3d = i+1+3d <= N => 3d <= N-1-i => d <= (N-1-i)//3
    for d in range(1, (N - 1 - pos) // 3 + 1):
        idx1 = pos
        idx2 = pos + d
        idx3 = pos + 2 * d
        idx4 = pos + 3 * d
        # Check if currently mono
        if c[idx1] == c[idx2] == c[idx3] == c[idx4]:
            delta -= 1
        # Check if becomes mono after flip
        if new_val == c[idx2] == c[idx3] == c[idx4]:
            delta += 1

    # Case 2: pos = a+d (second element)
    # a+d = pos+1 => a = pos+1-d. a >= 1 => d <= pos.
    # a+3d = pos+1-d+3d = pos+1+2d <= N => 2d <= N-1-pos => d <= (N-1-pos)//2
    for d in range(1, min(pos, (N - 1 - pos) // 2) + 1):
        idx1 = pos - d
        idx2 = pos
        idx3 = pos + d
        idx4 = pos + 2 * d
        if c[idx1] == c[idx2] == c[idx3] == c[idx4]:
            delta -= 1
        if c[idx1] == new_val == c[idx3] == c[idx4]:
            delta += 1

    # Case 3: pos = a+2d (third element)
    # a+2d = pos+1 => a = pos+1-2d. a >= 1 => 2d <= pos => d <= pos//2
    # a+3d = pos+1-2d+3d = pos+1+d <= N => d <= N-1-pos
    for d in range(1, min(pos // 2, N - 1 - pos) + 1):
        idx1 = pos - 2 * d
        idx2 = pos - d
        idx3 = pos
        idx4 = pos + d
        if c[idx1] == c[idx2] == c[idx3] == c[idx4]:
            delta -= 1
        if c[idx1] == c[idx2] == new_val == c[idx4]:
            delta += 1

    # Case 4: pos = a+3d (fourth element)
    # a+3d = pos+1 => a = pos+1-3d. a >= 1 => 3d <= pos => d <= pos//3
    for d in range(1, pos // 3 + 1):
        idx1 = pos - 3 * d
        idx2 = pos - 2 * d
        idx3 = pos - d
        idx4 = pos
        if c[idx1] == c[idx2] == c[idx3] == c[idx4]:
            delta -= 1
        if c[idx1] == c[idx2] == c[idx3] == new_val:
            delta += 1

    return delta

def local_search(coloring, deadline, max_iters=1000):
    """Hill-climbing local search with delta evaluation."""
    c = np.array(coloring, dtype=np.int8)
    score = count_mono_4ap_fast(c)
    best_c = c.copy()
    best_score = score
    
    it = 0
    while it < max_iters and time.time() < deadline:
        improved = False
        best_pos = -1
        best_delta = 0
        
        # Find the flip with the most negative delta
        # We can optimize this by sampling if it's too slow, but N=256 is small
        # 256 * ~100 ops = 25k ops per iteration. 1000 iters = 25M ops. Should be fine in Python if optimized.
        # Actually, flip_delta is Python loops. Let's see speed.
        # 256 positions * (avg ~20 d's) * 4 checks = ~20k operations per local search step.
        # 1000 steps = 20M operations. In Python this might take ~1-2 seconds.
        # We have 150s budget. So we can do many restarts.
        
        for pos in range(N):
            d = flip_delta(c, pos)
            if d < best_delta:
                best_delta = d
                best_pos = pos
        
        if best_pos >= 0 and best_delta < 0:
            c[best_pos] = 1 - c[best_pos]
            score += best_delta
            improved = True
            if score < best_score:
                best_score = score
                best_c = c.copy()
        else:
            # Try a few random flips to escape local minima
            for _ in range(5):
                if time.time() > deadline:
                    break
                pos = random.randint(0, N - 1)
                c[pos] = 1 - c[pos]
                # Recompute score for simplicity after random kick, or use delta
                # Using delta is safer
                # Wait, flip_delta assumes c is current state.
                # If we flip, we should update score.
                # But we already flipped in c.
                # Let's just recompute for the random kick to avoid bugs
                score = count_mono_4ap_fast(c)
                if score < best_score:
                    best_score = score
                    best_c = c.copy()
            break
        
        it += 1
    
    return best_c, best_score

def genetic_algorithm(deadline):
    """
    GA with local search polish.
    """
    # Population size
    pop_size = 50
    # Initialize population with diverse randoms and some structured patterns
    population = []
    
    # Add random individuals
    for _ in range(pop_size - 10):
        population.append(np.random.randint(0, 2, N).astype(np.int8))
    
    # Add structured individuals (Thue-Morse, Alternating, Blocks)
    # Thue-Morse
    tm = np.zeros(N, dtype=np.int8)
    for i in range(N):
        tm[i] = bin(i).count('1') % 2
    population.append(tm)
    
    # Alternating
    alt = np.array([i % 2 for i in range(N)], dtype=np.int8)
    population.append(alt)
    
    # Blocks of 2
    blk2 = np.array([i // 2 % 2 for i in range(N)], dtype=np.int8)
    population.append(blk2)
    
    # Blocks of 4
    blk4 = np.array([i // 4 % 2 for i in range(N)], dtype=np.int8)
    population.append(blk4)
    
    # Blocks of 8
    blk8 = np.array([i // 8 % 2 for i in range(N)], dtype=np.int8)
    population.append(blk8)
    
    # Evaluate initial population
    scores = [count_mono_4ap_fast(ind) for ind in population]
    
    best_individual = population[np.argmin(scores)].copy()
    best_score = min(scores)
    
    generation = 0
    while time.time() < deadline - 10:
        generation += 1
        
        # Selection: Tournament selection
        # Crossover: Uniform crossover
        # Mutation: Bit flip with low probability
        
        new_population = []
        
        # Keep the best individual (Elitism)
        idx_best = np.argmin(scores)
        new_population.append(population[idx_best].copy())
        new_population.append(population[idx_best].copy()) # Duplicate best
        
        # Generate offspring
        while len(new_population) < pop_size:
            if time.time() > deadline - 10:
                break
            
            # Select parents
            p1_idx = random.randint(0, pop_size - 1)
            p2_idx = random.randint(0, pop_size - 1)
            
            # Tournament selection for better quality
            # Pick 3 random, take best
            c1_idx = min(random.sample(range(pop_size), 3), key=lambda i: scores[i])
            c2_idx = min(random.sample(range(pop_size), 3), key=lambda i: scores[i])
            
            p1 = population[c1_idx]
            p2 = population[c2_idx]
            
            # Uniform Crossover
            mask = np.random.randint(0, 2, N)
            child = np.where(mask == 1, p1, p2).astype(np.int8)
            
            # Mutation
            # Low probability mutation
            if random.random() < 0.1:
                mut_pos = random.randint(0, N - 1)
                child[mut_pos] = 1 - child[mut_pos]
            
            new_population.append(child)
        
        # Evaluate new population
        # To save time, we can only evaluate some, but N=256 is fast enough.
        # 50 * count_mono_4ap_fast.
        # count_mono_4ap_fast is ~1ms?
        # 50 * 1ms = 50ms per generation.
        # 1000 generations = 50s.
        # We have ~140s. So ~2000 generations.
        
        new_scores = [count_mono_4ap_fast(ind) for ind in new_population]
        
        # Update best
        idx_new_best = np.argmin(new_scores)
        if new_scores[idx_new_best] < best_score:
            best_score = new_scores[idx_new_best]
            best_individual = new_population[idx_new_best].copy()
            
        # Local search polish on the best individual found so far
        # Do this every 10 generations
        if generation % 10 == 0 and time.time() < deadline - 5:
            polished, polished_score = local_search(best_individual, deadline - 5, max_iters=100)
            if polished_score < best_score:
                best_score = polished_score
                best_individual = polished
        
        population = new_population
        scores = new_scores
        
        # Inject best individual into population to ensure it survives
        # Find worst individual and replace with best
        idx_worst = np.argmax(scores)
        population[idx_worst] = best_individual.copy()
        scores[idx_worst] = best_score
        
    # Final polish
    if time.time() < deadline - 2:
        polished, polished_score = local_search(best_individual, deadline - 2, max_iters=500)
        if polished_score < best_score:
            best_score = polished_score
            best_individual = polished
            
    return best_individual

def construct():
    deadline = time.time() + 145  # Safety margin for 150s limit
    
    # Run GA
    best_c = genetic_algorithm(deadline)
    
    # Final check
    return list(best_c)

# EVOLVE-BLOCK-END
