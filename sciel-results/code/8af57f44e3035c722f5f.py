import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # 20s safety margin
    
    n = 256
    
    # Precompute 4-APs as 0-indexed indices
    # (a, a+d, a+2d, a+3d) -> (i, i+d, i+2d, i+3d)
    # i is 0-indexed, so i + 3d < 256
    ap_list = []
    for d in range(1, 86):
        max_i = n - 3*d
        for i in range(max_i):
            ap_list.append((i, i+d, i+2*d, i+3*d))
    
    ap_array = np.array(ap_list, dtype=np.int32)
    num_aps = ap_array.shape[0]
    
    # Helper to count monochromatic 4-APs
    def count_mono(colors):
        # colors is a numpy array of 0/1
        # A 4-AP is mono if all 4 colors are equal
        # colors[ap_array[:,0]] == colors[ap_array[:,1]] == ...
        # Efficient check: 
        # c0 == c1 == c2 == c3
        # This is equivalent to:
        # c0 == c1 AND c1 == c2 AND c2 == c3
        # Or simply:
        # np.all(colors[ap_array] == colors[ap_array[:, 0:1]], axis=1)
        
        # Let's use the broadcast comparison which is very fast
        c = colors[ap_array]
        # c shape (num_aps, 4)
        # Compare each column to the first column
        return np.sum(np.all(c == c[:, 0:1], axis=1))
    
    best_score = float('inf')
    best_colors = None
    
    # Multi-start hill climbing
    # We will run many restarts. Each restart does a local search until no improvement.
    # To make it fast, we can use a vectorized delta calculation for the whole state at once?
    # No, hill climbing needs sequential updates. But we can make the delta calculation fast.
    
    # Alternative: Use a "batch" hill climbing where we evaluate all possible single flips at once?
    # That would be O(n * num_aps) per step, which is 256 * ~16000 = 4M ops.
    # In NumPy, 4M ops is ~1ms. So we could do ~1000 steps per second.
    # With 100s budget, that's 100,000 steps.
    # But we have 256 variables. A full pass is 256 steps.
    # So we could do ~400 full passes per restart.
    # If we do 10 restarts, that's 4000 full passes.
    # This seems feasible.
    
    # Let's implement a fast local search that evaluates all flips at once.
    # State: colors (256,)
    # Objective: M = sum( mono(APs) )
    # Delta for flipping index k:
    # For each AP containing k, check if it was mono and if it becomes mono.
    # This is still dependent on the specific APs.
    
    # Let's stick to the standard hill climbing but optimize the inner loop with NumPy where possible.
    # Actually, the previous program's bottleneck was the Python loop in calc_delta.
    # We can precompute the "neighbors" for each index in a way that allows vectorized delta calculation?
    # 
    # Let's try a different approach: 
    # Random restarts + Fast Greedy Hill Climbing.
    # In each step, pick a random index, compute delta (using precomputed lists but optimized), 
    # accept if improves.
    
    # To beat the SA, we need to escape local minima. Random restarts help.
    # To be fast, we need fast delta.
    
    # Let's precompute for each index i:
    # aps_with_i: list of (ap_idx, pos)
    # We can store these as numpy arrays for faster access?
    
    aps_containing = [[] for _ in range(n)]
    for ap_idx, (i0, i1, i2, i3) in enumerate(ap_list):
        aps_containing[i0].append((ap_idx, 0))
        aps_containing[i1].append((ap_idx, 1))
        aps_containing[i2].append((ap_idx, 2))
        aps_containing[i3].append((ap_idx, 3))
    
    # Convert to numpy arrays for faster iteration?
    # Maybe just keep as lists of tuples, but optimize the check.
    
    def calc_delta(colors, flip_idx):
        """
        Calculate delta in mono count if we flip colors[flip_idx].
        colors is a numpy array.
        """
        old_color = colors[flip_idx]
        new_color = 1 - old_color
        delta = 0
        
        # Iterate over APs containing flip_idx
        for ap_idx, pos in aps_containing[flip_idx]:
            # Get the AP indices
            i0, i1, i2, i3 = ap_list[ap_idx]
            
            # We need to check the other 3 positions
            # Instead of building a list, we can just check the 3 specific indices
            
            # Indices of the other 3 elements
            # This is a bit messy in Python. Let's try to vectorize the check for a single flip?
            # No, the set of APs depends on flip_idx.
            
            # Let's just do the manual check but efficiently.
            # We can precompute the "other indices" for each (ap_idx, pos) pair?
            # There are ~16000 APs * 4 positions = 64000 pairs.
            # Storing 3 indices for each is 192000 ints. That's fine.
            
            pass
        
        return delta

    # Precompute "other indices" for fast delta
    # other_indices[ap_idx][pos] = (idx1, idx2, idx3)
    other_indices = [[None]*4 for _ in range(num_aps)]
    for ap_idx, (i0, i1, i2, i3) in enumerate(ap_list):
        idxs = [i0, i1, i2, i3]
        for pos in range(4):
            others = [idxs[j] for j in range(4) if j != pos]
            other_indices[ap_idx][pos] = others
    
    def calc_delta_fast(colors, flip_idx):
        old_color = colors[flip_idx]
        new_color = 1 - old_color
        delta = 0
        
        for ap_idx, pos in aps_containing[flip_idx]:
            others = other_indices[ap_idx][pos]
            
            # Check if the 3 others match old_color
            # If they do, the AP was mono (before flip). Flipping breaks it. delta -= 1
            # If they match new_color, the AP becomes mono (after flip). Flipping creates it. delta += 1
            
            # Vectorized check for the 3 others
            # colors[others] is an array of 3 elements
            # We want to know if all are == old_color
            # and if all are == new_color
            
            c_others = colors[others]
            
            if np.all(c_others == old_color):
                delta -= 1
            if np.all(c_others == new_color):
                delta += 1
                
        return delta

    # This is still a Python loop over ~10-20 APs per flip.
    # 256 flips * 20 APs * 1000 steps = 5M iterations.
    # Might be too slow in Python.
    
    # Let's try to vectorize the entire delta calculation for ALL indices at once?
    # That would allow us to pick the best flip (steepest descent) or random flip with known delta.
    
    # To calculate delta for all indices simultaneously:
    # For each AP, we can determine which indices would break/create mono status.
    # This is complex.
    
    # Let's stick to the random flip approach but optimize the inner loop as much as possible.
    # Or, use a different strategy: 
    # **Genetic Algorithm**?
    # Population of 50.
    # Evaluate all 50 (50 * 1ms = 50ms).
    # Crossover/Mutation.
    # GA might escape local minima better than hill climbing.
    
    # Let's try a GA.
    pop_size = 50
    population = np.array([np.random.randint(0, 2, n) for _ in range(pop_size)])
    
    # Evaluate population
    def evaluate_population(pop):
        scores = np.zeros(pop.shape[0])
        for i in range(pop.shape[0]):
            scores[i] = count_mono(pop[i])
        return scores
    
    scores = evaluate_population(population)
    best_idx = np.argmin(scores)
    best_score = scores[best_idx]
    best_colors = population[best_idx].copy()
    
    # GA Loop
    while time.time() < deadline:
        # Selection: Keep top half, replace bottom half with offspring
        # Or use tournament selection
        
        # Let's do a simple (mu+lambda) evolution strategy
        # Generate offspring by mutating parents
        
        num_offspring = pop_size
        offspring = np.zeros((num_offspring, n), dtype=np.int8)
        
        for i in range(num_offspring):
            # Pick a parent (tournament of 2)
            i1 = np.random.randint(0, pop_size)
            i2 = np.random.randint(0, pop_size)
            if scores[i1] < scores[i2]:
                parent = population[i1]
            else:
                parent = population[i2]
            
            # Mutation: Flip a few bits
            # Adaptive mutation rate?
            # Let's try flipping 1 bit (hill climb step) or 5 bits (larger jump)
            # Random choice
            if random.random() < 0.5:
                num_flips = 1
            else:
                num_flips = 5
                
            child = parent.copy()
            if num_flips > 0:
                flip_indices = np.random.choice(n, num_flips, replace=False)
                child[flip_indices] = 1 - child[flip_indices]
                
            offspring[i] = child
            
        # Evaluate offspring
        off_scores = evaluate_population(offspring)
        
        # Combine parents and offspring
        combined_pop = np.vstack([population, offspring])
        combined_scores = np.concatenate([scores, off_scores])
        
        # Select best pop_size
        # argsort is O(N log N)
        sorted_indices = np.argsort(combined_scores)
        new_pop_indices = sorted_indices[:pop_size]
        
        population = combined_pop[new_pop_indices]
        scores = combined_scores[new_pop_indices]
        
        # Update best
        current_best_idx = np.argmin(scores)
        if scores[current_best_idx] < best_score:
            best_score = scores[current_best_idx]
            best_colors = population[current_best_idx].copy()
            
        # Optional: Local search on the best individual
        # If we have time, polish the best solution
        if time.time() < deadline - 10:
            # Simple hill climbing on the best solution
            current = best_colors.copy()
            current_score = best_score
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                # Try flipping each bit? Or random bits?
                # Random bits is faster
                for _ in range(100):
                    if time.time() > deadline - 5:
                        break
                    idx = np.random.randint(0, n)
                    delta = calc_delta_fast(current, idx)
                    if delta < 0:
                        current[idx] = 1 - current[idx]
                        current_score += delta
                        improved = True
                        if current_score < best_score:
                            best_score = current_score
                            best_colors = current.copy()
                            
    return list(best_colors)

# For quick testing
if __name__ == "__main__":
    t0 = time.time()
    result = construct()
    t1 = time.time()
    print(f"Time: {t1-t0:.2f}s")
    print(f"Colors: {result[:20]}...")
    # Verify length and values
    assert len(result) == 256
    assert all(c in (0, 1) for c in result)
    
    # Quick score check
    import numpy as np
    c = np.array(result)
    ap_list = []
    for d in range(1, 86):
        max_i = 256 - 3*d
        for i in range(max_i):
            ap_list.append((i, i+d, i+2*d, i+3*d))
    ap_array = np.array(ap_list)
    mono = np.sum(np.all(c[ap_array] == c[ap_array[:, 0:1]], axis=1))
    print(f"Score: {mono}")

