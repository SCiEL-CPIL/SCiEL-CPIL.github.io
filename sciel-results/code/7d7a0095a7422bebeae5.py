"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    def time_left():
        return deadline - time.time()
    
    def solve_n(n, rng_seed=42):
        if n < 5:
            return None
        
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        
        # Generate K4s and K5s
        k4s = list(combinations(range(n), 4))
        k5s = list(combinations(range(n), 5))
        num_k4 = len(k4s)
        num_k5 = len(k5s)
        
        # Build adjacency: for each triple, which K4s and K5s contain it
        # We need a mapping from triple -> list of K4 indices and K5 indices
        # But for memory efficiency with large n, we might store the triples inside K4/K5
        # and use a reverse index.
        
        # Let's precompute the triple indices for each K4 and K5
        triple_idx = {t: i for i, t in enumerate(triples)}
        
        k4_triples = []
        for k4 in k4s:
            ts = []
            for tri in combinations(k4, 3):
                ts.append(triple_idx[tri])
            k4_triples.append(ts)
            
        k5_triples = []
        for k5 in k5s:
            ts = []
            for tri in combinations(k5, 3):
                ts.append(triple_idx[tri])
            k5_triples.append(ts)
            
        # Reverse index: triple_idx -> list of k4_idx, list of k5_idx
        triple_to_k4 = [[] for _ in range(num_triples)]
        for i, ts in enumerate(k4_triples):
            for t_idx in ts:
                triple_to_k4[t_idx].append(i)
                
        triple_to_k5 = [[] for _ in range(num_triples)]
        for i, ts in enumerate(k5_triples):
            for t_idx in ts:
                triple_to_k5[t_idx].append(i)
        
        # Convert to numpy arrays for fast access? 
        # Actually, lists of lists are okay if we convert the inner lists to np arrays?
        # No, the indices are sparse. 
        # Let's keep lists for now, but maybe convert to numpy arrays of variable length is hard.
        # We can pad them? No.
        # We can just use the lists. The overhead of python loop is the bottleneck.
        
        # Let's try a different approach:
        # Instead of iterating over all K4s/K5s containing the triple,
        # can we compute the delta faster?
        
        # When we flip triple t from color c to 1-c:
        # For every K4 containing t:
        #   If c was Red (0): red_count decreases by 1.
        #     If red_count was 4, it becomes 3 -> delta_red -= 1 (violation fixed)
        #     If red_count was 3, it becomes 4 -> delta_red += 1 (violation created)
        #   If c was Blue (1): red_count increases by 1.
        #     If red_count was 3, it becomes 4 -> delta_red += 1
        #     If red_count was 4, it becomes 5? No, max is 4.
        #     Wait, if c was Blue, it wasn't counted in red_count. So red_count doesn't change?
        #     Ah, red_count is the number of RED triples in the K4.
        #     If t is Blue, it is NOT in red_count. Flipping it to Red INCREASES red_count.
        
        # So:
        # old_color = colors[t]
        # For each k4 in triple_to_k4[t]:
        #   if old_color == 0 (Red -> Blue):
        #      red_count[k4] -= 1
        #      if red_count[k4] == 3: delta_red -= 1 (was 4, now 3)
        #      if red_count[k4] == 4: delta_red += 1 (was 3, now 4) -- Wait.
        #      Let's trace:
        #      old_count = red_count[k4]
        #      new_count = old_count - 1
        #      if old_count == 4 and new_count == 3: delta_red -= 1
        #      if old_count == 3 and new_count == 4: impossible (3-1=2)
        #      Actually:
        #      if old_count == 4: new_count = 3. Violation fixed. delta -= 1.
        #      if old_count == 3: new_count = 2. No violation change.
        #      
        #   if old_color == 1 (Blue -> Red):
        #      red_count[k4] += 1
        #      old_count = red_count[k4]
        #      new_count = old_count + 1
        #      if old_count == 3 and new_count == 4: delta_red += 1
        #      if old_count == 4 and new_count == 5: impossible (max 4)
        
        # Similar for K5s and Blue count.
        
        # Initialize colors
        # Use a randomized initialization based on a pseudo-random permutation or similar
        # to break symmetry and avoid the mod 3 trap which might be too structured.
        # Let's try random init with some bias? Or just random.
        
        # Actually, for n=34, random init has ~50% red.
        # Expected red K4s: C(34,4) * (0.5)^4 = 46376 * 1/16 ~ 2900.
        # Expected blue K5s: C(34,5) * (0.5)^10 = 278256 * 1/1024 ~ 271.
        # So we start with ~3000 red violations and ~300 blue violations.
        
        # We need to be efficient.
        # The bottleneck is the loop over triple_to_k4[t] and triple_to_k5[t].
        # Average size of triple_to_k4[t]:
        # Each triple is in (n-3) K4s. For n=34, that's 31.
        # Each triple is in C(n-3, 2) K5s. For n=34, C(31,2) = 465.
        # So per step, we do ~31 + 465 = 496 updates.
        # In Python, 1 million steps * 500 ops = 500 million ops. Too slow.
        # We need NumPy or a faster approach.
        
        # Let's use NumPy for the counts and try to vectorize the delta calculation?
        # But the indices are sparse (list of indices).
        # We can convert triple_to_k4[t] to a numpy array.
        
        # Let's precompute numpy arrays for triple_to_k4 and triple_to_k5?
        # They have variable length. We can pad them to max length.
        
        max_k4_per_triple = n - 3
        max_k5_per_triple = (n-3)*(n-4)//2
        
        # This might use a lot of memory if we pad.
        # num_triples * max_k5 ~ 6000 * 465 ~ 2.7 million ints. That's fine (22 MB).
        
        # Let's try this.
        
        # Initialize arrays
        # triple_to_k4_arr: (num_triples, max_k4_per_triple)
        # triple_to_k5_arr: (num_triples, max_k5_per_triple)
        # And lengths: len_k4[t], len_k5[t]
        
        triple_to_k4_arr = np.zeros((num_triples, max_k4_per_triple), dtype=np.int32)
        triple_to_k5_arr = np.zeros((num_triples, max_k5_per_triple), dtype=np.int32)
        len_k4 = np.zeros(num_triples, dtype=np.int32)
        len_k5 = np.zeros(num_triples, dtype=np.int32)
        
        for t_idx in range(num_triples):
            if triple_to_k4[t_idx]:
                k4_list = np.array(triple_to_k4[t_idx], dtype=np.int32)
                triple_to_k4_arr[t_idx, :len(k4_list)] = k4_list
                len_k4[t_idx] = len(k4_list)
            if triple_to_k5[t_idx]:
                k5_list = np.array(triple_to_k5[t_idx], dtype=np.int32)
                triple_to_k5_arr[t_idx, :len(k5_list)] = k5_list
                len_k5[t_idx] = len(k5_list)
        
        # Initialize colors and counts
        # Random init
        colors = np.random.randint(0, 2, size=num_triples)
        
        # Calculate initial counts
        red_k4_counts = np.zeros(num_k4, dtype=np.int32)
        blue_k5_counts = np.zeros(num_k5, dtype=np.int32)
        
        # For each triple, add to counts
        # This is still a loop in Python, but we do it once.
        # Or we can vectorize?
        # For each K4, count red triples.
        # K4 i has 4 triples.
        # red_k4_counts[i] = sum(colors[k4_triples[i]] == 0)
        # This is hard to vectorize without a loop over K4s.
        # But num_k4 is ~46k. A loop of 46k is fast.
        
        for i in range(num_k4):
            ts = k4_triples[i]
            red_k4_counts[i] = np.sum(colors[ts] == 0)
            
        for i in range(num_k5):
            ts = k5_triples[i]
            blue_k5_counts[i] = np.sum(colors[ts] == 1)
        
        red_violations = np.sum(red_k4_counts == 4)
        blue_violations = np.sum(blue_k5_counts == 10)
        
        if red_violations == 0 and blue_violations == 0:
            return colors.tolist()
        
        # Simulated Annealing
        T = 10.0
        cooling = 0.9995
        min_T = 0.1
        
        # Precompute random permutation of triples for iteration?
        # Or just random index.
        
        # We need to pick a triple.
        # Strategy: Pick a random K4. If it's a violation, pick a random triple from it.
        # Otherwise, pick a random K5. If violation, pick triple.
        # Otherwise, pick a random triple.
        
        # But checking if K4 is violation is O(1) with numpy array.
        
        # Let's try this selection strategy.
        
        # To speed up, we can use numpy.random for index generation.
        
        # Main loop
        steps = 0
        while time_left() > 5.0:
            # Pick a triple
            # 50% chance try K4, 50% K5
            if random.random() < 0.5:
                k4_idx = np.random.randint(0, num_k4)
                if red_k4_counts[k4_idx] == 4:
                    # Pick random triple from this K4
                    t_idx = k4_triples[k4_idx][np.random.randint(0, 4)]
                else:
                    t_idx = np.random.randint(0, num_triples)
            else:
                k5_idx = np.random.randint(0, num_k5)
                if blue_k5_counts[k5_idx] == 10:
                    t_idx = k5_triples[k5_idx][np.random.randint(0, 10)]
                else:
                    t_idx = np.random.randint(0, num_triples)
            
            old_color = colors[t_idx]
            new_color = 1 - old_color
            
            delta_red = 0
            delta_blue = 0
            
            # Update K4s
            l_k4 = len_k4[t_idx]
            if l_k4 > 0:
                k4_idxs = triple_to_k4_arr[t_idx, :l_k4]
                if old_color == 0: # Red -> Blue
                    old_counts = red_k4_counts[k4_idxs]
                    new_counts = old_counts - 1
                    red_k4_counts[k4_idxs] = new_counts
                    delta_red = np.sum(new_counts == 4) - np.sum(old_counts == 4)
                else: # Blue -> Red
                    old_counts = red_k4_counts[k4_idxs]
                    new_counts = old_counts + 1
                    red_k4_counts[k4_idxs] = new_counts
                    delta_red = np.sum(new_counts == 4) - np.sum(old_counts == 4)
            
            # Update K5s
            l_k5 = len_k5[t_idx]
            if l_k5 > 0:
                k5_idxs = triple_to_k5_arr[t_idx, :l_k5]
                if old_color == 1: # Blue -> Red
                    old_counts = blue_k5_counts[k5_idxs]
                    new_counts = old_counts - 1
                    blue_k5_counts[k5_idxs] = new_counts
                    delta_blue = np.sum(new_counts == 10) - np.sum(old_counts == 10)
                else: # Red -> Blue
                    old_counts = blue_k5_counts[k5_idxs]
                    new_counts = old_counts + 1
                    blue_k5_counts[k5_idxs] = new_counts
                    delta_blue = np.sum(new_counts == 10) - np.sum(old_counts == 10)
            
            total_delta = delta_red + delta_blue
            
            if total_delta < 0:
                colors[t_idx] = new_color
                red_violations += delta_red
                blue_violations += delta_blue
            else:
                # Accept with probability exp(-total_delta / T)
                if T > min_T:
                    prob = math.exp(-total_delta / T)
                    if random.random() < prob:
                        colors[t_idx] = new_color
                        red_violations += delta_red
                        blue_violations += delta_blue
                    else:
                        # Rollback
                        # We need to revert the counts
                        # This is the expensive part.
                        # Can we avoid rollback by copying? No.
                        # We have to revert.
                        if l_k4 > 0:
                            k4_idxs = triple_to_k4_arr[t_idx, :l_k4]
                            if old_color == 0:
                                red_k4_counts[k4_idxs] += 1
                            else:
                                red_k4_counts[k4_idxs] -= 1
                        if l_k5 > 0:
                            k5_idxs = triple_to_k5_arr[t_idx, :l_k5]
                            if old_color == 1:
                                blue_k5_counts[k5_idxs] += 1
                            else:
                                blue_k5_counts[k5_idxs] -= 1
            
            if red_violations == 0 and blue_violations == 0:
                return colors.tolist()
            
            T *= cooling
            steps += 1
            
        return None

    best_n = 9
    best_colors = [0 if (a+b+c) % 3 == 0 else 1 for a,b,c in combinations(range(9), 3)]
    
    # Try to solve for increasing n
    # We start from a smaller n to find a valid coloring, then try to extend?
    # Or just try each n independently.
    # Since the problem is hard, we might need to start from a known valid coloring
    # for a smaller n and extend it?
    # But the structure might not extend easily.
    
    # Let's try n=10, 11, ...
    # We give more time to larger n?
    # Or just let it run until deadline.
    
    # Actually, the current code tries n=10 to 34.
    # If n=10 is easy, it will be fast.
    # If n=20 is hard, it might take a long time.
    
    # We should probably try to find the max n we can solve.
    
    # Let's try a range of n, but prioritize larger n?
    # No, if we can't solve n=15, we likely can't solve n=20.
    # So we should go up sequentially.
    
    # But we also want to maximize n.
    
    # Let's try n from 10 to 34.
    
    # We can use multiple random seeds for each n to increase chance.
    
    for n in range(10, 35):
        if time_left() < 20.0:
            break
            
        # Try a few random seeds for this n
        solved = False
        for seed in range(10): # 10 restarts
            if time_left() < 10.0:
                break
            result = solve_n(n, rng_seed=seed)
            if result is not None:
                best_n = n
                best_colors = result
                solved = True
                break
        
        if not solved:
            # If we fail a small n, maybe larger n is also hard?
            # But maybe a specific n is easier? Unlikely.
            # Let's continue to the next n, but with less time?
            # Or break if n is small and we failed?
            # Let's just continue.
            pass

    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
