import time
import math
import random
import numpy as np

def construct():
    n = 256
    # 1. Continuous Optimization Phase
    # x is a numpy array of floats in [-1, 1]
    x = np.random.uniform(-1, 1, n)
    
    start_time = time.time()
    timeout = 50  # Leave 10s for refinement
    
    # Learning rate schedule
    eta = 0.1
    
    # Precompute APs? No, just iterate.
    # To speed up finding the max AP sum, we can just compute all sums.
    
    for step in range(10000):
        if time.time() - start_time > timeout:
            break
            
        # Compute all AP sums
        # We want to find the AP with the largest absolute sum.
        max_abs_sum = 0
        max_ap_indices = None
        max_sign = 1
        
        # Efficient calculation of AP sums
        # For a fixed step d, the APs are disjoint chains? No, they overlap.
        # But we can iterate d, then start.
        
        # To make this faster in Python, we might need to be careful.
        # Let's try to vectorize or just use pure python and hope it's fast enough.
        # n=256 is small.
        
        # Alternative: Use the fact that we only need the TOP 1 AP.
        # But we don't know which one is top without checking all.
        
        # Let's optimize the inner loop.
        # For each d in 1..n:
        #   For each start in 0..d-1:
        #     Sum x[start], x[start+d], ...
        
        # This is still O(n^2 log n).
        # 256^2 * 8 = 500k ops. In numpy, we can do better.
        
        # Let's try a numpy approach for the sums.
        # It's tricky to get all AP sums in one shot with numpy easily without loops.
        # Let's stick to python loops but optimize them.
        
        # Actually, let's just run the python loop.
        current_max_abs = 0
        best_d = 1
        best_start = 0
        best_sign = 1
        best_indices = []
        
        for d in range(1, n + 1):
            for start in range(d):
                # Sum x[start::d]
                # Using slicing creates a new array.
                # sum(x[start::d])
                s = np.sum(x[start::d])
                abs_s = abs(s)
                if abs_s > current_max_abs:
                    current_max_abs = abs_s
                    best_d = d
                    best_start = start
                    best_sign = 1 if s > 0 else -1
                    # Store indices for update
                    # We can reconstruct later or store now
                    best_indices = list(range(start, n, d))
        
        if current_max_abs == 0:
            break
            
        # Update x
        # Move x in the direction that reduces the sum
        # If sum is positive, we want to decrease it, so subtract from x_i
        # If sum is negative, we want to increase it, so add to x_i
        # Delta = -eta * sign(sum)
        delta = -eta * best_sign
        
        for idx in best_indices:
            x[idx] += delta
            if x[idx] > 1: x[idx] = 1
            elif x[idx] < -1: x[idx] = -1
            
        # Decay eta
        eta *= 0.999
        
    # 2. Discrete Refinement Phase
    # Convert to signs
    signs = [1 if v >= 0 else -1 for v in x]
    
    # Quick local search to fix rounding
    # We'll do a few iterations of "find worst AP, flip a random element in it"
    # This is much faster than full hill climbing if we limit iterations
    
    def get_max_ap_discrepancy_info(signs):
        max_abs = 0
        best_indices = []
        best_sign = 1
        for d in range(1, n + 1):
            for start in range(d):
                s = 0
                indices = []
                idx = start
                while idx < n:
                    s += signs[idx]
                    indices.append(idx)
                    idx += d
                abs_s = abs(s)
                if abs_s > max_abs:
                    max_abs = abs_s
                    best_indices = indices
                    best_sign = 1 if s > 0 else -1
        return max_abs, best_indices, best_sign

    current_max, _, _ = get_max_ap_discrepancy_info(signs)
    
    # Time left
    time_left = 60 - (time.time() - start_time)
    
    if time_left > 5:
        # Simulated Annealing / Random Flips
        # Since we have a good starting point from continuous optimization,
        # simple hill climbing might work if we are careful.
        # But let's do random flips and accept if better.
        
        # To speed up, we can just try flipping elements in the worst AP.
        
        for _ in range(1000):
            if time.time() - start_time > 58:
                break
                
            current_max, worst_indices, _ = get_max_ap_discrepancy_info(signs)
            
            if not worst_indices:
                break
                
            # Pick a random index from the worst AP
            # Or pick the index that, if flipped, reduces the sum the most?
            # Flipping an index changes the sum by -2 * sign.
            # If sum is positive, flipping a +1 to -1 reduces sum by 2.
            # Flipping a -1 to +1 increases sum by 2.
            # So we should flip a +1 if sum > 0, or a -1 if sum < 0.
            
            # Let's find such an index in the worst AP.
            flip_idx = -1
            if _ == 1: # sum > 0
                # Find an index in worst_indices where signs[i] == 1
                for i in worst_indices:
                    if signs[i] == 1:
                        flip_idx = i
                        break
            else: # sum < 0
                # Find an index in worst_indices where signs[i] == -1
                for i in worst_indices:
                    if signs[i] == -1:
                        flip_idx = i
                        break
                        
            if flip_idx != -1:
                # Try the flip
                signs[flip_idx] *= -1
                new_max, _, _ = get_max_ap_discrepancy_info(signs)
                if new_max < current_max:
                    current_max = new_max
                else:
                    signs[flip_idx] *= -1 # Revert
                    
    return signs
