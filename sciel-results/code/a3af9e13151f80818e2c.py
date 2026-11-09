import time
import random
import math
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    n = 20
    max_val = 10000
    
    # Precompute all pairs (i, j) with i <= j for n=20
    pairs_i = np.array([i for i in range(n) for j in range(i, n)], dtype=np.int32)
    pairs_j = np.array([j for i in range(n) for j in range(i, n)], dtype=np.int32)
    
    def evaluate(A):
        """Compute max(|A+A|, |A*A|) for set A using NumPy vectorization."""
        A_arr = np.array(A, dtype=np.int64)
        sums = A_arr[pairs_i] + A_arr[pairs_j]
        prods = A_arr[pairs_i] * A_arr[pairs_j]
        sumset_size = len(np.unique(sums))
        prodset_size = len(np.unique(prods))
        return max(sumset_size, prodset_size)
    
    # Generate the pool of 5-smooth numbers (Hamming numbers) <= max_val
    # Form: 2^a * 3^b * 5^c
    smooth_pool = set()
    for a in range(0, 14):  # 2^13 = 8192
        v2 = 1 << a
        if v2 > max_val: break
        for b in range(0, 9): # 3^8 = 6561
            v3 = v2 * (3 ** b)
            if v3 > max_val: break
            for c in range(0, 6): # 5^5 = 3125
                v5 = v3 * (5 ** c)
                if v5 > max_val: break
                smooth_pool.add(v5)
    
    smooth_list = sorted(list(smooth_pool))
    num_smooth = len(smooth_list)
    
    # Map values to exponent vectors (a, b, c)
    val_to_exp = {}
    for v in smooth_list:
        a = 0
        while v % 2 == 0:
            v //= 2
            a += 1
        b = 0
        while v % 3 == 0:
            v //= 3
            b += 1
        c = 0
        while v % 5 == 0:
            v //= 5
            c += 1
        # v should be 1 now
        val_to_exp[smooth_list[smooth_list.index(v * (2**a) * (3**b) * (5**c))] if False else next(x for x in smooth_list if x == (2**a)*(3**b)*(5**c))] = (a,b,c)
        
    # Re-do mapping correctly
    val_to_exp = {}
    exps = []
    for v in smooth_list:
        a = 0
        temp = v
        while temp % 2 == 0:
            temp //= 2
            a += 1
        b = 0
        while temp % 3 == 0:
            temp //= 3
            b += 1
        c = 0
        while temp % 5 == 0:
            temp //= 5
            c += 1
        val_to_exp[v] = (a, b, c)
        exps.append((a, b, c))
    
    exps_arr = np.array(exps, dtype=np.int32)
    
    # Geometric Proxy: Minimize the "spread" of exponent vectors.
    # A tight cluster of exponent vectors implies a small product set (Minkowski sum of exponents)
    # and often a smaller sum set due to the multiplicative structure keeping values close.
    
    def geometric_score(indices):
        """Calculate a proxy score based on the convex hull area/volume of exponent vectors."""
        if len(indices) < 3:
            return 0
        pts = exps_arr[indices]
        # Simple proxy: Variance of coordinates
        # Or better: Range in each dimension
        spread = 0
        for dim in range(3):
            spread += np.ptp(pts[:, dim])
        return spread

    # Initial Strategy: Greedy selection minimizing geometric spread
    # Start with the smallest number (1) and add numbers that minimize the increase in spread
    # This creates a "compact" set in the exponent lattice.
    
    current_indices = [0] # Start with 1
    current_vals = [smooth_list[0]]
    
    for _ in range(n - 1):
        best_idx = -1
        best_score = float('inf')
        # Try adding each unused smooth number
        for idx in range(num_smooth):
            if idx in current_indices:
                continue
            new_indices = current_indices + [idx]
            score = geometric_score(new_indices)
            # Add a small penalty for large values to keep sumset small?
            # The geometric score already biases towards small exponents.
            if score < best_score:
                best_score = score
                best_idx = idx
        current_indices.append(best_idx)
        current_vals.append(smooth_list[best_idx])
        
    # Sort indices to keep track
    current_indices.sort()
    current_vals = [smooth_list[i] for i in current_indices]
    
    current_score = evaluate(current_vals)
    best_set = current_vals[:]
    best_score = current_score
    
    # Local Search: Optimize the exact score, using geometric score for candidate generation
    # We perform hill climbing on the exact score, but we guide the search using the geometric structure.
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        # Try swapping each element with a candidate that improves or maintains geometric compactness
        # To speed up, we precompute a list of "good" neighbors based on geometric distance
        # But for simplicity, we scan the pool, prioritizing those close in exponent space.
        
        # Sort smooth numbers by their exponent vector distance to the centroid of current set
        centroid = np.mean(exps_arr[current_indices], axis=0)
        distances = np.linalg.norm(exps_arr - centroid, axis=1)
        sorted_candidates = np.argsort(distances)
        
        for i in range(n):
            old_idx = current_indices[i]
            old_val = current_vals[i]
            
            # Try candidates in order of geometric proximity
            for new_idx in sorted_candidates:
                new_idx = int(new_idx)
                if new_idx == old_idx:
                    continue
                if new_idx in current_indices:
                    continue
                
                new_val = smooth_list[new_idx]
                
                # Swap
                current_vals[i] = new_val
                current_indices[i] = new_idx
                
                new_score = evaluate(current_vals)
                
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    if current_score < best_score:
                        best_score = current_score
                        best_set = current_vals[:]
                else:
                    # Revert
                    current_vals[i] = old_val
                    current_indices[i] = old_idx
                    
                # Optional: Break if we found a significant improvement to save time?
                # No, keep searching for global best in this pass.
                
        # If stuck, perturb
        if not improved and time.time() < deadline - 10:
            # Randomly kick 1-2 elements to random nearby smooth numbers
            for _ in range(random.randint(1, 2)):
                idx_to_change = random.randint(0, n-1)
                # Pick a random neighbor in exponent space
                old_exp = exps_arr[current_indices[idx_to_change]]
                # Random walk in exponent space
                new_exp = old_exp + np.random.randint(-2, 3, size=3)
                # Find closest smooth number to this new exponent? 
                # Simpler: just pick a random smooth number that is distinct.
                new_idx = random.randint(0, num_smooth-1)
                if new_idx not in current_indices:
                    current_vals[idx_to_change] = smooth_list[new_idx]
                    current_indices[idx_to_change] = new_idx
                    current_score = evaluate(current_vals)
                    # Don't update best immediately, let next loop handle it
                    # But if this kick was good, we should note it? 
                    # The main loop will re-evaluate and accept if it's better than current_score 
                    # (which is now the kicked score). 
                    # Wait, if the kick is worse, current_score increases. 
                    # The next hill climbing pass will try to fix it.
                    # This is standard simulated annealing-like behavior with T=0 greedy recovery.

    return sorted(best_set)
