import time
import math
import random

def max_ap_discrepancy(signs):
    n = len(signs)
    max_abs = 0
    for d in range(1, n + 1):
        for start in range(d):
            max_ending_here = 0
            max_so_far = 0
            min_ending_here = 0
            min_so_far = 0
            idx = start
            while idx < n:
                val = signs[idx]
                # Max subarray sum
                max_ending_here = max(0, max_ending_here + val)
                if max_ending_here > max_so_far:
                    max_so_far = max_ending_here
                # Min subarray sum
                min_ending_here = min(0, min_ending_here + val)
                if min_ending_here < min_so_far:
                    min_so_far = min_ending_here
                idx += d
            if max_so_far > max_abs:
                max_abs = max_so_far
            if -min_so_far > max_abs:
                max_abs = -min_so_far
    return max_abs

def construct():
    n = 256
    signs = [0] * n
    
    # Greedy construction
    for i in range(n):
        # Evaluate choice +1
        cost_plus = 0
        # Evaluate choice -1
        cost_minus = 0
        
        # We need to check all APs ending at i
        # An AP ending at i has step d and start s.
        # s = i - k*d for k=1..floor(i/d)
        # The sum of the AP excluding the last term (at i) is needed.
        
        # To be efficient, we can just compute the sum of the prefix APs.
        # For each step d from 1 to i (if i>0) or just d=1 if i=0
        # Actually, for i=0, only d=1, start=0. Sum is s0.
        
        # Let's iterate through all possible steps d
        for d in range(1, i + 1):
            # Find the start index for the AP ending at i with step d
            # The AP is: i, i-d, i-2d, ...
            # We need the sum of signs[i-d], signs[i-2d], ...
            # Let's call this prev_sum
            
            prev_sum = 0
            idx = i - d
            while idx >= 0:
                prev_sum += signs[idx]
                idx -= d
            
            # If we choose +1, new sum is prev_sum + 1
            # If we choose -1, new sum is prev_sum - 1
            # We want to minimize the max absolute value over all such APs
            
            val_plus = abs(prev_sum + 1)
            val_minus = abs(prev_sum - 1)
            
            if val_plus > cost_plus:
                cost_plus = val_plus
            if val_minus > cost_minus:
                cost_minus = val_minus
        
        # Choose the sign with lower cost
        if cost_plus < cost_minus:
            signs[i] = 1
        elif cost_minus < cost_plus:
            signs[i] = -1
        else:
            # Tie-breaking: random or alternate
            signs[i] = 1 if i % 2 == 0 else -1
            
    # Local Search / Repair Phase
    # The greedy approach might get stuck. Let's do a simple hill climbing.
    current_max = max_ap_discrepancy(signs)
    best_signs = signs[:]
    best_max = current_max
    
    start_time = time.time()
    timeout = 55
    
    # Simple Hill Climbing with random restarts if stuck
    # But since we have time, we can do a more sophisticated local search.
    # Flip a sign if it reduces the global max discrepancy.
    
    while (time.time() - start_time) < timeout:
        improved = False
        # Try flipping each sign
        for i in range(n):
            signs[i] *= -1
            new_max = max_ap_discrepancy(signs)
            if new_max < current_max:
                current_max = new_max
                improved = True
                if current_max < best_max:
                    best_max = current_max
                    best_signs = signs[:]
            else:
                signs[i] *= -1 # Revert
                
        if not improved:
            # Stuck in local minimum. Do a random perturbation.
            # Flip 3-5 random signs
            for _ in range(5):
                i = random.randint(0, n - 1)
                signs[i] *= -1
            current_max = max_ap_discrepancy(signs)
            # If perturbation made it worse, we accept it to escape local min
            # (Simulated Annealing style but simpler)
            
    return best_signs
