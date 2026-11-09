import time
import random
from math import gcd

def construct():
    start_time = time.time()
    # Use 55s limit to be safe against the 60s kill
    deadline = start_time + 55.0
    
    n = 30
    
    # Calculate LCM of 1..30
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = 1
    for i in range(1, n + 1):
        L = lcm(L, i)
        
    # Weights w_i = L / i
    weights = [L // i for i in range(1, n + 1)]
    
    best_sum = None
    best_coeffs = None
    
    def evaluate(coeffs):
        s = 0
        for i in range(n):
            s += coeffs[i] * weights[i]
        return s

    def try_improve(coeffs):
        current_sum = evaluate(coeffs)
        # If current sum is 0, it's invalid. We need to move away.
        # If current sum is non-zero, we try to reduce abs(sum).
        
        improved = True
        # Limit iterations to avoid infinite loops or getting stuck
        max_iters = 1000 
        iter_count = 0
        
        while improved and iter_count < max_iters:
            iter_count += 1
            if time.time() > deadline:
                break
            improved = False
            # Shuffle indices to avoid bias? Or just iterate.
            # Iterating is faster.
            for i in range(n):
                if time.time() > deadline:
                    return coeffs, current_sum
                old_val = coeffs[i]
                # Try changing to other values
                for new_val in [-1, 0, 1]:
                    if new_val == old_val:
                        continue
                    delta = (new_val - old_val) * weights[i]
                    new_sum = current_sum + delta
                    
                    # Constraint: new_sum must be non-zero
                    if new_sum == 0:
                        # We hit zero. This is the best possible "target" but invalid.
                        # We should accept a move to zero only if we can immediately 
                        # find a good neighbor? 
                        # Actually, if we hit zero, we should stop improving this path 
                        # and return the current non-zero best if we had one, 
                        # or find the best neighbor of zero.
                        # For simplicity in the inner loop, if we hit 0, 
                        # we check if we have a valid best_sum.
                        # If we are currently at a non-zero sum, and we can move to 0,
                        # that's great progress, but we can't stay there.
                        # We should look for the smallest non-zero neighbor of 0.
                        # However, doing a full neighbor search here is expensive.
                        # Alternative: If new_sum == 0, we treat it as a "dead end" for improvement
                        # unless we are already at a very small sum.
                        # Let's just skip moves to 0 in the "improvement" phase.
                        # We will handle 0 specially at the end or with a dedicated check.
                        continue
                    
                    # If new_sum is non-zero
                    if abs(new_sum) < abs(current_sum):
                        # Accept the move
                        coeffs[i] = new_val
                        current_sum = new_sum
                        improved = True
                        break # Restart the for loop? Or continue? 
                        # Breaking the for loop and restarting the while loop is standard hill climbing.
                        # But we can just continue checking other indices.
                        # Let's break to restart the sweep to ensure we catch cascading improvements.
        
        return coeffs, current_sum

    # Main Search Loop
    num_random_starts = 0
    while time.time() < deadline:
        # Generate random candidate
        coeffs = [random.choice([-1, 0, 1]) for _ in range(n)]
        
        # Run local search
        improved_coeffs, s_imp = try_improve(coeffs)
        
        # Check if result is valid (non-zero)
        if s_imp != 0:
            if best_sum is None or abs(s_imp) < abs(best_sum):
                best_sum = s_imp
                best_coeffs = improved_coeffs[:]
        else:
            # If we ended up at 0 (unlikely with the 'continue' on new_sum==0 logic, 
            # but possible if initial was 0 and no move improved it? 
            # If initial is 0, try_improve returns 0.
            # We need to handle the case where we start at 0 or end at 0.
            # If s_imp == 0, find the best non-zero neighbor.
            best_neighbor_sum = None
            best_neighbor_coeffs = None
            for j in range(n):
                old_val = improved_coeffs[j]
                for new_val in [-1, 0, 1]:
                    if new_val == old_val:
                        continue
                    ns = s_imp + (new_val - old_val) * weights[j]
                    if ns != 0:
                        if best_neighbor_sum is None or abs(ns) < abs(best_neighbor_sum):
                            best_neighbor_sum = ns
                            best_neighbor_coeffs = improved_coeffs[:]
                            best_neighbor_coeffs[j] = new_val
            
            if best_neighbor_coeffs:
                if best_sum is None or abs(best_neighbor_sum) < abs(best_sum):
                    best_sum = best_neighbor_sum
                    best_coeffs = best_neighbor_coeffs
        
        num_random_starts += 1
        
        # Early exit if we found the theoretical minimum?
        # The minimum non-zero sum is likely related to 1/LCM or small combinations.
        # We can't easily know the global minimum, but we can stop if we find something very small.
        # For now, just run until deadline.

    if best_coeffs is None:
        # Fallback
        # Try to construct a known small sum, e.g., 1/2 - 1/3 = 1/6
        # Coeffs: [0, 1, -1, 0, ...]
        fallback = [0] * 30
        fallback[1] = 1  # 1/2
        fallback[2] = -1 # -1/3
        return fallback
        
    return best_coeffs
