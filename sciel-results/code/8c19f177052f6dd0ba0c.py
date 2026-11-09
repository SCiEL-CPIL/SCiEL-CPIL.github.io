import time
import random
import math
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    n = 30
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    # Calculate LCM(1..30)
    L = 1
    for i in range(1, n + 1):
        L = lcm(L, i)
        
    # Weights w[i] = L / (i+1) for i in 0..29
    # Sum S = sum c[i] * (i+1) / (L) ? No.
    # Sum S = sum_{k=1}^30 c_k / k.
    # Let N = S * L. Then N = sum c_k * (L/k).
    # We want to minimize |N| subject to N != 0.
    
    weights = [L // i for i in range(1, n + 1)]
    
    def evaluate(coeffs):
        s = 0
        for i in range(n):
            s += coeffs[i] * weights[i]
        return s

    def make_random_valid():
        while True:
            coeffs = [random.choice([-1, 0, 1]) for _ in range(n)]
            s = evaluate(coeffs)
            if s != 0:
                return coeffs, s

    def polish(coeffs, current_sum):
        # Local search: try all single moves, then all pair moves.
        # Repeat until no improvement in |sum| is found.
        improved = True
        while improved:
            improved = False
            
            # 1. Single moves
            for i in range(n):
                old_val = coeffs[i]
                for new_val in [-1, 0, 1]:
                    if new_val == old_val:
                        continue
                    delta = (new_val - old_val) * weights[i]
                    new_sum = current_sum + delta
                    if new_sum == 0:
                        continue
                    if abs(new_sum) < abs(current_sum):
                        coeffs[i] = new_val
                        current_sum = new_sum
                        improved = True
                        # Don't break, keep checking other values for this i? 
                        # Actually, if we improve, the state changed. 
                        # It's better to restart the loop or continue carefully.
                        # Given n=30, n*3 = 90 checks. It's fast.
                        # Let's just mark improved and continue scanning. 
                        # But current_sum changed. So subsequent deltas are based on new state.
                        # This is standard steepest descent.
            
            # 2. Pair moves
            # There are 30*29/2 * 3*3 ~ 3885 checks.
            # This might be slower. Let's do single first, then pair if single didn't improve in the last full pass?
            # Or just alternate.
            # Let's try pair moves only if single moves didn't find an improvement in the previous iteration?
            # No, let's just do single moves in a tight loop, and pair moves if stuck.
            
            if not improved:
                # Try pair moves
                for i in range(n):
                    if improved: break
                    old_val1 = coeffs[i]
                    for j in range(i + 1, n):
                        if improved: break
                        old_val2 = coeffs[j]
                        for new_val1 in [-1, 0, 1]:
                            if new_val1 == old_val1:
                                continue
                            for new_val2 in [-1, 0, 1]:
                                if new_val2 == old_val2:
                                    continue
                                delta = (new_val1 - old_val1) * weights[i] + (new_val2 - old_val2) * weights[j]
                                new_sum = current_sum + delta
                                if new_sum == 0:
                                    continue
                                if abs(new_sum) < abs(current_sum):
                                    coeffs[i] = new_val1
                                    coeffs[j] = new_val2
                                    current_sum = new_sum
                                    improved = True
                                    # Break out of loops?
                                    # Yes, restart the outer 'while improved' loop to re-evaluate everything from scratch.
                                    # This is safer and often faster due to simpler control flow.
                                    break
                            if improved: break
                        if improved: break
                    if improved: break
        
        return coeffs, current_sum

    best_coeffs = None
    best_sum = None
    best_abs_sum = float('inf')
    
    # Initial run
    for _ in range(5):
        if time.time() > deadline:
            break
        coeffs, s = make_random_valid()
        coeffs, s = polish(coeffs, s)
        if abs(s) < best_abs_sum:
            best_abs_sum = abs(s)
            best_sum = s
            best_coeffs = coeffs[:]
            
    # Iterated Local Search
    # Perturb the best solution and polish
    while time.time() < deadline:
        if best_coeffs is None:
            break
            
        # Create a perturbed copy
        current_coeffs = best_coeffs[:]
        current_sum = best_sum
        
        # Perturb: flip 2 to 4 random distinct indices
        num_flips = random.randint(2, 4)
        indices = random.sample(range(n), num_flips)
        for idx in indices:
            old_val = current_coeffs[idx]
            new_val = random.choice([-1, 0, 1])
            while new_val == old_val:
                new_val = random.choice([-1, 0, 1])
            delta = (new_val - old_val) * weights[idx]
            current_sum += delta
            current_coeffs[idx] = new_val
            
        # If perturbation led to 0, it's invalid.
        # We can skip polishing if it's 0, or just let polish handle it (polish avoids 0).
        # Actually, if current_sum is 0, polish will try to move away.
        
        current_coeffs, current_sum = polish(current_coeffs, current_sum)
        
        if abs(current_sum) < best_abs_sum:
            best_abs_sum = abs(current_sum)
            best_sum = current_sum
            best_coeffs = current_coeffs[:]
            
    # Final check and fallback
    if best_coeffs is None:
        # Should not happen, but just in case
        fallback = [0] * 30
        fallback[1] = 1
        fallback[2] = -1
        return fallback
        
    return best_coeffs
