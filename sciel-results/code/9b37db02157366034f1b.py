import time
import random
from math import gcd
from functools import reduce

def _lcm(a, b):
    return a * b // gcd(a, b)

def _compute_lcm_1_to_n(n):
    return reduce(_lcm, range(1, n + 1), 1)

def construct():
    N = 30
    L = _compute_lcm_1_to_n(N)
    weights = [L // i for i in range(1, N + 1)]
    
    # Initialize with seed
    best_coeffs = [0] * 28 + [1, -1]
    best_val_scaled = sum(c * w for c, w in zip(best_coeffs, weights))
    
    if best_val_scaled == 0:
        best_coeffs = [0] * 29 + [1]
        best_val_scaled = weights[-1]

    current_coeffs = best_coeffs[:]
    current_val_scaled = best_val_scaled
    
    start_time = time.time()
    deadline = start_time + 55  # 55s to be very safe
    
    # Local Search with Random Restarts
    while time.time() < deadline:
        # Try to improve current solution
        improved = False
        for i in range(N):
            old_c = current_coeffs[i]
            for new_c in [-1, 0, 1]:
                if new_c == old_c:
                    continue
                delta = (new_c - old_c) * weights[i]
                new_val = current_val_scaled + delta
                if new_val == 0:
                    continue
                if abs(new_val) < abs(current_val_scaled):
                    current_coeffs[i] = new_c
                    current_val_scaled = new_val
                    improved = True
                    if abs(new_val) < abs(best_val_scaled):
                        best_val_scaled = new_val
                        best_coeffs = current_coeffs[:]
                    break # Restart loop as indices changed? No, just move to next i
        
        # If no improvement found in a full pass, do a random kick
        if not improved:
            # Randomly perturb 2-3 positions
            num_kicks = random.randint(2, 4)
            for _ in range(num_kicks):
                idx = random.randint(0, N - 1)
                current_coeffs[idx] = random.choice([-1, 0, 1])
            current_val_scaled = sum(c * w for c, w in zip(current_coeffs, weights))
            if current_val_scaled == 0:
                idx = random.randint(0, N - 1)
                current_coeffs[idx] = 1
                current_val_scaled = sum(c * w for c, w in zip(current_coeffs, weights))
            
            if abs(current_val_scaled) < abs(best_val_scaled):
                best_val_scaled = current_val_scaled
                best_coeffs = current_coeffs[:]

    return best_coeffs
