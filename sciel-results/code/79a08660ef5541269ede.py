"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
from math import gcd

def construct():
    n = 30
    
    # Compute LCM of 1..n
    M = 1
    for i in range(1, n + 1):
        M = M * i // gcd(M, i)
    
    a = [M // i for i in range(1, n + 1)]
    
    best_val = None
    best_coeffs = None
    
    for trial in range(100):
        random.seed(trial)
        coeffs = [random.choice([-1, 0, 1]) for _ in range(n)]
        current_sum = sum(coeffs[i] * a[i] for i in range(n))
        
        if current_sum != 0:
            if best_val is None or abs(current_sum) < abs(best_val):
                best_val = current_sum
                best_coeffs = coeffs[:]
        
        # Local search
        for step in range(2000):
            improved = False
            # Randomly shuffle indices for diversity
            indices = list(range(n))
            random.shuffle(indices)
            for i in indices:
                old_coeff = coeffs[i]
                for new_coeff in [-1, 0, 1]:
                    if new_coeff == old_coeff:
                        continue
                    delta = (new_coeff - old_coeff) * a[i]
                    new_sum = current_sum + delta
                    if new_sum == 0:
                        continue
                    if best_val is None or abs(new_sum) < abs(best_val):
                        best_val = new_sum
                        best_coeffs = coeffs[:]
                        best_coeffs[i] = new_coeff
                    if abs(new_sum) < abs(current_sum):
                        coeffs[i] = new_coeff
                        current_sum = new_sum
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
    
    if best_coeffs is None:
        return [0] * 28 + [1, -1]
    
    s = sum(best_coeffs[i] / (i + 1) for i in range(n))
    if s == 0:
        return [0] * 28 + [1, -1]
    
    return best_coeffs
# EVOLVE-BLOCK-END
