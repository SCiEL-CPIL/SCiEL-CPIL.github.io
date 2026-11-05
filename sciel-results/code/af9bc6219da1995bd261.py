import time
import math
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 85.0  # 85 seconds to be safe against 90s limit
    
    n = 30
    
    # Precompute LCM(1..30)
    LCM = 1
    for i in range(1, n + 1):
        LCM = LCM * i // math.gcd(LCM, i)
    
    # term_nums[i] is the integer numerator of 1/(i+1) when expressed with denominator LCM
    term_nums = [LCM // (i + 1) for i in range(n)]
    
    def compute_sum(coeffs):
        """Compute the integer numerator of the signed sum."""
        s = 0
        for i, c in enumerate(coeffs):
            if c != 0:
                s += c * term_nums[i]
        return s
    
    def greedy_init():
        """Create a random initial configuration with several non-zero terms."""
        coeffs = [0] * n
        num_nonzero = random.randint(8, 20)
        indices = random.sample(range(n), num_nonzero)
        for idx in indices:
            coeffs[idx] = random.choice([-1, 1])
        return coeffs
    
    def local_search(coeffs, max_iters=5000):
        """Perform hill climbing to minimize |sum| while keeping it nonzero."""
        current_sum = compute_sum(coeffs)
        if current_sum == 0:
            # Perturb to get nonzero
            idx = random.randrange(n)
            coeffs[idx] = 1 if coeffs[idx] != 1 else -1
            current_sum = compute_sum(coeffs)
        
        best_sum = current_sum
        best_coeffs = coeffs[:]
        best_abs = abs(best_sum)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            # Try all single-coordinate changes
            best_delta = 0
            best_change = None
            best_new_abs = best_abs
            
            for i in range(n):
                for delta in [-1, 1]:
                    new_c = coeffs[i] + delta
                    if new_c not in (-1, 0, 1):
                        continue
                    new_sum = current_sum + delta * term_nums[i]
                    if new_sum == 0:
                        continue
                    new_abs = abs(new_sum)
                    if new_abs < best_new_abs:
                        best_new_abs = new_abs
                        best_delta = delta
                        best_change = i
                        best_sum = new_sum
            
            if best_change is not None:
                coeffs[best_change] += best_delta
                current_sum = best_sum
                best_abs = best_new_abs
                best_coeffs = coeffs[:]
            else:
                # No improving move found; try a random perturbation to escape local minimum
                num_perturb = random.randint(1, 3)
                for _ in range(num_perturb):
                    idx = random.randrange(n)
                    old_val = coeffs[idx]
                    choices = [c for c in (-1, 0, 1) if c != old_val]
                    coeffs[idx] = random.choice(choices)
                current_sum = compute_sum(coeffs)
                if current_sum == 0:
                    # Force nonzero
                    idx2 = random.randrange(n)
                    coeffs[idx2] = 1 if coeffs[idx2] != 1 else -1
                    current_sum = compute_sum(coeffs)
                best_abs = abs(current_sum)
                best_coeffs = coeffs[:]
        
        return best_coeffs, best_sum
    
    # Random restart hill climbing
    best_global_coeffs = None
    best_global_sum = None
    best_global_abs = float('inf')
    
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        coeffs = greedy_init()
        coeffs, s = local_search(coeffs, max_iters=3000)
        
        if s == 0:
            continue
        abs_s = abs(s)
        if abs_s < best_global_abs:
            best_global_abs = abs_s
            best_global_sum = s
            best_global_coeffs = coeffs[:]
        
        # If we've found a very good solution, we can reduce remaining time
        if best_global_abs <= 1:
            break
    
    if best_global_coeffs is None:
        # Fallback: simple 2-term solution
        return [0] * 28 + [1, -1]
    
    return best_global_coeffs

# EVOLVE-BLOCK-END
