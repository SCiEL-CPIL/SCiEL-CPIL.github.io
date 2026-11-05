"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline, 10s safety margin
    
    n = 30
    # Precompute unit fractions
    inv = [Fraction(1, i) for i in range(1, n + 1)]
    
    best_sum = None
    best_coeffs = None
    best_abs = None
    
    def evaluate(coeffs):
        s = Fraction(0)
        for i, c in enumerate(coeffs):
            if c != 0:
                s += c * inv[i]
        return s
    
    def local_search(coeffs, time_limit):
        nonlocal best_sum, best_coeffs, best_abs
        curr = list(coeffs)
        curr_sum = evaluate(curr)
        curr_abs = abs(curr_sum)
        
        if curr_sum == 0:
            return curr, curr_sum
        
        if best_abs is None or curr_abs < best_abs:
            best_sum = curr_sum
            best_coeffs = list(curr)
            best_abs = curr_abs
        
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            best_flip_idx = -1
            best_flip_val = None
            best_new_sum = None
            best_new_abs = curr_abs
            
            for i in range(n):
                for new_val in (-1, 0, 1):
                    if new_val == curr[i]:
                        continue
                    # Compute new sum
                    new_sum = curr_sum + (new_val - curr[i]) * inv[i]
                    if new_sum == 0:
                        continue
                    new_abs = abs(new_sum)
                    if new_abs < best_new_abs:
                        best_new_abs = new_abs
                        best_flip_idx = i
                        best_flip_val = new_val
                        best_new_sum = new_sum
            
            if best_flip_idx >= 0:
                curr[best_flip_idx] = best_flip_val
                curr_sum = best_new_sum
                curr_abs = best_new_abs
                improved = True
                if best_abs is None or curr_abs < best_abs:
                    best_sum = curr_sum
                    best_coeffs = list(curr)
                    best_abs = curr_abs
        
        return curr, curr_sum
    
    # Try multiple random initializations
    random.seed(42)
    
    # First try the seed solution
    seed_coeffs = [0] * 28 + [1, -1]
    local_search(seed_coeffs, deadline)
    
    # Try various structured initializations
    structured_inits = []
    
    # All ones then flip some
    for _ in range(50):
        coeffs = [random.choice([-1, 0, 1]) for _ in range(n)]
        structured_inits.append(coeffs)
    
    # Greedy initialization: try to cancel terms
    for _ in range(100):
        coeffs = [0] * n
        # Randomly assign
        for i in range(n):
            coeffs[i] = random.choice([-1, 0, 1])
        structured_inits.append(coeffs)
    
    # Try each initialization with local search
    for init_coeffs in structured_inits:
        if (time.time() - start_time) > deadline:
            break
        local_search(init_coeffs, deadline)
    
    # Additional random restarts if time permits
    restart_count = 0
    while (time.time() - start_time) < deadline and restart_count < 500:
        restart_count += 1
        coeffs = [random.choice([-1, 0, 1]) for _ in range(n)]
        local_search(coeffs, deadline)
    
    if best_coeffs is None:
        # Fallback to seed
        return [0] * 28 + [1, -1]
    
    return best_coeffs
# EVOLVE-BLOCK-END
