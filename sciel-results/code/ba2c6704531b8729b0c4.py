import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 30
    # Pre-compute weights as floats
    weights = [1.0 / i for i in range(1, n + 1)]
    
    best_abs_exact = float('inf')
    best_coeffs = None
    best_sum_exact = None
    
    # Helper to verify and update best
    def try_update_best(coeffs):
        nonlocal best_abs_exact, best_coeffs, best_sum_exact
        # Calculate exact sum
        s = Fraction(0, 1)
        for i, c in enumerate(coeffs):
            if c != 0:
                s += Fraction(c, i + 1)
        
        if s == 0:
            return False
            
        abs_s = abs(s)
        if abs_s < best_abs_exact:
            best_abs_exact = abs_s
            best_coeffs = coeffs[:]
            best_sum_exact = s
            return True
        return False

    # Initial valid solution
    init_coeffs = [0] * 28 + [1, -1]
    try_update_best(init_coeffs)
    
    def local_search_float(start_coeffs):
        nonlocal best_abs_exact, best_coeffs, best_sum_exact
        current_coeffs = list(start_coeffs)
        current_sum = sum(c * w for c, w in zip(current_coeffs, weights))
        
        # If start is zero, kick it
        if current_sum == 0:
            # Flip index 30 (smallest weight) to 1
            current_coeffs[29] = 1
            current_sum = sum(c * w for c, w in zip(current_coeffs, weights))
            
        # Check initial
        if current_sum != 0:
            try_update_best(current_coeffs)
            
        improved = True
        stall_count = 0
        max_stall = 100
        
        while improved and (time.time() < deadline - 2.0):
            improved = False
            indices = list(range(n))
            random.shuffle(indices)
            
            for i in indices:
                old_val = current_coeffs[i]
                for new_val in [-1, 0, 1]:
                    if new_val == old_val:
                        continue
                    
                    # Calculate change
                    delta = (new_val - old_val) * weights[i]
                    new_sum = current_sum + delta
                    
                    # Skip if zero (approx)
                    if abs(new_sum) < 1e-15:
                        continue
                        
                    if abs(new_sum) < abs(current_sum):
                        # Accept
                        current_coeffs[i] = new_val
                        current_sum = new_sum
                        improved = True
                        stall_count = 0
                        
                        # Check for global best
                        # To avoid too many Fraction calls, only check if float improvement is significant
                        # or if we are very close to best
                        if abs(new_sum) < best_abs_exact * 1.01:
                             try_update_best(current_coeffs)
                        break # Restart index loop? Or just continue?
                             # Breaking allows us to re-shuffle or just move to next index.
                             # Moving to next index is fine.
            
            if not improved:
                # Stuck
                stall_count += 1
                if stall_count > max_stall:
                    # Random kick
                    kick_idx = random.randint(0, n - 1)
                    old_val = current_coeffs[kick_idx]
                    choices = [v for v in [-1, 0, 1] if v != old_val]
                    new_val = random.choice(choices)
                    current_coeffs[kick_idx] = new_val
                    current_sum = sum(c * w for c, w in zip(current_coeffs, weights))
                    stall_count = 0
                    # Re-check best after kick?
                    if current_sum != 0:
                        try_update_best(current_coeffs)

        # Final check after local search ends
        if current_sum != 0:
            try_update_best(current_coeffs)

    # Main Loop
    restarts = 0
    while time.time() < deadline - 5.0:
        # Generate start
        if restarts == 0:
            coeffs = [0]*28 + [1, -1]
        elif restarts < 5:
            # Dense random signs
            coeffs = [random.choice([-1, 1]) for _ in range(n)]
        else:
            # Mix of sparse and dense
            if random.random() < 0.5:
                coeffs = [random.choice([-1, 1]) for _ in range(n)]
            else:
                coeffs = [random.choice([-1, 0, 1]) for _ in range(n)]
        
        local_search_float(coeffs)
        restarts += 1
        
        # If we found a very small value, we can stop
        if best_abs_exact < 1e-12:
            break

    if best_coeffs is None:
        best_coeffs = [0] * 28 + [1, -1]
        
    return best_coeffs
