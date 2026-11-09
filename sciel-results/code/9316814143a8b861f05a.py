"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    n = 30
    # Calculate LCM of 1..30
    lcm_val = 1
    for i in range(1, n + 1):
        lcm_val = lcm_val * i // gcd(lcm_val, i)
        
    # Weights for integer arithmetic: S = sum(c_i / i) = (1/LCM) * sum(c_i * LCM/i)
    weights = [lcm_val // i for i in range(1, n + 1)]
    
    best_abs_num = None
    best_coeffs = None
    
    # Initial candidate from seed
    seed_coeffs = [0] * (n - 2) + [1, -1]
    seed_num = sum(c * w for c, w in zip(seed_coeffs, weights))
    if seed_num != 0:
        best_abs_num = abs(seed_num)
        best_coeffs = seed_coeffs[:]
    
    def evaluate(coeffs):
        s = 0
        for c, w in zip(coeffs, weights):
            if c != 0:
                s += c * w
        return s
    
    # Random Local Search
    # We use a simple hill-climbing with random restarts
    # State is a list of ints in {-1, 0, 1}
    
    def random_state():
        return [random.choice([-1, 0, 1]) for _ in range(n)]
    
    def local_search(init_coeffs, time_limit):
        nonlocal best_abs_num, best_coeffs
        
        current_coeffs = init_coeffs[:]
        current_num = evaluate(current_coeffs)
        current_abs = abs(current_num) if current_num != 0 else float('inf')
        
        # If current is zero, we need to move away from zero
        # If current is non-zero, we try to reduce it
        
        while time.time() < time_limit:
            # Pick a random index to flip or change
            idx = random.randint(0, n - 1)
            
            # Try changing c[idx] to one of {-1, 0, 1} \ {current}
            possible_vals = [v for v in [-1, 0, 1] if v != current_coeffs[idx]]
            random.shuffle(possible_vals)
            
            moved = False
            for new_val in possible_vals:
                old_val = current_coeffs[idx]
                current_coeffs[idx] = new_val
                
                # Update sum incrementally
                delta = (new_val - old_val) * weights[idx]
                new_num = current_num + delta
                new_abs = abs(new_num) if new_num != 0 else float('inf')
                
                # Acceptance criterion:
                # 1. If new is non-zero and smaller than current, accept.
                # 2. If new is zero, accept if it allows escape (we don't want to stay at 0, but passing through 0 is fine if we land on small non-zero).
                #    Actually, if we land on 0, we must move again.
                #    Let's just accept if new_abs < current_abs (and new_num != 0).
                #    If new_num == 0, we accept it temporarily to get out of a local minimum? 
                #    No, better to reject 0 unless we are stuck?
                #    Let's define score: 0 is "infinite bad" for our objective of minimizing non-zero.
                #    So if new_num == 0, new_abs = inf.
                
                if new_abs < current_abs:
                    # Strict improvement
                    current_num = new_num
                    current_abs = new_abs
                    moved = True
                    
                    # Update global best
                    if best_abs_num is None or current_abs < best_abs_num:
                        best_abs_num = current_abs
                        best_coeffs = current_coeffs[:]
                    break
                else:
                    # Revert
                    current_coeffs[idx] = old_val
            
            if not moved:
                # Stuck in local minimum? Try a random jump (restart local search from random state or just keep trying)
                # To avoid infinite loops, we might just continue trying random moves which might break symmetry
                # Or we can perform a random jump: change 3-5 random indices
                if random.random() < 0.05:
                    for _ in range(random.randint(1, 3)):
                        j = random.randint(0, n - 1)
                        current_coeffs[j] = random.choice([-1, 0, 1])
                    current_num = evaluate(current_coeffs)
                    current_abs = abs(current_num) if current_num != 0 else float('inf')
        
        return current_coeffs

    # Run multiple restarts
    # Start with seed
    local_search(seed_coeffs, deadline)
    
    # Then random restarts
    restarts = 0
    while time.time() < deadline:
        init = random_state()
        local_search(init, deadline)
        restarts += 1
        # Optional: break if we found 1 (minimum possible non-zero numerator is 1/LCM? No, min non-zero integer is 1)
        # If best_abs_num == 1, we can't do better.
        if best_abs_num == 1:
            break

    # Return the best coefficients found
    if best_coeffs is None:
        # Fallback to seed
        return seed_coeffs
    return best_coeffs

# EVOLVE-BLOCK-END
