"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def construct():
    start = time.time()
    deadline = start + 85  # safety margin (90s kill, leave 5s)
    
    n = 30
    # weights[i] = 1/(i+1) for i in 0..29
    
    # Precompute LCM of 1..30 for exact arithmetic via integer scaling
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    lcm_val = 1
    for i in range(1, n + 1):
        lcm_val = lcm(lcm_val, i)
    
    # integer_weights[i] = lcm_val // (i+1)
    int_weights = [lcm_val // (i + 1) for i in range(n)]
    
    def eval_int(coeffs):
        """Evaluate sum using integer arithmetic (scaled by lcm_val)."""
        s = 0
        for c, w in zip(coeffs, int_weights):
            if c != 0:
                s += c * w
        return s
    
    best_abs_int = float('inf')
    best_coeffs = None
    
    def try_update(coeffs):
        nonlocal best_abs_int, best_coeffs
        val = eval_int(coeffs)
        if val == 0:
            return  # must be nonzero
        abs_val = abs(val)
        if abs_val < best_abs_int:
            best_abs_int = abs_val
            best_coeffs = list(coeffs)
    
    def local_search(coeffs, time_limit):
        """Greedy local search: flip coefficients to reduce |S|, avoiding zero."""
        nonlocal best_abs_int, best_coeffs
        coeffs = list(coeffs)
        current_sum = eval_int(coeffs)
        current_abs = abs(current_sum)
        if current_sum == 0:
            # Need to make it nonzero first
            idx = random.randint(0, n - 1)
            coeffs[idx] = -coeffs[idx] if coeffs[idx] != 0 else 1
            current_sum = eval_int(coeffs)
            current_abs = abs(current_sum)
        
        if current_abs < best_abs_int:
            best_abs_int = current_abs
            best_coeffs = list(coeffs)
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for i in range(n):
                w = int_weights[i]
                c = coeffs[i]
                # Try flipping: new_sum = current_sum - 2*c*w
                if c != 0:
                    new_sum = current_sum - 2 * c * w
                    if new_sum != 0:
                        new_abs = abs(new_sum)
                        if new_abs < current_abs:
                            coeffs[i] = -c
                            current_sum = new_sum
                            current_abs = new_abs
                            improved = True
                            if current_abs < best_abs_int:
                                best_abs_int = current_abs
                                best_coeffs = list(coeffs)
                else:
                    # Try setting to +1 or -1
                    for sign in (1, -1):
                        new_sum = current_sum + sign * w
                        if new_sum != 0:
                            new_abs = abs(new_sum)
                            if new_abs < current_abs:
                                coeffs[i] = sign
                                current_sum = new_sum
                                current_abs = new_abs
                                improved = True
                                if current_abs < best_abs_int:
                                    best_abs_int = current_abs
                                    best_coeffs = list(coeffs)
                                break
        
        return coeffs
    
    def random_start():
        """Generate a random initial configuration."""
        coeffs = [random.choice([-1, 0, 1]) for _ in range(n)]
        # Ensure at least one nonzero
        if all(c == 0 for c in coeffs):
            coeffs[random.randint(0, n - 1)] = 1
        return coeffs
    
    # Strategy: multiple random restarts with local search
    # Use a mix of random and structured starts
    
    num_restarts = 0
    while time.time() < deadline - 2:
        num_restarts += 1
        
        # Different initialization strategies
        if num_restarts % 3 == 1:
            # Random signs, no zeros
            coeffs = [random.choice([-1, 1]) for _ in range(n)]
        elif num_restarts % 3 == 2:
            # Random with some zeros
            coeffs = [random.choice([-1, 0, 1]) for _ in range(n)]
            if all(c == 0 for c in coeffs):
                coeffs[0] = 1
        else:
            # Start from best and perturb
            if best_coeffs is not None:
                coeffs = list(best_coeffs)
                # Flip 1-3 random coefficients
                for _ in range(random.randint(1, 3)):
                    idx = random.randint(0, n - 1)
                    coeffs[idx] = random.choice([-1, 0, 1])
            else:
                coeffs = [random.choice([-1, 0, 1]) for _ in range(n)]
        
        # Run local search with a time budget
        remaining = deadline - 2 - time.time()
        if remaining < 0.5:
            break
        # Use at most 3 seconds per local search, or remaining time
        ts_limit = min(3.0, remaining)
        local_search(coeffs, time.time() + ts_limit)
    
    # Final verification and return
    if best_coeffs is None:
        # Fallback: simple nonzero sum
        return [0] * 28 + [1, -1]
    
    # Verify the result is valid (nonzero)
    final_sum = eval_int(best_coeffs)
    if final_sum == 0:
        # Shouldn't happen, but fallback
        return [0] * 28 + [1, -1]
    
    return best_coeffs

# EVOLVE-BLOCK-END
