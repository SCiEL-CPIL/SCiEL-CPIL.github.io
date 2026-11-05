import time
import random
from math import gcd
from functools import reduce

def construct():
    deadline = time.time() + 85  # 90s limit, keep 5s margin
    
    # 1. Precompute weights
    # L = lcm(1, ..., 30)
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, range(1, 31))
    weights = [L // i for i in range(1, 31)] # weights[i-1] corresponds to 1/i
    
    # 2. Simulated Annealing
    # State: coeffs (list of 30 ints in {-1, 0, 1})
    # Value: N = sum(coeffs[i] * weights[i])
    
    best_nonzero_abs = None
    best_nonzero_coeffs = None
    
    def calculate_N(coeffs):
        return sum(c * w for c, w in zip(coeffs, weights))
    
    def update_best(coeffs, N):
        nonlocal best_nonzero_abs, best_nonzero_coeffs
        if N == 0:
            return
        abs_N = abs(N)
        if best_nonzero_abs is None or abs_N < best_nonzero_abs:
            best_nonzero_abs = abs_N
            best_nonzero_coeffs = list(coeffs)
            
    # Helper to get a random valid coefficient different from current
    def random_diff_val(current):
        options = [-1, 0, 1]
        options.remove(current)
        return random.choice(options)

    # Run multiple SA instances
    num_instances = 20
    random.seed(42)
    
    for inst in range(num_instances):
        if time.time() > deadline:
            break
            
        # Initialize
        # Strategy: Start with random signs, biased towards cancellation
        # e.g., c_i = 1 if i < 15 else -1, with some noise
        coeffs = [1 if i < 15 else -1 for i in range(30)]
        for i in range(30):
            if random.random() < 0.1:
                coeffs[i] = random.choice([-1, 0, 1])
        
        N = calculate_N(coeffs)
        update_best(coeffs, N)
        
        # Annealing parameters
        # We need to cool down enough to settle into a local minimum
        # Let's use a time-based cooling or step-based
        # With 90s and 20 instances, we have ~4.5s per instance.
        # Let's do a fixed number of steps per instance, say 50,000 steps.
        # Or just run until time slice is up.
        
        T = 1000.0 # Initial temperature (scale with max weight? max weight is L, which is huge. 
                   # Actually, delta N can be up to L. So T should be large.)
                   # Let's scale T based on the magnitude of weights.
                   # Max change in N is max(weights) ~ L.
                   # Let's set T = 1.0 * max(weights) initially? No, that's too high.
                   # Let's just use a geometric cooling schedule.
        
        # Better: Normalize weights? No, we need exact integer arithmetic.
        # Let's set T_start = max(weights) / 10.0
        T_start = max(weights) / 10.0
        T_end = 1e-6
        alpha = 0.99 # Cooling factor per step? 
        # If we run 10000 steps, T_end = T_start * (0.99)^10000 ~ 0. 
        # This is fine.
        
        steps = 20000
        for step in range(steps):
            if time.time() > deadline:
                break
                
            i = random.randint(0, 29)
            old_val = coeffs[i]
            new_val = random_diff_val(old_val)
            
            # Calculate delta N
            delta_N = (new_val - old_val) * weights[i]
            new_N = N + delta_N
            
            # Acceptance
            # If new_N is 0, we might want to reject it if we are looking for non-zero?
            # No, minimizing |N| leads to 0. If we hit 0, we should move away to the nearest non-zero.
            # But standard SA minimizes |N|. 
            # If we accept 0, N becomes 0. Next move will be from 0.
            # The cost is |N|. 
            # If current N != 0 and new N == 0, cost decreases (from |N| to 0). Accept.
            # If current N == 0 and new N != 0, cost increases (from 0 to |N|). 
            # We should accept with prob exp(-|N|/T).
            
            curr_cost = abs(N)
            new_cost = abs(new_N)
            delta_cost = new_cost - curr_cost
            
            if delta_cost < 0:
                # Improvement
                coeffs[i] = new_val
                N = new_N
                update_best(coeffs, N)
            else:
                # Worse or equal
                # If new_N is 0, cost is 0. If curr_cost > 0, delta_cost < 0 handled above.
                # If curr_cost == 0 and new_cost > 0, delta_cost > 0.
                # Probability of acceptance
                if T > 1e-9:
                    # Avoid overflow/underflow in exp
                    # If delta_cost is huge, exp is 0
                    if delta_cost / T < 50: 
                        p = __import__('math').exp(-delta_cost / T)
                        if random.random() < p:
                            coeffs[i] = new_val
                            N = new_N
                            update_best(coeffs, N)
            
            # Cool down
            T *= 0.998 # Slower cooling to explore more
            
            # Optional: If we hit 0, and T is low, force a move to non-zero?
            # No, let the annealing handle it. The "best_nonzero" tracker will keep the best non-zero seen.

    if best_nonzero_coeffs is None:
        # Fallback
        return [0] * 28 + [1, -1]
        
    return best_nonzero_coeffs
