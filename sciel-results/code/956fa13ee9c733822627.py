import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin

    def count_nonzero(poly):
        return int(np.count_nonzero(poly))

    def evaluate(coeffs):
        # coeffs is a list of 12 integers
        sq = np.convolve(coeffs, coeffs)
        return int(np.count_nonzero(sq))

    def validate(coeffs):
        if len(coeffs) != 12:
            return False
        if coeffs[-1] == 0:
            return False
        if coeffs[0] == 0:
            return False
        if not all(isinstance(c, int) for c in coeffs):
            return False
        if not all(-64 <= c <= 64 for c in coeffs):
            return False
        if sum(1 for c in coeffs if c != 0) != 12:
            return False
        return True

    best_score = float('inf')
    best_poly = None

    random.seed(42)

    # We assume the optimal structure has consecutive exponents 0..11
    # So we search for coefficients c[0..11] in [-64, 64] \ {0}
    
    def run_search(num_restarts=50, iters_per_restart=5000):
        nonlocal best_score, best_poly
        
        for r in range(num_restarts):
            if time.time() > deadline - 5:
                break
            
            # Initialize with random +/- 1
            curr = [random.choice([-1, 1]) for _ in range(12)]
            curr_score = evaluate(curr)
            
            if curr_score < best_score:
                best_score = curr_score
                best_poly = curr.copy()
                
            # Simulated Annealing
            T = 5.0
            T_min = 0.1
            cooling_rate = 0.999
            
            for i in range(iters_per_restart):
                if time.time() > deadline - 2:
                    return
                
                # Pick a random index to modify
                idx = random.randint(0, 11)
                old_val = curr[idx]
                
                # Choose a new value
                # Bias towards small integers for better cancellation potential
                # But allow larger range
                if random.random() < 0.7:
                    new_val = random.choice([-3, -2, -1, 1, 2, 3])
                else:
                    new_val = random.randint(-64, 64)
                
                if new_val == 0:
                    new_val = 1 # Must be non-zero
                
                if new_val == old_val:
                    continue
                    
                curr[idx] = new_val
                new_score = evaluate(curr)
                
                delta = new_score - curr_score
                
                # Acceptance criterion
                if delta < 0 or random.random() < np.exp(-delta / T):
                    curr_score = new_score
                    if curr_score < best_score:
                        best_score = curr_score
                        best_poly = curr.copy()
                        # Optional: restart cooling if we found a new best
                        # T = max(T_min, T * 0.9) 
                else:
                    curr[idx] = old_val
                    
                T *= cooling_rate
                if T < T_min:
                    T = T_min

    # Run search
    run_search(num_restarts=20, iters_per_restart=20000)
    
    # If time remains, do a more focused local search on the best found
    if best_poly is not None and time.time() < deadline - 5:
        curr = best_poly.copy()
        curr_score = best_score
        
        # Hill climbing with random restarts within the basin
        no_improve = 0
        max_no_improve = 5000
        
        while time.time() < deadline - 2 and no_improve < max_no_improve:
            idx = random.randint(0, 11)
            old_val = curr[idx]
            
            # Try a specific range of values around the current one or random
            candidates = set()
            candidates.add(old_val + 1)
            candidates.add(old_val - 1)
            candidates.add(old_val * -1)
            candidates.add(random.randint(-64, 64))
            
            improved = False
            for new_val in candidates:
                if new_val == 0:
                    continue
                if not (-64 <= new_val <= 64):
                    continue
                if new_val == old_val:
                    continue
                    
                curr[idx] = new_val
                new_score = evaluate(curr)
                
                if new_score < curr_score:
                    curr_score = new_score
                    improved = True
                    if curr_score < best_score:
                        best_score = curr_score
                        best_poly = curr.copy()
                        no_improve = 0
                    break
            if not improved:
                curr[idx] = old_val
                no_improve += 1
            else:
                no_improve = 0

    if best_poly is None:
        # Fallback
        best_poly = [1] * 12

    # Ensure validation
    if not validate(best_poly):
        best_poly = [1] * 12

    return best_poly
