import time
import numpy as np
from math import gcd
from functools import reduce

def construct():
    """
    Find 30 coefficients in {-1, 0, 1} such that sum(c[i]/(i+1)) is nonzero
    and has minimal absolute value.
    """
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 30
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, range(1, n + 1))
    weights = np.array([L // (i + 1) for i in range(n)], dtype=np.int64)
    
    best_abs_S = np.iinfo(np.int64).max
    best_coeffs = None
    rng = np.random.default_rng()
    
    # Strategy: Multiple restarts of Local Search
    # Each restart uses a different initialization strategy
    
    for restart in range(1000):
        if time.time() > deadline:
            break
            
        # Initialization
        if restart % 3 == 0:
            # Random
            c = rng.choice([-1, 0, 1], size=n)
            if np.all(c == 0): c[0] = 1
        elif restart % 3 == 1:
            # Greedy: Cancel largest weights
            c = np.zeros(n, dtype=np.int64)
            current_S = 0
            # Sort indices by weight magnitude descending
            order = np.argsort(weights)[::-1]
            for i in order:
                # Choose sign to minimize |current_S + c[i]*w[i]|
                # c[i] can be -1, 0, 1
                # We want to pick c[i] such that |current_S + c[i]*w[i]| is minimized
                # But we must ensure final sum != 0.
                # For now, just minimize.
                w_i = weights[i]
                # Options: -w_i, 0, w_i
                # New sums: current_S - w_i, current_S, current_S + w_i
                # Pick the one with smallest abs
                candidates = [current_S - w_i, current_S, current_S + w_i]
                abs_candidates = [abs(x) for x in candidates]
                min_idx = np.argmin(abs_candidates)
                if min_idx == 0:
                    c[i] = -1
                elif min_idx == 2:
                    c[i] = 1
                else:
                    c[i] = 0
                current_S = candidates[min_idx]
        else:
            # Random with bias towards cancellation
            c = np.zeros(n, dtype=np.int64)
            for i in range(n):
                r = rng.random()
                if r < 0.33:
                    c[i] = -1
                elif r < 0.66:
                    c[i] = 0
                else:
                    c[i] = 1
            if np.all(c == 0): c[0] = 1

        current_S = np.dot(c, weights)
        
        if current_S == 0:
            # Perturb
            c[rng.integers(n)] = 1
            current_S = np.dot(c, weights)
            
        # Local Search Refinement
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Shuffle order of indices to avoid bias?
            # For N=30, just iterate 0..29
            for i in range(n):
                if time.time() > deadline:
                    break
                old_val = c[i]
                for new_val in [-1, 0, 1]:
                    if new_val == old_val:
                        continue
                    delta = (new_val - old_val) * weights[i]
                    new_S = current_S + delta
                    
                    if new_S == 0:
                        continue
                        
                    if abs(new_S) < abs(current_S):
                        c[i] = new_val
                        current_S = new_S
                        improved = True
                        break
                
        if abs(current_S) < best_abs_S:
            best_abs_S = abs(current_S)
            best_coeffs = c.copy()
            
        if best_abs_S == 1:
            break

    if best_coeffs is None:
        best_coeffs = np.zeros(n, dtype=np.int64)
        best_coeffs[0] = 1

    return [int(c) for c in best_coeffs]
