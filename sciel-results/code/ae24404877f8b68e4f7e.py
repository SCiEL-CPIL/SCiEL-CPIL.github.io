import numpy as np
from math import gcd
from functools import reduce
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    n = 30
    L = 2329089562800  # LCM of 1..30
    weights = [L // i for i in range(1, n + 1)]
    
    def eval_abs_int(coeffs):
        """Return |sum c_i * w_i| where w_i = L/(i+1)."""
        s = 0
        for i, c in enumerate(coeffs):
            if c != 0:
                s += c * weights[i]
        return abs(s)
    
    def local_search(initial_coeffs, time_limit):
        """Greedy local search: flip one coefficient at a time to minimize |S|, S != 0."""
        best_coeffs = list(initial_coeffs)
        best_val = eval_abs_int(best_coeffs)
        
        # If best_val is 0, we need to find a nonzero solution
        if best_val == 0:
            for i in range(n):
                for v in [-1, 0, 1]:
                    if v == best_coeffs[i]:
                        continue
                    trial = list(best_coeffs)
                    trial[i] = v
                    val = eval_abs_int(trial)
                    if val > 0 and val < best_val:
                        best_val = val
                        best_coeffs = trial
                        break
                if best_val > 0:
                    break
        
        while time.time() < time_limit:
            improved = False
            for i in range(n):
                for v in [-1, 0, 1]:
                    if v == best_coeffs[i]:
                        continue
                    trial = list(best_coeffs)
                    trial[i] = v
                    val = eval_abs_int(trial)
                    if val > 0 and val < best_val:
                        best_val = val
                        best_coeffs = trial
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        
        return best_coeffs, best_val
    
    best_coeffs_overall = None
    best_val_overall = float('inf')
    
    # Random restarts with local search
    random.seed(42)
    
    # Try a few structured initializations first
    for i in range(n):
        for j in range(i + 1, n):
            coeffs = [0] * n
            coeffs[i] = 1
            coeffs[j] = -1
            val = eval_abs_int(coeffs)
            if val > 0 and val < best_val_overall:
                best_val_overall = val
                best_coeffs_overall = coeffs
    
    # Random restarts
    for r in range(100):
        if time.time() > deadline:
            break
        
        coeffs = [random.choice([-1, 0, 1]) for _ in range(n)]
        if all(c == 0 for c in coeffs):
            coeffs[0] = 1
        
        result_coeffs, result_val = local_search(coeffs, deadline)
        
        if result_val > 0 and result_val < best_val_overall:
            best_val_overall = result_val
            best_coeffs_overall = result_coeffs
    
    if best_coeffs_overall is None:
        return [0] * 28 + [1, -1]
    
    return best_coeffs_overall

# EVOLVE-BLOCK-END
