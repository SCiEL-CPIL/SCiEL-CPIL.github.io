import time
import random
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    n = 30
    # Calculate LCM of 1..30
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, range(1, n + 1))
    
    # Precompute weights: w[i] = L / (i+1) for i in 0..n-1
    # The sum is sum(c[i] / (i+1)) = (1/L) * sum(c[i] * (L/(i+1)))
    # Let W[i] = L // (i+1)
    W = [L // (i + 1) for i in range(n)]
    
    best_N = None
    best_coeffs = None
    best_abs_N = float('inf')
    
    def calc_N(coeffs):
        s = 0
        for i in range(n):
            s += coeffs[i] * W[i]
        return s

    # Random restarts with local search
    while time.time() < deadline:
        # Random initialization
        current_coeffs = [random.choice([-1, 0, 1]) for _ in range(n)]
        current_N = calc_N(current_coeffs)
        
        if current_N == 0:
            # Perturb to make it non-zero
            idx = random.randint(0, n - 1)
            current_coeffs[idx] = 1
            current_N += W[idx]
            
        # Local Search (Hill Climbing)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Shuffle indices to avoid bias
            indices = list(range(n))
            random.shuffle(indices)
            
            for i in indices:
                old_val = current_coeffs[i]
                # Try other values
                # We want to minimize abs(current_N + delta)
                # delta = (new_val - old_val) * W[i]
                # Possible deltas:
                # If old_val is -1: new can be 0 (delta=W), 1 (delta=2W)
                # If old_val is 0: new can be -1 (delta=-W), 1 (delta=W)
                # If old_val is 1: new can be -1 (delta=-2W), 0 (delta=-W)
                
                best_delta = None
                best_new_val = None
                best_new_N = None
                current_abs = abs(current_N)
                
                for new_val in [-1, 0, 1]:
                    if new_val == old_val:
                        continue
                    delta = (new_val - old_val) * W[i]
                    new_N = current_N + delta
                    if new_N == 0:
                        continue
                    new_abs = abs(new_N)
                    if new_abs < current_abs:
                        current_abs = new_abs
                        best_delta = delta
                        best_new_val = new_val
                        best_new_N = new_N
                        
                if best_new_val is not None:
                    current_coeffs[i] = best_new_val
                    current_N = best_new_N
                    improved = True
            
            if not improved:
                # Escape local minimum: random perturbation
                # Flip 1-3 random coefficients
                for _ in range(3):
                    idx = random.randint(0, n - 1)
                    old = current_coeffs[idx]
                    new = random.choice([-1, 0, 1])
                    if new != old:
                        delta = (new - old) * W[idx]
                        new_N = current_N + delta
                        if new_N != 0:
                            current_coeffs[idx] = new
                            current_N = new_N
                        else:
                            # If it became 0, try another value
                            new2 = random.choice([-1, 0, 1])
                            if new2 != old:
                                delta2 = (new2 - old) * W[idx]
                                new_N2 = current_N + delta2
                                if new_N2 != 0:
                                    current_coeffs[idx] = new2
                                    current_N = new_N2
                                else:
                                    # Revert if stuck at 0
                                    pass
                    # Check if we are still valid (non-zero)
                    if current_N == 0:
                        # Force fix
                        for k in range(n):
                            old_k = current_coeffs[k]
                            for v in [-1, 0, 1]:
                                if v == old_k: continue
                                d = (v - old_k) * W[k]
                                ns = current_N + d
                                if ns != 0:
                                    current_coeffs[k] = v
                                    current_N = ns
                                    break
                            if current_N != 0: break

        # Update best
        if abs(current_N) < best_abs_N:
            best_abs_N = abs(current_N)
            best_coeffs = current_coeffs[:]
            best_N = current_N

    # If no valid non-zero sum was found (extremely unlikely), return a default
    if best_coeffs is None:
        best_coeffs = [1] + [0] * (n - 1)
        
    return best_coeffs
