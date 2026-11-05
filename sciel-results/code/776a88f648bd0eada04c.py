import numpy as np
import time
import random

def _fast_sup_norm(coeffs, grid_size=2048):
    """
    Fast sup-norm evaluation.
    Returns an approximation of max |P(z)|.
    This is used for search.
    """
    n = len(coeffs)
    k = np.arange(n)
    theta = 2 * np.pi * np.arange(grid_size) / grid_size
    # Compute P(z_j) for all j
    # P(z_j) = sum_k coeffs[k] * exp(i*k*theta_j)
    # We can do this as coeffs @ exp(1j * outer(k, theta))
    # But constructing the outer product is expensive.
    # Instead, we can compute iteratively or use FFT? 
    # For n=24, coeffs @ matrix is fast enough if matrix is precomputed? 
    # No, theta is fixed, so we can precompute the matrix of exponentials!
    pass

# Precompute the evaluation matrix for a fixed grid
# Since the grid is fixed, we can precompute exp(i*k*theta_j)
_GRID_SIZE = 2048
_k = np.arange(24)
_theta = 2 * np.pi * np.arange(_GRID_SIZE) / _GRID_SIZE
# M[j, k] = exp(i * k * theta_j)
# P(z_j) = sum_k coeffs[k] * M[j, k]
# So P = M @ coeffs
_M = np.exp(1j * np.outer(_theta, _k)) # Shape (2048, 24)

def _eval_fast(coeffs):
    """
    Very fast evaluation using precomputed matrix.
    coeffs: array of 24 values
    Returns: max |P(z)| on the grid
    """
    P = _M @ coeffs
    return np.max(np.abs(P))

def _eval_refined(coeffs, top_k=5, iterations=40):
    """
    Refine the peaks found in the fast evaluation.
    """
    n = len(coeffs)
    k = np.arange(n)
    
    # Get peaks from fast eval
    P_grid = _M @ coeffs
    mags = np.abs(P_grid)
    top_idx = np.argsort(mags)[-top_k:]
    
    max_val = 0.0
    for i in top_idx:
        theta0 = _theta[i]
        a = theta0 - 2 * np.pi / _GRID_SIZE
        b = theta0 + 2 * np.pi / _GRID_SIZE
        
        def f(t):
            z = np.exp(1j * t)
            powers = z ** k
            val = np.dot(coeffs, powers)
            return abs(val)
        
        gr = (np.sqrt(5) - 1) / 2
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        fc = f(c)
        fd = f(d)
        
        for _ in range(iterations):
            if fc < fd:
                b = d
                d = c
                fd = fc
                c = b - gr * (b - a)
                fc = f(c)
            else:
                a = c
                c = d
                fc = fd
                d = a + gr * (b - a)
                fd = f(d)
        
        peak = max(fc, fd)
        if peak > max_val:
            max_val = peak
            
    return max_val

def _simulated_annealing_fast(initial_coeffs, time_budget=10.0):
    """
    Simulated annealing using fast evaluation.
    """
    n = len(initial_coeffs)
    current = initial_coeffs.copy()
    current_val = _eval_fast(current)
    
    best = current.copy()
    best_val = _eval_refined(current) # Ensure best is accurate
    
    start_time = time.time()
    
    T = 0.5
    alpha = 0.9999 # Slower cooling
    T_end = 0.01
    
    while time.time() - start_time < time_budget:
        # Propose move
        idx = random.randint(0, n - 1)
        new_coeffs = current.copy()
        new_coeffs[idx] *= -1
        
        new_val_fast = _eval_fast(new_coeffs)
        delta_fast = new_val_fast - current_val
        
        # Acceptance criterion based on fast val
        # If new_val_fast is significantly worse, reject immediately
        if delta_fast > 0.1:
            T *= alpha
            continue
            
        if delta_fast < 0 or random.random() < np.exp(-delta_fast / max(T, 1e-10)):
            current = new_coeffs
            current_val = new_val_fast
            
            # If fast val suggests improvement, check with refined eval
            if current_val < best_val * 0.99: # Heuristic: if fast val is much lower
                refined_val = _eval_refined(current)
                if refined_val < best_val:
                    best = current.copy()
                    best_val = refined_val
            elif current_val < best_val:
                # Periodically check if we are close to best
                if random.random() < 0.05:
                    refined_val = _eval_refined(current)
                    if refined_val < best_val:
                        best = current.copy()
                        best_val = refined_val
        
        T *= alpha
        if T < T_end:
            T = T_end
            
    return best, best_val

def construct():
    n = 24
    best_coeffs = None
    best_score = float('inf')
    
    start_time = time.time()
    time_limit = 115.0
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. All +1
    candidates.append(np.ones(n))
    
    # 2. Alternating
    candidates.append(np.array([1 if i % 2 == 0 else -1 for i in range(n)]))
    
    # 3. GRS-like sequences
    p, q = [1], [1]
    while len(p) < n + 10:
        p_new = p + q
        q_new = p + [-c for c in q]
        p, q = p_new, q_new
    candidates.append(np.array(p[:n], dtype=float))
    candidates.append(np.array(p[:n][::-1], dtype=float))
    
    # 4. Random candidates
    random.seed(12345)
    for _ in range(10):
        candidates.append(np.array([1 if random.random() < 0.5 else -1 for _ in range(n)], dtype=float))
    
    # 5. Candidates with specific structures
    c1 = np.ones(n)
    c1[n//2:] = -1
    candidates.append(c1)
    
    c2 = np.ones(n)
    c2[n//4:n//2] = -1
    c2[3*n//4:] = -1
    candidates.append(c2)
    
    # Run simulated annealing on each candidate
    for i, cand in enumerate(candidates):
        if time.time() - start_time > time_limit:
            break
        
        remaining = time_limit - (time.time() - start_time)
        n_remaining = len(candidates) - i
        time_per = max(remaining / n_remaining, 5.0)
        
        res_coeffs, res_score = _simulated_annealing_fast(cand.copy(), time_budget=time_per)
        
        if res_score < best_score:
            best_score = res_score
            best_coeffs = res_coeffs
    
    if best_coeffs is None:
        best_coeffs = np.ones(n)
    
    # Final refinement on the best candidate to ensure accuracy
    # The search score uses a raw value. The task says "score is max... divided by sqrt(24)".
    # The evaluator computes it with high accuracy.
    # We should return the coefficients that minimize the true sup norm.
    # Our best_score is from _eval_refined which is quite accurate (40 iter golden section).
    # Let's do one more high-precision check on the best candidate.
    
    final_val = _eval_refined(best_coeffs, top_k=10, iterations=60)
    
    return [int(c) for c in best_coeffs]
