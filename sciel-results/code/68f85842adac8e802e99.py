"""
Combined strategy:
1. Structural Initialization: Use Sidon sets to initialize the search. Sidon sets
   have distinct pairwise sums, which minimizes the peak of the autoconvolution
   relative to the L2 norm, providing a strong starting point for the ratio.
2. Smooth Optimization: Use L-BFGS-B with a log-parameterization (h = exp(x))
   and a smooth surrogate (log-sum-exp) for the supremum norm of the
   autoconvolution. This allows gradient-based optimization of the non-smooth
   max operator.
3. High-Precision Polish: Final coordinate descent to refine the discrete peaks.
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 50

def find_sidon_set(n, max_size=10):
    """
    Find a Sidon set of size up to max_size in {0, ..., n-1}.
    A Sidon set has distinct pairwise sums.
    """
    best_set = []
    
    def is_sidon(candidate):
        sums = set()
        for i in range(len(candidate)):
            for j in range(i, len(candidate)):
                s = candidate[i] + candidate[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    # Greedy approach with random restarts to find larger sets
    rng = np.random.RandomState(42)
    for _ in range(100):
        current = [0]
        sums = {0}
        # Try to add elements in random order or increasing order
        # Increasing order is usually denser
        for x in range(1, n):
            if len(current) >= max_size:
                break
            # Check if x can be added
            valid = True
            new_sums = set()
            for s in current:
                ns = x + s
                if ns in sums or ns in new_sums:
                    valid = False
                    break
                new_sums.add(ns)
            two_x = 2 * x
            if two_x in sums or two_x in new_sums:
                valid = False
            
            if valid:
                current.append(x)
                sums.update(new_sums)
                sums.add(two_x)
        
        if len(current) > len(best_set):
            best_set = current
            if len(best_set) >= max_size:
                break
                
    return best_set

def autoconv(h):
    return np.convolve(h, h, mode='full')

def score_fn(h):
    """Return the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf). Higher is better."""
    h = np.maximum(h, 0.0)
    norm2_sq = np.sum(h * h)
    if norm2_sq < 1e-30:
        return 0.0
    norm1 = np.sum(h)
    if norm1 < 1e-30:
        return 0.0
    ac = autoconv(h)
    ac_max = np.max(np.abs(ac))
    if ac_max < 1e-30:
        return 0.0
    return norm2_sq / (norm1 * ac_max)

def smooth_max(x, beta=100.0):
    """Smooth approximation of max(x) using log-sum-exp."""
    x_max = np.max(x)
    if x_max < 1e-30:
        return 1e-30
    shifted = x - x_max
    # Avoid overflow in exp
    log_sum_exp = x_max + np.log(np.sum(np.exp(beta * shifted))) / beta
    return log_sum_exp

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_h = None
    best_score = 0.0
    
    # Phase 1: Generate Sidon set initializations
    sidon_sets = []
    # Try to find sets of different sizes
    for size in [7, 8, 9, 10]:
        s = find_sidon_set(N, max_size=size)
        if len(s) > 0:
            sidon_sets.append(s)
    
    # If we didn't find good ones, use a simple sparse set
    if not sidon_sets:
        sidon_sets.append(list(range(0, N, 5)))
        
    # Add a uniform initialization as a fallback
    uniform_h = np.ones(N) / N
    sidon_h_list = []
    for s in sidon_sets:
        h = np.zeros(N)
        for idx in s:
            h[idx] = 1.0
        h = h / np.sum(h) # Normalize L1 to 1
        sidon_h_list.append(h)
    sidon_h_list.append(uniform_h)
    
    # Phase 2: Optimize each initialization using L-BFGS-B on log-h
    # Objective: minimize -score(h) where h = exp(x)
    # Note: score is scale invariant, so we don't need to constrain L1=1 strictly,
    # but it helps conditioning. We will just let L-BFGS find the direction.
    
    def neg_smooth_score(log_h, beta=50.0):
        h = np.exp(log_h)
        # Clip to prevent overflow/underflow issues in subsequent steps
        h = np.clip(h, 1e-10, 1e10)
        
        norm2_sq = np.sum(h * h)
        if norm2_sq < 1e-30:
            return 1e10
        norm1 = np.sum(h)
        if norm1 < 1e-30:
            return 1e10
        
        ac = autoconv(h)
        ac_max_smooth = smooth_max(ac, beta=beta)
        if ac_max_smooth < 1e-30:
            return 1e10
        
        return -(norm2_sq / (norm1 * ac_max_smooth))
    
    for h0 in sidon_h_list:
        if time.time() > deadline - 20:
            break
            
        h0 = np.maximum(h0, 1e-10)
        log_h0 = np.log(h0)
        
        # Run L-BFGS-B
        try:
            res = minimize(
                neg_smooth_score,
                log_h0,
                args=(50.0,),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 2000}
            )
            
            h_opt = np.exp(res.x)
            h_opt = np.maximum(h_opt, 0.0)
            # Normalize to L1=1 for consistent comparison, though score is invariant
            l1 = np.sum(h_opt)
            if l1 > 1e-10:
                h_opt = h_opt / l1
            
            s = score_fn(h_opt)
            if s > best_score:
                best_score = s
                best_h = h_opt.copy()
                
        except Exception:
            pass
            
    # Phase 3: Fine polish with coordinate descent on the best solution
    if best_h is not None and time.time() < deadline - 5:
        h_curr = best_h.copy()
        curr_score = score_fn(h_curr)
        
        # Adaptive step sizes
        for step in [0.01, 0.005, 0.001, 0.0005, 0.0001]:
            if time.time() > deadline - 2:
                break
            improved = True
            while improved and time.time() < deadline - 2:
                improved = False
                for i in range(N):
                    # Try increase
                    h_trial = h_curr.copy()
                    h_trial[i] += step
                    h_trial = np.maximum(h_trial, 0.0)
                    l1 = np.sum(h_trial)
                    if l1 > 0:
                        h_trial = h_trial / l1
                    s = score_fn(h_trial)
                    if s > curr_score + 1e-12:
                        h_curr = h_trial
                        curr_score = s
                        improved = True
                        continue
                        
                    # Try decrease
                    h_trial = h_curr.copy()
                    h_trial[i] -= step
                    h_trial = np.maximum(h_trial, 0.0)
                    l1 = np.sum(h_trial)
                    if l1 > 0:
                        h_trial = h_trial / l1
                    s = score_fn(h_trial)
                    if s > curr_score + 1e-12:
                        h_curr = h_trial
                        curr_score = s
                        improved = True
        
        if curr_score > best_score:
            best_score = curr_score
            best_h = h_curr.copy()

    if best_h is None:
        return [1.0/N] * N
        
    return [float(x) for x in best_h]

# EVOLVE-BLOCK-END
