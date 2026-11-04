"""Second autocorrelation inequality: maximize squared L2 norm / (L1 norm * sup norm of autoconvolution)."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _compute_ratio(heights):
    """Compute the objective: (L2 norm)^2 / (L1 norm * sup norm of autoconvolution).
    
    heights: array of N nonnegative values
    Returns the ratio value.
    """
    h = np.asarray(heights, dtype=np.float64)
    if np.any(h < -1e-12):
        return 0.0
    h = np.maximum(h, 0.0)
    
    l1 = np.sum(h)
    if l1 <= 1e-15:
        return 0.0
    
    l2_sq = np.sum(h * h)
    
    # Autoconvolution
    # Using FFT for speed if N is large, but N=50 is small. np.convolve is fine.
    autoconv = np.convolve(h, h, mode='full')
    sup_norm = np.max(autoconv)
    
    if sup_norm <= 1e-15:
        return 0.0
    
    return l2_sq / (l1 * sup_norm)

def _get_sidon_set(n, max_val):
    """Generate a Sidon set of size n within [0, max_val-1].
    A Sidon set has the property that all pairwise sums are distinct.
    This helps minimize the peak of the autoconvolution.
    """
    S = [0]
    sums = set([0])
    for i in range(1, max_val):
        if len(S) >= n:
            break
        # Check if i can be added
        # New sums will be i+s for s in S, and 2*i
        new_sums = set()
        valid = True
        for s in S:
            new_sum = i + s
            if new_sum in sums or new_sum in new_sums:
                valid = False
                break
            new_sums.add(new_sum)
        two_i = 2 * i
        if two_i in sums or two_i in new_sums:
            valid = False
        
        if valid:
            S.append(i)
            sums.update(new_sums)
            sums.add(two_i)
    return S

def _optimize_sparse_amplitudes(indices, deadline):
    """Optimize the amplitudes of a sparse set of indices."""
    if not indices:
        return np.zeros(N), 0.0
        
    k = len(indices)
    # We optimize the amplitudes. 
    # The objective is scale invariant, so we can fix sum(amplitudes) = 1.
    # Let x be the amplitudes. We want to maximize f(x) = sum(x^2) / max(autoconv(x))
    # subject to x >= 0 and sum(x) = 1.
    
    # Initial guess: uniform amplitudes
    x0 = np.ones(k) / k
    
    def objective(x):
        # x are the amplitudes
        h = np.zeros(N)
        h[indices] = x
        # Ensure non-negative (should be handled by bounds, but safe)
        x_pos = np.maximum(x, 0)
        h[indices] = x_pos
        return -_compute_ratio(h)
    
    # Bounds: amplitudes must be non-negative
    bounds = [(0, 1)] * k
    
    # We can't easily enforce sum(x)=1 with standard bounds in L-BFGS-B without reparameterization.
    # Instead, we can use a reparameterization: x_i = exp(y_i) / sum(exp(y_j)).
    # Or simply optimize x with penalty or just let it find the shape, then normalize.
    # Since the function is homogeneous of degree 0, we can just optimize x in [0, 1]^k
    # and normalize at the end? No, the ratio depends on the relative weights.
    # Let's use the softmax reparameterization for smooth optimization.
    
    def softmax_objective(y):
        # y are log-amplitudes
        # Subtract max for numerical stability
        y_max = np.max(y)
        exp_y = np.exp(y - y_max)
        x = exp_y / np.sum(exp_y)
        h = np.zeros(N)
        h[indices] = x
        return -_compute_ratio(h)
    
    # Initial y: log of uniform
    y0 = np.zeros(k)
    
    best_score = 0.0
    best_x = x0
    
    # Try a few random initializations for y
    rng = np.random.RandomState(42)
    for _ in range(5):
        y_init = rng.randn(k) * 0.5
        try:
            res = minimize(softmax_objective, y_init, method='L-BFGS-B', 
                           options={'maxiter': 100, 'ftol': 1e-12})
            if res.fun < -best_score: # res.fun is negative score
                best_score = -res.fun
                # Recover x from y
                y_opt = res.x
                y_max = np.max(y_opt)
                exp_y = np.exp(y_opt - y_max)
                best_x = exp_y / np.sum(exp_y)
        except Exception:
            pass
        
        if time.time() > deadline:
            break
            
    # Also try Nelder-Mead on the simplex directly (x >= 0, sum x = 1)
    # This is robust for small k
    def simplex_objective(x):
        # x might violate sum=1 slightly, normalize
        s = np.sum(x)
        if s < 1e-15:
            return 0.0
        x_norm = x / s
        h = np.zeros(N)
        h[indices] = x_norm
        return -_compute_ratio(h)
    
    # Start from best found so far or uniform
    x_start = best_x.copy()
    try:
        res_nm = minimize(simplex_objective, x_start, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
        if res_nm.fun < -best_score:
            best_score = -res_nm.fun
            x_final = np.maximum(res_nm.x, 0)
            s = np.sum(x_final)
            if s > 0:
                best_x = x_final / s
    except Exception:
        pass

    h_final = np.zeros(N)
    h_final[indices] = best_x
    # Normalize to L1=1 for consistency, though ratio is invariant
    s = np.sum(h_final)
    if s > 0:
        h_final = h_final / s
    
    return h_final, best_score

def construct():
    """Construct 50 nonnegative heights to maximize the ratio.
    
    Returns a list of N nonnegative step heights.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_h = np.ones(N)
    best_score = _compute_ratio(best_h)
    
    # Strategy:
    # 1. Generate candidate sparse index sets (Sidon sets, random sparse)
    # 2. Optimize amplitudes for each set
    # 3. Keep the best
    
    # Candidate sets of indices
    candidate_indices_sets = []
    
    # 1. Greedy Sidon sets of various sizes
    # Theoretical max ratio is roughly K for K non-overlapping peaks.
    # For N=50, we can have K up to ~10-12.
    for k in range(3, 13):
        indices = _get_sidon_set(k, N)
        if len(indices) == k:
            candidate_indices_sets.append(indices)
            
    # 2. Random sparse sets
    rng = np.random.RandomState(123)
    for _ in range(20):
        k = rng.randint(3, 11)
        indices = rng.choice(N, k, replace=False)
        candidate_indices_sets.append(indices)
        
    # 3. Structured sets: e.g., primes, squares (might have collisions, but worth trying)
    # Squares: 0, 1, 4, 9, 16, 25, 36, 49
    squares = [i for i in range(N) if int(np.sqrt(i))**2 == i]
    if len(squares) >= 3:
        candidate_indices_sets.append(squares)
        
    # Primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    primes_in_range = [p for p in primes if p < N]
    if len(primes_in_range) >= 3:
        candidate_indices_sets.append(primes_in_range)

    # Deduplicate index sets (sorted tuples)
    unique_sets = set()
    for s in candidate_indices_sets:
        unique_sets.add(tuple(sorted(s)))
    
    for indices in unique_sets:
        if time.time() > deadline - 10.0:
            break
            
        h_opt, score_opt = _optimize_sparse_amplitudes(list(indices), deadline)
        
        if score_opt > best_score:
            best_score = score_opt
            best_h = h_opt
    
    # Final check: ensure non-negative and return
    best_h = np.maximum(best_h, 0.0)
    
    if np.sum(best_h) <= 1e-15:
        best_h = np.ones(N)
        
    # The evaluator measures the raw value. 
    # The ratio is scale invariant. 
    # However, sometimes "sup norm of autoconvolution" might be sensitive to absolute scale if there's numerical noise, 
    # but mathematically it's invariant. 
    # We return the heights as they are (L1 normalized in _optimize_sparse_amplitudes).
    
    return [float(x) for x in best_h]
