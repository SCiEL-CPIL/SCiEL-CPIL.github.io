"""
Second autocorrelation inequality: choose 50 nonnegative heights on equal intervals of [-1/4, 1/4]
to maximize (sum h_i^2) / ((sum h_i) * sup_k C[k]) where C = h autoconvolution.

Strategy: 
1. Use sparse supports (few peaks) to minimize autoconvolution collisions
2. Optimize weights using gradient-based methods with proper handling of the max operator
3. Try various structured and random supports
4. Use a smooth approximation of max for gradient-based optimization
"""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
import sys

N = 50
DELTA = 1.0 / N  # = 0.02

def _compute_conv(h):
    """Compute autoconvolution of h."""
    return np.convolve(h, h, mode='full')

def _score_raw(h):
    """Compute the raw score (higher is better)."""
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 <= 0:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = _compute_conv(h)
    max_conv = np.max(h_conv)
    if max_conv <= 0:
        return 0.0
    return l2_sq / (DELTA * l1 * max_conv)

def _objective(h):
    """Objective function to minimize (negative of score)."""
    h = np.asarray(h, dtype=np.float64)
    if np.any(h < -1e-10):
        return 1e10
    h = np.maximum(h, 0)
    l1 = np.sum(h)
    if l1 <= 0:
        return 1e10
    h = h / l1
    l2_sq = np.sum(h**2)
    h_conv = _compute_conv(h)
    max_conv = np.max(h_conv)
    if max_conv <= 0:
        return 1e10
    return -l2_sq / (DELTA * max_conv)

def _smooth_max_conv(h, temperature=0.1):
    """Smooth approximation of max of autoconvolution using log-sum-exp."""
    h_conv = _compute_conv(h)
    # Use log-sum-exp for smooth max
    max_val = np.max(h_conv)
    if max_val <= 0:
        return 0.0
    # Soft max approximation
    scaled = h_conv / temperature
    exp_vals = np.exp(scaled - np.max(scaled))
    return max_val + temperature * np.log(np.sum(exp_vals))

def _smooth_objective(h, temperature=0.1):
    """Smooth version of objective for gradient-based optimization."""
    h = np.asarray(h, dtype=np.float64)
    if np.any(h < -1e-10):
        return 1e10
    h = np.maximum(h, 0)
    l1 = np.sum(h)
    if l1 <= 0:
        return 1e10
    h = h / l1
    l2_sq = np.sum(h**2)
    max_conv = _smooth_max_conv(h, temperature)
    if max_conv <= 0:
        return 1e10
    return -l2_sq / (DELTA * max_conv)

def _optimize_with_slsqp(h_init, maxiter=1000, temperature=0.05):
    """Optimize weights using SLSQP with smooth objective."""
    h0 = h_init.copy()
    h0 = np.maximum(h0, 0)
    l1 = np.sum(h0)
    if l1 <= 0:
        return h_init
    
    bounds = [(0.0, None)] * N
    cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]
    
    try:
        res = minimize(_smooth_objective, h0, method="SLSQP", bounds=bounds,
                       constraints=cons, options={'maxiter': maxiter, 'ftol': 1e-14},
                       args=(temperature,))
        if res.x is not None:
            hx = np.clip(res.x, 0.0, None)
            if np.sum(hx) > 0:
                hx = hx / np.sum(hx)
                return hx
    except Exception:
        pass
    
    return h_init

def _optimize_with_l_bfgs_b(h_init, maxiter=1000, temperature=0.05):
    """Optimize weights using L-BFGS-B with smooth objective."""
    h0 = h_init.copy()
    h0 = np.maximum(h0, 0)
    l1 = np.sum(h0)
    if l1 <= 0:
        return h_init
    
    bounds = [(0.0, None)] * N
    
    try:
        res = minimize(_smooth_objective, h0, method="L-BFGS-B", bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-14},
                       args=(temperature,))
        if res.x is not None:
            hx = np.clip(res.x, 0.0, None)
            if np.sum(hx) > 0:
                hx = hx / np.sum(hx)
                return hx
    except Exception:
        pass
    
    return h_init

def _generate_sidon_set(k, seed=0, n_trials=200):
    """Generate a Sidon-like set of size k in [0, N)."""
    rng = np.random.RandomState(seed)
    best_set = None
    best_min_gap = -1
    
    for attempt in range(n_trials):
        S = []
        sums = set()
        valid = True
        
        for _ in range(k):
            found = False
            for _trial in range(500):
                cand = rng.randint(0, N)
                if cand in S:
                    continue
                
                new_sums = set()
                conflict = False
                for s in S:
                    pair_sum = cand + s
                    if pair_sum in sums:
                        conflict = True
                        break
                    new_sums.add(pair_sum)
                if not conflict:
                    if cand + cand in sums:
                        conflict = True
                    else:
                        new_sums.add(cand + cand)
                    
                    if not conflict:
                        S.append(cand)
                        sums.update(new_sums)
                        found = True
                        break
            
            if not found:
                valid = False
                break
        
        if valid and len(S) == k:
            S_sorted = sorted(S)
            if len(S_sorted) > 1:
                gaps = np.diff(S_sorted)
                min_gap = np.min(gaps)
            else:
                min_gap = 0
            if min_gap > best_min_gap:
                best_min_gap = min_gap
                best_set = S_sorted
    
    if best_set is None:
        best_set = sorted(rng.choice(N, k, replace=False).tolist())
    
    return best_set

def _generate_random_sparse_support(k, seed=0):
    """Generate a random sparse support of size k."""
    rng = np.random.RandomState(seed)
    return sorted(rng.choice(N, k, replace=False).tolist())

def _generate_structured_support(pattern, params):
    """Generate a structured support based on pattern."""
    if pattern == "powers":
        # Powers of 2: 0, 1, 2, 4, 8, 16, 32
        return [2**i for i in range(int(np.log2(N)) + 1) if 2**i < N]
    elif pattern == "primes":
        # Primes in range
        def is_prime(n):
            if n < 2:
                return False
            for i in range(2, int(np.sqrt(n)) + 1):
                if n % i == 0:
                    return False
            return True
        return [i for i in range(N) if is_prime(i)]
    elif pattern == "arithmetic":
        start, gap = params
        return [i for i in range(start, N, gap)]
    elif pattern == "geometric":
        start, ratio = params
        result = [start]
        while result[-1] * ratio < N:
            result.append(int(result[-1] * ratio))
        return [i for i in result if i < N]
    elif pattern == "fibonacci":
        a, b = params
        result = [a, b]
        while result[-1] + result[-2] < N:
            result.append(result[-1] + result[-2])
        return [i for i in result if i < N]
    else:
        return []

def _try_support_with_weights(S, n_weight_configs=10, seed=0):
    """Try multiple weight configurations for a given support and return best h."""
    rng = np.random.RandomState(seed)
    best_h = None
    best_score = -1
    
    # Uniform weights
    h = np.zeros(N)
    for idx in S:
        h[idx] = 1.0
    h_opt = _optimize_with_slsqp(h, maxiter=500, temperature=0.05)
    s = _score_raw(h_opt)
    if s > best_score:
        best_score = s
        best_h = h_opt.copy()
    
    # Geometric weights
    for r in [0.5, 0.7, 0.8, 0.9, 0.95]:
        h = np.zeros(N)
        for i, idx in enumerate(sorted(S)):
            h[idx] = r**i
        h_opt = _optimize_with_slsqp(h, maxiter=500, temperature=0.05)
        s = _score_raw(h_opt)
        if s > best_score:
            best_score = s
            best_h = h_opt.copy()
    
    # Random weights
    for _ in range(n_weight_configs):
        h = np.zeros(N)
        for idx in S:
            h[idx] = rng.uniform(0.1, 1.0)
        h_opt = _optimize_with_slsqp(h, maxiter=300, temperature=0.05)
        s = _score_raw(h_opt)
        if s > best_score:
            best_score = s
            best_h = h_opt.copy()
    
    return best_h, best_score

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    best_h = None
    best_score = 0.0
    
    def update_best(h):
        nonlocal best_h, best_score
        h = np.asarray(h, dtype=np.float64)
        if np.any(h < 0) or not np.all(np.isfinite(h)):
            return
        h = np.maximum(h, 0)
        s = _score_raw(h)
        if s > best_score:
            best_score = s
            best_h = h.copy()
    
    def time_left():
        return deadline - (time.time() - start_time)
    
    # Phase 1: Try very sparse configurations (1-3 peaks)
    # Single peak
    for i in range(N):
        h = np.zeros(N)
        h[i] = 1.0
        update_best(h)
    
    # Two peaks
    for i in range(N):
        for j in range(i + 1, N):
            if time_left() < 10:
                return list(best_h) if best_h is not None else list(np.ones(N))
            h = np.zeros(N)
            h[i] = 1.0
            h[j] = 1.0
            update_best(h)
    
    # Three peaks
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                if time_left() < 15:
                    break
                h = np.zeros(N)
                h[i] = 1.0
                h[j] = 1.0
                h[k] = 1.0
                update_best(h)
            if time_left() < 15:
                break
        if time_left() < 15:
            break
    
    # Phase 2: Sidon sets with weight optimization
    for k in range(4, 15):
        for seed in range(10):
            if time_left() < 20:
                break
            S = _generate_sidon_set(k, seed=seed + k * 100, n_trials=100)
            h_opt, s = _try_support_with_weights(S, n_weight_configs=5, seed=seed)
            if h_opt is not None:
                update_best(h_opt)
    
    # Phase 3: Structured supports
    structured_patterns = [
        ("powers", None),
        ("primes", None),
        ("arithmetic", (0, 3)),
        ("arithmetic", (1, 3)),
        ("arithmetic", (0, 4)),
        ("arithmetic", (1, 4)),
        ("arithmetic", (0, 5)),
        ("arithmetic", (1, 5)),
        ("arithmetic", (0, 7)),
        ("arithmetic", (1, 7)),
        ("geometric", (1, 2)),
        ("geometric", (1, 1.5)),
        ("geometric", (2, 1.5)),
        ("fibonacci", (1, 2)),
        ("fibonacci", (1, 3)),
    ]
    
    for pattern, params in structured_patterns:
        if time_left() < 15:
            break
        S = _generate_structured_support(pattern, params)
        if len(S) >= 3:
            h_opt, s = _try_support_with_weights(S, n_weight_configs=8, seed=42)
            if h_opt is not None:
                update_best(h_opt)
    
    # Phase 4: Random sparse supports
    np.random.seed(12345)
    for trial in range(500):
        if time_left() < 20:
            break
        num_peaks = np.random.randint(4, 12)
        indices = np.random.choice(N, num_peaks, replace=False)
        S = sorted(indices.tolist())
        
        h = np.zeros(N)
        for idx in S:
            h[idx] = np.random.uniform(0.3, 2.0)
        
        h_opt = _optimize_with_slsqp(h, maxiter=300, temperature=0.05)
        update_best(h_opt)
    
    # Phase 5: Local refinement of best configuration
    if best_h is not None:
        h_curr = best_h.copy()
        
        for iteration in range(100):
            if time_left() < 10:
                break
            
            improved = False
            nonzero_idx = np.where(h_curr > 1e-6)[0]
            
            # Try perturbing each nonzero entry
            for idx in nonzero_idx:
                if time_left() < 5:
                    break
                
                orig_val = h_curr[idx]
                for factor in [0.5, 0.7, 0.8, 1.2, 1.3, 1.5, 2.0]:
                    h_try = h_curr.copy()
                    h_try[idx] = orig_val * factor
                    s = _score_raw(h_try)
                    if s > best_score:
                        best_score = s
                        best_h = h_try.copy()
                        h_curr = h_try.copy()
                        improved = True
                        break
                
                # Try moving the peak
                for offset in [-2, -1, 1, 2]:
                    new_idx = idx + offset
                    if 0 <= new_idx < N and h_curr[new_idx] < 1e-6:
                        h_try = h_curr.copy()
                        h_try[new_idx] = h_try[idx]
                        h_try[idx] = 0.0
                        s = _score_raw(h_try)
                        if s > best_score:
                            best_score = s
                            best_h = h_try.copy()
                            h_curr = h_try.copy()
                            improved = True
                            break
            
            # Try adding a new peak
            if not improved:
                zero_idx = np.where(h_curr < 1e-6)[0]
                if len(zero_idx) > 0:
                    new_idx = np.random.choice(zero_idx)
                    h_try = h_curr.copy()
                    h_try[new_idx] = 0.5
                    s = _score_raw(h_try)
                    if s > best_score:
                        best_score = s
                        best_h = h_try.copy()
                        h_curr = h_try.copy()
            
            # Try removing a peak
            if not improved and len(nonzero_idx) > 1:
                remove_idx = np.random.choice(nonzero_idx)
                h_try = h_curr.copy()
                h_try[remove_idx] = 0.0
                s = _score_raw(h_try)
                if s > best_score:
                    best_score = s
                    best_h = h_try.copy()
                    h_curr = h_try.copy()
            
            # Final optimization pass
            if time_left() > 15:
                h_opt = _optimize_with_slsqp(h_curr, maxiter=500, temperature=0.03)
                update_best(h_opt)
                h_curr = best_h.copy()
    
    # Phase 6: Try smooth bump functions
    for center in range(5, N - 5):
        for width in [2, 3, 4, 5, 7, 10, 15]:
            if time_left() < 5:
                break
            x = np.arange(N)
            h = np.exp(-0.5 * ((x - center) / width)**2)
            if np.sum(h) > 0:
                update_best(h)
                if time_left() > 10:
                    h_opt = _optimize_with_slsqp(h, maxiter=300, temperature=0.05)
                    update_best(h_opt)
    
    # Return best found
    if best_h is not None:
        return list(best_h)
    return list(np.ones(N))
