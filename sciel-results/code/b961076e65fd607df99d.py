"""
Second autocorrelation inequality: choose 50 nonnegative heights on equal intervals of [-1/4, 1/4]
to maximize (sum h_i^2) / ((sum h_i) * sup_k C[k]) where C = h autoconvolution.

The autoconvolution of the step function f(t) = sum h_i * 1_{[x_i, x_{i+1})}(t) with delta = 1/50:
- ||f||_1 = delta * sum(h_i)
- ||f||_2^2 = delta * sum(h_i^2)
- ||f*f||_inf = delta * max(conv(h,h))

Ratio = (delta * sum h^2) / (delta * sum h * delta * max conv) = sum h^2 / (delta * sum h * max conv)

The evaluator normalizes (likely L1=1), so we maximize: sum(h^2) / (delta * max(conv(h,h)))
subject to sum(h)=1, h>=0.

Strategy: Use sparse, well-separated peaks (Sidon-like sets) to minimize autoconvolution overlap,
then refine weights using SLSQP optimization.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DELTA = 1.0 / N  # = 0.02

def _compute_score(h):
    """Compute the score assuming L1 normalization."""
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 <= 0:
        return 0.0
    h_norm = h / l1
    l2_sq = np.sum(h_norm**2)
    h_conv = np.convolve(h_norm, h_norm, mode='full')
    max_conv = np.max(h_conv)
    if max_conv <= 0:
        return 0.0
    return l2_sq / (DELTA * max_conv)

def _score_raw(h):
    """Raw score without assuming normalization - used for internal comparison."""
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 <= 0:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    max_conv = np.max(h_conv)
    if max_conv <= 0:
        return 0.0
    # The actual ratio: sum h^2 / (delta * sum h * max conv)
    return l2_sq / (DELTA * l1 * max_conv)

def _neg_ratio(h):
    """Negative of the ratio for minimization."""
    h = np.asarray(h, dtype=np.float64)
    if np.any(h < -1e-10):
        return 1e10
    h = np.maximum(h, 0)
    l1 = np.sum(h)
    if l1 <= 0:
        return 1e10
    h = h / l1
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    max_conv = np.max(h_conv)
    if max_conv <= 0:
        return 1e10
    return -l2_sq / (DELTA * max_conv)

def _find_sidon_support(k, seed=0, n_trials=100):
    """
    Find a Sidon-like support set of size k in [0, N).
    A Sidon set has unique pairwise sums (i+j for i<=j).
    We use a greedy randomized approach.
    """
    rng = np.random.RandomState(seed)
    best_set = None
    best_score = -1
    
    for attempt in range(n_trials):
        S = []
        sums = set()
        valid = True
        
        # Try to build a set of size k
        for _ in range(k):
            # Try random candidates
            found = False
            for _trial in range(200):
                cand = rng.randint(0, N)
                if cand in S:
                    continue
                
                # Check if adding cand creates a collision
                new_sums = set()
                conflict = False
                for s in S:
                    pair_sum = cand + s
                    if pair_sum in sums:
                        conflict = True
                        break
                    new_sums.add(pair_sum)
                if not conflict:
                    # Also check cand+cand
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
            # Score: how "spread out" is the set? Prefer larger gaps
            S_sorted = sorted(S)
            gaps = np.diff(S_sorted)
            score = np.min(gaps) if len(gaps) > 0 else 0
            if score > best_score:
                best_score = score
                best_set = S_sorted
    
    if best_set is None:
        # Fallback: random set
        best_set = sorted(rng.choice(N, k, replace=False).tolist())
    
    return best_set

def _optimize_weights(h_init, maxiter=500):
    """
    Local refinement maximizing ratio under L1=1, h>=0 using SLSQP.
    """
    h0 = h_init / np.sum(h_init)
    bounds = [(0.0, None)] * N
    cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]
    try:
        res = minimize(_neg_ratio, h0, method="SLSQP", bounds=bounds,
                       constraints=cons, options={'maxiter': maxiter, 'ftol': 1e-12})
        if res.x is not None:
            hx = np.clip(res.x, 0.0, None)
            if np.sum(hx) > 0:
                hx = hx / np.sum(hx)
                if _score_raw(hx) > _score_raw(h0):
                    return hx
    except Exception:
        pass
    return h0

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
    
    # Phase 1: Try sparse configurations with few peaks
    # The key insight: fewer peaks with larger values -> higher sum(h^2)/sum(h) ratio
    # But we need to minimize max autoconvolution
    
    # Single peak (baseline)
    for i in range(N):
        h = np.zeros(N)
        h[i] = 1.0
        update_best(h)
    
    # Two peaks
    for i in range(N):
        for j in range(i+1, N):
            if time_left() < 10:
                return list(best_h) if best_h is not None else list(np.ones(N))
            h = np.zeros(N)
            h[i] = 1.0
            h[j] = 1.0
            update_best(h)
    
    # Phase 2: Sidon set search & weight optimization
    # The key insight: if support is Sidon, max_conv is dominated by the largest pairwise product
    # If we make one weight dominant, the ratio can be improved
    
    # Generate candidate Sidon sets
    candidate_supports = []
    
    # 1. Small Sidon sets (sparse)
    for k in range(4, 13):
        for seed in range(8):
            if time_left() < 15:
                break
            S = _find_sidon_support(k, seed=seed + k*100, n_trials=50)
            candidate_supports.append(S)
    
    # 2. Specific known Sidon sets (e.g. powers of 2, primes)
    # Powers of 2: 0, 1, 2, 4, 8, 16, 32
    candidate_supports.append([0, 1, 2, 4, 8, 16, 32])
    # Primes in range
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    candidate_supports.append([p for p in primes if p < N])
    # Another structured set
    candidate_supports.append([0, 1, 3, 7, 15, 31])
    candidate_supports.append([0, 2, 5, 11, 23, 47])
    
    # 3. Random smooth sets
    np.random.seed(42)
    for _ in range(20):
        if time_left() < 15:
            break
        a = np.random.uniform(0.5, 5.0)
        c = np.random.uniform(4, N - 4)
        h_bump = np.exp(-a * (np.arange(N) - c)**2)
        # Threshold to get support
        threshold = np.percentile(h_bump, 50)
        support = np.where(h_bump > threshold)[0].tolist()
        if len(support) > 3:
            candidate_supports.append(support)
    
    # Optimize weights for each support
    for S in candidate_supports:
        if time_left() < 10:
            break
        
        # Create initial vector with ones on support
        h_init = np.zeros(N)
        for idx in S:
            h_init[idx] = 1.0
        
        # Try different initial weight distributions
        h_cands = [h_init.copy()]
        
        # Geometric distribution
        for r in [0.5, 0.7, 0.8, 0.9]:
            h_geo = np.zeros(N)
            for i, idx in enumerate(sorted(S)):
                h_geo[idx] = r**i
            h_cands.append(h_geo)
        
        # Random weights
        for _ in range(5):
            h_rand = np.zeros(N)
            for idx in S:
                h_rand[idx] = np.random.uniform(0.1, 1.0)
            h_cands.append(h_rand)
        
        # Optimize each candidate
        for h_c in h_cands:
            if time_left() < 5:
                break
            h_opt = _optimize_weights(h_c, maxiter=300)
            update_best(h_opt)
    
    # Phase 3: Random sparse configurations
    np.random.seed(12345)
    for _ in range(2000):
        if time_left() < 15:
            break
        num_peaks = np.random.randint(3, 10)
        indices = np.random.choice(N, num_peaks, replace=False)
        h = np.zeros(N)
        h[indices] = np.random.uniform(0.3, 2.0, size=num_peaks)
        update_best(h)
        
        # Also try optimized version
        if time_left() < 10:
            continue
        h_opt = _optimize_weights(h, maxiter=200)
        update_best(h_opt)
    
    # Phase 4: Local refinement of best configuration
    if best_h is not None:
        h_curr = best_h.copy()
        for _ in range(50):
            if time_left() < 5:
                break
            
            improved = False
            # Try perturbing each nonzero entry
            nonzero_idx = np.where(h_curr > 1e-6)[0]
            for idx in nonzero_idx:
                if time_left() < 3:
                    break
                
                orig_val = h_curr[idx]
                # Try scaling up and down
                for factor in [0.5, 0.7, 1.3, 1.5, 2.0]:
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
            if time_left() > 10:
                h_opt = _optimize_weights(h_curr, maxiter=500)
                update_best(h_opt)
                h_curr = best_h.copy()
    
    # Phase 5: Try structured patterns
    # Alternating peaks
    for start in range(N):
        for gap in [2, 3, 4, 5, 7, 10]:
            if time_left() < 5:
                break
            h = np.zeros(N)
            for i in range(start, N, gap):
                h[i] = 1.0
            if np.sum(h) > 0:
                update_best(h)
                if time_left() > 5:
                    h_opt = _optimize_weights(h, maxiter=200)
                    update_best(h_opt)
    
    # Gaussian-like with varying widths
    for center in range(5, N-5):
        for width in [2, 3, 4, 5, 7, 10, 15]:
            if time_left() < 5:
                break
            x = np.arange(N)
            h = np.exp(-0.5 * ((x - center) / width)**2)
            if np.sum(h) > 0:
                update_best(h)
                if time_left() > 5:
                    h_opt = _optimize_weights(h, maxiter=200)
                    update_best(h_opt)
    
    # Return best found
    if best_h is not None:
        return list(best_h)
    return list(np.ones(N))
