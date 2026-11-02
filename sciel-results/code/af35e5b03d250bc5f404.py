import numpy as np
import time
from scipy.optimize import minimize

N = 50
DELTA = 1.0 / N  # step size

def _autoconv(h):
    """Full autoconvolution of h (length 2N-1)."""
    return np.correlate(h, h, mode='full')

def _ratio(h):
    """
    Compute the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf).
    h: array of nonnegative heights.
    Returns the ratio value (higher is better).
    """
    h = np.asarray(h, dtype=float)
    if len(h) != N:
        return 0.0
    if np.any(h < -1e-12):
        return 0.0
    h = np.maximum(h, 0.0)
    
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    
    l2_sq = np.sum(h**2)
    
    c = _autoconv(h)
    sup_c = np.max(c)
    
    if sup_c < 1e-15:
        return 0.0
    
    return l2_sq / (l1 * sup_c)

def _negative_ratio(h):
    return -_ratio(h)

def _reweight_fixed_peak(h, peak_lag, iters=200, tol=1e-10):
    """
    Iterative reweighting: fix which lag k gives the sup norm of autoconvolution,
    then solve for optimal amplitudes under that constraint.
    
    For fixed peak at lag k, the problem becomes:
    maximize L2^2 / (L1 * c[k])  subject to c[j] <= c[k] for all j.
    
    We use multiplicative updates to find the optimum.
    """
    h = np.maximum(h.copy(), 0.0)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return h
    
    # Normalize to L1 = 1 for the reweighting
    h = h / l1
    
    for it in range(iters):
        c = _autoconv(h)
        c_peak = c[peak_lag + N - 1]  # index in full correlation
        
        if c_peak < 1e-15:
            break
        
        # The constraint is c[j] <= c_peak for all j.
        # At optimum, the active constraints determine the solution.
        # Use gradient-based update: increase h[i] if it helps ratio.
        
        l2_sq = np.sum(h**2)
        ratio = l2_sq / c_peak  # since L1=1
        
        # Gradient of ratio w.r.t. h[i] (with L1=1 constraint, use Lagrange)
        # d(ratio)/d(h[i]) = (2*h[i]*c_peak - l2_sq * dc_peak/d(h[i])) / c_peak^2
        # dc_peak/d(h[i]) = sum_j h[j] * delta_{i+j, peak_lag or -peak_lag}
        
        # For lag k (0-indexed in full, actual lag = k-(N-1)):
        actual_lag = peak_lag  # this is the lag value
        
        # dc/d(h[i]) for autoconv at lag L:
        # c[L] = sum_i h[i] h[i+L] for L >= 0
        # dc[L]/dh[i] = h[i-L] + h[i+L] (with boundary handling)
        
        dc_dh = np.zeros(N)
        L = actual_lag
        if L >= 0:
            for i in range(N):
                val = 0.0
                if i - L >= 0 and i - L < N:
                    val += h[i - L]
                if i + L >= 0 and i + L < N:
                    val += h[i + L]
                dc_dh[i] = val
        else:
            L2 = -L
            for i in range(N):
                val = 0.0
                if i - L2 >= 0 and i - L2 < N:
                    val += h[i - L2]
                if i + L2 >= 0 and i + L2 < N:
                    val += h[i + L2]
                dc_dh[i] = val
        
        # Gradient ascent on ratio (with L1 constraint via normalization)
        grad = 2.0 * h * c_peak - l2_sq * dc_dh
        grad = grad / (c_peak * c_peak)
        
        # Multiplicative update
        step = 0.1
        h_new = h * np.exp(step * grad / (np.max(np.abs(grad)) + 1e-15))
        h_new = np.maximum(h_new, 0.0)
        l1_new = np.sum(h_new)
        if l1_new < 1e-15:
            break
        h_new = h_new / l1_new
        
        change = np.max(np.abs(h_new - h))
        h = h_new
        
        if change < tol:
            break
    
    return h

def _try_support(support, time_left):
    """
    Try to optimize amplitudes on a given support (list of indices).
    Uses coordinate descent and local optimization.
    """
    n_sup = len(support)
    if n_sup < 1:
        return None, 0.0
    
    # Initialize with uniform amplitudes on support
    h = np.zeros(N)
    for idx in support:
        h[idx] = 1.0 / n_sup
    
    best_h = h.copy()
    best_score = _ratio(h)
    
    # Coordinate descent: adjust one amplitude at a time
    for coord_iter in range(100):
        if time.time() > time_left:
            break
        improved = False
        for idx in support:
            # Try different amplitudes at this coordinate
            current_val = h[idx]
            if current_val < 1e-15:
                continue
            
            # Try scaling this coordinate
            for scale in [0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 2.0]:
                h_trial = h.copy()
                h_trial[idx] = current_val * scale
                s = _ratio(h_trial)
                if s > best_score + 1e-12:
                    best_score = s
                    best_h = h_trial.copy()
                    improved = True
                h = h_trial
            # Reset to best
            h = best_h.copy()
        
        if not improved:
            break
    
    # Local optimization on the support
    if time.time() < time_left:
        try:
            # Create a sub-problem with only the support variables
            def sub_cost(x):
                h_full = np.zeros(N)
                for i, idx in enumerate(support):
                    h_full[idx] = max(0.0, x[i])
                return -_ratio(h_full)
            
            x0 = np.array([best_h[idx] for idx in support])
            bounds_sub = [(0.0, 10.0)] * n_sup
            
            res = minimize(sub_cost, x0, method='L-BFGS-B', bounds=bounds_sub,
                          options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12})
            
            if res.fun < 0:
                score = -res.fun
                if score > best_score:
                    h_full = np.zeros(N)
                    for i, idx in enumerate(support):
                        h_full[idx] = max(0.0, res.x[i])
                    best_score = score
                    best_h = h_full.copy()
        except Exception:
            pass
    
    return best_h, best_score

def _find_good_supports(n_supports, n_per_group, time_limit):
    """
    Generate diverse support patterns and evaluate them.
    """
    supports = []
    
    # Consecutive blocks
    for size in range(1, min(n_per_group + 1, N)):
        for start in range(0, N - size + 1, max(1, (N - size) // n_per_group + 1)):
            supports.append(list(range(start, start + size)))
    
    # Sparse: pick every k-th element
    for k in range(2, 10):
        idxs = list(range(0, N, k))
        if len(idxs) >= 2:
            supports.append(idxs)
    
    # Random subsets
    np.random.seed(12345)
    for _ in range(n_supports):
        size = np.random.randint(2, min(N, 15) + 1)
        idxs = np.random.choice(N, size=size, replace=False)
        supports.append(sorted(idxs.tolist()))
    
    # Sidon-like: quadratic residues mod p
    for p in [53, 59, 61, 67]:
        if p > N:
            break
        residues = [i * i % p for i in range(p)]
        residues = sorted(set(r for r in residues if r < N))
        if len(residues) >= 2:
            supports.append(residues)
    
    return supports

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    best_h = np.zeros(N)
    best_h[0] = 1.0
    best_score = _ratio(best_h)
    
    # Strategy 1: Try various structured supports
    supports = _find_good_supports(30, 8, deadline)
    
    # Evaluate each support quickly
    for support in supports:
        if time.time() > deadline - 30:
            break
        if len(support) < 1 or len(support) > 20:
            continue
        
        h_candidate, score = _try_support(support, deadline - 10)
        if h_candidate is not None and score > best_score:
            best_score = score
            best_h = h_candidate.copy()
    
    # Strategy 2: Full optimization from best found so far
    if time.time() < deadline - 20:
        try:
            bounds = [(0.0, 5.0)] * N
            res = minimize(_negative_ratio, best_h, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.fun < 0:
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_h = res.x.copy()
        except Exception:
            pass
    
    # Strategy 3: Random restarts with local optimization
    np.random.seed(42)
    n_restarts = 15
    for r in range(n_restarts):
        if time.time() > deadline - 10:
            break
        
        # Generate random initial point
        if r % 3 == 0:
            # Sparse random
            n_nonzero = np.random.randint(2, 8)
            idxs = np.random.choice(N, size=n_nonzero, replace=False)
            x0 = np.zeros(N)
            x0[idxs] = np.random.uniform(0.1, 1.0, size=n_nonzero)
        elif r % 3 == 1:
            # Smooth random
            x0 = np.random.exponential(0.3, size=N)
        else:
            # Bimodal
            x0 = np.zeros(N)
            center1 = np.random.randint(0, N // 2)
            center2 = np.random.randint(N // 2, N)
            for i in range(N):
                x0[i] = np.exp(-0.5 * ((i - center1) / 3.0)**2) + np.exp(-0.5 * ((i - center2) / 3.0)**2)
        
        try:
            res = minimize(_negative_ratio, x0, method='L-BFGS-B', 
                          bounds=[(0.0, 5.0)] * N,
                          options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.fun < 0:
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_h = res.x.copy()
        except Exception:
            pass
    
    # Final polish
    if time.time() < deadline - 5:
        try:
            res = minimize(_negative_ratio, best_h, method='L-BFGS-B',
                          bounds=[(0.0, 5.0)] * N,
                          options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-13})
            if res.fun < 0:
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_h = res.x.copy()
        except Exception:
            pass
    
    return list(best_h)
