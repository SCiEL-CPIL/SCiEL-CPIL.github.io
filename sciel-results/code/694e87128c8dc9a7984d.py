# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy import optimize


def _compute_P(s, w):
    """Exact P[X1+X2+X3 < 2*X4] for support s, normalized weights w (strict <)."""
    n = s.size
    if n <= 1:
        return 0.0
    # Build all n^2 pairwise sums (for X1+X2)
    pair_sums = s[:, None] + s[None, :]
    pair_weights = w[:, None] * w[None, :]
    # Sort pair sums
    flat_s = pair_sums.ravel()
    flat_w = pair_weights.ravel()
    order = np.argsort(flat_s)
    sorted_sums = flat_s[order]
    sorted_ws = flat_w[order]
    cumsum = np.cumsum(sorted_ws)
    # For each l, k: need sum of w_i*w_j where s_i+s_j < 2*s_l - s_k
    # Total = sum_k w_k * sum_l w_l * H(2*s_l - s_k)
    # where H(t) = sum of pair weights with pair_sum < t
    # Use searchsorted for efficiency
    total = 0.0
    for k in range(n):
        thresholds = 2.0 * s - s[k]
        # For each l, count pair sums < 2*s_l - s_k
        idx = np.searchsorted(sorted_sums, thresholds, side='left')
        H = np.zeros(n)
        mask = idx > 0
        H[mask] = cumsum[idx[mask] - 1]
        total += w[k] * np.sum(w * H)
    return float(total)


def _compute_P_fast(s, w):
    """Faster version using vectorized approach."""
    n = s.size
    if n <= 1:
        return 0.0
    # Pair sums: n x n matrix
    pair_sums = s[:, None] + s[None, :]  # n x n
    pair_w = w[:, None] * w[None, :]     # n x n
    
    # Flatten and sort
    flat_sums = pair_sums.ravel()
    flat_w = pair_w.ravel()
    order = np.argsort(flat_sums)
    sorted_sums = flat_sums[order]
    sorted_w = flat_w[order]
    cumsum = np.cumsum(sorted_w)
    
    # For each (k, l): threshold = 2*s[l] - s[k]
    # Build all thresholds as n x n
    # thresholds[k, l] = 2*s[l] - s[k]
    thresholds = 2.0 * s[None, :] - s[:, None]  # shape (n, n), [k, l]
    
    # Searchsorted for each threshold
    # searchsorted expects 1D, so ravel
    idx = np.searchsorted(sorted_sums, thresholds.ravel(), side='left')
    H = np.zeros(n * n)
    mask = idx > 0
    H[mask] = cumsum[idx[mask] - 1]
    H = H.reshape(n, n)  # H[k, l] = P[X1+X2 < 2*s_l - s_k]
    
    # Total = sum_k sum_l w[k]*w[l]*H[k,l]
    total = np.sum(w[:, None] * w[None, :] * H)
    return float(total)


def _compute_P_smooth(s, w, alpha=50.0):
    """Smooth surrogate for P using soft indicator via log-sum-exp.
    
    Replaces the hard step function 1[s_i+s_j < t] with a smooth sigmoid:
    1/(1+exp(alpha*(s_i+s_j - t)))
    
    This is differentiable everywhere, allowing gradient-based optimizers
    to navigate the landscape. As alpha -> inf, it converges to the hard step.
    """
    n = s.size
    if n <= 1:
        return 0.0
    # Pair sums: n x n
    pair_sums = s[:, None] + s[None, :]  # (n, n) [i, j]
    pair_w = w[:, None] * w[None, :]     # (n, n)
    
    # Thresholds: (n, n) [k, l] = 2*s_l - s_k
    thresholds = 2.0 * s[None, :] - s[:, None]  # (n, n) [k, l]
    
    # For each (k, l): H[k,l] = sum_{i,j} pair_w[i,j] * sigmoid(alpha*(t_kl - pair_sums[i,j]))
    # To avoid overflow, compute in chunks or use safe sigmoid
    # diff[k, l, i, j] = thresholds[k,l] - pair_sums[i,j]  -> too big for large n
    # Instead: for each (k,l), compute H[k,l] = sum_ij pair_w_ij * sig(alpha * (th[k,l] - ps[i,j]))
    
    # Vectorized: expand to (n, n, n) where dim0=k, dim1=l, dim2=flattened (i,j)
    # This is n^3 which is fine for n <= 10
    if n <= 10:
        # H[k,l] = sum_ij pair_w[i,j] * sigmoid(alpha * (th[k,l] - ps[i,j]))
        # Shape: th is (n,n) [k,l], ps is (n*n) [i,j]
        # diff: (n, n, n*n)
        diff = thresholds[:, :, None] - pair_sums.ravel()[None, None, :]  # (n, n, n^2)
        # Clip to avoid overflow in exp
        diff_clipped = np.clip(diff, -50.0, 50.0)
        # sigmoid(alpha * diff) = 1/(1+exp(-alpha*diff))
        sig = 1.0 / (1.0 + np.exp(-alpha * diff_clipped))
        # H[k,l] = sum_j sig[k,l,j] * pair_w_flat[j]
        H = np.sum(pair_w.ravel()[None, None, :] * sig, axis=2)  # (n, n)
    else:
        # Fallback to exact for large n
        return _compute_P_fast(s, w)
    
    total = np.sum(w[:, None] * w[None, :] * H)
    return float(total)


def _negP(params):
    """Objective: minimize -P. params = [s_1,...,s_n, logw_1,...,logw_n]"""
    n = len(params) // 2
    raw_s = params[:n]
    raw_w = params[n:]
    # s must be >= 0; use softplus-like or just exp
    s = np.exp(raw_s)  # ensures positive
    # Normalize weights
    w = np.exp(raw_w - np.max(raw_w))
    w = w / w.sum()
    P = _compute_P_fast(s, w)
    return -P


def _negP_constrained(params):
    """For scipy minimize with bounds: params directly encode s>=0 and log-weights."""
    n = len(params) // 2
    s = params[:n]  # already >= 0 due to bounds
    raw_w = params[n:]
    w = np.exp(raw_w - np.max(raw_w))
    w = w / w.sum()
    P = _compute_P_fast(s, w)
    return -P


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    t0 = time.time()
    deadline = 580.0  # safety margin
    rng = np.random.default_rng(42)
    
    best_P = -1.0
    best_s = None
    best_w = None
    
    def consider(s, w):
        nonlocal best_P, best_s, best_w
        if s.size < 1:
            return
        if not (np.all(np.isfinite(s)) and np.all(np.isfinite(w))):
            return
        if np.any(s < -1e-12) or np.any(w < -1e-12):
            return
        if w.sum() <= 1e-12:
            return
        w = w / w.sum()
        P = _compute_P_fast(s, w)
        if P > best_P:
            best_P = P
            best_s = s.copy()
            best_w = w.copy()
    
    # --- Analytical 2-point structure ---
    # For a < b/2: P = (1-p)(3p^2 - 2p^3)
    # dP/dp = 6p - 15p^2 + 8p^3 = p(6 - 15p + 8p^2)
    # 8p^2 - 15p + 6 = 0 -> p = (15 ± sqrt(33))/16
    p_opt = (15.0 - np.sqrt(33.0)) / 16.0
    P_2pt = (1-p_opt)*(3*p_opt**2 - 2*p_opt**3)
    
    # Use multiple scale factors to explore the (a,b) space
    for a in [1e-8, 1e-6, 1e-4, 1e-3, 0.01, 0.1, 0.5]:
        for b in [1.0, 5.0, 10.0, 50.0, 100.0, 500.0]:
            if a < b/2:
                s_test = np.array([a, b])
                w_test = np.array([p_opt, 1-p_opt])
                consider(s_test, w_test)
    
    # Also try the other regime: a >= b/2, P = (1-p)*p^3, p* = 3/4
    p_opt2 = 0.75
    for a in [0.6, 0.7, 0.8, 1.0]:
        for b in [1.0, 2.0, 5.0]:
            if a >= b/2:
                s_test = np.array([a, b])
                w_test = np.array([p_opt2, 1-p_opt2])
                consider(s_test, w_test)
    
    # --- Now optimize 3-point and 4-point structures ---
    # Strategy: fix the "structure" (which points are small, which large) and optimize
    
    def optimize_n(n, s0, w0, bounds_s=None, maxiter=5000):
        """Optimize n-point measure."""
        # Parameterize: s_i >= 0 (use log for optimization), weights via softmax
        # params: [log(s_1),...,log(s_n), logw_1,...,logw_n]
        def objective(params):
            raw_s = params[:n]
            raw_w = params[n:]
            s = np.exp(raw_s)
            w = np.exp(raw_w - np.max(raw_w))
            w = w / w.sum()
            return _compute_P_fast(s, w)
        
        def neg_obj(params):
            return -objective(params)
        
        log_s0 = np.log(np.maximum(s0, 1e-12))
        log_w0 = np.log(np.maximum(w0, 1e-12))
        x0 = np.concatenate([log_s0, log_w0])
        
        # Bounds: log_s in [-25, 25], log_w in [-25, 25]
        bounds = [(-25.0, 25.0)] * (2 * n)
        
        try:
            res = optimize.minimize(neg_obj, x0, method='L-BFGS-B', bounds=bounds,
                                    options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-9})
            if res.x is not None and np.all(np.isfinite(res.x)):
                s = np.exp(res.x[:n])
                raw_w = res.x[n:]
                w = np.exp(raw_w - np.max(raw_w))
                w = w / w.sum()
                consider(s, w)
                return s, w
        except Exception:
            pass
        
        # Try Nelder-Mead on raw params (s >= 0, weights normalized)
        def neg_obj_raw(params):
            s = np.maximum(params[:n], 0.0)
            w = np.maximum(params[n:], 0.0)
            if w.sum() < 1e-15:
                return 1.0
            w = w / w.sum()
            return -_compute_P_fast(s, w)
        
        x0_raw = np.concatenate([s0, w0])
        try:
            res2 = optimize.minimize(neg_obj_raw, x0_raw, method='Nelder-Mead',
                                     options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-11})
            if res2.x is not None and np.all(np.isfinite(res2.x)):
                s = np.maximum(res2.x[:n], 0.0)
                w = np.maximum(res2.x[n:], 0.0)
                if w.sum() > 1e-15:
                    w = w / w.sum()
                    consider(s, w)
                    return s, w
        except Exception:
            pass
        return s0, w0
    
    # 3-point: [small, medium, large]
    for trial in range(30):
        if time.time() - t0 > deadline:
            break
        # Diverse starting points
        a = rng.uniform(1e-8, 0.5)
        b = rng.uniform(0.5, 5.0)
        c = rng.uniform(5.0, 200.0)
        pa = rng.uniform(0.2, 0.8)
        pb = rng.uniform(0.05, 0.4)
        pc = 1.0 - pa - pb
        if pc < 0.01:
            pc = 0.01
            pa = (1.0 - pc - pb) / 2
            pb = (1.0 - pc - pa)
        s3 = np.array([a, b, c])
        w3 = np.array([pa, pb, pc])
        w3 = w3 / w3.sum()
        consider(s3, w3)
        optimize_n(3, s3, w3)
    
    # 4-point: [tiny, small, medium, large]
    for trial in range(20):
        if time.time() - t0 > deadline:
            break
        a = rng.uniform(1e-8, 1e-2)
        b = rng.uniform(0.01, 1.0)
        c = rng.uniform(1.0, 10.0)
        d = rng.uniform(10.0, 500.0)
        wa = rng.uniform(0.2, 0.7)
        wb = rng.uniform(0.1, 0.4)
        wc = rng.uniform(0.05, 0.25)
        wd = 1.0 - wa - wb - wc
        if wd < 0.01:
            wd = 0.01
            wa = (1.0 - wd - wb - wc) / 3
            wb = (1.0 - wd - wa - wc) / 2
            wc = 1.0 - wd - wa - wb
        s4 = np.array([a, b, c, d])
        w4 = np.array([wa, wb, wc, wd])
        w4 = w4 / w4.sum()
        consider(s4, w4)
        optimize_n(4, s4, w4)
    
    # 5-point
    for trial in range(12):
        if time.time() - t0 > deadline:
            break
        s5 = np.sort(rng.uniform(1e-6, 100.0, 5))
        w5 = rng.dirichlet(np.ones(5) * 0.3)
        consider(s5, w5)
        optimize_n(5, s5, w5)
    
    # 6-point for more exploration
    for trial in range(8):
        if time.time() - t0 > deadline:
            break
        s6 = np.sort(rng.uniform(1e-7, 200.0, 6))
        w6 = rng.dirichlet(np.ones(6) * 0.2)
        consider(s6, w6)
        optimize_n(6, s6, w6)
    
    # Final polish: take best found and do fine optimization with multiple restarts
    if best_s is not None:
        n_best = best_s.size
        if time.time() - t0 < deadline - 60:
            # First: high-precision Nelder-Mead on the best solution
            def neg_obj_final(params):
                s = np.maximum(params[:n_best], 0.0)
                w = np.maximum(params[n_best:], 0.0)
                if w.sum() < 1e-15:
                    return 1.0
                w = w / w.sum()
                return -_compute_P_fast(s, w)
            
            x0 = np.concatenate([best_s, best_w])
            try:
                resf = optimize.minimize(neg_obj_final, x0, method='Nelder-Mead',
                                         options={'maxiter': 100000, 'xatol': 1e-14, 'fatol': 1e-15})
                if resf.x is not None and np.all(np.isfinite(resf.x)):
                    s = np.maximum(resf.x[:n_best], 0.0)
                    w = np.maximum(resf.x[n_best:], 0.0)
                    if w.sum() > 1e-15:
                        w = w / w.sum()
                        consider(s, w)
            except Exception:
                pass
            
            # Second: perturb the best solution and re-optimize to escape local minima
            if time.time() - t0 < deadline - 30:
                for pert_trial in range(5):
                    if time.time() - t0 > deadline - 10:
                        break
                    # Small random perturbation in log-space
                    log_s = np.log(np.maximum(best_s, 1e-15))
                    log_w = np.log(np.maximum(best_w, 1e-15))
                    noise_s = rng.normal(0, 0.1, n_best)
                    noise_w = rng.normal(0, 0.1, n_best)
                    pert_s = np.exp(log_s + noise_s)
                    pert_w = np.exp(log_w + noise_w)
                    pert_w = pert_w / pert_w.sum()
                    consider(pert_s, pert_w)
                    # Re-optimize from perturbed point
                    s_pert, w_pert = optimize_n(n_best, pert_s, pert_w, maxiter=3000)
    
    # Prepare final answer
    s = np.ascontiguousarray(best_s, dtype=np.float64).ravel()
    w = np.ascontiguousarray(best_w, dtype=np.float64).ravel()
    s = np.maximum(s, 0.0)
    w = np.maximum(w, 0.0)
    w = w / w.sum()
    
    # Final sanity check: if something went wrong, fall back to the analytic 2-point solution
    if not (np.all(np.isfinite(s)) and np.all(np.isfinite(w))):
        p_fallback = (15.0 - np.sqrt(33.0)) / 16.0
        s = np.array([1e-6, 1.0], dtype=np.float64)
        w = np.array([p_fallback, 1.0 - p_fallback], dtype=np.float64)
    
    assert s.size == w.size >= 1
    assert np.all(np.isfinite(s)) and np.all(np.isfinite(w))
    assert np.all(s >= -1e-9) and np.all(w >= -1e-9) and w.sum() > 1e-9
    return s, w

# EVOLVE-BLOCK-END
