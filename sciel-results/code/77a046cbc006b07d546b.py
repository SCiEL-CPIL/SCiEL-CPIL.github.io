# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy import optimize


def _compute_P_fast(s, w):
    """Fast exact P[X1+X2+X3 < 2*X4] for support s, normalized weights w (strict <)."""
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
    thresholds = 2.0 * s[None, :] - s[:, None]  # shape (n, n), [k, l]
    
    # Searchsorted for each threshold
    idx = np.searchsorted(sorted_sums, thresholds.ravel(), side='left')
    H = np.zeros(n * n)
    mask = idx > 0
    H[mask] = cumsum[idx[mask] - 1]
    H = H.reshape(n, n)  # H[k, l] = P[X1+X2 < 2*s_l - s_k]
    
    # Total = sum_k sum_l w[k]*w[l]*H[k,l]
    total = np.sum(w[:, None] * w[None, :] * H)
    return float(total)


def _decode(params, n):
    """Decode raw params to (s, w) with s >= 0 and w normalized."""
    raw_s = params[:n]
    raw_w = params[n:]
    s = np.exp(raw_s)
    w = np.exp(raw_w - np.max(raw_w))
    w = w / w.sum()
    return s, w


def _negP(params, n):
    """Objective: minimize -P. params = [log(s_1),...,log(s_n), logw_1,...,logw_n]"""
    s, w = _decode(params, n)
    return -_compute_P_fast(s, w)


def _compute_P_smooth(s, w, sigma=0.01):
    """Smooth surrogate using sigmoid instead of step function."""
    n = s.size
    if n <= 1:
        return 0.0
    pair_sums = s[:, None] + s[None, :]
    pair_w = w[:, None] * w[None, :]
    
    thresholds = 2.0 * s[None, :] - s[:, None]  # (n, n) [k, l]
    
    # For each pair (i,j) and each (k,l), compute sigmoid((2*s_l - s_k - s_i - s_j)/sigma)
    # This is n^4 which is too slow for large n, so we use a different approach.
    # Instead, we approximate by evaluating the smooth CDF at each threshold.
    
    flat_sums = pair_sums.ravel()
    flat_w = pair_w.ravel()
    order = np.argsort(flat_sums)
    sorted_sums = flat_sums[order]
    sorted_w = flat_w[order]
    cumsum = np.cumsum(sorted_w)
    
    # Smoothed version: use interpolation with a small window
    # For simplicity, use the exact version as a fallback for smooth surrogate
    return _compute_P_fast(s, w)


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    t0 = time.time()
    deadline = 610.0  # safety margin (evaluator kills at 630s)
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
    p_opt = (15.0 - np.sqrt(33.0)) / 16.0
    for a in [1e-8, 1e-6, 1e-4, 1e-3, 0.01, 0.1, 0.5]:
        for b in [1.0, 5.0, 10.0, 50.0, 100.0, 500.0]:
            if a < b/2:
                s_test = np.array([a, b])
                w_test = np.array([p_opt, 1-p_opt])
                consider(s_test, w_test)
    
    p_opt2 = 0.75
    for a in [0.6, 0.7, 0.8, 1.0]:
        for b in [1.0, 2.0, 5.0]:
            if a >= b/2:
                s_test = np.array([a, b])
                w_test = np.array([p_opt2, 1-p_opt2])
                consider(s_test, w_test)
    
    # --- Differential Evolution for global search ---
    def de_optimize(n, pop_size=50, max_iter=200, tol=1e-10, mutation=(0.5, 1.0), recombination=0.8):
        """Run differential evolution in log-space."""
        def neg_obj_de(params):
            s, w = _decode(params, n)
            return -_compute_P_fast(s, w)
        
        # Bounds in log-space
        bounds = [(-20.0, 20.0)] * (2 * n)
        
        try:
            res = optimize.differential_evolution(
                neg_obj_de, bounds,
                popsize=pop_size,
                maxiter=max_iter,
                tol=tol,
                mutation=mutation,
                recombination=recombination,
                seed=int(rng.integers(0, 2**31)),
                polish=False,
                init='latinhypercube'
            )
            if res.x is not None and np.all(np.isfinite(res.x)):
                s, w = _decode(res.x, n)
                consider(s, w)
                return s, w
        except Exception:
            pass
        return None, None
    
    # DE for n=3, 4, 5, 6
    for n in [3, 4, 5, 6]:
        if time.time() - t0 > deadline - 60:
            break
        for pop, iters in [(30, 100), (50, 150)]:
            if time.time() - t0 > deadline - 30:
                break
            s_de, w_de = de_optimize(n, pop_size=pop, max_iter=iters)
    
    # --- Local refinement with Powell on best solutions ---
    def powell_refine(s0, w0, maxiter=5000):
        """Refine using Powell method in raw space."""
        n = s0.size
        def neg_obj_raw(params):
            s = np.maximum(params[:n], 0.0)
            w = np.maximum(params[n:], 0.0)
            if w.sum() < 1e-15:
                return 1.0
            w = w / w.sum()
            return -_compute_P_fast(s, w)
        
        x0 = np.concatenate([s0, w0])
        try:
            res = optimize.minimize(neg_obj_raw, x0, method='Powell',
                                    options={'maxiter': maxiter, 'xtol': 1e-12, 'ftol': 1e-14})
            if res.x is not None and np.all(np.isfinite(res.x)):
                s = np.maximum(res.x[:n], 0.0)
                w = np.maximum(res.x[n:], 0.0)
                if w.sum() > 1e-15:
                    w = w / w.sum()
                    consider(s, w)
                    return s, w
        except Exception:
            pass
        return s0, w0
    
    # Refine the best solution found so far
    if best_s is not None:
        n_best = best_s.size
        if time.time() - t0 < deadline - 30:
            s_ref, w_ref = powell_refine(best_s, best_w, maxiter=10000)
            
            # Nelder-Mead polish
            if time.time() - t0 < deadline - 15:
                def neg_obj_nm(params):
                    s = np.maximum(params[:n_best], 0.0)
                    w = np.maximum(params[n_best:], 0.0)
                    if w.sum() < 1e-15:
                        return 1.0
                    w = w / w.sum()
                    return -_compute_P_fast(s, w)
                
                x0 = np.concatenate([best_s, best_w])
                try:
                    res_nm = optimize.minimize(neg_obj_nm, x0, method='Nelder-Mead',
                                               options={'maxiter': 50000, 'xatol': 1e-14, 'fatol': 1e-15})
                    if res_nm.x is not None and np.all(np.isfinite(res_nm.x)):
                        s = np.maximum(res_nm.x[:n_best], 0.0)
                        w = np.maximum(res_nm.x[n_best:], 0.0)
                        if w.sum() > 1e-15:
                            w = w / w.sum()
                            consider(s, w)
                except Exception:
                    pass
    
    # --- Multi-start local optimization from diverse points ---
    def local_optimize(s0, w0, method='L-BFGS-B', maxiter=5000):
        """Local optimization from a starting point."""
        n = s0.size
        log_s0 = np.log(np.maximum(s0, 1e-12))
        log_w0 = np.log(np.maximum(w0, 1e-12))
        x0 = np.concatenate([log_s0, log_w0])
        bounds = [(-20.0, 20.0)] * (2 * n)
        
        try:
            res = optimize.minimize(lambda p: _negP(p, n), x0, method=method, bounds=bounds,
                                    options={'maxiter': maxiter, 'ftol': 1e-13, 'gtol': 1e-10})
            if res.x is not None and np.all(np.isfinite(res.x)):
                s, w = _decode(res.x, n)
                consider(s, w)
                return s, w
        except Exception:
            pass
        return s0, w0
    
    # Generate diverse starting points and locally optimize
    for n in [3, 4, 5]:
        for trial in range(10):
            if time.time() - t0 > deadline - 20:
                break
            if n == 3:
                s0 = np.sort(rng.uniform(1e-8, 50.0, 3))
            elif n == 4:
                s0 = np.sort(rng.uniform(1e-8, 100.0, 4))
            else:
                s0 = np.sort(rng.uniform(1e-8, 200.0, 5))
            w0 = rng.dirichlet(np.ones(n) * 0.5)
            consider(s0, w0)
            local_optimize(s0, w0, method='L-BFGS-B', maxiter=3000)
    
    # Final high-precision polish
    if best_s is not None:
        n_best = best_s.size
        if time.time() - t0 < deadline - 10:
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
                                         options={'maxiter': 100000, 'xatol': 1e-15, 'fatol': 1e-16})
                if resf.x is not None and np.all(np.isfinite(resf.x)):
                    s = np.maximum(resf.x[:n_best], 0.0)
                    w = np.maximum(resf.x[n_best:], 0.0)
                    if w.sum() > 1e-15:
                        w = w / w.sum()
                        consider(s, w)
            except Exception:
                pass
    
    # Prepare final answer
    if best_s is None:
        p_fallback = (15.0 - np.sqrt(33.0)) / 16.0
        best_s = np.array([1e-6, 1.0], dtype=np.float64)
        best_w = np.array([p_fallback, 1.0 - p_fallback], dtype=np.float64)
    
    s = np.ascontiguousarray(best_s, dtype=np.float64).ravel()
    w = np.ascontiguousarray(best_w, dtype=np.float64).ravel()
    s = np.maximum(s, 0.0)
    w = np.maximum(w, 0.0)
    w = w / w.sum()
    
    # Final sanity check
    if not (np.all(np.isfinite(s)) and np.all(np.isfinite(w))):
        p_fallback = (15.0 - np.sqrt(33.0)) / 16.0
        s = np.array([1e-6, 1.0], dtype=np.float64)
        w = np.array([p_fallback, 1.0 - p_fallback], dtype=np.float64)
    
    assert s.size == w.size >= 1
    assert np.all(np.isfinite(s)) and np.all(np.isfinite(w))
    assert np.all(s >= -1e-9) and np.all(w >= -1e-9) and w.sum() > 1e-9
    return s, w

# EVOLVE-BLOCK-END
