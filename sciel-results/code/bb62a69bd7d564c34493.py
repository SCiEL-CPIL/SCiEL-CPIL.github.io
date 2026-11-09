# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy import optimize
from scipy.optimize import minimize_scalar


def _compute_P(s, w, _cache=None):
    """Exact P[X1+X2+X3 < 2*X4] for support s, normalized weights w (strict <).
    
    P = sum_{k,l} w_k * w_l * G(2*s_l - s_k)
    where G(t) = sum_{i,j: s_i+s_j < t} w_i*w_j.
    
    Uses optional cache for pairwise sums to avoid recomputation.
    """
    n = s.size
    if n == 1:
        return 0.0  # s+s+s < 2*s => s < 0, impossible since s>=0
    
    if _cache is not None and _cache.get('s') is not None:
        if np.array_equal(_cache['s'], s):
            # Support unchanged, only weights changed
            pf_s = _cache['pf_s']
            order = _cache['order']
            n2 = n * n
            pw = (w[:, None] * w[None, :]).ravel()
            cum = np.cumsum(pw[order])
            T = (2.0 * s[None, :] - s[:, None]).ravel()
            idx = np.searchsorted(pf_s, T, side='left')
            G_vals = np.zeros(n2, dtype=np.float64)
            mask = idx > 0
            if mask.any():
                G_vals[mask] = cum[idx[mask] - 1]
            G_vals = G_vals.reshape(n, n)
            P = np.sum(w[:, None] * w[None, :] * G_vals)
            return float(P)
    
    ps = s[:, None] + s[None, :]
    pw = w[:, None] * w[None, :]
    pf = ps.ravel()
    wf = pw.ravel()
    order = np.argsort(pf, kind='mergesort')
    pf_s = pf[order]
    cum = np.cumsum(wf[order])
    # Vectorize over all (k, l) pairs
    T = 2.0 * s[None, :] - s[:, None]  # T[k, l] = 2*s_l - s_k
    idx = np.searchsorted(pf_s, T.ravel(), side='left')
    # G_values[k, l] = cum[idx-1] if idx>0 else 0
    G_vals = np.zeros(T.size, dtype=np.float64)
    mask = idx > 0
    if mask.any():
        G_vals[mask] = cum[idx[mask] - 1]
    G_vals = G_vals.reshape(n, n)
    P = np.sum(w[:, None] * w[None, :] * G_vals)
    return float(P)


def _decode(params, n):
    raw_s = params[:n]
    raw_w = params[n:2 * n]
    s = np.logaddexp(0.0, raw_s)          # softplus -> (0, inf), smooth
    mx = np.max(raw_w)
    w = np.exp(raw_w - mx)
    w = w / w.sum()                        # softmax -> positive, sums to 1
    return s, w


def _negP(params, n, _cache=None):
    s, w = _decode(params, n)
    return -_compute_P(s, w, _cache=_cache)


def _softplus_inv(s):
    s = np.maximum(s, 1e-9)
    return np.log(np.expm1(s))


def _P_two_point(p, a, b):
    """Compute P for a 2-point distribution with support {a, b} and weight p on a."""
    s_arr = np.array([a, b])
    w_arr = np.array([p, 1.0 - p])
    return _compute_P(s_arr, w_arr)


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    t0 = time.time()
    deadline = 590.0  # safety margin
    rng = np.random.default_rng(20240517)

    best_P = -1.0
    best_s = None
    best_w = None
    best_params = None
    best_n = None

    def consider(params, n):
        nonlocal best_P, best_s, best_w, best_params, best_n
        s, w = _decode(params, n)
        if not (np.all(np.isfinite(s)) and np.all(np.isfinite(w))):
            return
        P = _compute_P(s, w)
        if P > best_P + 1e-13:
            best_P = P
            best_s = s.copy()
            best_w = w.copy()
            best_params = params.copy()
            best_n = n

    # --- Phase 1: Analytical two-point solution ---
    for a in [0.0, 1e-6, 1e-4, 1e-3, 0.01, 0.1, 0.5, 1.0]:
        for b in [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 500.0, 1000.0, 5000.0]:
            if b - a < 1e-10:
                continue
            try:
                res_ps = minimize_scalar(lambda pp: -_P_two_point(pp, a, b),
                                         bounds=(1e-10, 1.0 - 1e-10), method='bounded',
                                         options={'xatol': 1e-14})
                p_opt = res_ps.x
                raw_s = np.array([_softplus_inv(max(a, 1e-12)), _softplus_inv(b)])
                raw_w = np.array([np.log(p_opt), np.log(1.0 - p_opt)])
                params = np.concatenate([raw_s, raw_w])
                consider(params, 2)
            except Exception:
                pass
    
    # --- Phase 1b: 3-point analytical scan ---
    # Try 3-point distributions with structured support
    for a in [0.0, 1e-3, 0.01, 0.1, 1.0]:
        for b in [1.0, 5.0, 10.0, 50.0, 100.0, 500.0]:
            for c in [10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0]:
                if b - a < 1e-10 or c - b < 1e-10:
                    continue
                # Optimize weights for fixed support
                def _negP3(w_params):
                    w3 = np.exp(w_params)
                    w3 = w3 / w3.sum()
                    s3 = np.array([a, b, c])
                    return -_compute_P(s3, w3)
                try:
                    res_w = minimize_scalar(_negP3, bounds=(-10, 10), method='bounded',
                                            options={'xatol': 1e-12})
                    # Try 2D weight optimization
                    from scipy.optimize import minimize
                    def _negP3_2d(w_params):
                        w3 = np.exp(w_params)
                        w3 = np.concatenate([w3, [1.0]])
                        w3 = w3 / w3.sum()
                        s3 = np.array([a, b, c])
                        return -_compute_P(s3, w3)
                    res_w2 = minimize(_negP3_2d, np.array([0.0, 0.0]), method='Nelder-Mead',
                                      options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12})
                    if res_w2.success or res_w2.fun < res_w.fun:
                        w3 = np.exp(res_w2.x)
                        w3 = np.concatenate([w3, [1.0]])
                        w3 = w3 / w3.sum()
                    else:
                        w3 = np.exp(np.array([res_w.x, 0.0]))
                        w3 = np.concatenate([w3, [1.0]])
                        w3 = w3 / w3.sum()
                    s3 = np.array([a, b, c])
                    P3 = _compute_P(s3, w3)
                    if P3 > best_P + 1e-13:
                        best_P = P3
                        best_s = s3.copy()
                        best_w = w3.copy()
                        raw_s3 = _softplus_inv(s3)
                        raw_w3 = np.log(w3)
                        best_params = np.concatenate([raw_s3, raw_w3])
                        best_n = 3
                except Exception:
                    pass
                if time.time() - t0 > deadline - 120:
                    break
            if time.time() - t0 > deadline - 120:
                break
        if time.time() - t0 > deadline - 120:
            break

    # Start optimization from the best solution
    if best_params is not None:
        # Use cache for faster evaluation
        _cache = {}
        s_best, w_best = _decode(best_params, best_n)
        _cache['s'] = s_best.copy()
        
        def _negP_cached(params, n):
            s, w = _decode(params, n)
            return -_compute_P(s, w, _cache=_cache)
        
        # Refine with Nelder-Mead
        try:
            res = optimize.minimize(_negP_cached, best_params, args=(best_n,), method='Nelder-Mead',
                                    options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10})
            if res.x is not None and np.all(np.isfinite(res.x)):
                consider(res.x, best_n)
        except Exception:
            pass

    # Now try 3-point distributions with multiple initializations
    for n_try in [3, 4, 5]:
        # Multiple initialization strategies
        inits = []
        
        # Geometric spacing
        s_init = np.logspace(np.log10(1e-4), np.log10(10.0), n_try)
        w_init = np.ones(n_try) / n_try
        raw_s = _softplus_inv(s_init)
        raw_w = np.log(w_init)
        inits.append(np.concatenate([raw_s, raw_w]))
        
        # Uniform spacing
        s_init = np.linspace(1e-3, 10.0, n_try)
        w_init = np.ones(n_try) / n_try
        raw_s = _softplus_inv(s_init)
        raw_w = np.log(w_init)
        inits.append(np.concatenate([raw_s, raw_w]))
        
        # Based on best solution, add points
        if best_params is not None:
            s_best, w_best = _decode(best_params, best_n)
            for new_s in [0.0, s_best[0]/2, (s_best[0]+s_best[-1])/2, s_best[-1]*2, s_best[-1]*10]:
                if new_s < 1e-9:
                    new_s = 1e-6
                s_new = np.sort(np.concatenate([s_best, [new_s]]))
                if len(s_new) > n_try:
                    s_new = s_new[:n_try]
                w_new = np.concatenate([w_best, [0.01]])
                if len(w_new) > n_try:
                    w_new = w_new[:n_try]
                w_new = w_new / w_new.sum()
                raw_s = _softplus_inv(s_new)
                raw_w = np.log(w_new)
                inits.append(np.concatenate([raw_s, raw_w]))
        
        for params in inits:
            if time.time() - t0 > deadline - 60:
                break
            # Powell optimization with cache
            _cache_local = {}
            s_init_dec, _ = _decode(params, n_try)
            _cache_local['s'] = s_init_dec.copy()
            
            def _negP_cached_local(params, n, _cl=_cache_local):
                s, w = _decode(params, n)
                return -_compute_P(s, w, _cache=_cl)
            
            try:
                res = optimize.minimize(_negP_cached_local, params, args=(n_try,), method='Powell',
                                        options={'maxiter': 500, 'xtol': 1e-9, 'ftol': 1e-10})
                if res.x is not None and np.all(np.isfinite(res.x)):
                    consider(res.x, n_try)
            except Exception:
                pass
            
            # Nelder-Mead polish if this initialization improved
            if best_params is not None and best_n == n_try:
                _cache_nm = {}
                s_nm, _ = _decode(best_params, n_try)
                _cache_nm['s'] = s_nm.copy()
                
                def _negP_cached_nm(params, n, _cn=_cache_nm):
                    s, w = _decode(params, n)
                    return -_compute_P(s, w, _cache=_cn)
                
                try:
                    res2 = optimize.minimize(_negP_cached_nm, best_params, args=(n_try,), method='Nelder-Mead',
                                             options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-10})
                    if res2.x is not None and np.all(np.isfinite(res2.x)):
                        consider(res2, n_try)
                except Exception:
                    pass

    # Multiple random restarts for n=3,4,5
    for attempt in range(30):
        if time.time() - t0 > deadline - 30:
            break
        n = 3 + (attempt % 3)  # cycle through 3, 4, 5
        raw_s = rng.uniform(-4.0, 5.0, size=n)
        raw_w = rng.uniform(-2.0, 2.0, size=n)
        params = np.concatenate([raw_s, raw_w])
        
        _cache_rr = {}
        s_rr, _ = _decode(params, n)
        _cache_rr['s'] = s_rr.copy()
        
        def _negP_cached_rr(params, n, _cr=_cache_rr):
            s, w = _decode(params, n)
            return -_compute_P(s, w, _cache=_cr)
        
        try:
            res = optimize.minimize(_negP_cached_rr, params, args=(n,), method='Powell',
                                    options={'maxiter': 300, 'xtol': 1e-8, 'ftol': 1e-9})
            if res.x is not None and np.all(np.isfinite(res.x)):
                consider(res.x, n)
        except Exception:
            pass

    # Final high-precision polish: L-BFGS-B then Nelder-Mead with caching
    if best_params is not None and time.time() - t0 < deadline - 20:
        _cache_final = {}
        s_final, _ = _decode(best_params, best_n)
        _cache_final['s'] = s_final.copy()
        
        def _negP_final(params, n, _cf=_cache_final):
            s, w = _decode(params, n)
            return -_compute_P(s, w, _cache=_cf)
        
        # L-BFGS-B for smooth convergence
        try:
            res_lb = optimize.minimize(_negP_final, best_params, args=(best_n,),
                                       method='L-BFGS-B',
                                       options={'maxiter': 500, 'gtol': 1e-14, 'ftol': 1e-14, 'maxls': 50})
            if res_lb.x is not None and np.all(np.isfinite(res_lb.x)):
                consider(res_lb.x, best_n)
        except Exception:
            pass
    
    # Nelder-Mead polish with very tight tolerances
    if best_params is not None and time.time() - t0 < deadline - 5:
        _cache_polish = {}
        s_polish, _ = _decode(best_params, best_n)
        _cache_polish['s'] = s_polish.copy()
        
        def _negP_polish(params, n, _cp=_cache_polish):
            s, w = _decode(params, n)
            return -_compute_P(s, w, _cache=_cp)
        
        try:
            resf = optimize.minimize(_negP_polish, best_params, args=(best_n,),
                                     method='Nelder-Mead',
                                     options={'maxiter': 20000, 'xatol': 1e-14, 'fatol': 1e-14,
                                              'adaptive': True})
            if resf.x is not None and np.all(np.isfinite(resf.x)):
                consider(resf.x, best_n)
        except Exception:
            pass

    s = np.ascontiguousarray(best_s, dtype=np.float64).ravel()
    w = np.ascontiguousarray(best_w, dtype=np.float64).ravel()
    s = np.maximum(s, 0.0)
    w = np.maximum(w, 0.0)
    w = w / w.sum()
    assert s.size == w.size >= 1
    assert np.all(np.isfinite(s)) and np.all(np.isfinite(w))
    assert np.all(s >= -1e-9) and np.all(w >= -1e-9) and w.sum() > 1e-9
    return s, w

# EVOLVE-BLOCK-END
