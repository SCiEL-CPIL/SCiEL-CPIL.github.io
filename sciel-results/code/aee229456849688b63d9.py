# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy import optimize
from scipy.optimize import minimize_scalar


def _compute_P(s, w):
    """Exact P[X1+X2+X3 < 2*X4] for support s, normalized weights w (strict <).
    
    P = sum_{k,l} w_k * w_l * G(2*s_l - s_k)
    where G(t) = sum_{i,j: s_i+s_j < t} w_i*w_j.
    
    O(n^2 log n^2) via precomputed sorted pairwise sums.
    """
    n = s.size
    if n == 1:
        return 0.0
    ps = s[:, None] + s[None, :]
    pw = w[:, None] * w[None, :]
    pf = ps.ravel()
    wf = pw.ravel()
    order = np.argsort(pf, kind='mergesort')
    pf_s = pf[order]
    cum = np.cumsum(wf[order])
    T = 2.0 * s[None, :] - s[:, None]
    idx = np.searchsorted(pf_s, T.ravel(), side='left')
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
    s = np.logaddexp(0.0, raw_s)
    mx = np.max(raw_w)
    w = np.exp(raw_w - mx)
    w = w / w.sum()
    return s, w


def _negP(params, n):
    s, w = _decode(params, n)
    return -_compute_P(s, w)


def _softplus_inv(s):
    s = np.maximum(s, 1e-9)
    return np.log(np.expm1(s))


def _make_gamma_init(beta, lam, n, s_min=1e-9):
    """Create n support points and weights approximating Gamma(beta+1, lam)."""
    # Quantiles of Gamma distribution
    if beta + 1 <= 0 or lam <= 0:
        s = np.linspace(s_min, 1.0, n)
        w = np.ones(n) / n
        return s, w
    
    # Use scipy.stats.gamma quantiles
    from scipy.stats import gamma as gamma_dist
    # n equally spaced quantiles
    qs = np.linspace(1e-6, 1 - 1e-6, n)
    s = gamma_dist.ppf(qs, a=beta + 1, scale=1.0/lam)
    s = np.maximum(s, s_min)
    s = np.sort(s)
    # Weights proportional to pdf at midpoints
    midqs = np.linspace(0, 1, n + 1)
    mids = 0.5 * (s + np.concatenate([[s[0]], s[:-1]]))
    pdf_vals = gamma_dist.pdf(mids, a=beta + 1, scale=1.0/lam)
    w = pdf_vals / pdf_vals.sum()
    return s, w


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    t0 = time.time()
    deadline = 590.0
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
        if np.any(s < -1e-9) or np.any(w < -1e-9):
            return
        P = _compute_P(s, w)
        if P > best_P + 1e-14:
            best_P = P
            best_s = s.copy()
            best_w = w.copy()
            best_params = params.copy()
            best_n = n

    def elapsed():
        return time.time() - t0

    # --- Phase 1: Analytical two-point solution ---
    p_opt = (15.0 - np.sqrt(33.0)) / 16.0
    a_val = 1e-9
    b_val = 1.0
    raw_s = np.array([_softplus_inv(a_val), _softplus_inv(b_val)])
    raw_w = np.array([np.log(p_opt), np.log(1.0 - p_opt)])
    params = np.concatenate([raw_s, raw_w])
    consider(params, 2)

    # Try a few different (a, b) pairs
    for a_try, b_try in [(1e-6, 1.0), (0.01, 5.0), (0.1, 10.0), (1.0, 100.0),
                         (1e-4, 0.5), (0.05, 2.0), (0.5, 20.0)]:
        raw_s2 = np.array([_softplus_inv(max(a_try, 1e-9)), _softplus_inv(b_try)])
        raw_w2 = np.array([np.log(p_opt), np.log(1.0 - p_opt)])
        params2 = np.concatenate([raw_s2, raw_w2])
        consider(params2, 2)

    # Refine best 2-point
    if best_params is not None and elapsed() < deadline - 30:
        try:
            res = optimize.minimize(_negP, best_params, args=(2,), method='Nelder-Mead',
                                    options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12,
                                             'adaptive': True})
            if res.x is not None and np.all(np.isfinite(res.x)):
                consider(res.x, 2)
        except Exception:
            pass

    # --- Phase 2: Gamma-like initializations for n=3,4,5,6 ---
    gamma_configs = []
    for beta in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]:
        for lam in [0.5, 1.0, 2.0, 5.0, 10.0]:
            gamma_configs.append((beta, lam))

    for n_try in [3, 4, 5, 6]:
        if elapsed() > deadline - 40:
            break
        for beta, lam in gamma_configs:
            if elapsed() > deadline - 40:
                break
            try:
                s_init, w_init = _make_gamma_init(beta, lam, n_try)
                if not (np.all(np.isfinite(s_init)) and np.all(np.isfinite(w_init))):
                    continue
                raw_s = _softplus_inv(s_init)
                raw_w = np.log(np.maximum(w_init, 1e-15))
                params_i = np.concatenate([raw_s, raw_w])
                
                res = optimize.minimize(_negP, params_i, args=(n_try,), method='Powell',
                                        options={'maxiter': 500, 'xtol': 1e-10, 'ftol': 1e-11})
                if res.x is not None and np.all(np.isfinite(res.x)):
                    consider(res.x, n_try)
            except Exception:
                pass

    # --- Phase 3: Extend best solution to higher dimensions ---
    if best_params is not None and elapsed() < deadline - 35:
        s_b, w_b = _decode(best_params, best_n)
        for n_new in range(best_n + 1, min(best_n + 4, 8)):
            if elapsed() > deadline - 35:
                break
            inits = []
            # Add points at various positions
            positions = [s_b[0] * 0.5, s_b[-1] * 1.5, s_b[-1] * 3.0,
                         s_b[0] + 0.3 * (s_b[-1] - s_b[0]),
                         s_b[0] + 0.7 * (s_b[-1] - s_b[0])]
            for pos in positions:
                if pos < 1e-9:
                    pos = 1e-6
                s_new = np.sort(np.concatenate([s_b, [pos]]))
                w_new = np.concatenate([w_b * 0.85, [0.05]])
                w_new = w_new / w_new.sum()
                raw_s = _softplus_inv(s_new)
                raw_w = np.log(np.maximum(w_new, 1e-15))
                inits.append(np.concatenate([raw_s, raw_w]))
            
            for params_i in inits:
                if elapsed() > deadline - 35:
                    break
                try:
                    res = optimize.minimize(_negP, params_i, args=(n_new,), method='Powell',
                                            options={'maxiter': 400, 'xtol': 1e-10, 'ftol': 1e-11})
                    if res.x is not None and np.all(np.isfinite(res.x)):
                        consider(res.x, n_new)
                except Exception:
                    pass

    # --- Phase 4: Random restarts ---
    max_attempts = 30
    for attempt in range(max_attempts):
        if elapsed() > deadline - 25:
            break
        n = 3 + (attempt % 4)  # cycle through 3, 4, 5, 6
        raw_s = rng.uniform(-2.0, 3.0, size=n)
        raw_w = rng.uniform(-2.0, 2.0, size=n)
        params = np.concatenate([raw_s, raw_w])
        try:
            res = optimize.minimize(_negP, params, args=(n,), method='Powell',
                                    options={'maxiter': 200, 'xtol': 1e-9, 'ftol': 1e-10})
            if res.x is not None and np.all(np.isfinite(res.x)):
                consider(res.x, n)
        except Exception:
            pass

    # --- Phase 5: Final high-precision polish ---
    if best_params is not None and elapsed() < deadline - 15:
        try:
            res_lb = optimize.minimize(_negP, best_params, args=(best_n,),
                                       method='L-BFGS-B',
                                       options={'maxiter': 200, 'gtol': 1e-14, 'ftol': 1e-14, 'maxls': 50})
            if res_lb.x is not None and np.all(np.isfinite(res_lb.x)):
                consider(res_lb.x, best_n)
        except Exception:
            pass

    if best_params is not None and elapsed() < deadline - 5:
        try:
            resf = optimize.minimize(_negP, best_params, args=(best_n,),
                                     method='Nelder-Mead',
                                     options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-14,
                                              'adaptive': True})
            if resf.x is not None and np.all(np.isfinite(resf.x)):
                consider(resf.x, best_n)
        except Exception:
            pass

    # Fallback
    if best_s is None or best_w is None:
        s = np.array([0.0, 1.0], dtype=np.float64)
        w = np.array([p_opt, 1.0 - p_opt], dtype=np.float64)
        w = w / w.sum()
        return s, w

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
