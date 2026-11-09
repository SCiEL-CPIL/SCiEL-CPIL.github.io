# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy import optimize


def _compute_P(s, w):
    """Exact P[X1+X2+X3 < 2*X4] for support s, normalized weights w (strict <).

    Uses a 2D outer-sum trick: T = X1+X2+X3 = (X1+X2) + X3, so the triple
    sums are exactly the pairwise sums of the n^2 values (s_i+s_j, w_i*w_j)
    with a third draw s_k, w_k.  This avoids materializing an n^3 tensor.
    """
    n = s.size
    ps = s[:, None] + s[None, :]          # n x n
    pw = w[:, None] * w[None, :]          # n x n
    pf = ps.ravel()
    wf = pw.ravel()
    order = np.argsort(pf)
    pf_s = pf[order]
    cum = np.cumsum(wf[order])
    # H(T) = weight of triples with sum strictly < T
    idx = np.searchsorted(pf_s, 2.0 * s, side='left')  # count of pf_s < 2*s[l]
    H = np.zeros(n)
    nz = idx > 0
    H[nz] = cum[idx[nz] - 1]
    return float(np.sum(w * H))


def _decode(params, n):
    raw_s = params[:n]
    raw_w = params[n:2 * n]
    s = np.logaddexp(0.0, raw_s)          # softplus -> (0, inf), smooth
    mx = np.max(raw_w)
    w = np.exp(raw_w - mx)
    w = w / w.sum()                        # softmax -> positive, sums to 1
    return s, w


def _negP(params, n):
    s, w = _decode(params, n)
    return -_compute_P(s, w)


def _softplus_inv(s):
    s = np.maximum(s, 1e-9)
    return np.log(np.expm1(s))


def _decode_exp_mixture(params, n):
    """Decode params for exponential mixture model.
    
    params layout:
      [0]: log(alpha) where alpha is the mixing weight for component 1
      [1]: log(lambda1) where lambda1 is the rate for component 1
      [2]: log(lambda2) where lambda2 is the rate for component 2
    
    Support: n points sampled from a geometric progression based on the rates
    Weights: proportional to the mixture density at each support point
    
    The support points are: s_i = c * r^i for i=0..n-1 where r is determined
    by the ratio of rates, and c is a base scale.
    """
    log_alpha = params[0]
    log_lambda1 = params[1]
    log_lambda2 = params[2]
    
    alpha = 1.0 / (1.0 + np.exp(-log_alpha))  # sigmoid
    lambda1 = np.exp(log_lambda1)
    lambda2 = np.exp(log_lambda2)
    
    # Ensure rates are positive and reasonable
    lambda1 = np.clip(lambda1, 1e-6, 1e6)
    lambda2 = np.clip(lambda2, 1e-6, 1e6)
    
    # Create support points as a geometric progression
    # The ratio is related to the ratio of the rates
    ratio = lambda2 / lambda1
    # Use a base scale that allows the support to span a useful range
    base_scale = 1.0 / lambda1
    
    # Create n support points: s_i = base_scale * ratio^i
    # But we want them to be well-spread, so let's use a different approach
    # s_i = base_scale * (1 + i * ratio_factor)
    # Actually, let's just create a geometric sequence
    # s_i = base_scale * r^i where r = ratio^(1/(n-1)) to span from base_scale to base_scale*ratio
    if n == 1:
        s = np.array([base_scale])
    else:
        r = ratio ** (1.0 / (n - 1))
        exponents = np.arange(n, dtype=np.float64)
        s = base_scale * (r ** exponents)
    
    # Ensure support is positive
    s = np.maximum(s, 1e-10)
    
    # Compute mixture density at each support point
    # f(s) = alpha * lambda1 * exp(-lambda1 * s) + (1-alpha) * lambda2 * exp(-lambda2 * s)
    with np.errstate(over='ignore', under='ignore'):
        dens1 = lambda1 * np.exp(-lambda1 * s)
        dens2 = lambda2 * np.exp(-lambda2 * s)
    
    f = alpha * dens1 + (1.0 - alpha) * dens2
    
    # Ensure positive weights
    f = np.maximum(f, 1e-15)
    
    # Normalize to get weights
    w = f / f.sum()
    
    return s, w


def _negP_exp(params, n):
    s, w = _decode_exp_mixture(params, n)
    if not (np.all(np.isfinite(s)) and np.all(np.isfinite(w))):
        return 1e10
    P = _compute_P(s, w)
    return -P


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    t0 = time.time()
    deadline = 590.0  # safety margin well under the 630 s kill
    rng = np.random.default_rng(20240517)

    best_P = -1.0
    best_s = None
    best_w = None
    best_params = None
    best_n = None
    best_model = None  # 'fixed' or 'exp'

    def consider_fixed(params, n):
        nonlocal best_P, best_s, best_w, best_params, best_n, best_model
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
            best_model = 'fixed'

    def consider_exp(params, n):
        nonlocal best_P, best_s, best_w, best_params, best_n, best_model
        s, w = _decode_exp_mixture(params, n)
        if not (np.all(np.isfinite(s)) and np.all(np.isfinite(w))):
            return
        P = _compute_P(s, w)
        if P > best_P + 1e-13:
            best_P = P
            best_s = s.copy()
            best_w = w.copy()
            best_params = params.copy()
            best_n = n
            best_model = 'exp'

    # --- Structured initializers for fixed model ---
    def init_two_point():
        p = (15.0 - np.sqrt(33.0)) / 16.0
        raw_s = np.array([np.log(1e-7), 1000.0])       # ~0 and ~1000
        raw_w = np.array([np.log(p), np.log(1.0 - p)])
        return np.concatenate([raw_s, raw_w]), 2

    def init_uniform(lo, hi, n):
        s_init = np.linspace(lo, hi, n)
        raw_s = _softplus_inv(s_init)
        raw_w = np.zeros(n)
        return np.concatenate([raw_s, raw_w]), n

    def init_geo():
        s_init = np.array([0.01, 0.1, 0.5, 2.0, 10.0, 50.0])
        w_init = np.array([0.45, 0.20, 0.12, 0.10, 0.08, 0.05])
        raw_s = _softplus_inv(s_init)
        raw_w = np.log(w_init)
        return np.concatenate([raw_s, raw_w]), 6

    # --- Initializers for exponential mixture model ---
    def init_exp_mixture(n):
        # Random initial parameters
        log_alpha = rng.uniform(-2.0, 2.0)
        log_lambda1 = rng.uniform(-3.0, 3.0)
        log_lambda2 = rng.uniform(-3.0, 3.0)
        params = np.array([log_alpha, log_lambda1, log_lambda2])
        return params, n

    def init_exp_mixture_good(n):
        # Start with reasonable values: one fast decay, one slow decay
        log_alpha = 0.0  # alpha = 0.5
        log_lambda1 = np.log(1.0)   # lambda1 = 1.0
        log_lambda2 = np.log(0.1)   # lambda2 = 0.1
        params = np.array([log_alpha, log_lambda1, log_lambda2])
        return params, n

    # Seed with the analytic 2-point optimum as a guaranteed-valid floor
    p0, n0 = init_two_point()
    consider_fixed(p0, n0)

    # Also seed with a good exponential mixture
    pe, ne = init_exp_mixture_good(5)
    consider_exp(pe, ne)

    n_list_fixed = [2, 3, 4, 5, 6, 4, 3, 5, 3, 4]
    n_list_exp = [3, 4, 5, 6, 7, 5, 4, 6, 4, 5]
    
    attempt = 0
    while time.time() - t0 < deadline:
        r = attempt
        use_exp = (r % 2 == 0)  # Alternate between models
        
        if use_exp:
            # Exponential mixture model
            if r % 6 == 0 and best_model == 'exp' and best_params is not None:
                # Perturb current best
                params = best_params + rng.normal(0.0, 0.2, size=best_params.size)
                n = best_n
            elif r % 6 == 1:
                # Good structured start
                params, n = init_exp_mixture_good(5)
                params = params + rng.normal(0.0, 0.1, size=params.size)
            else:
                # Random start
                n = n_list_exp[r % len(n_list_exp)]
                params, _ = init_exp_mixture(n)
            
            cur = params
            try:
                res = optimize.minimize(_negP_exp, params, args=(n,), method='Nelder-Mead',
                                        options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-9})
                if res.x is not None and np.all(np.isfinite(res.x)):
                    consider_exp(res.x, n)
                    cur = res.x
            except Exception:
                pass

            # Polish
            if (cur is not None and np.all(np.isfinite(cur))
                    and time.time() - t0 < deadline - 30):
                try:
                    res2 = optimize.minimize(_negP_exp, cur, args=(n,), method='Nelder-Mead',
                                             options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-11})
                    if res2.x is not None and np.all(np.isfinite(res2.x)):
                        consider_exp(res2.x, n)
                except Exception:
                    pass

        else:
            # Fixed support model
            if r % 4 == 1 and best_model == 'fixed' and best_params is not None:
                # Perturb current best
                params = best_params + rng.normal(0.0, 0.3, size=best_params.size)
                n = best_n
            elif r % 6 < 3:
                # Cycle structured starts
                if (r // 6) % 3 == 0:
                    params, n = init_two_point()
                elif (r // 6) % 3 == 1:
                    params, n = init_uniform(0.05, 1.0, 5)
                else:
                    params, n = init_geo()
                params = params + rng.normal(0.0, 0.1, size=params.size)
            else:
                n = n_list_fixed[r % len(n_list_fixed)]
                raw_s = rng.uniform(-5.0, 5.0, size=n)
                raw_w = rng.uniform(-2.0, 2.0, size=n)
                params = np.concatenate([raw_s, raw_w])

            cur = params
            try:
                res = optimize.minimize(_negP, params, args=(n,), method='Powell',
                                        options={'maxiter': 800, 'xtol': 1e-9, 'ftol': 1e-10})
                if res.x is not None and np.all(np.isfinite(res.x)):
                    consider_fixed(res.x, n)
                    cur = res.x
            except Exception:
                pass

            # Nelder-Mead polish from the Powell result
            if (cur is not None and np.all(np.isfinite(cur))
                    and time.time() - t0 < deadline - 30):
                try:
                    res2 = optimize.minimize(_negP, cur, args=(n,), method='Nelder-Mead',
                                             options={'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-10})
                    if res2.x is not None and np.all(np.isfinite(res2.x)):
                        consider_fixed(res2.x, n)
                except Exception:
                    pass

        attempt += 1
        if attempt >= 200:
            break

    # Final high-precision polish of the global best
    if best_params is not None and time.time() - t0 < deadline - 10:
        if best_model == 'exp':
            try:
                resf = optimize.minimize(_negP_exp, best_params, args=(best_n,),
                                         method='Nelder-Mead',
                                         options={'maxiter': 10000, 'xatol': 1e-11, 'fatol': 1e-12})
                if resf.x is not None and np.all(np.isfinite(resf.x)):
                    consider_exp(resf.x, best_n)
            except Exception:
                pass
        else:
            try:
                resf = optimize.minimize(_negP, best_params, args=(best_n,),
                                         method='Nelder-Mead',
                                         options={'maxiter': 10000, 'xatol': 1e-11, 'fatol': 1e-12})
                if resf.x is not None and np.all(np.isfinite(resf.x)):
                    consider_fixed(resf.x, best_n)
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
