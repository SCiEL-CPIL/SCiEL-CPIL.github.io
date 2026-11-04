"""Second autocorrelation inequality: maximize ||h||^2_2 / (||h||_1 * ||h*h||_inf).

Discrete formulation on 50 equal intervals of [-1/4, 1/4], spacing dh = 1/20.
The continuous autoconvolution sampled at the grid is approximated by the
discrete convolution  (h*h)(x_i) = dh * sum_j h_j * h_k  over j+k = i + OFFSET,
where OFFSET = 1/2/dh = 10  (because x_j + x_k = x_i  <=>  j + k = i + 10).

The objective is scale-invariant, so we fix ||h||_2 = 1 and minimize
||h*h||_inf  (equivalently maximize  1 / (||h||_1 * ||h*h||_inf)).
"""

import time
import math
import numpy as np
from scipy.optimize import minimize

N = 50
DH = 1.0 / 20.0
OFFSET = 10  # 1/2 / DH

# Wall-clock deadline with safety margin (evaluator kills at 150 s).
_DEADLINE = 150.0
_SAFETY = 15.0
_start = time.time()


def _elapsed():
    return time.time() - _start


def _time_left():
    return _DEADLINE - _SAFETY - _elapsed()


def _autoconv(h):
    """Discrete autoconvolution sampled on the same grid (length N)."""
    h = np.asarray(h, dtype=float)
    full = np.convolve(h, h, mode='full')  # length 2N-1, index j+k
    # (h*h)(x_i) uses j+k = i + OFFSET  ->  full[i + OFFSET]
    out = full[OFFSET:OFFSET + N]
    return DH * out


def _ratio(h):
    """Return the objective value R = ||h||^2_2 / (||h||_1 * ||h*h||_inf)."""
    h = np.asarray(h, dtype=float)
    l2sq = float(np.dot(h, h))
    l1 = float(np.sum(h))
    if l1 <= 0.0 or l2sq <= 0.0:
        return 0.0
    ac = _autoconv(h)
    acmax = float(np.max(ac))
    if acmax <= 0.0:
        return 0.0
    return l2sq / (l1 * acmax)


def _ratio_neg(h):
    return -_ratio(h)


def _ratio_and_grad_smooth(h):
    """
    Computes the ratio R and the gradient of log(R) w.r.t h.
    Uses the active set approximation for the max term in the denominator.
    """
    h = np.asarray(h, dtype=float)
    l2sq = float(np.dot(h, h))
    l1 = float(np.sum(h))
    if l1 <= 0.0 or l2sq <= 0.0:
        return 0.0, np.zeros_like(h)
    
    ac = _autoconv(h)
    acmax = float(np.max(ac))
    if acmax <= 0.0:
        return 0.0, np.zeros_like(h)
    
    # Find index of max in ac
    idx_max = int(np.argmax(ac))
    
    # Corresponding index in the full convolution
    M = OFFSET + idx_max
    
    # Gradient of log(R)
    # log R = log(l2sq) - log(l1) - log(acmax)
    # grad log(l2sq) = 2*h / l2sq
    # grad log(l1) = ones / l1
    # grad log(acmax) = grad(acmax) / acmax
    # grad(acmax) = 2 * h_shifted, where h_shifted[i] = h[M-i]
    
    grad_l2 = 2.0 * h / l2sq
    grad_l1 = np.ones_like(h) / l1
    
    # Construct h_shifted
    h_shifted = np.zeros_like(h)
    i_start = max(0, M - (N - 1))
    i_end = min(N, M + 1)
    
    if i_start < i_end:
        i_range = np.arange(i_start, i_end)
        j_range = M - i_range
        h_shifted[i_range] = h[j_range]
        
    grad_ac = 2.0 * h_shifted / acmax
    
    grad_logR = grad_l2 - grad_l1 - grad_ac
    
    val = l2sq / (l1 * acmax)
    return val, grad_logR


def _normalize(h):
    s = float(np.linalg.norm(h))
    if s <= 0:
        return h
    return h / s


def _local_optimize(h0, time_budget):
    """Run L-BFGS-B on the ratio with nonnegativity; return best h found."""
    h0 = np.maximum(np.asarray(h0, dtype=float), 0.0)
    h0 = _normalize(h0)
    best_h = h0.copy()
    best_v = _ratio(h0)
    t0 = time.time()

    def neg_logR(x):
        x = np.maximum(x, 0.0)
        v, _ = _ratio_and_grad_smooth(x)
        if v <= 0.0:
            return 1e12
        return -math.log(v)

    def neg_logR_grad(x):
        x = np.maximum(x, 0.0)
        v, g = _ratio_and_grad_smooth(x)
        if v <= 0.0:
            return 1e12, np.zeros_like(x)
        return -math.log(v), -g

    bounds = [(0.0, None)] * N
    try:
        res = minimize(
            neg_logR, h0,
            method='L-BFGS-B',
            jac=neg_logR_grad,
            bounds=bounds,
            options={'maxiter': 1000, 'maxfun': 10000, 'ftol': 1e-14,
                     'gtol': 1e-10, 'disp': False},
            callback=lambda x: (np.nan if (time.time() - t0) > time_budget else x),
        )
        if res.x is not None:
            rx = np.maximum(res.x, 0.0)
            rv = _ratio(rx)
            if rv > best_v:
                best_v = rv
                best_h = rx.copy()
    except Exception:
        pass

    # Polish with Nelder-Mead
    try:
        for _ in range(2):
            if _time_left() <= 5.0:
                break
            res2 = minimize(
                _ratio_neg, best_h,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-11,
                         'adaptive': True, 'disp': False},
            )
            rx = np.maximum(res2.x, 0.0)
            rv = _ratio(rx)
            if rv > best_v:
                best_v = rv
                best_h = rx.copy()
    except Exception:
        pass

    return best_h, best_v


def _initializations():
    """Yield a set of promising starting points."""
    inits = []
    rng = np.random.default_rng(12345)
    
    # Single spikes
    for pos in range(N):
        s = np.zeros(N); s[pos] = 1.0
        inits.append(s)
        
    # Two spikes
    for a in range(0, N, 2):
        for b in range(a+2, N, 2):
            s = np.zeros(N); s[a] = 1.0; s[b] = 1.0
            inits.append(s)
            
    # Random sparse
    for _ in range(100):
        k = rng.integers(2, 15)
        idx = rng.choice(N, size=k, replace=False)
        s = np.zeros(N)
        s[idx] = 1.0
        inits.append(s)
        
    # Random dense
    for _ in range(50):
        s = rng.random(N)
        inits.append(s)
        
    # Structured: Arithmetic Progressions
    for start in range(0, 10):
        for step in range(2, 10):
            idx = np.arange(start, N, step)
            s = np.zeros(N)
            s[idx] = 1.0
            inits.append(s)
            
    return inits


def construct():
    global _start
    _start = time.time()

    best_h = None
    best_v = 0.0

    inits = _initializations()
    # Pre-evaluate and sort initializations by ratio (best first) to spend
    # optimization budget where it pays.
    scored = []
    for s in inits:
        v = _ratio(s)
        scored.append((v, s))
        if v > best_v:
            best_v = v
            best_h = s.copy()
    scored.sort(key=lambda t: -t[0])

    # Optimize the top initializations within the time budget.
    budget_per = 6.0
    for v0, s in scored:
        if _time_left() <= budget_per + 2.0:
            break
        # Only bother if this init is plausibly competitive.
        if v0 < best_v * 0.5 and best_v > 0:
            # still try a few, but skip very weak ones after the first pass
            if best_v > 20.0:
                continue
        h_opt, v_opt = _local_optimize(s, time_budget=budget_per)
        if v_opt > best_v:
            best_v = v_opt
            best_h = h_opt.copy()
        if _time_left() <= 20.0:
            break

    # Final high-precision polish on the global best if time remains.
    if best_h is not None and _time_left() > 15.0:
        try:
            h_pol, v_pol = _local_optimize(best_h, time_budget=_time_left() - 18.0)
            if v_pol > best_v:
                best_v = v_pol
                best_h = h_pol.copy()
        except Exception:
            pass

    if best_h is None:
        best_h = np.ones(N)
    best_h = np.maximum(best_h, 0.0)
    # Return as a plain sequence of floats (step heights).
    return [float(x) for x in best_h]


if __name__ == '__main__':
    # quick self-test / timing
    t0 = time.time()
    h = construct()
    print("N =", len(h))
    print("ratio =", _ratio(h))
    print("time =", time.time() - t0)
