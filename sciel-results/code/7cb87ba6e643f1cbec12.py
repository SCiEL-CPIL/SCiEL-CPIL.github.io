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
from scipy.optimize import minimize, LinearConstraint

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


def _soft_logR_and_grad(h, beta=50.0):
    """
    Smooth surrogate: log R_smooth = log(l2sq) - log(l1) - logsumexp(ac, beta)
    where logsumexp(ac, beta) = (1/beta)*log(sum(exp(beta*ac))) approximates log(max(ac)).
    """
    h = np.asarray(h, dtype=float)
    l2sq = float(np.dot(h, h))
    l1 = float(np.sum(h))
    if l1 <= 0.0 or l2sq <= 0.0:
        return 1e12, np.zeros_like(h)

    ac = _autoconv(h)
    ac_max = float(np.max(ac))
    ac_shifted = ac - ac_max
    exp_ac = np.exp(beta * ac_shifted)
    sum_exp = float(np.sum(exp_ac))
    if sum_exp <= 0.0:
        return 1e12, np.zeros_like(h)
    logsumexp = ac_max + (1.0 / beta) * math.log(sum_exp)

    if logsumexp <= 0.0:
        return 1e12, np.zeros_like(h)

    # Softmax weights
    w = exp_ac / sum_exp

    # Gradient of logsumexp wrt h:
    # d(logsumexp)/d(h_j) = sum_i w_i * d(ac_i)/d(h_j)
    # d(ac_i)/d(h_j) = 2*DH*h[i + OFFSET - j] if 0 <= i + OFFSET - j < N else 0
    d_logsumexp_dh = np.zeros(N)
    for j in range(N):
        s = 0.0
        for i in range(N):
            k = i + OFFSET - j
            if 0 <= k < N:
                s += w[i] * h[k]
        d_logsumexp_dh[j] = 2.0 * DH * s

    g = np.zeros(N)
    g += 2.0 * h / l2sq
    g -= 1.0 / l1
    g -= d_logsumexp_dh / logsumexp

    logR = math.log(l2sq) - math.log(l1) - logsumexp
    return logR, g


def _normalize(h):
    s = float(np.linalg.norm(h))
    if s <= 0:
        return h
    return h / s


def _local_optimize_trust(h0, time_budget):
    """Run trust-region optimization on the ratio with nonnegativity; return best h found."""
    h0 = np.maximum(np.asarray(h0, dtype=float), 0.0)
    h0 = _normalize(h0)
    best_h = h0.copy()
    best_v = _ratio(h0)
    t0 = time.time()

    # Use a smooth surrogate: maximize log R = log l2sq - log l1 - logsumexp(ac, beta).
    # We optimize -log R via analytical gradient with trust-constr, nonnegativity.
    def neg_logR(x):
        x = np.maximum(x, 0.0)
        v = _ratio(x)
        if v <= 0.0:
            return 1e12
        return -math.log(v)

    def neg_logR_grad(x):
        x = np.maximum(x, 0.0)
        logR, g = _soft_logR_and_grad(x, beta=50.0)
        if logR > 1e11:
            return 1e12, np.zeros_like(x)
        return -logR, -g

    # Use trust-constr with analytic gradient
    bounds = [(0.0, None)] * N
    try:
        res = minimize(
            neg_logR, h0,
            method='trust-constr',
            jac=neg_logR_grad,
            bounds=bounds,
            options={'maxiter': 200, 'gtol': 1e-10, 'disp': False,
                     'verbose': 0, 'barrier_tol': 1e-10},
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

    # A couple of Nelder-Mead polish passes on the best so far.
    try:
        for _ in range(3):
            if _time_left() <= 5.0:
                break
            res2 = minimize(
                _ratio_neg, best_h,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12,
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


def _local_optimize_lbfgsb(h0, time_budget):
    """Run L-BFGS-B on the ratio with nonnegativity; return best h found."""
    h0 = np.maximum(np.asarray(h0, dtype=float), 0.0)
    h0 = _normalize(h0)
    best_h = h0.copy()
    best_v = _ratio(h0)
    t0 = time.time()

    def neg_logR(x):
        x = np.maximum(x, 0.0)
        v = _ratio(x)
        if v <= 0.0:
            return 1e12
        return -math.log(v)

    def neg_logR_grad(x):
        x = np.maximum(x, 0.0)
        logR, g = _soft_logR_and_grad(x, beta=50.0)
        if logR > 1e11:
            return 1e12, np.zeros_like(x)
        return -logR, -g

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

    # A couple of Nelder-Mead polish passes on the best so far.
    try:
        for _ in range(3):
            if _time_left() <= 5.0:
                break
            res2 = minimize(
                _ratio_neg, best_h,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12,
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
    # Single spike at center
    for pos in [0, N // 2, N - 1, N // 4, 3 * N // 4]:
        s = np.zeros(N); s[pos] = 1.0
        inits.append(s)
    # Two spikes (spread to reduce autoconv peak overlap)
    for a in range(0, N, 10):
        for b in range(a + 10, N, 10):
            s = np.zeros(N); s[a] = 1.0; s[b] = 1.0
            inits.append(s)
    # Three spikes
    for a in range(0, N, 15):
        for b in range(a + 15, N, 15):
            for c in range(b + 15, N, 15):
                s = np.zeros(N); s[a] = 1.0; s[b] = 1.0; s[c] = 1.0
                inits.append(s)
    # Random sparse profiles
    rng = np.random.default_rng(12345)
    for _ in range(40):
        k = rng.integers(1, 12)
        idx = rng.choice(N, size=k, replace=False)
        s = np.zeros(N)
        s[idx] = 1.0
        inits.append(s)
    # Random dense (soft) profiles
    for _ in range(40):
        s = rng.random(N) ** 3.0  # sparse-ish
        inits.append(s)
    # Gaussian-ish bumps
    for center in range(0, N, 7):
        x = np.arange(N)
        s = np.exp(-0.5 * ((x - center) / 3.0) ** 2)
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
    # Use trust-constr for the first few, then L-BFGS-B for the rest.
    budget_per = 4.0
    for idx, (v0, s) in enumerate(scored):
        if _time_left() <= budget_per + 2.0:
            break
        # Only bother if this init is plausibly competitive.
        if v0 < best_v * 0.5 and best_v > 0:
            if best_v > 20.0:
                continue
        # Use trust-constr for the top 5, L-BFGS-B for the rest
        if idx < 5:
            h_opt, v_opt = _local_optimize_trust(s, time_budget=budget_per)
        else:
            h_opt, v_opt = _local_optimize_lbfgsb(s, time_budget=budget_per)
        if v_opt > best_v:
            best_v = v_opt
            best_h = h_opt.copy()
        if _time_left() <= 20.0:
            break

    # Final high-precision polish on the global best if time remains.
    if best_h is not None and _time_left() > 15.0:
        try:
            h_pol, v_pol = _local_optimize_trust(best_h, time_budget=_time_left() - 18.0)
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
