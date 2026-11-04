"""Second autocorrelation inequality -- MODEL-AND-OPTIMIZE rewrite.

Discrete setting: 50 nonnegative step heights on equal intervals of [-1/4, 1/4],
dh = 1/20.  The discrete autoconvolution sampled on the same grid is
    (h*h)(x_i) = dh * sum_{j+k = i + OFFSET} h_j h_k,   OFFSET = 10.

Objective (scale-invariant):  R = ||h||_2^2 / (||h||_1 * ||h*h||_inf).

Instead of optimizing the 50 heights directly (non-smooth due to max(0,.) and
non-negativity), we MODEL h as a composition of smooth Gaussian bumps
    h(x) = max(0, sum_m a_m * exp(-0.5*((x-c_m)/w_m)^2))
with 3M parameters.  We fit/optimise the parameters directly against the
task objective using the analytic gradient of log R w.r.t. the parameters,
which keeps the search smooth and fast.  Multiple global starts (coarse grid
+ random) each get a local L-BFGS-B polish; the best is returned.
"""

import time
import math
import numpy as np
from scipy.optimize import minimize

N = 50
DH = 1.0 / 20.0
OFFSET = 10  # 1/2 / DH

# Grid positions (cell centers / sample positions) in index space.
_IDX = np.arange(N, dtype=float)

_DEADLINE = 150.0
_SAFETY = 18.0


def _elapsed():
    return time.time() - _start


def _time_left():
    return _DEADLINE - _SAFETY - _elapsed()


# --------------------------------------------------------------------------- #
#  Basis (model) definition
# --------------------------------------------------------------------------- #
def _make_basis():
    """Gaussian bump basis.  M centers spread over the grid, 2 widths each."""
    M = 10
    centers = np.linspace(3.0, N - 4.0, M)
    widths = [2.2, 4.5]
    basis = []
    for c in centers:
        for w in widths:
            basis.append(np.exp(-0.5 * ((_IDX - c) / w) ** 2))
    B = np.stack(basis, axis=1)  # shape (N, P)
    return B


B = _make_basis()
P = B.shape[1]  # number of parameters


def _forward(params):
    """h = max(0, B @ params)."""
    h = B @ params
    np.maximum(h, 0.0, out=h)
    return h


def _ratio_from_h(h):
    """R = ||h||_2^2 / (||h||_1 * ||h*h||_inf) from a precomputed nonneg h."""
    l2sq = float(np.dot(h, h))
    l1 = float(np.sum(h))
    if l1 <= 1e-300 or l2sq <= 1e-300:
        return 0.0
    full = np.convolve(h, h, mode='full')
    ac = DH * full[OFFSET:OFFSET + N]
    acmax = float(np.max(ac))
    if acmax <= 1e-300:
        return 0.0
    return l2sq / (l1 * acmax)


def _ratio(params):
    return _ratio_from_h(_forward(params))


def _logR(params, smooth_beta=50.0):
    h = _forward(params)
    l2sq = float(np.dot(h, h))
    l1 = float(np.sum(h))
    if l1 <= 1e-300 or l2sq <= 1e-300:
        return -1e30
    full = np.convolve(h, h, mode='full')
    ac = DH * full[OFFSET:OFFSET + N]
    # Soft-max: log-sum-exp for smooth approximation of log(max(ac))
    ac_pos = np.maximum(ac, 1e-300)
    ac_max = np.max(ac_pos)
    if ac_max <= 1e-300:
        return -1e30
    # log-sum-exp: log(sum(exp(beta*(ac - ac_max))))/beta + ac_max
    scaled = beta_safe = 1.0 / smooth_beta
    shifted = ac_pos - ac_max
    log_sum_exp = ac_max + math.log(np.sum(np.exp(shifted / scaled))) * scaled
    if log_sum_exp <= 1e-300:
        return -1e30
    R = l2sq / (l1 * math.exp(log_sum_exp))
    if R <= 1e-300:
        return -1e30
    return math.log(R)


def _logR_and_grad(params, smooth_beta=50.0):
    """Analytic gradient of log R with soft-max smoothing w.r.t. params.

    Uses the chain rule through h = max(0, B a).  The active (positive) set
    determines the Jacobian dh/da = B_active.
    
    For the soft-max: log(acmax_smooth) = logsumexp(ac/beta)/beta
    d log(acmax_smooth) / d ac_i = (1/beta) * exp(ac_i/beta - logsumexp(ac/beta))
                                 = (1/beta) * softmax(ac/beta)_i
    """
    h_raw = B @ params
    pos = h_raw > 0.0
    h = np.where(pos, h_raw, 0.0)

    l2sq = float(np.dot(h, h))
    l1 = float(np.sum(h))
    if l1 <= 1e-300 or l2sq <= 1e-300:
        return -1e30, np.zeros_like(params)

    full = np.convolve(h, h, mode='full')
    ac = DH * full[OFFSET:OFFSET + N]
    ac_pos = np.maximum(ac, 1e-300)
    ac_max = np.max(ac_pos)
    if ac_max <= 1e-300:
        return -1e30, np.zeros_like(params)

    # Soft-max computation
    inv_beta = 1.0 / smooth_beta
    shifted = ac_pos - ac_max
    exp_shifted = np.exp(shifted * inv_beta)
    sum_exp = np.sum(exp_shifted)
    if sum_exp <= 1e-300:
        return -1e30, np.zeros_like(params)
    log_sum_exp = ac_max + inv_beta * math.log(sum_exp)
    
    R = l2sq / (l1 * math.exp(log_sum_exp))
    if R <= 1e-300:
        return -1e30, np.zeros_like(params)
    logR = math.log(R)

    # d logR / d h_i:
    # logR = log l2sq - log l1 - log(acmax_smooth)
    # d log l2sq / dh_i = 2 h_i / l2sq
    # d log l1 / dh_i = 1 / l1
    # d log(acmax_smooth) / dh_i = (1/beta) * softmax(ac/beta)_i * (d ac_i / dh_i)
    # where softmax_i = exp(ac_i/beta) / sum(exp(ac_j/beta))
    
    softmax_weights = exp_shifted / sum_exp  # shape (N,)
    
    # d ac_i / dh_k = 2 * DH * h_{i-k} if 0 <= i-k < N else 0
    # For each i, d ac_i / dh is a vector. We need:
    # d log(acmax_smooth) / dh_k = sum_i softmax_weights[i] * (1/beta) * d ac_i / dh_k
    # = (1/beta) * sum_i softmax_weights[i] * 2 * DH * h_{i-k} (if valid)
    
    # Build d log(acmax_smooth) / dh as a vector of length N
    dlog_acmax_dh = np.zeros(N)
    # For each k, dlog_acmax_dh[k] = (1/beta) * 2*DH * sum_i softmax_weights[i] * h_{i-k}
    # This is (1/beta) * 2*DH * (softmax_weights * h) convolved appropriately
    # ac_i = DH * sum_j h_j h_{i-j}, so d ac_i / dh_k = 2*DH*h_{i-k} (symmetric)
    # dlog_acmax_dh[k] = (1/beta) * sum_i softmax_weights[i] * 2*DH*h_{i-k}
    
    # Efficient: for each k, sum over i where i-k is valid
    # dlog_acmax_dh[k] = (2*DH/beta) * sum_i softmax_weights[i] * h[max(0, min(N-1, i-k))]
    # But we need to be careful: h_{i-k} is only defined if 0 <= i-k < N
    
    # Use convolution: (softmax_weights * h)[k] = sum_i softmax_weights[i] * h[i-k] 
    # This is cross-correlation. Let's compute directly.
    
    # For k in 0..N-1:
    #   dlog_acmax_dh[k] = (2*DH/beta) * sum_{i: 0<=i-k<N} softmax_weights[i] * h[i-k]
    
    # Vectorized: for each offset d = i-k, i = k+d, so d ranges such that 0<=k+d<N
    # dlog_acmax_dh[k] = (2*DH/beta) * sum_{d: -k <= d <= N-1-k} softmax_weights[k+d] * h[d]
    
    # This is a convolution of softmax_weights with h, evaluated at specific points.
    # Let's just do it with a loop over the N values since N=50 is small.
    for k in range(N):
        s = 0.0
        for i in range(N):
            j = i - k
            if 0 <= j < N:
                s += softmax_weights[i] * h[j]
        dlog_acmax_dh[k] = (2.0 * DH / smooth_beta) * s

    dlogR_dh = (2.0 * h / l2sq) - (1.0 / l1) - dlog_acmax_dh

    # Chain: d logR / d a_p = sum_i dlogR_dh_i * dh_i/da_p = sum_{i in active} dlogR_dh_i * B[i,p]
    dlogR_dh_active = np.where(pos, dlogR_dh, 0.0)
    grad = B.T @ dlogR_dh_active

    # Sanity fallback: if anything non-finite, use numerical gradient.
    if not np.all(np.isfinite(grad)):
        grad = np.zeros_like(params)
        eps = 1e-6
        for p in range(P):
            pp = params.copy(); pp[p] += eps
            pm = params.copy(); pm[p] -= eps
            lp = _logR(pp); lm = _logR(pm)
            grad[p] = (lp - lm) / (2.0 * eps)
    return logR, grad


def _neg_logR(params):
    v = _logR(params)
    return -v


def _neg_logR_jac(params):
    v, g = _logR_and_grad(params)
    return v, -g


# --------------------------------------------------------------------------- #
#  Optimization
# --------------------------------------------------------------------------- #
def _polish(params0, budget):
    """Local L-BFGS-B polish of log R w.r.t. params; return best params/value."""
    best_p = np.array(params0, dtype=float)
    best_v = _logR(best_p)
    t0 = time.time()
    bounds = [(None, None)] * P  # params free (nonneg handled by max at eval)
    try:
        res = minimize(
            _neg_logR, best_p,
            method='L-BFGS-B',
            jac=_neg_logR_jac,
            bounds=bounds,
            options={'maxiter': 300, 'maxfun': 4000, 'ftol': 1e-12,
                     'gtol': 1e-9, 'disp': False},
            callback=lambda x: (np.nan if (time.time() - t0) > budget else x),
        )
        if res.x is not None:
            rv = _logR(res.x)
            if rv > best_v:
                best_v = rv
                best_p = res.x.copy()
    except Exception:
        pass
    return best_p, best_v, math.exp(best_v) if best_v > -1e29 else 0.0


def _coordinate_descent(params, budget):
    """Simple cyclic coordinate descent fallback on log R w.r.t. params."""
    best_p = np.array(params, dtype=float)
    best_v = _logR(best_p)
    t0 = time.time()
    step = 0.5
    for _ in range(4000):
        if (time.time() - t0) > budget:
            break
        improved = False
        for p in range(P):
            # golden-section / ternary search over this coordinate
            lo = best_p[p] - step
            hi = best_p[p] + step
            for _ in range(24):
                m1 = lo + (hi - lo) / 3.0
                m2 = hi - (hi - lo) / 3.0
                tm1 = best_p.copy(); tm1[p] = m1
                tm2 = best_p.copy(); tm2[p] = m2
                v1 = _logR(tm1); v2 = _logR(tm2)
                if v1 < v2:
                    hi = m2
                else:
                    lo = m1
            cand = best_p.copy()
            cand[p] = (lo + hi) / 2.0
            vc = _logR(cand)
            if vc > best_v:
                best_v = vc
                best_p = cand
                improved = True
        if not improved:
            break
    return best_p, best_v, math.exp(best_v) if best_v > -1e29 else 0.0


# --------------------------------------------------------------------------- #
#  Global search over the parameter space
# --------------------------------------------------------------------------- #
def _global_starts():
    """Yield diverse starting points in parameter space.
    
    Key insight: to minimize ||h*h||_inf relative to ||h||_2^2/||h||_1,
    we want h to have multiple peaks spread out so that the autoconvolution
    peak is reduced. Triangular or multi-bump shapes are good.
    """
    rng = np.random.default_rng(20240501)
    starts = []

    # (a) Sparse single dominant bump (positive alpha on one basis, zero else).
    for p in range(P):
        a = np.zeros(P)
        a[p] = 1.0
        starts.append(a)

    # (b) Pairs of bumps (spreads support to reduce peak overlap).
    for p1 in range(0, P, 2):
        for p2 in range(p1 + 2, P, 2):
            a = np.zeros(P)
            a[p1] = 1.0
            a[p2] = 1.0
            starts.append(a)

    # (c) Random Latin-hypercube-ish spreads.
    for _ in range(60):
        a = rng.normal(0.0, 1.0, size=P)
        starts.append(a)

    # (d) Alternating sign patterns (to create multi-peak after max(0,.)).
    for _ in range(40):
        a = rng.choice([-1.0, 1.0], size=P)
        a *= rng.uniform(0.5, 1.5)
        starts.append(a)

    # (e) Triangular / tent shapes: positive on two adjacent wide bumps
    #     with opposite signs to create a flat-top or multi-peak
    M = 10
    for i in range(M - 1):
        a = np.zeros(P)
        # Use two adjacent basis functions with specific signs
        a[2*i] = 1.0
        a[2*i + 1] = -0.5  # create a valley
        starts.append(a)
        a2 = np.zeros(P)
        a2[2*i] = 1.0
        a2[2*i + 1] = 0.5   # create a plateau
        starts.append(a2)

    # (f) Symmetric multi-peak: try patterns with 3, 4, 5 peaks
    for npeaks in [3, 4, 5]:
        for trial in range(20):
            a = np.zeros(P)
            peak_indices = np.linspace(0, P-1, npeaks).astype(int)
            for pi in peak_indices:
                a[pi] = rng.uniform(0.5, 1.5)
                # Add small negative neighbors to create valleys
                if pi > 0:
                    a[pi-1] -= rng.uniform(0.2, 0.6)
                if pi < P-1:
                    a[pi+1] -= rng.uniform(0.2, 0.6)
            starts.append(a)

    # (g) Direct height-based starts: convert known good shapes to params
    #     via least squares fit to the basis
    good_shapes = []
    # Triangle
    tri = np.linspace(1, 0, N)
    tri = np.maximum(tri, 0)
    good_shapes.append(tri)
    # Inverted V
    invv = np.abs(np.linspace(-1, 1, N))
    invv = np.maximum(invv, 0)
    good_shapes.append(invv)
    # Sinc-like
    x = np.linspace(-2, 2, N)
    sinc = np.sinc(x / np.pi)
    sinc = np.maximum(sinc, 0)
    good_shapes.append(sinc)
    # Multi-bump: 3 Gaussian bumps
    for centers in [(10, 25, 40), (15, 35), (10, 30, 50)]:
        h_shape = np.zeros(N)
        for c in centers:
            h_shape += np.exp(-0.5 * ((np.arange(N) - c) / 5.0) ** 2)
        good_shapes.append(h_shape)
    
    for h_shape in good_shapes:
        # Solve least squares: B @ a ≈ h_shape
        try:
            a, _, _, _ = np.linalg.lstsq(B, h_shape, rcond=None)
            starts.append(a)
            # Also try scaled versions
            for scale in [0.5, 2.0]:
                a2, _, _, _ = np.linalg.lstsq(B, h_shape * scale, rcond=None)
                starts.append(a2)
        except:
            pass

    return starts


def construct():
    global _start
    _start = time.time()

    # Quick sanity: seed with a couple of starts to get a baseline fast.
    best_params = None
    best_log = -1e30
    best_R = 0.0

    starts = _global_starts()

    # Score all starts cheaply, sort best-first.
    scored = []
    for a in starts:
        v = _logR(a)
        scored.append((v, a))
        if v > best_log:
            best_log = v
            best_params = a.copy()
            best_R = math.exp(v) if v > -1e29 else 0.0
    scored.sort(key=lambda t: -t[0])

    # Polish the top starts within budget.  Budget per polish shrinks as time runs out.
    top_k = min(len(scored), 60)
    for k in range(top_k):
        v0, a = scored[k]
        if _time_left() <= 6.0:
            break
        # Skip starts clearly worse than current best (log gap > 2).
        if v0 < best_log - 2.0 and best_log > -1e29:
            continue
        budget = min(4.0, max(1.0, _time_left() - 8.0))
        try:
            p_opt, v_opt, R_opt = _polish(a, budget=budget)
            if v_opt > best_log:
                best_log = v_opt
                best_params = p_opt.copy()
                best_R = R_opt
        except Exception:
            pass
        if _time_left() <= 20.0:
            break

    # Coordinate-descent fallback on the global best if it's still weak.
    if best_params is not None and _time_left() > 15.0:
        try:
            p_cd, v_cd, R_cd = _coordinate_descent(best_params, budget=min(10.0, _time_left() - 18.0))
            if v_cd > best_log:
                best_log = v_cd
                best_params = p_cd.copy()
                best_R = R_cd
        except Exception:
            pass

    # Final high-precision polish on the global best.
    if best_params is not None and _time_left() > 12.0:
        try:
            p_f, v_f, R_f = _polish(best_params, budget=max(1.0, _time_left() - 18.0))
            if v_f > best_log:
                best_log = v_f
                best_params = p_f.copy()
                best_R = R_f
        except Exception:
            pass

    # Direct optimization on the 50 heights (not through basis) for final polish
    if best_params is not None and _time_left() > 15.0:
        h_init = _forward(best_params)
        h_init = np.maximum(h_init, 0.0)
        if float(np.sum(h_init)) > 0:
            try:
                def _direct_logR(h):
                    h = np.maximum(h, 0.0)
                    l2sq = float(np.dot(h, h))
                    l1 = float(np.sum(h))
                    if l1 <= 1e-300 or l2sq <= 1e-300:
                        return -1e30
                    full = np.convolve(h, h, mode='full')
                    ac = DH * full[OFFSET:OFFSET + N]
                    ac_pos = np.maximum(ac, 1e-300)
                    ac_max = np.max(ac_pos)
                    if ac_max <= 1e-300:
                        return -1e30
                    # Soft-max for smoothness
                    inv_beta = 1.0 / 50.0
                    shifted = ac_pos - ac_max
                    exp_shifted = np.exp(shifted * inv_beta)
                    sum_exp = np.sum(exp_shifted)
                    if sum_exp <= 1e-300:
                        return -1e30
                    log_sum_exp = ac_max + inv_beta * math.log(sum_exp)
                    R = l2sq / (l1 * math.exp(log_sum_exp))
                    if R <= 1e-300:
                        return -1e30
                    return math.log(R)
                
                def _direct_neg_logR(h):
                    return -_direct_logR(h)
                
                def _direct_grad(h):
                    h_pos = np.maximum(h, 0.0)
                    l2sq = float(np.dot(h_pos, h_pos))
                    l1 = float(np.sum(h_pos))
                    if l1 <= 1e-300 or l2sq <= 1e-300:
                        return np.zeros(N)
                    full = np.convolve(h_pos, h_pos, mode='full')
                    ac = DH * full[OFFSET:OFFSET + N]
                    ac_pos = np.maximum(ac, 1e-300)
                    ac_max = np.max(ac_pos)
                    if ac_max <= 1e-300:
                        return np.zeros(N)
                    inv_beta = 1.0 / 50.0
                    shifted = ac_pos - ac_max
                    exp_shifted = np.exp(shifted * inv_beta)
                    sum_exp = np.sum(exp_shifted)
                    if sum_exp <= 1e-300:
                        return np.zeros(N)
                    softmax_weights = exp_shifted / sum_exp
                    # d log(acmax_smooth) / dh_k
                    dlog_acmax_dh = np.zeros(N)
                    for k in range(N):
                        s = 0.0
                        for i in range(N):
                            j = i - k
                            if 0 <= j < N:
                                s += softmax_weights[i] * h_pos[j]
                        dlog_acmax_dh[k] = (2.0 * DH * inv_beta) * s
                    grad = (2.0 * h_pos / l2sq) - (1.0 / l1) - dlog_acmax_dh
                    # Zero out gradient where h is zero (non-differentiable, but we clip)
                    grad[h <= 1e-10] = 0.0
                    return grad
                
                def _direct_neg_logR_jac(h):
                    return -_direct_logR(h), -_direct_grad(h)
                
                bounds_direct = [(0.0, None)] * N
                t0_direct = time.time()
                budget_direct = min(_time_left() - 20.0, 30.0)
                res_direct = minimize(
                    _direct_neg_logR, h_init,
                    method='L-BFGS-B',
                    jac=_direct_neg_logR_jac,
                    bounds=bounds_direct,
                    options={'maxiter': 200, 'maxfun': 3000, 'ftol': 1e-12,
                             'gtol': 1e-9, 'disp': False},
                    callback=lambda x: (np.nan if (time.time() - t0_direct) > budget_direct else x),
                )
                if res_direct.x is not None:
                    h_direct = np.maximum(res_direct.x, 0.0)
                    R_direct = _ratio_from_h(h_direct)
                    if R_direct > best_R:
                        best_R = R_direct
                        best_log = math.log(R_direct) if R_direct > 0 else -1e30
                        # Store the direct solution by fitting to basis
                        try:
                            a_direct, _, _, _ = np.linalg.lstsq(B, h_direct, rcond=None)
                            best_params = a_direct
                        except:
                            pass
            except Exception:
                pass

    # Materialize the height vector.
    if best_params is None:
        h = np.ones(N)
    else:
        h = _forward(best_params)
        if not np.all(np.isfinite(h)) or float(np.sum(h)) <= 0.0:
            h = np.ones(N)
    h = np.maximum(h, 0.0)
    return [float(x) for x in h]


if __name__ == '__main__':
    t0 = time.time()
    h = construct()
    print("N =", len(h))
    print("ratio =", _ratio_from_h(h))
    print("time =", time.time() - t0)
