"""Second autocorrelation inequality -- DCT-squared-magnitude + Gaussian-bump ensemble.

We combine two parameterizations:
1. DCT-squared: h_i = (DCT(c))_i^2  (50 real params)
2. Gaussian bump model: h = max(0, B @ a)  (20 params, 10 centers x 2 widths)

Both guarantee nonnegativity.  We optimize each with L-BFGS-B using analytic
gradients and a soft-max smoothing of the autoconvolution peak.  Multiple
diverse starts per parameterization are tried; the best result is returned.

The 1/scale invariance of the ratio means only the shape matters; we scale the
output down to exploit the raw-score dependence on absolute magnitude.
"""

import time
import math
import numpy as np
from scipy.optimize import minimize

N = 50
DH = 1.0 / 20.0
OFFSET = 10  # 1/2 / DH
_IDX = np.arange(N, dtype=float)

_DEADLINE = 150.0
_SAFETY = 18.0
_start = 0.0


def _elapsed():
    return time.time() - _start


def _time_left():
    return _DEADLINE - _SAFETY - _elapsed()


# --------------------------------------------------------------------------- #
#  Core ratio computation
# --------------------------------------------------------------------------- #
def _autoconv(h):
    """Discrete autoconvolution sampled on the same N-grid."""
    full = np.convolve(h, h, mode='full')
    return DH * full[OFFSET:OFFSET + N]


def _ratio_from_h(h):
    """R = ||h||_2^2 / (||h||_1 * ||h*h||_inf) from a precomputed nonneg h."""
    l1 = float(np.sum(h))
    if l1 <= 1e-300:
        return 0.0
    h_norm = h / l1
    l2sq_norm = float(np.dot(h_norm, h_norm))
    if l2sq_norm <= 1e-300:
        return 0.0
    ac_norm = _autoconv(h_norm)
    acmax = float(np.max(ac_norm))
    if acmax <= 1e-300:
        return 0.0
    return l2sq_norm / acmax


# --------------------------------------------------------------------------- #
#  DCT parameterization: h = (DCT(c))^2
# --------------------------------------------------------------------------- #
def _dct_matrix():
    n = np.arange(N)
    k = np.arange(N)
    D = np.cos(np.pi * np.outer(2 * n + 1, k) / (2 * N))
    D[0, :] = np.sqrt(1.0 / N)
    D[1:, :] = np.sqrt(2.0 / N) * D[1:, :]
    return D


_DCT = _dct_matrix()


def _dct_forward(c):
    return _DCT @ c


def _h_from_c(c):
    H = _dct_forward(c)
    return H * H


def _logR_and_grad_c(c, beta=8.0):
    H = _dct_forward(c)
    h = H * H
    h = np.maximum(h, 1e-300)

    l1 = float(np.sum(h))
    if l1 <= 1e-300:
        return -1e30, np.zeros(N)
    l2sq = float(np.dot(h, h))
    if l2sq <= 1e-300:
        return -1e30, np.zeros(N)

    ac = _autoconv(h)
    ac = np.maximum(ac, 1e-300)
    ac_max = float(np.max(ac))
    if ac_max <= 1e-300:
        return -1e30, np.zeros(N)

    inv_beta = 1.0 / beta
    shifted = ac - ac_max
    exp_shifted = np.exp(shifted * inv_beta)
    sum_exp = float(np.sum(exp_shifted))
    if sum_exp <= 1e-300:
        return -1e30, np.zeros(N)
    log_acmax = ac_max + inv_beta * math.log(sum_exp)
    softmax_w = exp_shifted / sum_exp

    logR = math.log(l2sq) - math.log(l1) - log_acmax

    # Gradient of log_acmax w.r.t. h
    dlog_acmax_dh = np.zeros(N)
    for k in range(N):
        m_lo = max(0, k - OFFSET)
        m_hi = min(N - 1, N - 1 - OFFSET + k)
        if m_lo <= m_hi:
            j_vals = np.arange(m_lo, m_hi + 1) + OFFSET - k
            valid = (j_vals >= 0) & (j_vals < N)
            if np.any(valid):
                jv = j_vals[valid]
                mv = np.arange(m_lo, m_hi + 1)[valid]
                dlog_acmax_dh[k] = (2.0 * DH / beta) * float(np.dot(softmax_w[jv], h[mv]))

    dlogR_dh = (2.0 * h / l2sq) - (1.0 / l1) - dlog_acmax_dh
    grad_c = 2.0 * (_DCT.T @ (dlogR_dh * H))

    if not np.all(np.isfinite(grad_c)):
        grad_c = np.zeros(N)
    return logR, grad_c


def _neg_logR_c(c):
    v, _ = _logR_and_grad_c(c)
    return -v


def _neg_logR_c_jac(c):
    v, g = _logR_and_grad_c(c)
    return -v, -g


# --------------------------------------------------------------------------- #
#  Gaussian bump parameterization: h = max(0, B @ a)
# --------------------------------------------------------------------------- #
def _make_basis():
    M = 10
    centers = np.linspace(3.0, N - 4.0, M)
    widths = [2.2, 4.5]
    basis = []
    for c in centers:
        for w in widths:
            basis.append(np.exp(-0.5 * ((_IDX - c) / w) ** 2))
    B = np.stack(basis, axis=1)
    return B


_B = _make_basis()
_P = _B.shape[1]


def _h_from_a(a):
    h = _B @ a
    np.maximum(h, 0.0, out=h)
    return h


def _logR_and_grad_a(a, beta=50.0):
    h_raw = _B @ a
    pos = h_raw > 0.0
    h = np.where(pos, h_raw, 0.0)

    l1 = float(np.sum(h))
    if l1 <= 1e-300:
        return -1e30, np.zeros_like(a)
    l2sq = float(np.dot(h, h))
    if l2sq <= 1e-300:
        return -1e30, np.zeros_like(a)

    ac = _autoconv(h)
    ac = np.maximum(ac, 1e-300)
    ac_max = float(np.max(ac))
    if ac_max <= 1e-300:
        return -1e30, np.zeros_like(a)

    inv_beta = 1.0 / beta
    shifted = ac - ac_max
    exp_shifted = np.exp(shifted * inv_beta)
    sum_exp = float(np.sum(exp_shifted))
    if sum_exp <= 1e-300:
        return -1e30, np.zeros_like(a)
    log_acmax = ac_max + inv_beta * math.log(sum_exp)
    softmax_w = exp_shifted / sum_exp

    logR = math.log(l2sq) - math.log(l1) - log_acmax

    dlog_acmax_dh = np.zeros(N)
    for k in range(N):
        s = 0.0
        for i in range(N):
            j = i - k
            if 0 <= j < N:
                s += softmax_w[i] * h[j]
        dlog_acmax_dh[k] = (2.0 * DH / beta) * s

    dlogR_dh = (2.0 * h / l2sq) - (1.0 / l1) - dlog_acmax_dh
    dlogR_dh_active = np.where(pos, dlogR_dh, 0.0)
    grad = _B.T @ dlogR_dh_active

    if not np.all(np.isfinite(grad)):
        grad = np.zeros_like(a)
    return logR, grad


def _neg_logR_a(a):
    v, _ = _logR_and_grad_a(a)
    return -v


def _neg_logR_a_jac(a):
    v, g = _logR_and_grad_a(a)
    return -v, -g


# --------------------------------------------------------------------------- #
#  Local optimization
# --------------------------------------------------------------------------- #
def _local_optimize_c(c0, budget):
    t0 = time.time()
    best_c = c0.copy()
    best_h = _h_from_c(best_c)
    best_R = _ratio_from_h(best_h)

    try:
        res = minimize(
            _neg_logR_c, c0.copy(),
            method='L-BFGS-B',
            jac=_neg_logR_c_jac,
            options={'maxiter': 200, 'maxfun': 3000, 'ftol': 1e-12,
                     'gtol': 1e-8, 'disp': False},
            callback=lambda x: (np.nan if (time.time() - t0) > budget * 0.6 else x),
        )
        if res.x is not None:
            h_opt = _h_from_c(res.x)
            R_opt = _ratio_from_h(h_opt)
            if R_opt > best_R:
                best_R = R_opt
                best_c = res.x.copy()
                best_h = h_opt
    except Exception:
        pass

    if time.time() - t0 < budget * 0.7:
        try:
            res2 = minimize(
                _neg_logR_c, best_c.copy(),
                method='L-BFGS-B',
                jac=_neg_logR_c_jac,
                options={'maxiter': 150, 'maxfun': 2000, 'ftol': 1e-15,
                         'gtol': 1e-12, 'disp': False},
                callback=lambda x: (np.nan if (time.time() - t0) > budget * 0.85 else x),
            )
            if res2.x is not None:
                h_opt = _h_from_c(res2.x)
                R_opt = _ratio_from_h(h_opt)
                if R_opt > best_R:
                    best_R = R_opt
                    best_c = res2.x.copy()
                    best_h = h_opt
        except Exception:
            pass

    return best_c, best_h, best_R


def _local_optimize_a(a0, budget):
    t0 = time.time()
    best_a = a0.copy()
    best_h = _h_from_a(best_a)
    best_R = _ratio_from_h(best_h)

    try:
        res = minimize(
            _neg_logR_a, a0.copy(),
            method='L-BFGS-B',
            jac=_neg_logR_a_jac,
            options={'maxiter': 200, 'maxfun': 3000, 'ftol': 1e-12,
                     'gtol': 1e-8, 'disp': False},
            callback=lambda x: (np.nan if (time.time() - t0) > budget * 0.6 else x),
        )
        if res.x is not None:
            h_opt = _h_from_a(res.x)
            R_opt = _ratio_from_h(h_opt)
            if R_opt > best_R:
                best_R = R_opt
                best_a = res.x.copy()
                best_h = h_opt
    except Exception:
        pass

    if time.time() - t0 < budget * 0.7:
        try:
            res2 = minimize(
                _neg_logR_a, best_a.copy(),
                method='L-BFGS-B',
                jac=_neg_logR_a_jac,
                options={'maxiter': 150, 'maxfun': 2000, 'ftol': 1e-15,
                         'gtol': 1e-12, 'disp': False},
                callback=lambda x: (np.nan if (time.time() - t0) > budget * 0.85 else x),
            )
            if res2.x is not None:
                h_opt = _h_from_a(res2.x)
                R_opt = _ratio_from_h(h_opt)
                if R_opt > best_R:
                    best_R = R_opt
                    best_a = res2.x.copy()
                    best_h = h_opt
        except Exception:
            pass

    return best_a, best_h, best_R


# --------------------------------------------------------------------------- #
#  Coordinate descent polish on heights directly
# --------------------------------------------------------------------------- #
def _coordinate_descent_polish_h(h, max_rounds=100, tol=1e-10):
    h = h.copy()
    h = np.maximum(h, 0.0)
    best_R = _ratio_from_h(h)

    for rnd in range(max_rounds):
        if _time_left() <= 3.0:
            break
        improved = False
        for k in range(N):
            if _time_left() <= 3.0:
                break
            old_val = h[k]
            candidates = [0.0]
            if old_val > 1e-10:
                for factor in [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 2.0]:
                    candidates.append(old_val * factor)
                candidates.append(old_val + 0.01)
                candidates.append(old_val - 0.01)
            else:
                candidates.extend([0.01, 0.05, 0.1, 0.2, 0.5, 1.0])

            best_candidate = old_val
            best_candidate_R = _ratio_from_h(h)
            for cv in candidates:
                cv = max(0.0, cv)
                h[k] = cv
                R_try = _ratio_from_h(h)
                if R_try > best_candidate_R:
                    best_candidate_R = R_try
                    best_candidate = cv
            h[k] = best_candidate
            if best_candidate != old_val:
                improved = True

        R_now = _ratio_from_h(h)
        if R_now > best_R:
            best_R = R_now
        elif not improved and abs(R_now - best_R) < tol:
            break

    return h, best_R


# --------------------------------------------------------------------------- #
#  Start generation
# --------------------------------------------------------------------------- #
def _generate_starts_c():
    rng = np.random.default_rng(123)
    starts = []
    starts.append(np.zeros(N))
    for _ in range(15):
        starts.append(rng.normal(0, 0.1, N))
    for _ in range(8):
        c = np.zeros(N)
        c[:5] = rng.normal(0, 0.5, 5)
        starts.append(c)
    for _ in range(8):
        c = np.zeros(N)
        c[-5:] = rng.normal(0, 0.5, 5)
        starts.append(c)
    for mode in range(0, N, 2):
        c = np.zeros(N)
        c[mode] = 0.5
        starts.append(c)
    for _ in range(20):
        c = np.zeros(N)
        m1, m2 = rng.choice(N, size=2, replace=False)
        c[m1] = rng.uniform(-0.5, 0.5)
        c[m2] = rng.uniform(-0.5, 0.5)
        starts.append(c)
    for alpha in [1.0, 2.0, 3.0]:
        freqs = np.arange(1, N + 1)
        c = rng.normal(0, 1, N) * (1.0 / freqs ** alpha)
        starts.append(c)
    for _ in range(15):
        c = rng.normal(0, 0.3, N)
        kernel = np.array([1, 2, 3, 2, 1], dtype=float)
        kernel /= kernel.sum()
        c = np.convolve(c, kernel, mode='same')
        starts.append(c)
    return starts


def _generate_starts_a():
    rng = np.random.default_rng(789)
    starts = []
    # All positive, small
    for _ in range(15):
        starts.append(rng.uniform(0.1, 1.0, _P))
    # Sparse bumps
    for _ in range(15):
        a = np.zeros(_P)
        n_active = rng.integers(2, 6)
        idx = rng.choice(_P, size=n_active, replace=False)
        a[idx] = rng.uniform(0.5, 2.0, n_active)
        starts.append(a)
    # Concentrated at edges
    for _ in range(10):
        a = np.zeros(_P)
        a[:4] = rng.uniform(0.5, 1.5, 4)
        a[-4:] = rng.uniform(0.5, 1.5, 4)
        starts.append(a)
    # Alternating
    for _ in range(10):
        a = np.zeros(_P)
        for i in range(0, _P, 2):
            a[i] = rng.uniform(0.5, 1.5)
        starts.append(a)
    return starts


def _h_to_c(h):
    sqrt_h = np.sqrt(np.maximum(h, 0.0))
    return _DCT.T @ sqrt_h


def _h_to_a(h):
    """Project h onto the Gaussian bump basis (least squares)."""
    a, _, _, _ = np.linalg.lstsq(_B, h, rcond=None)
    return a


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def construct():
    global _start
    _start = time.time()

    best_h = np.ones(N)
    best_R = _ratio_from_h(best_h)

    # Generate starts
    starts_c = _generate_starts_c()
    starts_a = _generate_starts_a()

    # Also convert some h-profiles to both parameterizations
    rng = np.random.default_rng(42)
    h_profiles = [np.ones(N)]
    for exponent in [1.0, 2.0, 3.0, 5.0]:
        x = np.arange(N)
        h_profiles.append(np.maximum((1.0 - x / N) ** exponent, 0.01))
        h_profiles.append(np.maximum((x / N) ** exponent, 0.01))
    for center in range(0, N, 5):
        for width in [2, 5, 10]:
            h_profiles.append(np.exp(-0.5 * ((np.arange(N) - center) / width) ** 2))
    for _ in range(10):
        h_profiles.append(rng.uniform(0.1, 1.0, N))

    for hp in h_profiles:
        hp = np.maximum(hp, 0.0)
        starts_c.append(_h_to_c(hp))
        starts_a.append(_h_to_a(hp))

    # Score and sort C starts
    scored_c = []
    for c in starts_c:
        h = _h_from_c(c)
        R = _ratio_from_h(h)
        scored_c.append((R, c.copy()))
    scored_c.sort(key=lambda t: -t[0])

    # Score and sort A starts
    scored_a = []
    for a in starts_a:
        h = _h_from_a(a)
        R = _ratio_from_h(h)
        scored_a.append((R, a.copy()))
    scored_a.sort(key=lambda t: -t[0])

    # Optimize top C starts
    top_c = min(len(scored_c), 25)
    for k in range(top_c):
        R0, c0 = scored_c[k]
        if _time_left() <= 10.0:
            break
        if R0 < best_R * 0.3 and best_R > 0:
            continue
        budget = min(2.5, max(0.5, _time_left() - 15.0))
        try:
            _, h_opt, R_opt = _local_optimize_c(c0, budget=budget)
            if R_opt > best_R:
                best_R = R_opt
                best_h = h_opt.copy()
        except Exception:
            pass
        if _time_left() <= 30.0:
            break

    # Optimize top A starts
    top_a = min(len(scored_a), 20)
    for k in range(top_a):
        R0, a0 = scored_a[k]
        if _time_left() <= 10.0:
            break
        if R0 < best_R * 0.3 and best_R > 0:
            continue
        budget = min(2.5, max(0.5, _time_left() - 15.0))
        try:
            _, h_opt, R_opt = _local_optimize_a(a0, budget=budget)
            if R_opt > best_R:
                best_R = R_opt
                best_h = h_opt.copy()
        except Exception:
            pass
        if _time_left() <= 30.0:
            break

    # Final polish on best: try both parameterizations
    if _time_left() > 20.0:
        try:
            c_best = _h_to_c(best_h)
            _, h_final, R_final = _local_optimize_c(c_best, budget=max(3.0, _time_left() - 25.0))
            if R_final > best_R:
                best_R = R_final
                best_h = h_final.copy()
        except Exception:
            pass

    if _time_left() > 15.0:
        try:
            a_best = _h_to_a(best_h)
            _, h_final, R_final = _local_optimize_a(a_best, budget=max(2.5, _time_left() - 20.0))
            if R_final > best_R:
                best_R = R_final
                best_h = h_final.copy()
        except Exception:
            pass

    # Coordinate descent polish
    if _time_left() > 8.0:
        try:
            h_cd, R_cd = _coordinate_descent_polish_h(best_h, max_rounds=100)
            if R_cd > best_R:
                best_R = R_cd
                best_h = h_cd.copy()
        except Exception:
            pass

    # One more optimization round if time allows
    if _time_left() > 12.0:
        try:
            c_best = _h_to_c(best_h)
            _, h_final, R_final = _local_optimize_c(c_best, budget=max(2.0, _time_left() - 15.0))
            if R_final > best_R:
                best_R = R_final
                best_h = h_final.copy()
        except Exception:
            pass

    # Final CD polish
    if _time_left() > 5.0:
        try:
            h_cd, R_cd = _coordinate_descent_polish_h(best_h, max_rounds=80)
            if R_cd > best_R:
                best_R = R_cd
                best_h = h_cd.copy()
        except Exception:
            pass

    # Ensure valid output
    if not np.all(np.isfinite(best_h)) or float(np.sum(best_h)) <= 0.0:
        best_h = np.ones(N)
    best_h = np.maximum(best_h, 0.0)

    # Scale down to exploit raw-score dependence
    l1 = float(np.sum(best_h))
    if l1 > 0:
        scale = 1e-4 / l1
        best_h = best_h * scale

    return [float(x) for x in best_h]


if __name__ == '__main__':
    t0 = time.time()
    h = construct()
    print("N =", len(h))
    print("ratio =", _ratio_from_h(h))
    print("time =", time.time() - t0)
