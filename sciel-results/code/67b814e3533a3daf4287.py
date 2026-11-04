"""Second autocorrelation inequality -- DFT-MAGNITUDE parameterization.

Discrete setting: 50 nonnegative step heights on equal intervals of [-1/4, 1/4],
dh = 1/20.  The discrete autoconvolution sampled on the same grid is
    (h*h)(x_i) = dh * sum_{j+k = i + OFFSET} h_j h_k,   OFFSET = 10.

Objective (scale-invariant):  R = ||h||_2^2 / (||h||_1 * ||h*h||_inf).

Key idea: parameterize h as the squared magnitude of the DFT of a complex
vector z:  h = |DFT(z)|^2.  This guarantees h >= 0 and gives a smooth,
unconstrained optimization manifold (2N real parameters for z).  The
autoconvolution peak can be computed efficiently via FFT since
DFT(h) = DFT(|DFT(z)|^2) relates to the circular autocorrelation of h.

We optimize the complex vector z using L-BFGS-B on the log R objective,
with analytic gradients computed through the chain rule.
"""

import time
import math
import numpy as np
from scipy.optimize import minimize

N = 50
DH = 1.0 / 20.0
OFFSET = 10  # 1/2 / DH

# Grid positions in index space.
_IDX = np.arange(N, dtype=float)

_DEADLINE = 150.0
_SAFETY = 18.0


def _elapsed():
    return time.time() - _start


def _time_left():
    return _DEADLINE - _SAFETY - _elapsed()


# --------------------------------------------------------------------------- #
#  DFT-based forward map and objective
# --------------------------------------------------------------------------- #
def _h_from_z(z):
    """h = |DFT(z)|^2, normalized so that sum(h) = 1 for scale invariance.
    We return the unnormalized h; the ratio is scale-invariant anyway."""
    H = np.fft.fft(z)
    h = np.real(H * np.conj(H))
    h = np.maximum(h, 0.0)
    return h


def _autoconv(h):
    """Discrete autoconvolution sampled on the same N-grid."""
    full = np.convolve(h, h, mode='full')
    ac = DH * full[OFFSET:OFFSET + N]
    return ac


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
    # R = l2sq / (l1 * acmax)
    # l2sq = l2sq_norm * l1^2
    # acmax = acmax_norm * l1^2
    # R = (l2sq_norm * l1^2) / (l1 * acmax_norm * l1^2) = l2sq_norm / (l1 * acmax_norm)
    # Wait.
    # ||h||_2^2 = l1^2 * ||h_norm||_2^2
    # ||h||_1 = l1
    # ||h*h||_inf = l1^2 * ||h_norm*h_norm||_inf
    # R = (l1^2 * L2n) / (l1 * l1^2 * Amaxn) = L2n / (l1 * Amaxn)
    # This still has l1 in denominator.
    # Let's just compute logR and exp it, or use the normalized form directly if possible.
    # Actually, R is scale invariant.
    # R(h) = R(h_norm).
    # So R = ||h_norm||_2^2 / (||h_norm||_1 * ||h_norm*h_norm||_inf)
    # ||h_norm||_1 = 1.
    # R = l2sq_norm / (1 * acmax) = l2sq_norm / acmax
    return l2sq_norm / acmax


def _logR_from_z(z, smooth_beta=100.0):
    """Compute log R for h = |DFT(z)|^2 with soft-max smoothing on ac peak."""
    h = _h_from_z(z)
    l1 = float(np.sum(h))
    if l1 <= 1e-300:
        return -1e30
    # Normalize h to sum to 1 to prevent overflow in intermediate calculations.
    # R is scale-invariant.
    h_norm = h / l1
    l2sq_norm = float(np.dot(h_norm, h_norm))
    if l2sq_norm <= 1e-300:
        return -1e30
    ac_norm = _autoconv(h_norm)
    ac_pos = np.maximum(ac_norm, 1e-300)
    ac_max = float(np.max(ac_pos))
    if ac_max <= 1e-300:
        return -1e30
    # Soft-max via log-sum-exp for smooth max
    inv_beta = 1.0 / smooth_beta
    shifted = ac_pos - ac_max
    exp_shifted = np.exp(shifted * inv_beta)
    sum_exp = float(np.sum(exp_shifted))
    if sum_exp <= 1e-300:
        return -1e30
    log_acmax_smooth = ac_max + inv_beta * math.log(sum_exp)
    # log R = log(||h||_2^2) - log(||h||_1) - log(||h*h||_inf)
    # With normalized h (sum=1):
    # ||h||_1 = 1
    # ||h||_2^2 = l2sq_norm
    # ||h*h||_inf = ac_max (from _autoconv(h_norm))
    # Wait, _autoconv(h_norm) computes the autoconvolution of h_norm.
    # The peak of this is ac_max.
    # So log R = log(l2sq_norm) - log(1) - log(ac_max) = log(l2sq_norm) - log(ac_max)
    return math.log(l2sq_norm) - log_acmax_smooth


def _logR_and_grad_z(z, smooth_beta=100.0):
    """Analytic gradient of log R w.r.t. the real and imaginary parts of z."""
    h = _h_from_z(z)
    l1 = float(np.sum(h))
    if l1 <= 1e-300:
        return -1e30, np.zeros(2 * N)
    
    # Normalize h for stable logR calculation
    h_norm = h / l1
    l2sq_norm = float(np.dot(h_norm, h_norm))
    if l2sq_norm <= 1e-300:
        return -1e30, np.zeros(2 * N)
        
    ac_norm = _autoconv(h_norm)
    ac_pos = np.maximum(ac_norm, 1e-300)
    ac_max = float(np.max(ac_pos))
    if ac_max <= 1e-300:
        return -1e30, np.zeros(2 * N)

    inv_beta = 1.0 / smooth_beta
    shifted = ac_pos - ac_max
    exp_shifted = np.exp(shifted * inv_beta)
    sum_exp = float(np.sum(exp_shifted))
    if sum_exp <= 1e-300:
        return -1e30, np.zeros(2 * N)
    log_acmax_smooth = ac_max + inv_beta * math.log(sum_exp)
    
    # log R = log(l2sq_norm) - log_acmax_smooth
    logR = math.log(l2sq_norm) - log_acmax_smooth

    # Gradient calculation using unnormalized h derivatives
    # d logR / dh_k = 2*h_k/l2sq - 1/l1 - dlog_acmax_dh_k
    # Note: dlog_acmax_dh is computed with respect to unnormalized h?
    # log_acmax_smooth is based on ac_norm.
    # ac_norm = _autoconv(h/l1).
    # d ac_norm_i / dh_k = (1/l1^2) * d _autoconv(h)_i / dh_k = (1/l1^2) * 2*DH*h_{i-k}
    # So d log_acmax_smooth / dh_k = sum_i softmax_w[i] * (1/l1^2) * 2*DH*h_{i-k}
    # The previous code computed dlog_acmax_dh using h directly in the sum, but forgot the scaling?
    # Let's re-derive carefully.
    # Let f(h) = log(max(autoconv(h))).
    # If we use normalized h' = h/l1, then f(h) = f(h') because max(autoconv(h/l1)) = max(autoconv(h))/l1^2?
    # No. autoconv(h) is quadratic. autoconv(h/l1) = autoconv(h)/l1^2.
    # So max(autoconv(h/l1)) = max(autoconv(h))/l1^2.
    # log(max(autoconv(h/l1))) = log(max(autoconv(h))) - 2*log(l1).
    # This is NOT the same as log(max(autoconv(h))).
    # So we cannot just substitute h_norm into the gradient formula for unnormalized h.
    # We must compute the gradient of log R(h) with respect to h.
    # log R(h) = log(||h||_2^2) - log(||h||_1) - log(||h*h||_inf)
    # This function is scale invariant.
    # The gradient d log R / dh is well-defined.
    # We can compute it using the unnormalized h.
    # d log(||h||_2^2)/dh_k = 2 h_k / ||h||_2^2
    # d log(||h||_1)/dh_k = 1 / ||h||_1
    # d log(||h*h||_inf)/dh_k:
    # Let M(h) = ||h*h||_inf.
    # d M / dh_k = 2 * DH * sum_{i: i-k in [0,N)} w_i h_{i-k} where w_i are softmax weights of M(h).
    # So d log M / dh_k = (1/M) * 2 * DH * sum ...
    
    # Let's compute M and weights from unnormalized h.
    l2sq = float(np.dot(h, h))
    if l2sq <= 1e-300:
        return logR, np.zeros(2*N) # logR might be valid but grad is 0?
        
    ac_unnorm = _autoconv(h)
    ac_unnorm_pos = np.maximum(ac_unnorm, 1e-300)
    ac_unnorm_max = float(np.max(ac_unnorm_pos))
    if ac_unnorm_max <= 1e-300:
        return logR, np.zeros(2*N)
        
    # Softmax weights based on unnormalized ac
    shifted_u = ac_unnorm_pos - ac_unnorm_max
    exp_shifted_u = np.exp(shifted_u * inv_beta)
    sum_exp_u = float(np.sum(exp_shifted_u))
    if sum_exp_u <= 1e-300:
        return logR, np.zeros(2*N)
    softmax_w_u = exp_shifted_u / sum_exp_u
    
    dlog_acmax_dh = np.zeros(N)
    for k in range(N):
        s = 0.0
        for i in range(N):
            j = i - k
            if 0 <= j < N:
                s += softmax_w_u[i] * h[j]
        dlog_acmax_dh[k] = (2.0 * DH / ac_unnorm_max) * s
        # Note: d log(ac_max) / dh_k = (1/ac_max) * d ac_max / dh_k
        # d ac_max / dh_k approximated by sum w_i d ac_i / dh_k
        # d ac_i / dh_k = 2 DH h_{i-k}
        
    dlogR_dh = (2.0 * h / l2sq) - (1.0 / l1) - dlog_acmax_dh
    
    # Chain rule to z
    H = np.fft.fft(z)
    W = np.exp(-2j * np.pi * np.outer(np.arange(N), np.arange(N)) / N)
    
    v = dlogR_dh * H
    dlogR_dz = np.conj(W.T) @ v
    
    v2 = dlogR_dh * np.conj(H)
    dlogR_dconjz = W.T @ v2
    
    grad_x = np.real(dlogR_dz + dlogR_dconjz)
    grad_y = np.real(1j * (dlogR_dz - dlogR_dconjz))
    
    grad = np.concatenate([grad_x, grad_y])
    
    if not np.all(np.isfinite(grad)):
        grad = np.zeros(2 * N)
        eps = 1e-6
        for p in range(2 * N):
            zp = z.copy(); zp[p % N] += eps * (1 if p < N else 1j)
            zm = z.copy(); zm[p % N] -= eps * (1 if p < N else 1j)
            lp = _logR_from_z(zp); lm = _logR_from_z(zm)
            grad[p] = (lp - lm) / (2.0 * eps)
            
    return logR, grad


def _neg_logR_z(x):
    """Objective for L-BFGS-B: x = [Re(z), Im(z)]."""
    z = x[:N] + 1j * x[N:]
    return -_logR_from_z(z)


def _neg_logR_z_jac(x):
    z = x[:N] + 1j * x[N:]
    v, g = _logR_and_grad_z(z)
    return v, -g


# --------------------------------------------------------------------------- #
#  Optimization helpers
# --------------------------------------------------------------------------- #
def _polish_z(z0, budget):
    """Local L-BFGS-B polish on the DFT-magnitude parameterization."""
    x0 = np.concatenate([z0.real, z0.imag])
    best_x = x0.copy()
    best_v = _logR_from_z(z0)
    t0 = time.time()
    bounds = [(None, None)] * (2 * N)
    try:
        res = minimize(
            _neg_logR_z, best_x,
            method='L-BFGS-B',
            jac=_neg_logR_z_jac,
            bounds=bounds,
            options={'maxiter': 200, 'maxfun': 3000, 'ftol': 1e-12,
                     'gtol': 1e-9, 'disp': False},
            callback=lambda x: (np.nan if (time.time() - t0) > budget else x),
        )
        if res.x is not None:
            z_res = res.x[:N] + 1j * res.x[N:]
            rv = _logR_from_z(z_res)
            if rv > best_v:
                best_v = rv
                best_x = res.x.copy()
    except Exception:
        pass
    z_best = best_x[:N] + 1j * best_x[N:]
    R_best = math.exp(best_v) if best_v > -1e29 else 0.0
    return z_best, best_v, R_best


def _coordinate_descent_z(z, budget):
    """Cyclic coordinate descent on the DFT-magnitude parameters."""
    best_z = z.copy()
    best_v = _logR_from_z(best_z)
    t0 = time.time()
    step = 0.5
    for _ in range(2000):
        if (time.time() - t0) > budget:
            break
        improved = False
        for p in range(2 * N):
            idx = p % N
            is_real = (p < N)
            direction = 1.0 if is_real else 1j
            lo = best_z[idx].real - step if is_real else best_z[idx].imag - step
            hi = best_z[idx].real + step if is_real else best_z[idx].imag + step
            for _ in range(20):
                m1 = lo + (hi - lo) / 3.0
                m2 = hi - (hi - lo) / 3.0
                z1 = best_z.copy(); z2 = best_z.copy()
                if is_real:
                    z1[idx] = complex(m1, best_z[idx].imag)
                    z2[idx] = complex(m2, best_z[idx].imag)
                else:
                    z1[idx] = complex(best_z[idx].real, m1)
                    z2[idx] = complex(best_z[idx].real, m2)
                v1 = _logR_from_z(z1); v2 = _logR_from_z(z2)
                if v1 < v2:
                    hi = m2
                else:
                    lo = m1
            cand = best_z.copy()
            mid = (lo + hi) / 2.0
            if is_real:
                cand[idx] = complex(mid, best_z[idx].imag)
            else:
                cand[idx] = complex(best_z[idx].real, mid)
            vc = _logR_from_z(cand)
            if vc > best_v:
                best_v = vc
                best_z = cand
                improved = True
        if not improved:
            break
    R_best = math.exp(best_v) if best_v > -1e29 else 0.0
    return best_z, best_v, R_best


# --------------------------------------------------------------------------- #
#  Global search
# --------------------------------------------------------------------------- #
def _global_starts_z():
    """Yield diverse starting points in the complex z space."""
    rng = np.random.default_rng(20240501)
    starts = []

    # (a) Random complex vectors
    for _ in range(80):
        z = rng.normal(0, 1, N) + 1j * rng.normal(0, 1, N)
        starts.append(z)

    # (b) Sparse z (few nonzero components)
    for _ in range(40):
        z = np.zeros(N, dtype=complex)
        idx = rng.choice(N, size=rng.integers(2, 8), replace=False)
        z[idx] = rng.normal(0, 1, len(idx)) + 1j * rng.normal(0, 1, len(idx))
        starts.append(z)

    # (c) Structured: real z (so h is symmetric)
    for _ in range(30):
        z = rng.normal(0, 1, N).astype(complex)
        starts.append(z)

    # (d) Known good h shapes -> invert via IFFT(sqrt(h))
    good_h = []
    # Triangle
    tri = np.linspace(1, 0, N)
    tri = np.maximum(tri, 0)
    good_h.append(tri)
    # Inverted V
    invv = np.abs(np.linspace(-1, 1, N))
    good_h.append(invv)
    # Sinc-like
    x = np.linspace(-2, 2, N)
    sinc = np.sinc(x / np.pi)
    sinc = np.maximum(sinc, 0)
    good_h.append(sinc)
    # Multi-bump
    for centers in [(10, 25, 40), (15, 35), (10, 30, 45)]:
        h_shape = np.zeros(N)
        for c in centers:
            h_shape += np.exp(-0.5 * ((np.arange(N) - c) / 5.0) ** 2)
        good_h.append(h_shape)
    # Uniform
    good_h.append(np.ones(N))
    # Two peaks
    two = np.zeros(N)
    two[10] = 1.0; two[40] = 1.0
    good_h.append(two)

    for h_target in good_h:
        # Find z such that |FFT(z)|^2 ≈ h_target
        # h_target = |FFT(z)|^2, so FFT(z) = sqrt(h_target) * phase
        # Try with random phases
        for trial in range(3):
            phase = np.exp(2j * np.pi * rng.uniform(0, 1, N))
            H_target = np.sqrt(np.maximum(h_target, 0)) * phase
            z_candidate = np.fft.ifft(H_target)
            starts.append(z_candidate)

    # (e) Low-frequency z (smooth h)
    for _ in range(20):
        z = np.zeros(N, dtype=complex)
        k = rng.integers(1, 5)
        z[:k] = rng.normal(0, 1, k) + 1j * rng.normal(0, 1, k)
        # Symmetrize for real h
        if k > 1:
            z[N - k + 1:] = np.conj(z[1:k][::-1])
        starts.append(z)

    return starts


def construct():
    global _start
    _start = time.time()

    best_z = None
    best_log = -1e30
    best_R = 0.0
    best_h = None

    starts = _global_starts_z()

    # Score all starts, sort best-first
    scored = []
    for z in starts:
        v = _logR_from_z(z)
        scored.append((v, z))
        if v > best_log:
            best_log = v
            best_z = z.copy()
            best_R = math.exp(v) if v > -1e29 else 0.0
    scored.sort(key=lambda t: -t[0])

    # Polish top starts
    top_k = min(len(scored), 50)
    for k in range(top_k):
        v0, z = scored[k]
        if _time_left() <= 6.0:
            break
        if v0 < best_log - 2.0 and best_log > -1e29:
            continue
        budget = min(3.0, max(0.5, _time_left() - 10.0))
        try:
            z_opt, v_opt, R_opt = _polish_z(z, budget=budget)
            if v_opt > best_log:
                best_log = v_opt
                best_z = z_opt.copy()
                best_R = R_opt
        except Exception:
            pass
        if _time_left() <= 25.0:
            break

    # Coordinate descent on global best
    if best_z is not None and _time_left() > 20.0:
        try:
            z_cd, v_cd, R_cd = _coordinate_descent_z(best_z, budget=min(8.0, _time_left() - 25.0))
            if v_cd > best_log:
                best_log = v_cd
                best_z = z_cd.copy()
                best_R = R_cd
        except Exception:
            pass

    # Final polish
    if best_z is not None and _time_left() > 15.0:
        try:
            z_f, v_f, R_f = _polish_z(best_z, budget=max(1.0, _time_left() - 20.0))
            if v_f > best_log:
                best_log = v_f
                best_z = z_f.copy()
                best_R = R_f
        except Exception:
            pass

    # Direct optimization on heights as final fallback
    if best_z is not None and _time_left() > 15.0:
        h_init = _h_from_z(best_z)
        if float(np.sum(h_init)) > 0:
            try:
                def _direct_logR(h):
                    h = np.maximum(h, 0.0)
                    l2sq = float(np.dot(h, h))
                    l1 = float(np.sum(h))
                    if l1 <= 1e-300 or l2sq <= 1e-300:
                        return -1e30
                    ac = _autoconv(h)
                    ac_pos = np.maximum(ac, 1e-300)
                    ac_max = float(np.max(ac_pos))
                    if ac_max <= 1e-300:
                        return -1e30
                    inv_beta = 1.0 / 100.0
                    shifted = ac_pos - ac_max
                    exp_shifted = np.exp(shifted * inv_beta)
                    sum_exp = float(np.sum(exp_shifted))
                    if sum_exp <= 1e-300:
                        return -1e30
                    log_acmax = ac_max + inv_beta * math.log(sum_exp)
                    R = l2sq / (l1 * math.exp(log_acmax))
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
                    ac = _autoconv(h_pos)
                    ac_pos = np.maximum(ac, 1e-300)
                    ac_max = float(np.max(ac_pos))
                    if ac_max <= 1e-300:
                        return np.zeros(N)
                    inv_beta = 1.0 / 100.0
                    shifted = ac_pos - ac_max
                    exp_shifted = np.exp(shifted * inv_beta)
                    sum_exp = float(np.sum(exp_shifted))
                    if sum_exp <= 1e-300:
                        return np.zeros(N)
                    softmax_w = exp_shifted / sum_exp
                    dlog_acmax_dh = np.zeros(N)
                    for k in range(N):
                        s = 0.0
                        for i in range(N):
                            j = i - k
                            if 0 <= j < N:
                                s += softmax_w[i] * h_pos[j]
                        dlog_acmax_dh[k] = (2.0 * DH * inv_beta) * s
                    grad = (2.0 * h_pos / l2sq) - (1.0 / l1) - dlog_acmax_dh
                    grad[h <= 1e-10] = 0.0
                    return grad

                def _direct_neg_logR_jac(h):
                    return -_direct_logR(h), -_direct_grad(h)

                bounds_direct = [(0.0, None)] * N
                t0_direct = time.time()
                budget_direct = min(_time_left() - 20.0, 25.0)
                res_direct = minimize(
                    _direct_neg_logR, h_init,
                    method='L-BFGS-B',
                    jac=_direct_neg_logR_jac,
                    bounds=bounds_direct,
                    options={'maxiter': 150, 'maxfun': 2000, 'ftol': 1e-12,
                             'gtol': 1e-9, 'disp': False},
                    callback=lambda x: (np.nan if (time.time() - t0_direct) > budget_direct else x),
                )
                if res_direct.x is not None:
                    h_direct = np.maximum(res_direct.x, 0.0)
                    R_direct = _ratio_from_h(h_direct)
                    if R_direct > best_R:
                        best_R = R_direct
                        best_log = math.log(R_direct) if R_direct > 0 else -1e30
                        best_h = h_direct
            except Exception:
                pass

    # Materialize the height vector
    if best_h is not None:
        h = best_h
    elif best_z is not None:
        h = _h_from_z(best_z)
    else:
        h = np.ones(N)

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
