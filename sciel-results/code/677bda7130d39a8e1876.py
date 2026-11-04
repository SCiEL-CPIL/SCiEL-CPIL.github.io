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
_DEADLINE = 145.0
_SAFETY = 10.0
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


def _normalize(h):
    s = float(np.linalg.norm(h))
    if s <= 0:
        return h
    return h / s


def _h_from_coeffs(c):
    """
    Convert complex coefficients c (length N) to real non-negative heights h.
    h_k = |c_k|^2.
    """
    h = np.abs(c) ** 2
    return h


def _c_from_h(h):
    """
    Convert real non-negative heights h to complex coefficients c.
    c_k = sqrt(h_k) * exp(i * phase_k).
    We initialize phases randomly or based on a simple pattern.
    """
    h = np.asarray(h, dtype=float)
    h = np.maximum(h, 0.0)
    magnitudes = np.sqrt(h)
    # Simple phase initialization: linear ramp
    phases = np.linspace(0, 2 * np.pi, N, endpoint=False)
    c = magnitudes * np.exp(1j * phases)
    return c


def _objective_complex(c):
    """
    Objective function for complex coefficients c.
    We want to maximize R(h) where h = |c|^2.
    Since the objective is scale-invariant, we can fix ||h||_2 = 1 or just work with raw h.
    However, for optimization stability, let's minimize -log(R(h)).
    
    We use a smooth approximation for the max term to get gradients.
    """
    h = _h_from_coeffs(c)
    l2sq = float(np.sum(h * h)) # Wait, h_k = |c_k|^2, so ||h||_2^2 = sum h_k^2 = sum |c_k|^4
    # Correction: The task defines h as the heights.
    # L2 norm of h: ||h||_2 = (sum h_k^2)^0.5. So ||h||_2^2 = sum h_k^2.
    # L1 norm of h: ||h||_1 = sum h_k.
    # Autoconvolution peak: max(ac).
    
    # My previous _ratio uses:
    # l2sq = np.dot(h, h) -> sum h_k^2
    # l1 = np.sum(h) -> sum h_k
    # ac = _autoconv(h)
    # R = l2sq / (l1 * max(ac))
    
    # Let's compute R directly.
    l1 = float(np.sum(h))
    if l1 <= 0.0:
        return 1e12, np.zeros_like(c)
    
    l2sq = float(np.dot(h, h))
    if l2sq <= 0.0:
        return 1e12, np.zeros_like(c)
        
    ac = _autoconv(h)
    acmax = float(np.max(ac))
    if acmax <= 0.0:
        return 1e12, np.zeros_like(c)
        
    R = l2sq / (l1 * acmax)
    if R <= 0.0:
        return 1e12, np.zeros_like(c)
        
    # We minimize -log(R)
    val = -math.log(R)
    
    # Gradient w.r.t c
    # This is tricky. We need dR/dc.
    # h_k = c_k * conj(c_k)
    # dh_k/dc_k = conj(c_k)
    # dh_k/dconj(c_k) = c_k
    
    # Let's use the chain rule.
    # R = L2 / (L1 * A)
    # log R = log L2 - log L1 - log A
    
    # grad log L2 wrt h: 2*h / L2
    # grad log L1 wrt h: ones / L1
    # grad log A wrt h: approximated by active set
    
    # grad log R wrt h:
    # g_h = 2*h/L2 - ones/L1 - grad_A
    
    # grad_A wrt h:
    # A = max(ac). Let idx be argmax.
    # ac_i = DH * sum_j h_j h_{M-i-j} (with appropriate indices)
    # grad ac_i / grad h_k = DH * (h_{M-i-k} + h_{k}) if k is in range?
    # Actually, ac is a convolution.
    # Let's reuse the logic from _ratio_and_grad_smooth but adapted.
    
    # However, computing exact gradients for complex variables is prone to error.
    # Alternative: Use a black-box optimizer like Nelder-Mead or L-BFGS-B on the real/imag parts?
    # L-BFGS-B is better if we have gradients.
    # Let's try to compute gradients numerically or analytically.
    
    # Analytic gradient w.r.t h is available in _ratio_and_grad_smooth logic.
    # Let's extract that.
    
    idx_max = int(np.argmax(ac))
    M = OFFSET + idx_max
    
    # grad A wrt h
    grad_A = np.zeros(N)
    # ac_M = DH * sum_{j=0}^{N-1} h_j h_{M-j} (indices must be valid)
    # d(ac_M)/d(h_k) = DH * (h_{M-k} + h_k) if 0 <= M-k < N else DH * h_k (if M-k out of bounds, term is 0)
    # Wait, convolution: (h*h)[M] = sum_{j} h[j] h[M-j].
    # If we change h[k], we affect terms where j=k (h[k]h[M-k]) and j=M-k (h[M-k]h[k]).
    # So d/dh[k] = h[M-k] + h[k] (with 0 if index out of bounds).
    
    k_range = np.arange(N)
    partner = M - k_range
    valid = (partner >= 0) & (partner < N)
    
    grad_A[valid] = DH * (h[partner[valid]] + h[k_range[valid]])
    # If partner is invalid, the term h[k]h[partner] doesn't exist, but does h[k] appear elsewhere?
    # No, in the sum for ac[M], h[k] only appears paired with h[M-k].
    # So if M-k is invalid, the derivative is 0?
    # Wait, if M-k < 0, then h[M-k] is 0. So the term is 0.
    # If M-k >= N, then h[M-k] is 0.
    # So grad_A[k] = DH * (h[M-k] if valid else 0) + (h[k] if valid else 0)?
    # No, the term is h[k]*h[M-k]. If M-k is invalid, the product is 0.
    # So the derivative is 0.
    
    # So:
    grad_A = np.zeros(N)
    for k in range(N):
        p = M - k
        if 0 <= p < N:
            grad_A[k] = DH * (h[p] + h[k])
            
    # Normalize by A for log gradient
    grad_log_A = grad_A / acmax
    
    grad_log_L2 = 2.0 * h / l2sq
    grad_log_L1 = np.ones(N) / l1
    
    grad_log_R_h = grad_log_L2 - grad_log_L1 - grad_log_A
    
    # Now map gradient from h to c.
    # h_k = |c_k|^2 = c_k * conj(c_k)
    # Let c_k = u_k + i v_k.
    # h_k = u_k^2 + v_k^2.
    # dh_k/du_k = 2 u_k = 2 Re(c_k)
    # dh_k/dv_k = 2 v_k = 2 Im(c_k)
    
    # grad_u = grad_log_R_h * 2 * Re(c)
    # grad_v = grad_log_R_h * 2 * Im(c)
    
    # We are minimizing -log(R). So we want gradient of -log(R).
    # grad_neg_log_R_h = -grad_log_R_h
    # grad_neg_log_R_u = -grad_log_R_h * 2 * Re(c)
    # grad_neg_log_R_v = -grad_log_R_h * 2 * Im(c)
    
    # Flatten c to real vector for scipy
    # c_flat = [Re(c_0), Im(c_0), Re(c_1), Im(c_1), ...]
    # grad_flat = [grad_u_0, grad_v_0, ...]
    
    grad_u = -grad_log_R_h * 2.0 * np.real(c)
    grad_v = -grad_log_R_h * 2.0 * np.imag(c)
    
    grad_flat = np.zeros(2 * N)
    grad_flat[0::2] = grad_u
    grad_flat[1::2] = grad_v
    
    return val, grad_flat


def _objective_complex_flat(x):
    """Wrapper for scipy minimize taking a flat real vector."""
    c = np.zeros(N, dtype=complex)
    c.real = x[0::2]
    c.imag = x[1::2]
    val, grad = _objective_complex(c)
    return val, grad


def _optimize_complex(c0, time_budget):
    """Optimize complex coefficients c."""
    best_c = c0.copy()
    best_h = _h_from_coeffs(c0)
    best_v = _ratio(best_h)
    
    x0 = np.zeros(2 * N)
    x0[0::2] = np.real(c0)
    x0[1::2] = np.imag(c0)
    
    t0 = time.time()
    
    try:
        res = minimize(
            _objective_complex_flat, x0,
            method='L-BFGS-B',
            options={'maxiter': 1000, 'maxfun': 5000, 'ftol': 1e-12,
                     'gtol': 1e-8, 'disp': False},
            callback=lambda x: (np.nan if (time.time() - t0) > time_budget else x),
        )
        if res.x is not None:
            c_new = np.zeros(N, dtype=complex)
            c_new.real = res.x[0::2]
            c_new.imag = res.x[1::2]
            h_new = _h_from_coeffs(c_new)
            v_new = _ratio(h_new)
            if v_new > best_v:
                best_v = v_new
                best_c = c_new
                best_h = h_new
    except Exception:
        pass
        
    return best_h, best_v


def _make_gaussian_lobes(centers, widths, heights):
    """Create a signal from Gaussian lobes at given centers with given widths and heights."""
    x = np.linspace(-0.25, 0.25, N)
    h = np.zeros(N)
    for c, w, ht in zip(centers, widths, heights):
        if w <= 0: continue
        h += ht * np.exp(-((x - c) ** 2) / (2 * w ** 2))
    return h


def construct():
    global _start
    _start = time.time()

    best_h = None
    best_v = 0.0
    
    rng = np.random.default_rng(42)
    
    candidates = []
    
    # 1. Generate diverse initial h profiles
    # 1a. Gaussian lobes
    for n_lobe in [1, 2, 3, 4]:
        for trial in range(10):
            if n_lobe == 1:
                centers = [0.0]
                widths = [0.02 + 0.01 * (trial % 5)]
                heights = [1.0]
            else:
                if trial == 0:
                    centers = np.linspace(-0.2, 0.2, n_lobe)
                else:
                    centers = np.sort(rng.uniform(-0.22, 0.22, n_lobe))
                widths = rng.uniform(0.01, 0.04, n_lobe)
                heights = rng.uniform(0.5, 1.5, n_lobe)
            
            h = _make_gaussian_lobes(centers, widths, heights)
            if np.sum(h) > 0:
                h = _normalize(h)
                v = _ratio(h)
                if v > 0:
                    candidates.append((v, h))
    
    # 1b. Random smooth signals
    for trial in range(50):
        h_raw = rng.random(N)
        kernel = np.array([1, 2, 3, 2, 1]) / 9.0
        h_smooth = np.convolve(h_raw, kernel, mode='same')
        h = np.maximum(h_smooth, 0.0)
        if np.sum(h) > 0:
            h = _normalize(h)
            v = _ratio(h)
            if v > 0:
                candidates.append((v, h))
                
    # 1c. Sparse spikes
    for pos in range(0, N, 5):
        s = np.zeros(N); s[pos] = 1.0
        candidates.append((_ratio(s), s))
        
    # Sort candidates
    candidates.sort(key=lambda x: -x[0])
    
    top_k = 15
    if len(candidates) > top_k:
        candidates = candidates[:top_k]
    
    if candidates:
        best_v = candidates[0][0]
        best_h = candidates[0][1].copy()
    
    # Phase 2: Optimize using complex coefficient representation
    total_time_budget = 110.0
    time_per_candidate = total_time_budget / len(candidates) if candidates else 0
    
    for v0, h0 in candidates:
        if _time_left() < time_per_candidate + 5.0:
            break
            
        t_budget = time_per_candidate
        
        # Initialize complex coefficients from h0
        c0 = _c_from_h(h0)
        
        h_opt, v_opt = _optimize_complex(c0, time_budget=t_budget)
        
        if v_opt > best_v:
            best_v = v_opt
            best_h = h_opt.copy()
            
    # Phase 3: Final Polish on the global best using complex optimization
    if best_h is not None and _time_left() > 10.0:
        try:
            c_best = _c_from_h(best_h)
            h_pol, v_pol = _optimize_complex(c_best, time_budget=_time_left() - 12.0)
            if v_pol > best_v:
                best_v = v_pol
                best_h = h_pol.copy()
        except Exception:
            pass

    if best_h is None:
        best_h = np.ones(N)
    best_h = np.maximum(best_h, 0.0)
    
    return [float(x) for x in best_h]


if __name__ == '__main__':
    # quick self-test / timing
    t0 = time.time()
    h = construct()
    print("N =", len(h))
    print("ratio =", _ratio(h))
    print("time =", time.time() - t0)
