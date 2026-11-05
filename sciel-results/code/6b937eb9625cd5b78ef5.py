"""Combined program: DCT spectral shaping + Golay complementary pair + windowed Zadoff-Chu +
full-dimensional coordinate descent + smooth-max refinement.

Strategy:
- Primary: DCT spectral shaping with L-BFGS-B on smooth max surrogate (multiple restarts).
- Secondary: Golay complementary pair (two chirps with different rates) where the sum of
  individual autocorrelations cancels out off-peak values.
- Tertiary: Windowed Zadoff-Chu sequences with global optimization.
- Fourth: Full-dimensional coordinate descent with smooth logsumexp surrogate to escape
  Fourier subspace restriction, allowing high-frequency corrections.
- Refine best candidates with L-BFGS-B on DFT basis and Nelder-Mead polish.
- Deadline guard ensures we return before 150s wall-clock.
"""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.fft import dct as scipy_dct, idct as scipy_idct

N = 400
_t0 = time.time()
_DEADLINE = 135.0  # safety margin under 150s wall-clock
_K_DCT = 40
_K_DFT = 20

# Precompute common arrays
_i_arr = np.arange(N, dtype=float)

def _value(h):
    """Compute the task value (lower is better)."""
    h = np.asarray(h, dtype=float)
    s = float(np.sum(h))
    if s == 0.0 or not np.all(np.isfinite(h)):
        return np.inf
    conv = np.convolve(h, h)
    m = float(np.max(conv))
    return abs(2.0 * N * m / (s * s))

def _fft_convolve(h):
    """Fast autoconvolution via FFT."""
    n = len(h)
    size = 2 * n
    H = np.fft.rfft(h, size)
    conv = np.fft.irfft(H * H, size)[:2 * n - 1]
    return conv

def _convolve_incremental(h, i, delta):
    """
    Compute the change in autoconvolution when h[i] changes by delta.
    Returns the new convolution (not the delta).
    Full conv = old_conv + 2*delta*h shifted + delta^2 at lag i
    Actually: conv_new[k] = sum_j h_new[j]*h_new[j+k]
    The change from modifying h[i] by delta:
    conv_new[k] = conv_old[k] + delta*(h[i-k] + h[i+k]) + delta^2*(indicator)
    where we need to be careful with bounds.
    
    More precisely, for full convolution of length 2N-1:
    conv[k] = sum_j h[j]*h[j+k] for k in [-(N-1), N-1]
    (using numpy convention: conv[k] = sum_j h[j]*h[j+k-N+1]... let me think)
    
    numpy convolve(h,h)[k] = sum_j h[j]*h[k-j] for j where both indices valid.
    So conv[k] = sum_{j=max(0,k-N+1)}^{min(N-1,k)} h[j]*h[k-j]
    
    If we change h[i] -> h[i]+delta, then:
    new_conv[k] = old_conv[k] + delta * (sum of terms involving h[i])
    Terms involving h[i]: h[i]*h[k-i] and h[k-i]*h[i] (if k-i != i)
    So: new_conv[k] = old_conv[k] + delta*(h[k-i] + h[k-i]) if k-i != i
                     = old_conv[k] + delta*(2*h[k-i]) if k-i != i and 0<=k-i<N
                     = old_conv[k] + delta*(h[i] + h[i]) + delta^2 if k-i == i (i.e., k=2i)
    
    Wait, more carefully:
    old_conv[k] = sum_j h[j]*h[k-j]
    new_conv[k] = sum_j h_new[j]*h_new[k-j]
    where h_new[j] = h[j] + delta*(j==i)
    
    new_conv[k] = sum_j (h[j] + delta*I(j=i))*(h[k-j] + delta*I(k-j=i))
    = sum_j h[j]*h[k-j] 
      + delta * sum_j I(j=i)*h[k-j]
      + delta * sum_j h[j]*I(k-j=i)
      + delta^2 * sum_j I(j=i)*I(k-j=i)
    = old_conv[k]
      + delta * h[k-i]  (if 0 <= k-i < N)
      + delta * h[k-i]  (if 0 <= k-i < N)  [since I(k-j=i) means j=k-i]
      + delta^2 * I(0<=i<N and 0<=k-i<N and i==k-i)
    = old_conv[k] + 2*delta*h[k-i] + delta^2*I(k==2i)
    (where h[k-i] is 0 if k-i out of range)
    """
    n = len(h)
    conv = _fft_convolve(h)
    # Compute the delta contribution
    # For each lag k in [0, 2n-2]:
    #   delta_conv[k] = 2*delta*h[k-i] + delta^2*(k==2i)
    # where h[k-i] is 0 if k-i < 0 or k-i >= n
    k_idx = np.arange(2 * n - 1)
    j_idx = k_idx - i
    # h[k-i] is valid when 0 <= k-i < n
    valid = (j_idx >= 0) & (j_idx < n)
    h_vals = np.zeros(2 * n - 1)
    h_vals[valid] = h[j_idx[valid]]
    delta_conv = 2.0 * delta * h_vals
    if 0 <= 2 * i < 2 * n - 1:
        delta_conv[2 * i] += delta * delta
    return conv + delta_conv

def _smooth_max_value(h, beta=80.0):
    """Smooth approximation of max of autoconvolution using log-sum-exp."""
    h = np.asarray(h, dtype=float)
    s = float(np.sum(h))
    if s == 0.0 or not np.all(np.isfinite(h)):
        return np.inf
    conv = _fft_convolve(h)
    m = np.max(conv)
    scaled = beta * (conv - m)
    scaled = np.clip(scaled, -100, 100)
    log_sum_exp = m + np.log(np.sum(np.exp(scaled))) / beta
    return abs(2.0 * N * log_sum_exp / (s * s))

def _dct_basis_matrix(K, N):
    """Build DCT-II basis matrix of shape (N, K) for first K modes."""
    n = np.arange(N)
    k = np.arange(K)
    angles = np.pi * (2 * n[:, None] + 1) * k[None, :] / (2 * N)
    basis = np.cos(angles)
    norms = np.sqrt(np.sum(basis ** 2, axis=0))
    norms[norms == 0] = 1.0
    basis = basis / norms
    return basis

_DCT_BASIS = _dct_basis_matrix(_K_DCT, N)
_DCT_BASIS_T = _DCT_BASIS.T

def _dct_h(coeffs):
    """Generate h from DCT coefficients."""
    return _DCT_BASIS @ coeffs

def _dct_smooth_obj(coeffs, beta=80.0):
    """Smooth objective for DCT coefficients."""
    h = _dct_h(coeffs)
    return _smooth_max_value(h, beta)

def _dct_obj(coeffs):
    """Exact objective for DCT coefficients."""
    h = _dct_h(coeffs)
    return _value(h)

# Low-frequency DFT basis for refinement
_n_idx = np.arange(N)
_basis = np.zeros((2 * _K_DFT, N))
for k in range(1, _K_DFT + 1):
    angle = 2.0 * np.pi * k * _n_idx / N
    _basis[2 * (k - 1), :] = np.cos(angle)
    _basis[2 * (k - 1) + 1, :] = np.sin(angle)
_BASIS_T = _basis.T

def _dft_h(coeffs, base_val):
    """Generate h from DFT coefficients and a base value."""
    h = base_val + _BASIS_T @ coeffs
    return h

def _dft_obj(coeffs_base):
    """Objective for DFT parameters: [coeffs..., base]."""
    coeffs = coeffs_base[:-1]
    base_val = coeffs_base[-1]
    h = _dft_h(coeffs, base_val)
    return _value(h)

def _dft_smooth_obj(coeffs_base, beta=80.0):
    """Smooth objective for DFT parameters."""
    coeffs = coeffs_base[:-1]
    base_val = coeffs_base[-1]
    h = _dft_h(coeffs, base_val)
    return _smooth_max_value(h, beta)

def _golay_complementary_h(params):
    """
    Golay complementary pair: h = base + amp1*cos(quad_phase1) + amp2*cos(quad_phase2)
    Two chirps with different rates can have complementary autocorrelation properties.
    params: [phase1, rate1, amp1, phase2, rate2, amp2, base]
    """
    phase1, rate1, amp1, phase2, rate2, amp2, base = params
    i = _i_arr
    ang1 = 2.0 * np.pi * rate1 * i * (i + 1) / N + phase1
    ang2 = 2.0 * np.pi * rate2 * i * (i + 1) / N + phase2
    h = base + amp1 * np.cos(ang1) + amp2 * np.cos(ang2)
    return h

def _golay_obj(params):
    """Objective for Golay complementary parameters."""
    h = _golay_complementary_h(params)
    return _value(h)

def _golay_smooth_obj(params, beta=80.0):
    """Smooth objective for Golay complementary parameters."""
    h = _golay_complementary_h(params)
    return _smooth_max_value(h, beta)

def _windowed_zadoff_chu_h(params):
    """Windowed Zadoff-Chu with smooth window to suppress sidelobes."""
    phase, rate, amp, base = params[0], params[1], params[2], params[3]
    win_width = params[4] if len(params) > 4 else 1.0
    win_center = params[5] if len(params) > 5 else 0.5
    win_type = params[6] if len(params) > 6 else 0.0
    
    i = _i_arr
    ang = 2.0 * np.pi * rate * i * (i + 1) / N + phase
    rel_pos = i / N
    
    if win_type < 0.5:
        x = (rel_pos - win_center) / (0.5 * win_width)
        window = np.where(np.abs(x) <= 1.0, 0.5 * (1.0 + np.cos(np.pi * x)), 0.0)
    else:
        sigma = win_width / 4.0
        window = np.exp(-0.5 * ((rel_pos - win_center) / sigma) ** 2)
    
    h = base + amp * window * np.cos(ang)
    return h

def _windowed_zc_obj(params):
    """Objective for windowed Zadoff-Chu parameters."""
    h = _windowed_zadoff_chu_h(params)
    return _value(h)

def _windowed_zc_smooth_obj(params, beta=80.0):
    """Smooth objective for windowed Zadoff-Chu parameters."""
    h = _windowed_zadoff_chu_h(params)
    return _smooth_max_value(h, beta)

def _chirp_h(params):
    """Generate h from chirp parameters: [phase, rate, amp, base]."""
    phase, rate, amp, base = params
    i = _i_arr
    ang = phase + rate * (i / N) ** 2
    h = base + amp * np.cos(ang)
    return h

def _chirp_obj(params):
    """Objective for chirp parameters."""
    h = _chirp_h(params)
    return _value(h)

def _project_dft(h):
    """Project h onto DFT basis + base, return (coeffs, base_val)."""
    mean_h = float(np.mean(h))
    target = h - mean_h
    # Use the precomputed basis matrix for a single lstsq
    try:
        coeffs, _, _, _ = np.linalg.lstsq(_BASIS_T, target, rcond=None)
    except:
        coeffs = np.zeros(2 * _K_DFT)
    return coeffs, mean_h

def _full_dim_smooth_obj(h, beta=80.0):
    """Smooth objective for full-dimensional h."""
    return _smooth_max_value(h, beta)

def _smooth_max_value_from_conv(conv, s, beta=80.0):
    """Smooth max value from precomputed convolution and sum."""
    if s == 0.0:
        return np.inf
    m = np.max(conv)
    scaled = beta * (conv - m)
    scaled = np.clip(scaled, -100, 100)
    log_sum_exp = m + np.log(np.sum(np.exp(scaled))) / beta
    return abs(2.0 * N * log_sum_exp / (s * s))

def _full_dim_coord_descent(h0, max_iter=200, step_size=0.01, beta=80.0, deadline_fn=None):
    """
    Full-dimensional coordinate descent with incremental convolution update.
    Uses the structure of autoconvolution to update in O(N) per coordinate.
    """
    h = h0.copy()
    s = float(np.sum(h))
    conv = _fft_convolve(h)
    best_v = _smooth_max_value_from_conv(conv, s, beta)
    
    for it in range(max_iter):
        if deadline_fn and not deadline_fn():
            break
        improved = False
        for i in range(N):
            orig = h[i]
            for delta in [step_size, -step_size]:
                new_s = s + delta
                if new_s == 0.0:
                    continue
                new_conv = _convolve_incremental(h, i, delta)
                v_trial = _smooth_max_value_from_conv(new_conv, new_s, beta)
                if v_trial < best_v:
                    h[i] = orig + delta
                    s = new_s
                    conv = new_conv
                    best_v = v_trial
                    improved = True
                    break
                else:
                    h[i] = orig
        if not improved:
            step_size *= 0.5
            if step_size < 1e-6:
                break
    
    return h, best_v

def construct():
    global _t0
    _t0 = time.time()
    
    best_h = np.ones(N)
    best_v = _value(best_h)
    
    def deadline_ok():
        return (time.time() - _t0) < _DEADLINE
    
    def update_best(h):
        nonlocal best_h, best_v
        v = _value(h)
        if v < best_v:
            best_h = h.copy()
            best_v = v
    
    # Strategy 1: DCT spectral shaping
    for restart in range(5):
        if not deadline_ok():
            break
        try:
            # Random initial DCT coefficients
            rng = np.random.RandomState(restart * 123 + 456)
            coeffs0 = rng.randn(_K_DCT) * 0.1
            res = minimize(_dct_smooth_obj, coeffs0, method='L-BFGS-B',
                          options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            h = _dct_h(res.x)
            update_best(h)
        except Exception:
            pass
    
    # Strategy 2: Golay complementary pair
    for restart in range(3):
        if not deadline_ok():
            break
        try:
            rng = np.random.RandomState(restart * 789 + 123)
            params0 = np.array([
                rng.uniform(0, 2*np.pi), rng.uniform(0.5, 3.0), rng.uniform(0.1, 0.5),
                rng.uniform(0, 2*np.pi), rng.uniform(0.5, 3.0), rng.uniform(0.1, 0.5),
                1.0
            ])
            res = minimize(_golay_smooth_obj, params0, method='L-BFGS-B',
                          options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            h = _golay_complementary_h(res.x)
            update_best(h)
        except Exception:
            pass
    
    # Strategy 3: Windowed Zadoff-Chu
    for restart in range(3):
        if not deadline_ok():
            break
        try:
            rng = np.random.RandomState(restart * 321 + 654)
            params0 = np.array([
                rng.uniform(0, 2*np.pi), rng.uniform(0.5, 3.0), rng.uniform(0.1, 0.5),
                1.0, 1.0, 0.5, rng.uniform(0, 1)
            ])
            res = minimize(_windowed_zc_smooth_obj, params0, method='L-BFGS-B',
                          options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            h = _windowed_zadoff_chu_h(res.x)
            update_best(h)
        except Exception:
            pass
    
    # Strategy 4: Full-dimensional coordinate descent from best candidate
    if deadline_ok():
        try:
            h_cd, v_cd = _full_dim_coord_descent(best_h, max_iter=200, step_size=0.01, beta=80.0, deadline_fn=deadline_ok)
            update_best(h_cd)
        except Exception:
            pass
    
    # Strategy 5: DFT refinement
    if deadline_ok():
        try:
            coeffs, base_val = _project_dft(best_h)
            coeffs_base = np.append(coeffs, base_val)
            res = minimize(_dft_smooth_obj, coeffs_base, method='L-BFGS-B',
                          options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            h = _dft_h(res.x[:-1], res.x[-1])
            update_best(h)
        except Exception:
            pass
    
    # Strategy 6: Targeted gradient-based polish at the max location
    if deadline_ok():
        try:
            h = best_h.copy()
            s = float(np.sum(h))
            conv = _fft_convolve(h)
            max_idx = int(np.argmax(conv))
            
            # Compute gradient of the exact objective w.r.t. h
            # value = 2*N*conv[max_idx] / s^2 (assuming max is positive)
            # d(value)/d(h[i]) = 2*N/s^2 * d(conv[max_idx])/d(h[i]) - 4*N*conv[max_idx]/s^3 * d(s)/d(h[i])
            # d(conv[k])/d(h[i]) = h[k-i] + h[k-i] = 2*h[k-i] (if valid)
            # d(s)/d(h[i]) = 1
            
            # Only update coordinates that affect conv[max_idx]
            k = max_idx
            # h[i] affects conv[k] when i is in [max(0, k-N+1), min(N-1, k)]
            i_lo = max(0, k - N + 1)
            i_hi = min(N - 1, k)
            
            # Line search: move in negative gradient direction
            for ls_iter in range(50):
                if not deadline_ok():
                    break
                # Compute gradient at current h
                grad = np.zeros(N)
                for i in range(i_lo, i_hi + 1):
                    j = k - i
                    if 0 <= j < N:
                        grad[i] += 2.0 * h[j]
                # d(conv[k])/d(h[i]) = 2*h[k-i] for valid k-i
                # value = 2*N*conv[k]/s^2
                # d(value)/d(h[i]) = 2*N/s^2 * 2*h[k-i] - 4*N*conv[k]/s^3
                dval_dh = (4.0 * N / (s * s)) * grad - (4.0 * N * conv[k] / (s * s * s))
                
                # Normalize gradient
                gnorm = np.linalg.norm(dval_dh)
                if gnorm < 1e-12:
                    break
                direction = -dval_dh / gnorm
                
                # Backtracking line search
                alpha = 0.01
                improved = False
                for _ in range(20):
                    h_trial = h + alpha * direction
                    v_trial = _value(h_trial)
                    v_curr = _value(h)
                    if v_trial < v_curr:
                        h = h_trial
                        improved = True
                        break
                    alpha *= 0.5
                if not improved:
                    break
            
            update_best(h)
        except Exception:
            pass
    
    # Final refinement with coordinate descent (more sweeps possible with incremental update)
    if deadline_ok():
        try:
            h_cd2, v_cd2 = _full_dim_coord_descent(best_h, max_iter=300, step_size=0.005, beta=100.0, deadline_fn=deadline_ok)
            update_best(h_cd2)
        except Exception:
            pass
    
    # Additional gradient polish after CD
    if deadline_ok():
        try:
            h = best_h.copy()
            s = float(np.sum(h))
            conv = _fft_convolve(h)
            max_idx = int(np.argmax(conv))
            k = max_idx
            i_lo = max(0, k - N + 1)
            i_hi = min(N - 1, k)
            
            for ls_iter in range(100):
                if not deadline_ok():
                    break
                grad = np.zeros(N)
                for i in range(i_lo, i_hi + 1):
                    j = k - i
                    if 0 <= j < N:
                        grad[i] += 2.0 * h[j]
                dval_dh = (4.0 * N / (s * s)) * grad - (4.0 * N * conv[k] / (s * s * s))
                gnorm = np.linalg.norm(dval_dh)
                if gnorm < 1e-14:
                    break
                direction = -dval_dh / gnorm
                alpha = 0.01
                improved = False
                for _ in range(30):
                    h_trial = h + alpha * direction
                    v_trial = _value(h_trial)
                    v_curr = _value(h)
                    if v_trial < v_curr:
                        h = h_trial
                        s = float(np.sum(h))
                        conv = _fft_convolve(h)
                        max_idx = int(np.argmax(conv))
                        k = max_idx
                        i_lo = max(0, k - N + 1)
                        i_hi = min(N - 1, k)
                        improved = True
                        break
                    alpha *= 0.5
                if not improved:
                    break
            
            update_best(h)
        except Exception:
            pass
    
    # Ensure we return a valid sequence
    if not np.all(np.isfinite(best_h)) or np.sum(best_h) == 0:
        return np.ones(N)
    
    return best_h
