"""
Structural change: Sparse analytic signal representation (squared magnitude of linear combination of complex exponentials)
+ Multi-start local search (Nelder-Mead) on the low-dimensional coefficient space.

This represents h[i] = | sum_{k=0}^M c_k exp(i * 2 * pi * f_k * i / N) |^2.
This ensures h[i] >= 0 by construction.
The optimization is performed on the real and imaginary parts of the coefficients c_k.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 600
W = 1.0 / 1200.0  # width of each interval

# Fine grid for accurate autoconvolution
FINE_N = 6000
X_FINE = np.linspace(-0.25, 0.25, FINE_N, endpoint=False)
DX = 0.5 / FINE_N
CELL = np.clip(((X_FINE + 0.25) / W).astype(np.int64), 0, N - 1)
N_FFT = 1 << (2 * FINE_N - 1).bit_length()

# Precompute indices for signal generation
I_IDX = np.arange(N)

def ratio_from_h(h):
    """
    Compute the normalized maximum of the autoconvolution.
    Normalization: max((f*f)(t)) / (integral f)^2
    """
    h = np.asarray(h, dtype=np.float64)
    if h.size != N:
        h = np.resize(h, N)
    h = np.clip(h, 0.0, None)
    s = h.sum()
    if s < 1e-15:
        return float('inf')

    # Map step heights to fine grid
    f_fine = h[CELL]

    # FFT-based autoconvolution
    F = np.fft.rfft(f_fine, N_FFT)
    conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1]
    conv *= DX  # scale by dx for continuous integral

    max_conv = np.max(conv)

    # Normalization: (integral f)^2 = (sum(h) * W)^2
    integral_sq = (s * W) ** 2
    if integral_sq < 1e-15:
        return float('inf')

    return max_conv / integral_sq

def coeffs_to_heights(coeffs, freqs):
    """
    Convert complex coefficients to nonnegative step heights.
    coeffs: array of complex numbers [c0, c1, ..., cM]
    freqs: array of frequencies [f0, f1, ..., fM]
    h[i] = | sum_{k} c_k * exp(i * 2 * pi * f_k * i / N) |^2
    """
    signal = np.zeros(N, dtype=np.complex128)
    for k, c in enumerate(coeffs):
        if c != 0:
            signal += c * np.exp(1j * 2.0 * np.pi * freqs[k] * I_IDX / N)
    heights = np.abs(signal) ** 2
    return heights

def objective_from_coeffs(coeffs, freqs):
    """
    Objective function for optimization on coefficients.
    """
    h = coeffs_to_heights(coeffs, freqs)
    return ratio_from_h(h)

def construct():
    """
    Construct a nonnegative step function with 600 equal intervals on [-1/4, 1/4]
    to minimize the normalized maximum of its autoconvolution.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    # Initial best: uniform (corresponds to c0=1, others 0)
    best_h = np.ones(N, dtype=np.float64)
    best_val = ratio_from_h(best_h)

    np.random.seed(42)

    # Number of basis functions (sparse)
    M = 6
    # Use low frequencies to capture global shape
    freqs = np.array([0.0] + np.arange(1, M + 1, dtype=np.float64))
    # Note: freqs[0] is 0, so exp(0) = 1. c0 is the DC term.
    # The representation is h[i] = | c0 + sum_{k=1}^M c_k e^{i 2 pi k i / N} |^2
    
    # Number of real parameters: 2*(M+1)
    n_params = 2 * (M + 1)

    # Multi-start local search
    n_starts = 50
    max_iter_per_start = 200

    for s in range(n_starts):
        if time.time() > deadline - 10.0:
            break

        # Generate random initialization
        # c0 should be real and positive to ensure large base level
        # Other coefficients can be complex
        c0_real = np.random.uniform(0.5, 2.0)
        c0_imag = 0.0
        params = np.zeros(n_params)
        params[0] = c0_real
        params[1] = c0_imag
        
        for k in range(1, M + 1):
            # Smaller coefficients for higher frequencies
            scale = 0.5 / np.sqrt(k)
            params[2*k] = np.random.normal(0, scale)
            params[2*k+1] = np.random.normal(0, scale)

        # Reshape to complex coefficients
        def unpack(params):
            coeffs = np.zeros(M + 1, dtype=np.complex128)
            coeffs[0] = params[0] + 1j * params[1]
            for k in range(1, M + 1):
                coeffs[k] = params[2*k] + 1j * params[2*k+1]
            return coeffs

        def packed_objective(p):
            coeffs = unpack(p)
            return objective_from_coeffs(coeffs, freqs)

        try:
            # Use Nelder-Mead which doesn't require gradients and is robust for low-dim
            res = minimize(
                packed_objective,
                params,
                method='Nelder-Mead',
                options={'maxiter': max_iter_per_start, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            
            if np.isfinite(res.fun):
                best_coeffs = unpack(res.x)
                h_candidate = coeffs_to_heights(best_coeffs, freqs)
                val = ratio_from_h(h_candidate)
                if val < best_val and np.isfinite(val):
                    best_val = val
                    best_h = h_candidate.copy()
                    
        except Exception:
            pass

    # Polish with direct coordinate descent if time remains
    if time.time() < deadline - 5.0:
        step_size = 0.1
        for it in range(5000):
            if time.time() > deadline - 2.0:
                break

            n_perturb = np.random.randint(2, 10)
            idx = np.random.choice(N, n_perturb, replace=False)
            h_try = best_h.copy()
            h_try[idx] += np.random.randn(n_perturb) * step_size
            h_try = np.clip(h_try, 0, None)

            val = ratio_from_h(h_try)
            if val < best_val:
                best_val = val
                best_h = h_try.copy()
                step_size = max(step_size * 0.9, 0.01)
            else:
                step_size = min(step_size * 1.1, 0.5)

    return list(best_h)
