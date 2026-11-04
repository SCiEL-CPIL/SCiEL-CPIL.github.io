import numpy as np
import time
from scipy.optimize import minimize

N = 600
W = 1.0 / 1200.0  # width of each interval

# Fine grid for accurate autoconvolution (final verification)
FINE_N = 6000
X_FINE = np.linspace(-0.25, 0.25, FINE_N, endpoint=False)
DX = 0.5 / FINE_N
CELL = np.clip(((X_FINE + 0.25) / W).astype(np.int64), 0, N - 1)
N_FFT = 1 << (2 * FINE_N - 1).bit_length()

# Precompute indices
I_IDX = np.arange(N)

def ratio_from_h_fine(h):
    """
    Compute the normalized maximum of the autoconvolution using fine grid FFT.
    This is the accurate score.
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

def ratio_from_h_coarse_lse(h, tau=0.05):
    """
    Fast surrogate using coarse grid convolution and log-sum-exp smoothing.
    Used for optimization.
    """
    h = np.asarray(h, dtype=np.float64)
    if h.size != N:
        h = np.resize(h, N)
    h = np.clip(h, 0.0, None)
    s = h.sum()
    if s < 1e-15:
        return 1e10

    # Coarse convolution on the 600 bins
    # We use np.convolve for speed, mode 'full'
    conv = np.convolve(h, h, mode='full') * W
    
    # Log-sum-exp for smooth max
    m = np.max(conv)
    if m < 1e-15:
        return 1e10
    
    # Scale tau relative to typical conv magnitude to ensure smoothness
    # Typical conv magnitude is ~ s^2 * W. 
    # If s ~ 600, W ~ 1/1200, s^2 W ~ 300.
    # tau should be small enough to approximate max, large enough for gradient.
    # Let's use a dynamic tau or a fixed one.
    lse = m + tau * np.log(np.sum(np.exp((conv - m) / tau)))

    integral_sq = (s * W) ** 2
    if integral_sq < 1e-15:
        return 1e10

    return lse / integral_sq

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

def objective_smooth(params, freqs, tau):
    """
    Objective function for optimization on coefficients using LSE surrogate.
    """
    coeffs = unpack_coeffs(params, len(freqs))
    h = coeffs_to_heights(coeffs, freqs)
    return ratio_from_h_coarse_lse(h, tau)

def unpack_coeffs(params, M_plus_1):
    """
    Unpack real vector params into complex coefficients.
    params = [Re(c0), Im(c0), Re(c1), Im(c1), ...]
    """
    coeffs = np.zeros(M_plus_1, dtype=np.complex128)
    for k in range(M_plus_1):
        coeffs[k] = params[2*k] + 1j * params[2*k+1]
    return coeffs

def construct():
    """
    Construct a nonnegative step function with 600 equal intervals on [-1/4, 1/4]
    to minimize the normalized maximum of its autoconvolution.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    # Initial best: uniform (corresponds to c0=1, others 0)
    best_h = np.ones(N, dtype=np.float64)
    best_val = ratio_from_h_fine(best_h)

    np.random.seed(42)

    # Number of basis functions (sparse)
    M = 12
    # Use a range of frequencies. 
    # To allow for "chirp-like" behavior, we can use non-integer frequencies 
    # or just integer harmonics. Integer harmonics are safer for periodicity 
    # but the function is not periodic, it's on [-0.25, 0.25].
    # Let's use integer frequencies 0..M for stability.
    freqs = np.arange(M + 1, dtype=np.float64)
    
    n_params = 2 * (M + 1)

    # Multi-start local search with L-BFGS-B
    n_starts = 30
    max_iter_per_start = 300
    tau = 0.1 # Smoothing parameter

    for s in range(n_starts):
        if time.time() > deadline - 20.0:
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
            scale = 0.3 / np.sqrt(k)
            params[2*k] = np.random.normal(0, scale)
            params[2*k+1] = np.random.normal(0, scale)

        try:
            # Use L-BFGS-B
            res = minimize(
                objective_smooth,
                params,
                args=(freqs, tau),
                method='L-BFGS-B',
                options={'maxiter': max_iter_per_start, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if np.isfinite(res.fun):
                best_coeffs = unpack_coeffs(res.x, M + 1)
                h_candidate = coeffs_to_heights(best_coeffs, freqs)
                # Verify with fine grid
                val = ratio_from_h_fine(h_candidate)
                if val < best_val and np.isfinite(val):
                    best_val = val
                    best_h = h_candidate.copy()
                    
        except Exception:
            pass

    # Polish with direct coordinate descent if time remains
    if time.time() < deadline - 5.0:
        step_size = 0.05
        for it in range(5000):
            if time.time() > deadline - 2.0:
                break

            n_perturb = np.random.randint(2, 10)
            idx = np.random.choice(N, n_perturb, replace=False)
            h_try = best_h.copy()
            h_try[idx] += np.random.randn(n_perturb) * step_size
            h_try = np.clip(h_try, 0, None)

            val = ratio_from_h_fine(h_try)
            if val < best_val:
                best_val = val
                best_h = h_try.copy()
                step_size = max(step_size * 0.95, 0.01)
            else:
                step_size = min(step_size * 1.05, 0.2)

    return list(best_h)
