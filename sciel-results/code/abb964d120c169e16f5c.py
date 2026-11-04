import numpy as np
import time
from scipy.optimize import minimize

N = 600
W = 0.5 / N  # Width of each interval
# Centers of the intervals
CENTERS = -0.25 + (np.arange(N) + 0.5) * W

# Precompute Fourier basis
# We use K modes. h_i = a_0 + sum_{k=1}^K (a_k cos(2*pi*k*i/N) + b_k sin(2*pi*k*i/N))
K = 40
I = np.arange(N)
# Shape (K, N)
COS = np.cos(2.0 * np.pi * I[None, :] * np.arange(1, K + 1)[:, None] / N)
SIN = np.sin(2.0 * np.pi * I[None, :] * np.arange(1, K + 1)[:, None] / N)

def fourier_to_heights(params):
    """
    Convert Fourier parameters to nonnegative step heights.
    params: [a0, a1, b1, a2, b2, ..., aK, bK]
    """
    a0 = params[0]
    # a_k is params[2*k-1], b_k is params[2*k]
    a_coeffs = params[1::2][:K]
    b_coeffs = params[2::2][:K]
    
    heights = a0 + a_coeffs @ COS + b_coeffs @ SIN
    return np.clip(heights, 0.0, None)

def heights_to_fourier(heights):
    """
    Compute Fourier coefficients of a height vector.
    Returns [a0, a1, b1, ..., aK, bK]
    """
    a0 = np.mean(heights)
    a_coeffs = np.zeros(K)
    b_coeffs = np.zeros(K)
    for k in range(1, K + 1):
        a_coeffs[k-1] = (2.0 / N) * np.sum(heights * np.cos(2.0 * np.pi * k * I / N))
        b_coeffs[k-1] = (2.0 / N) * np.sum(heights * np.sin(2.0 * np.pi * k * I / N))
    
    params = np.zeros(2 * K + 1)
    params[0] = a0
    params[1::2] = a_coeffs
    params[2::2] = b_coeffs
    return params

def compute_score_fine(heights):
    """
    Compute the normalized maximum of the autoconvolution using a fine grid FFT.
    This is the accurate score used for final evaluation.
    """
    h = np.asarray(heights, dtype=np.float64)
    if h.size != N:
        h = np.resize(h, N)
    h = np.clip(h, 0.0, None)
    s = h.sum()
    if s < 1e-15:
        return float('inf')

    # Map step heights to a fine grid for accurate convolution
    # We use a grid 10x finer than the step function to capture the max better
    FINE_N = N * 10
    X_FINE = np.linspace(-0.25, 0.25, FINE_N, endpoint=False)
    DX = 0.5 / FINE_N
    # Map fine grid points to step indices
    # (x + 0.25) / W gives the index in the step function
    # Clip to valid range
    cell_indices = np.clip(((X_FINE + 0.25) / W).astype(np.int64), 0, N - 1)
    f_fine = h[cell_indices]

    N_FFT = 1 << (2 * FINE_N - 1).bit_length()
    F = np.fft.rfft(f_fine, N_FFT)
    conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1]
    conv *= DX  # Scale by dx for continuous integral approximation

    max_conv = np.max(conv)
    integral_sq = (s * W) ** 2
    
    if integral_sq < 1e-15:
        return float('inf')
        
    return max_conv / integral_sq

def compute_score_coarse(heights, tau=0.01):
    """
    Fast surrogate using coarse grid convolution and log-sum-exp smoothing.
    Used for optimization.
    """
    h = np.asarray(heights, dtype=np.float64)
    if h.size != N:
        h = np.resize(h, N)
    h = np.clip(h, 0.0, None)
    s = h.sum()
    if s < 1e-15:
        return 1e10

    # Coarse convolution on the original N grid
    # This is an approximation of the continuous convolution
    conv = np.convolve(h, h, mode='full') * W
    
    # Log-sum-exp for smooth max
    m = np.max(conv)
    if m < 1e-15:
        return 1e10
    
    # Scale tau relative to typical conv magnitude
    # conv values are roughly O(s^2 * W)
    # We want the LSE to approximate max closely but be smooth
    # If tau is too small, it's non-differentiable. If too large, it's too smooth.
    # Let's use a fixed small tau in units of the conv values.
    # The max conv value is the target.
    lse = m + tau * np.log(np.sum(np.exp((conv - m) / tau)))

    integral_sq = (s * W) ** 2
    if integral_sq < 1e-15:
        return 1e10

    return lse / integral_sq

def smooth_objective_fourier(params):
    """
    Smooth surrogate for Fourier parameters using coarse grid.
    """
    h = fourier_to_heights(params)
    return compute_score_coarse(h, tau=0.01)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_h = np.ones(N, dtype=np.float64)
    best_val = compute_score_fine(best_h)
    
    np.random.seed(42)

    # Generate diverse initial Fourier parameter vectors
    inits = []
    
    # 1. Uniform (Zero frequency)
    p0_uniform = np.zeros(2 * K + 1)
    p0_uniform[0] = 1.0
    inits.append(p0_uniform)
    
    # 2. Gaussian-like (Low frequency decay)
    # Approximate a Gaussian by summing cosines with decaying amplitudes
    p0_gauss = np.zeros(2 * K + 1)
    p0_gauss[0] = 1.0
    sigma = 0.1
    for k in range(1, K + 1):
        # Fourier coeff of exp(-x^2/2sigma^2) is ~ exp(-k^2 sigma^2 / 2)
        amp = np.exp(-k**2 * sigma**2 / 2)
        p0_gauss[2*k-1] = amp
        # b_k = 0 for symmetric
    inits.append(p0_gauss)
    
    # 3. Top-hat approximation
    # A top-hat has sinc-like Fourier coefficients
    p0_top = np.zeros(2 * K + 1)
    p0_top[0] = 0.5
    width = 0.2
    for k in range(1, K + 1):
        # sinc(k * width)
        val = np.sinc(k * width) # np.sinc(x) = sin(pi x)/(pi x)
        p0_top[2*k-1] = val
    inits.append(p0_top)
    
    # 4. Random low-frequency perturbations
    for _ in range(10):
        p0 = np.zeros(2 * K + 1)
        p0[0] = np.random.uniform(0.5, 1.5)
        # Only use low frequencies for smoothness
        num_modes = np.random.randint(5, 15)
        for k in range(1, num_modes + 1):
            p0[2*k-1] = np.random.normal(0, 0.1)
            p0[2*k] = np.random.normal(0, 0.1)
        inits.append(p0)
        
    # 5. Symmetric "bell" shapes with varying widths
    for sigma in [0.05, 0.1, 0.15, 0.2]:
        p0 = np.zeros(2 * K + 1)
        p0[0] = 1.0
        for k in range(1, K + 1):
            amp = np.exp(-k**2 * sigma**2 / 2)
            p0[2*k-1] = amp
        inits.append(p0)

    # Optimize using L-BFGS-B on the coarse surrogate
    for p0 in inits:
        if time.time() > deadline - 30.0:
            break
        try:
            res = minimize(
                smooth_objective_fourier,
                p0,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            if np.isfinite(res.fun):
                h_candidate = fourier_to_heights(res.x)
                # Verify with fine grid
                val = compute_score_fine(h_candidate)
                if val < best_val and np.isfinite(val):
                    best_val = val
                    best_h = h_candidate.copy()
        except Exception:
            pass

    # Local optimization on the best found heights
    # Use a finer Fourier representation or direct coordinate descent?
    # Direct coordinate descent on 600 variables is slow.
    # Let's do a few more Fourier optimizations from the best solution with higher K?
    # Or just refine the current best_h with small random perturbations on the coarse grid.
    
    if best_h is not None and time.time() < deadline - 10.0:
        current = best_h.copy()
        current_val = best_val
        # Normalize to keep sum constant? The objective is scale invariant, so sum doesn't matter for the ratio.
        # But keeping sum ~ N helps numerical stability.
        if current.sum() > 0:
            current = current * (N / current.sum())
        
        np.random.seed(123)
        # Multi-scale local optimization
        scales = [0.05, 0.02, 0.01, 0.005, 0.002]
        for scale in scales:
            if time.time() > deadline - 5:
                break
            for _ in range(200):
                if time.time() > deadline - 5:
                    break
                mask = np.random.random(N) < 0.05
                if mask.sum() == 0:
                    continue
                perturbation = (np.random.random(mask.sum()) - 0.5) * scale
                new_heights = current.copy()
                new_heights[mask] += perturbation
                new_heights = np.maximum(new_heights, 0)
                
                val = compute_score_coarse(new_heights, tau=0.01)
                if val < current_val:
                    # Check with fine score occasionally to avoid surrogate drift
                    if _ % 10 == 0 or _ == 199:
                        fine_val = compute_score_fine(new_heights)
                        if fine_val < best_val:
                            best_val = fine_val
                            best_h = new_heights.copy()
                            current_val = fine_val
                    else:
                        current_val = val
                    
                    current = new_heights

    if best_h is None:
        return [1.0] * N
    return list(best_h)
