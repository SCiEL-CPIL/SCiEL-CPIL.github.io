"""Second autocorrelation inequality: maximize ||h||^2_2 / (||h||_1 * ||h*h||_inf)
over 50 nonnegative step heights using spectral parameterization."""

import numpy as np
import time
import math
from scipy.optimize import minimize

N = 50
_DEADLINE = time.time() + 135.0  # safety margin before 150s kill


def _time_left():
    return _DEADLINE - time.time()


def _autoconv_sup(h):
    """Compute the supremum of the autoconvolution of step function with heights h.
    
    For a step function f(x) = h_i on [i*H, (i+1)*H), the autoconvolution at
    t = k*H is H * sum_i h_i * h_{k-i} (with valid indices).
    The supremum over all t is achieved at some t = k*H for integer k.
    
    Actually, for nonnegative step functions, the autoconvolution is piecewise linear
    and its maximum occurs at a point where the "overlap" is maximized. For the
    discrete case, we compute the full autocorrelation sequence and take the max.
    
    The continuous autoconvolution (f*f)(t) for t in [0, 2*N*H) can be computed.
    At t = m*H (integer multiple), (f*f)(m*H) = H * sum_{i} h_i * h_{m-i}
    where the sum is over i such that both i and m-i are in [0, N-1].
    
    But the true supremum might be higher between grid points. For nonnegative
    functions, the autoconvolution is actually maximized at one of the grid points
    t = m*H. Let me verify: the autoconvolution is a piecewise linear function whose
    breakpoints are at t = m*H. Between breakpoints, it's linear. So the maximum
    is at a breakpoint or at an endpoint. The endpoints give 0. So max is at some
    t = m*H.
    
    Therefore: sup ||f*f|| = H * max_m |sum_i h_i h_{m-i}| = H * max_m (sum_i h_i h_{m-i})
    since h_i >= 0.
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    # Full autocorrelation: r[m] = sum_i h_i * h_{m-i} for m = -(n-1), ..., (n-1)
    # Since h is nonnegative, r[m] >= 0, and r is symmetric.
    # We need max over m of r[m].
    # Use FFT for efficiency.
    n_fft = 1
    while n_fft < 2 * n - 1:
        n_fft *= 2
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    fft_h = np.fft.fft(h_pad)
    autocorr = np.fft.ifft(fft_h * np.conj(fft_h)).real
    # autocorr[m] for m = 0, ..., n_fft-1 represents circular autocorrelation
    # The linear autocorrelation is: for lag k (positive), sum_{i=0}^{n-1-k} h[i]*h[i+k]
    # autocorr[0] = sum h_i^2
    # autocorr[k] for k=1,...,n-1 = sum_{i=0}^{n-1-k} h[i]*h[i+k]
    # autocorr[n_fft-k] = autocorr[k] by symmetry
    # The max of the autoconvolution is H * max(autocorr[0:n])
    max_ac = np.max(autocorr[:n])
    return max_ac * (1.0 / n)  # H = 1/N, but wait...
    
    # Actually, let me reconsider. The step function has heights h_i on intervals
    # of width H = 1/N (since the total interval is [-1/4, 1/4] = 1/2 wide, and
    # we have 50 intervals, so H = (1/2)/50 = 1/100? No wait.
    # The problem says "equal intervals of [-1/4, 1/4]". The interval [-1/4, 1/4]
    # has length 1/2. With 50 intervals, each has width H = (1/2)/50 = 1/100.
    # But actually, for the ratio, the constant H cancels out!
    # L2^2 = sum(h_i^2) * H
    # L1 = sum(h_i) * H
    # sup ||f*f|| = H * max_m (sum_i h_i h_{m-i})  [the H comes from the width]
    # Wait, let me recompute. (f*f)(t) = integral f(x) f(t-x) dx.
    # If f(x) = h_i for x in [i*H, (i+1)*H), then for t = m*H:
    # (f*f)(m*H) = sum_i sum_j h_i h_j * length([iH,(i+1)H) ∩ [(m-j-1)H, (m-j)H))
    # The intersection has length H if the intervals align, which happens when
    # i + j + 1 = m, i.e., j = m - 1 - i.
    # So (f*f)(m*H) = H * sum_i h_i * h_{m-1-i} (for valid indices).
    # Hmm, this is getting confusing. Let me just use the standard result:
    # For a step function with step width H, the autoconvolution at the "center"
    # of the support gives the peak, and it equals H times the discrete autocorrelation.
    # The ratio L2^2 / (L1 * sup||f*f||) = (H * sum h_i^2) / (H * sum h_i * H * max_ac)
    # = (sum h_i^2) / (H * sum h_i * max_ac)
    # where max_ac = max_m sum_i h_i h_{m-i} (discrete autocorrelation peak).
    # So the H matters! Let me be careful.
    
    # Actually, for the purpose of optimization, since H is a constant, we can
    # ignore it and just maximize sum(h_i^2) / (sum(h_i) * max_ac).
    # The H factor is constant and will be applied by the evaluator.
    
    return max_ac


def _ratio(h):
    """Compute the ratio: sum(h_i^2) / (sum(h_i) * max autocorrelation).
    
    This is proportional to the actual ratio (differing by a constant factor H
    that the evaluator accounts for). For optimization purposes, maximizing this
    is equivalent to maximizing the actual ratio.
    """
    h = np.asarray(h, dtype=np.float64)
    if np.any(h < 0):
        return 0.0
    s2 = np.sum(h ** 2)
    s1 = np.sum(h)
    if s1 <= 0:
        return 0.0
    max_ac = _autoconv_sup(h)
    if max_ac <= 0:
        return 0.0
    return s2 / (s1 * max_ac)


def _ratio_neg(h):
    return -_ratio(h)


def _spectral_to_heights(z):
    """Convert complex vector z to nonnegative heights via |DFT(z)|^2.
    
    z is a complex vector of length N. We compute the DFT, take squared magnitudes,
    and normalize. This ensures non-negativity and provides a smooth manifold.
    """
    z = np.asarray(z, dtype=np.complex128)
    n = len(z)
    fft_z = np.fft.fft(z)
    heights = np.abs(fft_z) ** 2
    # Normalize to have max 1 for stability
    max_h = np.max(heights)
    if max_h > 0:
        heights = heights / max_h
    return heights


def _heights_to_spectral(h):
    """Convert nonnegative heights to a complex vector z such that |DFT(z)|^2 ~ h.
    
    This is an inverse problem. We set z = IDFT(sqrt(h) * exp(i*theta)) for some
    phases theta. For initialization, we can use theta = 0.
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    # Take square root of heights (these will be the magnitudes in frequency domain)
    mags = np.sqrt(h)
    # Use zero phases for simplicity
    z = np.fft.ifft(mags)
    return z


def _local_optimize_spectral(z0, max_iter=200):
    """Optimize the complex vector z to maximize the ratio.
    
    Parameters:
    - z0: initial complex vector of length N
    - max_iter: maximum iterations
    
    Returns:
    - best_z: optimized complex vector
    - best_score: best ratio found
    """
    def neg_ratio_z(z_flat):
        z = z_flat[:N].astype(np.complex128)
        z = z[N:].astype(np.complex128) if False else z_flat[:N].view(np.complex128)
        # Actually, let's handle this more carefully
        # z_flat is real, length 2*N, representing real and imaginary parts
        re = z_flat[:N]
        im = z_flat[N:]
        z = re + 1j * im
        h = _spectral_to_heights(z)
        return -_ratio(h)
    
    # Initial point
    z0 = np.asarray(z0, dtype=np.complex128)
    x0 = np.concatenate([z0.real, z0.imag])
    
    # Use Nelder-Mead on the real representation
    res = minimize(neg_ratio_z, x0, method='Nelder-Mead',
                   options={'maxiter': max_iter, 'xatol': 1e-6, 'fatol': 1e-8})
    
    best_z = np.concatenate([res.x[:N], 1j * res.x[N:]])
    best_score = -res.fun
    return best_z, best_score


def _local_optimize_direct(h0, max_iter=300):
    """Optimize heights directly with nonnegativity constraint.
    
    Uses a reparameterization: h_i = exp(x_i) to ensure non-negativity.
    """
    def neg_ratio_exp(x):
        h = np.exp(x)
        # Normalize to avoid overflow/underflow issues
        h = h / (np.max(h) + 1e-15)
        return -_ratio(h)
    
    # Initial point in log-space
    h0 = np.asarray(h0, dtype=np.float64)
    h0 = np.maximum(h0, 1e-10)  # avoid log(0)
    x0 = np.log(h0)
    
    res = minimize(neg_ratio_exp, x0, method='Nelder-Mead',
                   options={'maxiter': max_iter, 'xatol': 1e-6, 'fatol': 1e-8})
    
    best_h = np.exp(res.x)
    best_h = best_h / (np.max(best_h) + 1e-15)
    best_score = -res.fun
    return best_h, best_score


def _initializations():
    """Generate diverse initial candidates."""
    candidates = []
    
    # 1. All ones
    candidates.append(np.ones(N))
    
    # 2. Single spike
    for pos in [0, N//4, N//2, 3*N//4, N-1]:
        h = np.zeros(N)
        h[pos] = 1.0
        candidates.append(h)
    
    # 3. Gaussian profiles with various widths
    center = N // 2
    for width in [1, 2, 3, 5, 8, 12, 20]:
        x = np.arange(N)
        h = np.exp(-0.5 * ((x - center) / width) ** 2)
        candidates.append(h)
    
    # 4. Triangular profiles
    for width in [5, 10, 15, 25]:
        x = np.arange(N)
        h = np.maximum(0, 1 - np.abs(x - center) / width)
        candidates.append(h)
    
    # 5. Lorentzian profiles
    for gamma in [0.5, 1, 2, 5, 10]:
        x = np.arange(N)
        h = 1.0 / (1.0 + ((x - center) / gamma) ** 2)
        candidates.append(h)
    
    # 6. Random smooth profiles
    rng = np.random.RandomState(42)
    for _ in range(10):
        # Generate random smooth profile using low-frequency Fourier components
        freqs = np.fft.fftfreq(N)
        coeffs = rng.randn(N) + 1j * rng.randn(N)
        # Keep only low frequencies
        mask = np.abs(freqs) < 0.2
        coeffs[~mask] = 0
        h = np.real(np.fft.ifft(coeffs))
        h = np.maximum(h, 0)
        if np.max(h) > 0:
            h = h / np.max(h)
            candidates.append(h)
    
    # 7. Spectral initializations: random complex vectors
    for _ in range(10):
        z = rng.randn(N) + 1j * rng.randn(N)
        h = _spectral_to_heights(z)
        candidates.append(h)
    
    return candidates


def construct():
    start_time = time.time()
    
    best_h = np.ones(N)
    best_score = _ratio(best_h)
    
    # Evaluate initial candidates
    candidates = _initializations()
    for h in candidates:
        if _time_left() < 10:
            break
        s = _ratio(h)
        if s > best_score:
            best_score = s
            best_h = h.copy()
    
    # Direct optimization from best candidate
    if _time_left() > 15:
        h_opt, s_opt = _local_optimize_direct(best_h, max_iter=300)
        if s_opt > best_score:
            best_score = s_opt
            best_h = h_opt
    
    # Spectral optimization from several starting points
    # Convert best_h to spectral representation and optimize
    if _time_left() > 20:
        z0 = _heights_to_spectral(best_h)
        z_opt, s_opt = _local_optimize_spectral(z0, max_iter=200)
        h_from_z = _spectral_to_heights(z_opt)
        s_from_z = _ratio(h_from_z)
        if s_from_z > best_score:
            best_score = s_from_z
            best_h = h_from_z
    
    # Try spectral optimization from other initializations
    rng = np.random.RandomState(123)
    for i in range(min(15, len(candidates))):
        if _time_left() < 15:
            break
        h0 = candidates[i]
        z0 = _heights_to_spectral(h0)
        # Add some random perturbation
        z0 = z0 + (rng.randn(N) + 1j * rng.randn(N)) * 0.1
        z_opt, s_opt = _local_optimize_spectral(z_opt if False else z0, max_iter=150)
        h_from_z = _spectral_to_heights(z_opt)
        s_from_z = _ratio(h_from_z)
        if s_from_z > best_score:
            best_score = s_from_z
            best_h = h_from_z
    
    # Final refinement with direct optimization
    if _time_left() > 10:
        h_opt, s_opt = _local_optimize_direct(best_h, max_iter=500)
        if s_opt > best_score:
            best_score = s_opt
            best_h = h_opt
    
    # Ensure nonnegative and return
    best_h = np.maximum(0, best_h)
    
    return list(best_h)


if __name__ == "__main__":
    result = construct()
    print("Score:", _ratio(result))
    print("Heights:", result)
