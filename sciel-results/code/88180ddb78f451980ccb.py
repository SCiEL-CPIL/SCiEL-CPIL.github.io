import numpy as np
import time
import scipy.optimize as opt
from scipy.optimize import differential_evolution, minimize

N = 600
W = 0.5 / N
X = -0.25 + (np.arange(N) + 0.5) * W

# Precompute fine grid for accurate convolution
FINE_N = 6000
X_FINE = np.linspace(-0.25, 0.25, FINE_N, endpoint=False)
DX = 0.5 / FINE_N
CELL = np.clip(((X_FINE + 0.25) / W).astype(np.int64), 0, N - 1)
N_FFT = 1 << (2 * FINE_N - 1).bit_length()

# Precompute Fourier basis for parameterization
K_FOURIER = 40
I = np.arange(N)
FREQS = np.arange(1, K_FOURIER + 1)
COS_BASIS = np.cos(2.0 * np.pi * FREQS[:, None] * I[None, :] / N)
SIN_BASIS = np.sin(2.0 * np.pi * FREQS[:, None] * I[None, :] / N)


def compute_ratio_fine(h):
    """Compute max(autoconv) / (int f)^2 using FFT on fine grid."""
    h = np.asarray(h, dtype=np.float64)
    if h.size != N:
        h = np.resize(h, N)
    h = np.clip(h, 0.0, None)
    s = h.sum() * W
    if s < 1e-15:
        return float('inf')
    f = h[CELL]
    F = np.fft.rfft(f, N_FFT)
    conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1] * DX
    max_conv = np.max(conv)
    if max_conv < 1e-15:
        return float('inf')
    return float(max_conv / (s * s))


def compute_ratio_coarse(h, tau=0.005):
    """Fast smooth surrogate using coarse grid + log-sum-exp for max."""
    h = np.asarray(h, dtype=np.float64)
    if h.size != N:
        h = np.resize(h, N)
    h = np.clip(h, 0.0, None)
    s = h.sum() * W
    if s < 1e-15:
        return 1e10

    conv = np.convolve(h, h, mode='full') * W

    m = np.max(conv)
    if m < 1e-15:
        return 1e10

    scaled = (conv - m) / tau
    scaled = np.clip(scaled, -50, 50)
    lse = m + tau * np.log(np.sum(np.exp(scaled)))

    integral_sq = (s * s)
    if integral_sq < 1e-15:
        return 1e10

    return float(lse / integral_sq)


def fourier_to_h(params):
    """Convert Fourier parameters to nonnegative step heights."""
    a0 = params[0]
    a_coeffs = params[1::2][:K_FOURIER]
    b_coeffs = params[2::2][:K_FOURIER]
    heights = a0 + a_coeffs @ COS_BASIS + b_coeffs @ SIN_BASIS
    return np.clip(heights, 0.0, None)


def h_to_fourier(heights):
    """Compute Fourier coefficients of a height vector."""
    a0 = np.mean(heights)
    a_coeffs = np.zeros(K_FOURIER)
    b_coeffs = np.zeros(K_FOURIER)
    for k in range(K_FOURIER):
        a_coeffs[k] = (2.0 / N) * np.sum(heights * COS_BASIS[k])
        b_coeffs[k] = (2.0 / N) * np.sum(heights * SIN_BASIS[k])
    params = np.zeros(2 * K_FOURIER + 1)
    params[0] = a0
    params[1::2] = a_coeffs
    params[2::2] = b_coeffs
    return params


def smooth_fourier_ratio(params, tau=0.005):
    """Smooth surrogate for Fourier parameters using coarse grid LSE."""
    h = fourier_to_h(params)
    return compute_ratio_coarse(h, tau=tau)


def smooth_gaussian_ratio(params, tau=0.005):
    """Gaussian superposition model."""
    base = params[0]
    h = np.full(N, base)
    for k in range(1, len(params), 3):
        amp = params[k]
        mu = params[k + 1]
        sig = max(params[k + 2], 1e-4)
        h += amp * np.exp(-0.5 * ((X - mu) / sig) ** 2)
    return compute_ratio_coarse(np.clip(h, 0.0, None), tau=tau)


def chirp_to_h(params):
    """
    Convert chirp parameters to nonnegative step heights.
    params: [sigma, f0, chirp_rate, phase, amp]
    h[k] = amp * exp(-((X[k])/(2*sigma))^2) * cos(f0*X[k] + 0.5*chirp_rate*X[k]^2 + phase)^2
    This ensures non-negativity by construction.
    """
    sigma = max(params[0], 1e-4)
    f0 = params[1]
    chirp = params[2]
    phase = params[3]
    amp = max(params[4], 1e-6)
    
    envelope = np.exp(-0.5 * (X / sigma) ** 2)
    carrier = np.cos(f0 * X + 0.5 * chirp * X ** 2 + phase)
    h = amp * envelope * carrier ** 2
    return np.clip(h, 0.0, None)


def smooth_chirp_ratio(params, tau=0.005):
    """Smooth surrogate for chirp parameters."""
    h = chirp_to_h(params)
    return compute_ratio_coarse(h, tau=tau)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    best_h = np.ones(N)
    best_val = compute_ratio_fine(best_h)

    def update_best(h, val):
        nonlocal best_h, best_val
        if val < best_val and np.isfinite(val):
            best_val = val
            best_h = h.copy()

    rng = np.random.default_rng(42)

    # === Phase 1: Fourier parameterization with L-BFGS-B on smooth surrogate ===
    inits = []

    # 1a. Uniform
    p0 = np.zeros(2 * K_FOURIER + 1)
    p0[0] = 1.0
    inits.append(p0)

    # 1b. Gaussian-like (low frequency decay)
    for sigma in [0.05, 0.08, 0.1, 0.12, 0.15, 0.2]:
        p0 = np.zeros(2 * K_FOURIER + 1)
        p0[0] = 1.0
        for k in range(K_FOURIER):
            amp = np.exp(-((k + 1) ** 2) * sigma ** 2)
            p0[2 * k + 1] = amp
        inits.append(p0)

    # 1c. Random low-frequency perturbations
    for _ in range(15):
        p0 = np.zeros(2 * K_FOURIER + 1)
        p0[0] = float(rng.uniform(0.5, 1.5))
        num_modes = int(rng.integers(5, 25))
        for k in range(min(num_modes, K_FOURIER)):
            p0[2 * k + 1] = float(rng.normal(0, 0.15))
            p0[2 * k + 2] = float(rng.normal(0, 0.15))
        inits.append(p0)

    # 1d. Symmetric "dip" shapes (lower in center)
    for dip_width in [0.08, 0.12, 0.15, 0.2, 0.25]:
        p0 = np.zeros(2 * K_FOURIER + 1)
        p0[0] = 1.0
        for k in range(K_FOURIER):
            amp = -0.5 * np.exp(-((k + 1) ** 2) * dip_width ** 2)
            p0[2 * k + 1] = amp
        inits.append(p0)

    # 1e. Bimodal shapes (two peaks)
    for sep in [0.1, 0.15, 0.2]:
        p0 = np.zeros(2 * K_FOURIER + 1)
        p0[0] = 0.5
        for k in range(K_FOURIER):
            amp = 0.8 * np.cos(2 * np.pi * (k + 1) * sep)
            p0[2 * k + 1] = amp
        inits.append(p0)

    # 1f. Cosine window shapes
    for freq in [1, 2, 3, 4]:
        p0 = np.zeros(2 * K_FOURIER + 1)
        p0[0] = 0.8
        p0[2 * freq] = 0.3
        inits.append(p0)

    # Optimize each initialization with L-BFGS-B
    for i, p0 in enumerate(inits):
        if time.time() > deadline - 40.0:
            break
        try:
            res = opt.minimize(
                smooth_fourier_ratio,
                p0,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-13, 'gtol': 1e-9, 'maxls': 50}
            )
            if np.isfinite(res.fun):
                h_candidate = fourier_to_h(res.x)
                val = compute_ratio_fine(h_candidate)
                if val < best_val and np.isfinite(val):
                    update_best(h_candidate, val)
        except Exception:
            pass

    # === Phase 2: Chirp parameterization with Nelder-Mead ===
    # Chirp signals have low autocorrelation properties.
    # We use h(x) = A * exp(-(x/sigma)^2 / 2) * cos(f0*x + beta*x^2/2 + phi)^2
    # This is always non-negative.
    
    chirp_inits = []
    for sigma in [0.05, 0.08, 0.1, 0.15]:
        for f0 in [0, 10, 20, 40]:
            for beta in [0, 50, 100, 200]:
                for phi in [0, np.pi/2, np.pi]:
                    p0 = np.array([sigma, f0, beta, phi, 1.0])
                    chirp_inits.append(p0)
    
    # Add some random chirp inits
    for _ in range(20):
        sigma = float(rng.uniform(0.03, 0.2))
        f0 = float(rng.uniform(0, 50))
        beta = float(rng.uniform(0, 300))
        phi = float(rng.uniform(0, 2*np.pi))
        amp = float(rng.uniform(0.5, 2.0))
        chirp_inits.append(np.array([sigma, f0, beta, phi, amp]))

    for p0 in chirp_inits:
        if time.time() > deadline - 30.0:
            break
        try:
            result = opt.minimize(
                smooth_chirp_ratio,
                p0,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
            )
            if np.isfinite(result.fun):
                h_opt = chirp_to_h(result.x)
                val = compute_ratio_fine(h_opt)
                if val < best_val and np.isfinite(val):
                    update_best(h_opt, val)
        except Exception:
            pass

    # === Phase 3: Refine best solution in Fourier space with L-BFGS-B ===
    if time.time() < deadline - 20.0:
        p_best = h_to_fourier(best_h)
        try:
            res = opt.minimize(
                smooth_fourier_ratio,
                p_best,
                method='L-BFGS-B',
                options={'maxiter': 600, 'ftol': 1e-15, 'gtol': 1e-11, 'maxls': 60}
            )
            if np.isfinite(res.fun):
                h_candidate = fourier_to_h(res.x)
                val = compute_ratio_fine(h_candidate)
                if val < best_val and np.isfinite(val):
                    update_best(h_candidate, val)
        except Exception:
            pass

    # Final safety check
    best_h = np.maximum(best_h, 0.0)
    return list(best_h)
