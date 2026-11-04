import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 600
W = 1.0 / 1200.0  # width of each interval
DEADLINE = 135.0  # safety margin before 150s kill

# Fine grid for accurate autoconvolution
FINE_N = 6000
X_FINE = np.linspace(-0.25, 0.25, FINE_N, endpoint=False)
DX = 0.5 / FINE_N
CELL = np.clip(((X_FINE + 0.25) / W).astype(np.int64), 0, N - 1)
N_FFT = 1 << (2 * FINE_N - 1).bit_length()

# Precompute Fourier basis for fast conversion
K = 20
I = np.arange(N)
COS = np.cos(2.0 * np.pi * I[None, :] * np.arange(1, K + 1)[:, None] / N)
SIN = np.sin(2.0 * np.pi * I[None, :] * np.arange(1, K + 1)[:, None] / N)


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


def ratio_from_h_coarse(h, tau=0.01):
    """
    Fast surrogate using coarse grid convolution and log-sum-exp smoothing.
    This is used for optimization.
    """
    h = np.asarray(h, dtype=np.float64)
    if h.size != N:
        h = np.resize(h, N)
    h = np.clip(h, 0.0, None)
    s = h.sum()
    if s < 1e-15:
        return 1e10

    # Coarse convolution
    conv = np.convolve(h, h, mode='full') * W

    # Log-sum-exp for smooth max
    m = np.max(conv)
    if m < 1e-15:
        return 1e10
    lse = m + tau * np.log(np.sum(np.exp((conv - m) / tau)))

    integral_sq = (s * W) ** 2
    if integral_sq < 1e-15:
        return 1e10

    return lse / integral_sq


def boxcar_to_heights(params):
    """
    Convert boxcar parameters to nonnegative step heights.
    params: [amp0, pos0, width0, amp1, pos1, width1, ..., ampM, posM, widthM]
    Each boxcar is centered at pos (in [0,1]), has width in [0.01, 1.0], amplitude in [0, 3].
    The height at bin i is the sum of amplitudes of all boxcars covering bin i.
    """
    M = len(params) // 3
    heights = np.zeros(N, dtype=np.float64)
    bin_centers = (np.arange(N) + 0.5) / N  # bin centers in [0,1]

    for m in range(M):
        amp = params[3 * m]
        pos = params[3 * m + 1]
        width = params[3 * m + 2]

        # Boxcar covers [pos - width/2, pos + width/2] (circular)
        left = pos - width / 2.0
        right = pos + width / 2.0

        # Handle circular wrap
        if left < 0:
            # Part wraps around
            mask1 = bin_centers >= (left + 1.0)
            mask2 = bin_centers <= right
            heights[mask1] += amp
            heights[mask2] += amp
        elif right > 1:
            # Part wraps around
            mask1 = bin_centers >= left
            mask2 = bin_centers <= (right - 1.0)
            heights[mask1] += amp
            heights[mask2] += amp
        else:
            mask = (bin_centers >= left) & (bin_centers <= right)
            heights[mask] += amp

    return np.clip(heights, 0.0, None)


def smooth_objective_boxcar(params, tau=0.01):
    """
    Smooth surrogate: log-sum-exp of conv / (integral)^2.
    Uses coarse grid for speed.
    """
    h = boxcar_to_heights(params)
    return ratio_from_h_coarse(h, tau)


def fourier_to_heights(params, K):
    """
    Convert Fourier parameters to nonnegative step heights.
    params: [a0, a1, b1, a2, b2, ..., aK, bK]
    h[i] = max(0, a0 + sum_{k=1}^{K} a_k cos(2*pi*k*i/N) + b_k sin(2*pi*k*i/N))
    """
    a0 = params[0]
    a_coeffs = params[1::2][:K]
    b_coeffs = params[2::2][:K]

    heights = a0 + a_coeffs @ COS + b_coeffs @ SIN
    heights = np.clip(heights, 0.0, None)
    return heights


def heights_to_fourier(heights, K):
    """
    Compute Fourier coefficients of a height vector.
    Returns [a0, a1, b1, ..., aK, bK]
    """
    a0 = np.mean(heights)
    a_coeffs = np.zeros(K)
    b_coeffs = np.zeros(K)
    for k in range(1, K + 1):
        a_coeffs[k - 1] = (2.0 / N) * np.sum(heights * np.cos(2.0 * np.pi * k * I / N))
        b_coeffs[k - 1] = (2.0 / N) * np.sum(heights * np.sin(2.0 * np.pi * k * I / N))

    params = np.zeros(2 * K + 1)
    params[0] = a0
    params[1::2] = a_coeffs
    params[2::2] = b_coeffs
    return params


def smooth_objective_fourier(params, K, tau=0.01):
    """
    Smooth surrogate: log-sum-exp of conv / (integral)^2.
    Uses coarse grid for speed.
    """
    h = fourier_to_heights(params, K)
    return ratio_from_h_coarse(h, tau)


def construct():
    """
    Construct a nonnegative step function with 600 equal intervals on [-1/4, 1/4]
    to minimize the normalized maximum of its autoconvolution.
    """
    start_time = time.time()
    deadline = start_time + DEADLINE

    # Initial best: uniform
    best_h = np.ones(N, dtype=np.float64)
    best_val = ratio_from_h(best_h)

    np.random.seed(42)

    # Strategy 1: Boxcar representation with Differential Evolution
    # M boxcars, each with (amp, pos, width)
    M = 6
    bounds = []
    for _ in range(M):
        bounds.append((0.0, 3.0))   # amplitude
        bounds.append((0.0, 1.0))   # position
        bounds.append((0.01, 0.5))  # width

    if time.time() < deadline - 45.0:
        try:
            de_result = differential_evolution(
                smooth_objective_boxcar,
                bounds,
                args=(0.01,),
                seed=42,
                maxiter=80,
                popsize=12,
                tol=1e-5,
                polish=False
            )
            if de_result.success and np.isfinite(de_result.fun):
                h_candidate = boxcar_to_heights(de_result.x)
                val = ratio_from_h(h_candidate)
                if val < best_val and np.isfinite(val):
                    best_val = val
                    best_h = h_candidate.copy()
        except Exception:
            pass

    # Strategy 2: Fourier representation with L-BFGS-B
    K = 20
    inits = []

    # Near-uniform with small perturbations
    for _ in range(4):
        p0 = np.zeros(2 * K + 1)
        p0[0] = 1.0
        for k in range(1, K + 1):
            p0[2 * k - 1] = np.random.normal(0, 0.1)
            p0[2 * k] = np.random.normal(0, 0.1)
        inits.append(p0)

    # Stronger modulation
    for _ in range(4):
        p0 = np.zeros(2 * K + 1)
        p0[0] = 1.0
        for k in range(1, K + 1):
            p0[2 * k - 1] = np.random.normal(0, 0.3)
            p0[2 * k] = np.random.normal(0, 0.3)
        inits.append(p0)

    # Dominant low-frequency modes
    for k_dom in [1, 2, 3]:
        p0 = np.zeros(2 * K + 1)
        p0[0] = 1.0
        p0[2 * k_dom - 1] = np.random.uniform(0.2, 0.5)
        p0[2 * k_dom] = np.random.uniform(0.2, 0.5)
        for k in range(1, K + 1):
            if k != k_dom:
                p0[2 * k - 1] = np.random.normal(0, 0.05)
                p0[2 * k] = np.random.normal(0, 0.05)
        inits.append(p0)

    # Random
    for _ in range(4):
        p0 = np.zeros(2 * K + 1)
        p0[0] = np.random.uniform(0.5, 1.5)
        for k in range(1, K + 1):
            p0[2 * k - 1] = np.random.normal(0, 0.4)
            p0[2 * k] = np.random.normal(0, 0.4)
        inits.append(p0)

    # Initialize from current best_h
    p0_best = heights_to_fourier(best_h, K)
    inits.append(p0_best)
    p0_perturbed = p0_best + np.random.normal(0, 0.05, size=2 * K + 1)
    inits.append(p0_perturbed)

    for p0 in inits:
        if time.time() > deadline - 40.0:
            break
        try:
            res = minimize(
                smooth_objective_fourier,
                p0,
                args=(K, 0.01),
                method='L-BFGS-B',
                options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 40}
            )
            if np.isfinite(res.fun):
                h_candidate = fourier_to_heights(res.x, K)
                val = ratio_from_h(h_candidate)
                if val < best_val and np.isfinite(val):
                    best_val = val
                    best_h = h_candidate.copy()
        except Exception:
            pass

    # Strategy 3: Direct L-BFGS-B on heights with coarse surrogate
    if time.time() < deadline - 25.0:
        h_start = best_h.copy()
        s = h_start.sum()
        if s > 0:
            h_start = h_start / s * N  # normalize to sum = N

        try:
            bounds = [(0, None)] * N
            res = minimize(
                lambda h: ratio_from_h_coarse(h, 0.01),
                h_start,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 80, 'ftol': 1e-10, 'gtol': 1e-6, 'maxls': 30}
            )
            if np.isfinite(res.fun):
                h_candidate = res.x
                val = ratio_from_h(h_candidate)
                if val < best_val and np.isfinite(val):
                    best_val = val
                    best_h = h_candidate.copy()
        except Exception:
            pass

    # Strategy 4: Coordinate descent polish on the 600 bin heights
    if time.time() < deadline - 5.0:
        h = best_h.copy()
        s = h.sum()
        if s > 0:
            h = h / s * N  # normalize to sum = N

        step_size = 0.05
        for it in range(3000):
            if time.time() > deadline - 1.0:
                break
            # Pick a random coordinate
            i = np.random.randint(N)
            best_val_i = ratio_from_h(h)
            best_h_i = h[i]

            for delta in [-2 * step_size, -1.5 * step_size, -step_size, -0.5 * step_size,
                          0.5 * step_size, step_size, 1.5 * step_size, 2 * step_size]:
                h[i] = max(0.0, best_h_i + delta)
                val = ratio_from_h(h)
                if val < best_val_i:
                    best_val_i = val
                    best_h_i = h[i]

            h[i] = best_h_i
            if best_val_i < best_val:
                best_val = best_val_i
                best_h = h.copy()
                # Reduce step size on improvement
                step_size *= 0.95

            # Occasionally try larger jumps
            if it % 100 == 0:
                i = np.random.randint(N)
                h[i] = max(0.0, h[i] + np.random.uniform(-1.0, 1.0))
                val = ratio_from_h(h)
                if val < best_val:
                    best_val = val
                    best_h = h.copy()

    # Final check
    final_score = ratio_from_h(best_h)
    if final_score < best_val:
        best_val = final_score

    return list(best_h)
