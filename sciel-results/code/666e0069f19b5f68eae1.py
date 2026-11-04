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

# Fourier basis
K = 30
I = np.arange(N)
COS = np.cos(2.0 * np.pi * I[None, :] * np.arange(1, K + 1)[:, None] / N)
SIN = np.sin(2.0 * np.pi * I[None, :] * np.arange(1, K + 1)[:, None] / N)


def fourier_to_heights(params):
    """Convert Fourier parameters to nonnegative step heights."""
    a0 = params[0]
    a_coeffs = params[1::2][:K]
    b_coeffs = params[2::2][:K]
    heights = a0 + a_coeffs @ COS + b_coeffs @ SIN
    return np.clip(heights, 0.0, None)


def heights_to_fourier(heights):
    """Compute Fourier coefficients of a height vector."""
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


def ratio_from_h_fine(h):
    """Accurate score using fine grid FFT."""
    h = np.asarray(h, dtype=np.float64)
    if h.size != N:
        h = np.resize(h, N)
    h = np.clip(h, 0.0, None)
    s = h.sum()
    if s < 1e-15:
        return float('inf')

    f_fine = h[CELL]
    F = np.fft.rfft(f_fine, N_FFT)
    conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1]
    conv *= DX

    max_conv = np.max(conv)
    integral_sq = (s * W) ** 2
    if integral_sq < 1e-15:
        return float('inf')

    return max_conv / integral_sq


def ratio_coarse_lse(h, tau=0.01):
    """Fast surrogate using coarse grid convolution and log-sum-exp."""
    h = np.asarray(h, dtype=np.float64)
    if h.size != N:
        h = np.resize(h, N)
    h = np.clip(h, 0.0, None)
    s = h.sum()
    if s < 1e-15:
        return 1e10

    conv = np.convolve(h, h, mode='full') * W

    m = np.max(conv)
    if m < 1e-15:
        return 1e10
    # Scale tau relative to conv magnitude for numerical stability
    tau_eff = max(tau, m * 1e-4)
    lse = m + tau_eff * np.log(np.sum(np.exp((conv - m) / tau_eff)))

    integral_sq = (s * W) ** 2
    if integral_sq < 1e-15:
        return 1e10

    return lse / integral_sq


def smooth_fourier_objective(params, tau=0.01):
    """Smooth surrogate for Fourier parameters using coarse grid."""
    h = fourier_to_heights(params)
    return ratio_coarse_lse(h, tau)


def make_double_box(m):
    """Create a symmetric double-box structure with gap 2*m."""
    heights = np.zeros(N)
    gap = 2 * m
    non_zero_count = N - gap
    if non_zero_count <= 0 or non_zero_count % 2 != 0:
        return None
    block_len = non_zero_count // 2
    if block_len == 0:
        return None
    H = float(N) / float(non_zero_count)
    heights[0:block_len] = H
    heights[N - block_len:N] = H
    return heights


def make_multi_box(n_boxes, gap):
    """Create n_boxes equal blocks separated by gaps."""
    heights = np.zeros(N)
    total_gap = (n_boxes - 1) * gap
    non_zero_count = N - total_gap
    if non_zero_count <= 0:
        return None
    block_len = non_zero_count // n_boxes
    if block_len == 0:
        return None
    H = float(N) / float(non_zero_count)
    for i in range(n_boxes):
        start = i * (block_len + gap)
        end = start + block_len
        if end <= N:
            heights[start:end] = H
    return heights


def construct():
    """
    Construct a nonnegative step function with 600 equal intervals on [-1/4, 1/4]
    to minimize the normalized maximum of its autoconvolution.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    # Initial best: uniform
    best_h = np.ones(N, dtype=np.float64)
    best_val = ratio_from_h_fine(best_h)

    np.random.seed(42)

    # Generate diverse initializations in Fourier space
    inits = []

    # 1. Double-box structures converted to Fourier
    for m in [10, 20, 30, 50, 75, 100, 150, 200]:
        h = make_double_box(m)
        if h is not None:
            inits.append(heights_to_fourier(h))

    # 2. Multi-box structures
    for n_boxes in [3, 4, 5]:
        for gap in [5, 10, 20, 30]:
            h = make_multi_box(n_boxes, gap)
            if h is not None:
                inits.append(heights_to_fourier(h))

    # 3. Cosine-tapered boxes
    for m in [20, 50, 100]:
        h = make_double_box(m)
        if h is not None:
            gap = 2 * m
            non_zero_count = N - gap
            block_len = non_zero_count // 2
            taper = 0.5 * (1 + np.cos(np.pi * np.arange(block_len) / max(block_len - 1, 1)))
            h[0:block_len] *= taper
            h[N - block_len:N] *= taper[::-1]
            s = h.sum()
            if s > 1e-10:
                h *= N / s
            inits.append(heights_to_fourier(h))

    # 4. Random smooth perturbations of uniform
    for _ in range(8):
        h = np.ones(N)
        for k in [1, 2, 3, 4]:
            amp = np.random.uniform(0.1, 0.4)
            phase = np.random.uniform(0, 2 * np.pi)
            h += amp * np.cos(2 * np.pi * k * np.arange(N) / N + phase)
        h = np.clip(h, 0.0, None)
        s = h.sum()
        if s > 1e-10:
            h *= N / s
        inits.append(heights_to_fourier(h))

    # 5. Random Fourier initializations
    for _ in range(10):
        p0 = np.zeros(2 * K + 1)
        p0[0] = np.random.uniform(0.5, 1.5)
        p0[1:] = np.random.normal(0, 0.2, size=2 * K)
        inits.append(p0)

    # 6. Symmetric low-frequency init
    p0_sym = np.zeros(2 * K + 1)
    p0_sym[0] = 1.0
    p0_sym[1] = 0.5
    p0_sym[3] = 0.2
    inits.append(p0_sym)

    # Evaluate all initializations and keep the best few
    scored_inits = []
    for p in inits:
        h = fourier_to_heights(p)
        val = ratio_from_h_fine(h)
        scored_inits.append((val, p))
        if val < best_val:
            best_val = val
            best_h = h.copy()

    scored_inits.sort(key=lambda x: x[0])

    # Optimize from the top initializations in Fourier space
    for idx, (init_val, p0) in enumerate(scored_inits[:10]):
        if time.time() > deadline - 30.0:
            break

        try:
            res = minimize(
                smooth_fourier_objective,
                p0,
                args=(0.01,),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )

            if np.isfinite(res.fun):
                h_candidate = fourier_to_heights(res.x)
                # Renormalize to maintain L1 norm
                s = h_candidate.sum()
                if s > 1e-10:
                    h_candidate *= N / s
                val = ratio_from_h_fine(h_candidate)
                if val < best_val and np.isfinite(val):
                    best_val = val
                    best_h = h_candidate.copy()
        except Exception:
            pass

    # Final polish: Direct L-BFGS-B on heights with coarse surrogate
    if time.time() < deadline - 20.0:
        h_start = best_h.copy()
        s = h_start.sum()
        if s > 1e-10:
            h_start = h_start * (N / s)

        try:
            bounds = [(0.0, None)] * N
            res = minimize(
                lambda h: ratio_coarse_lse(h, 0.01),
                h_start,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 30}
            )
            if np.isfinite(res.fun):
                h_candidate = np.clip(res.x, 0.0, None)
                s = h_candidate.sum()
                if s > 1e-10:
                    h_candidate *= N / s
                val = ratio_from_h_fine(h_candidate)
                if val < best_val and np.isfinite(val):
                    best_val = val
                    best_h = h_candidate.copy()
        except Exception:
            pass

    # Ensure nonnegative and return as list
    best_h = np.clip(best_h, 0.0, None)
    return best_h.tolist()
