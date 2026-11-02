import numpy as np
import time

N = 400
DELTA = 0.5 / N


def _compute_c3_bound(heights):
    """Compute an upper bound on C3 based on the autoconvolution L2 norm."""
    n_fft = 1 << (2 * N).bit_length()
    h_pad = np.zeros(n_fft)
    h_pad[:N] = heights
    H = np.fft.fft(h_pad)
    conv = np.fft.ifft(H * np.conj(H)).real
    l2_sq = np.sum(conv ** 2)
    return DELTA ** 1.5 * np.sqrt(l2_sq)


def construct():
    """Construct a signed step function to minimize the autoconvolution upper bound."""
    start_time = time.time()
    DEADLINE = 145

    best_bound = float('inf')
    best_heights = np.ones(N)

    x = -0.25 + (np.arange(N) + 0.5) * DELTA

    # Sine-based sign patterns with varying frequency and phase
    for n_half in range(1, 40):
        if time.time() - start_time > DEADLINE:
            break
        for phase_idx in range(16):
            if time.time() - start_time > DEADLINE:
                break
            phase = 2 * np.pi * phase_idx / 16
            arg = 2 * np.pi * n_half * x + phase
            signs = np.sign(np.sin(arg))
            signs[signs == 0] = 1.0
            bound = _compute_c3_bound(signs)
            if bound < best_bound:
                best_bound = bound
                best_heights = signs.copy()

    # Random sign patterns
    rng = np.random.default_rng(424242)
    for _ in range(200):
        if time.time() - start_time > DEADLINE:
            break
        heights = rng.choice([-1.0, 1.0], size=N)
        bound = _compute_c3_bound(heights)
        if bound < best_bound:
            best_bound = bound
            best_heights = heights.copy()

    return best_heights
