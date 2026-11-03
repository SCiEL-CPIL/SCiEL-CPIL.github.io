import numpy as np
import time

N = 600
W = 1.0 / 1200.0

def compute_ratio(h, n_grid=4000):
    """Compute max(autoconv) / l2_sq for step function h.

    h: array of length N, step heights on equal intervals of [-1/4, 1/4].
    Returns the ratio of the maximum of the autoconvolution to the L2 norm squared.
    """
    h = np.asarray(h, dtype=np.float64)
    h = np.clip(h, 0.0, None)
    if np.sum(h) < 1e-15:
        return float('inf')

    # Upsample to a fine grid for accurate convolution peak detection
    x_fine = np.linspace(-0.25, 0.25, n_grid, endpoint=False)
    # Map each fine point to its step interval index
    cell_idx = np.clip(((x_fine + 0.25) / W).astype(np.int32), 0, N - 1)
    f_fine = h[cell_idx]

    # Linear autoconvolution via FFT (zero-padded)
    n_fft = 1 << (2 * n_grid).bit_length()
    F = np.fft.rfft(f_fine, n_fft)
    conv = np.fft.irfft(F * F, n_fft)[:2 * n_grid - 1]
    dx = 0.5 / n_grid
    conv *= dx
    max_conv = np.max(conv)

    l2_sq = np.sum(f_fine ** 2) * dx
    if l2_sq < 1e-15:
        return float('inf')
    return max_conv / l2_sq


def construct():
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    rng = np.random.default_rng(42)

    i = np.arange(N)
    t = i / N  # 0 to ~1

    candidates = []

    # 1. Uniform (baseline, theoretical ratio = 1.0)
    candidates.append(np.ones(N))

    # 2. Disjoint pulse patterns (key structural family)
    # Place k equal-height pulses with gaps
    for n_pulses in [2, 3, 4, 5, 6, 8, 10]:
        pulse_width = max(1, N // (n_pulses * 2))
        gap = N // n_pulses - pulse_width
        if gap < 0:
            gap = 0
        h = np.zeros(N)
        for p in range(n_pulses):
            start = p * (N // n_pulses)
            end = min(start + pulse_width, N)
            h[start:end] = 1.0
        candidates.append(h)

    # 3. Alternating block patterns
    for block_size in [10, 20, 30, 50, 100]:
        h = np.zeros(N)
        for b in range(0, N, 2 * block_size):
            h[b:b + block_size] = 1.0
        candidates.append(h)

    # 4. Smooth bell shapes
    candidates.append(0.5 * (1 + np.cos(2 * np.pi * t)))
    candidates.append(np.maximum(0, np.cos(np.pi * (t - 0.5))))
    candidates.append(np.maximum(0, 1 - 2 * np.abs(t - 0.5)))

    # 5. Power law shapes
    for p in [0.5, 2.0, 4.0]:
        base = 1 - np.abs(2 * t - 1)
        candidates.append(np.maximum(0, base ** p))

    # 6. Gaussian-like
    for sigma in [0.1, 0.2, 0.3, 0.5]:
        candidates.append(np.exp(-0.5 * ((t - 0.5) / sigma) ** 2))

    # 7. Random smooth functions (low-pass filtered noise)
    for trial in range(20):
        noise = rng.standard_normal(N)
        # Apply simple low-pass via moving average
        k = 20
        kernel = np.ones(k) / k
        h = np.convolve(noise, kernel, mode='same')
        h = np.clip(h, 0, None)
        if np.sum(h) > 1e-10:
            h /= np.max(h)
            candidates.append(h)

    # 8. Random disjoint pulses with random positions and widths
    for trial in range(30):
        h = np.zeros(N)
        n_pulses = rng.integers(2, 10)
        for _ in range(n_pulses):
            width = rng.integers(5, 50)
            start = rng.integers(0, N - width)
            h[start:start + width] = 1.0
        if np.sum(h) > 1e-10:
            candidates.append(h)

    # Evaluate all candidates
    best_h = np.ones(N)
    best_ratio = compute_ratio(best_h)

    for h in candidates:
        h = np.clip(h, 0.0, None)
        if np.sum(h) < 1e-15:
            continue
        r = compute_ratio(h)
        if r < best_ratio:
            best_ratio = r
            best_h = h.copy()

    # Local refinement using coordinate descent
    h = best_h.copy()
    step_size = max(0.01, np.max(h) * 0.01)
    no_improve_count = 0

    for it in range(500):
        if time.time() > deadline:
            break

        improved = False
        # Try perturbing a subset of coordinates
        n_trials = 50
        idxs = rng.choice(N, size=n_trials, replace=False)

        for idx in idxs:
            h_old = h[idx]
            h[idx] = h_old + step_size
            r_plus = compute_ratio(h)
            h[idx] = h_old - step_size
            r_minus = compute_ratio(h)
            h[idx] = h_old

            if r_plus < best_ratio:
                h[idx] = h_old + step_size
                best_ratio = r_plus
                improved = True
            elif r_minus < best_ratio:
                h[idx] = h_old - step_size
                best_ratio = r_minus
                improved = True

        if improved:
            no_improve_count = 0
        else:
            no_improve_count += 1
            step_size *= 0.8
            if step_size < 1e-6 or no_improve_count > 20:
                break

    # Final evaluation
    final_r = compute_ratio(h)
    if final_r < best_ratio:
        best_ratio = final_r
        best_h = h.copy()

    return best_h.tolist()
