import numpy as np
import time

N = 400


def construct():
    """
    Construct a sequence of N step heights minimizing:
        value = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h)**2)

    Strategy:
    1. Represent h[i] = a * cos(2*pi*f*t[i] + 0.5*chirp*t[i]^2 + phase) + dc
    2. Grid search over f, chirp, dc to find good basins
    3. Local refinement with Nelder-Mead and coordinate descent
    """
    start_time = time.time()
    DEADLINE = 142  # 8s safety margin before 150s kill

    best_value = np.inf
    best_h = np.ones(N).astype(float)

    def fast_convolve(h):
        fft_size = 1
        while fft_size < 2 * N - 1:
            fft_size *= 2
        H = np.fft.rfft(h, n=fft_size)
        conv = np.fft.irfft(H * H, n=fft_size)
        return conv[:2 * N - 1]

    def evaluate(h):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf, False
        conv = fast_convolve(h)
        max_conv = np.max(conv)
        val = abs(2.0 * N * max_conv / (s * s))
        valid = np.all(np.isfinite(h)) and abs(s) > 1e-12
        return val, valid

    # Evaluate seed
    val_seed, _ = evaluate(best_h)
    best_value = val_seed

    t = np.linspace(0, 1, N, endpoint=False)
    t2 = t * t

    # --- Grid search over chirp parameters ---
    # h[i] = a * cos(2*pi*f*t[i] + 0.5*chirp*t2[i] + phase) + dc
    # We want large negative DC to push autoconvolution peak negative
    candidates = []

    freq_range = np.arange(2, 50, 1)  # frequencies 2 to 49
    chirp_range = np.arange(-200, 201, 20)  # chirp rates
    dc_range = np.arange(-3.0, 0.51, 0.5)  # DC offsets (negative to push peak down)
    amp_range = [1.0, 0.5, 2.0]
    phase_range = [0.0, np.pi / 2, np.pi]

    for f in freq_range:
        for cr in chirp_range:
            for dc in dc_range:
                for a in amp_range:
                    for ph in phase_range:
                        h = a * np.cos(2 * np.pi * f * t + 0.5 * cr * t2 + ph) + dc
                        s = np.sum(h)
                        if abs(s) < 1e-12:
                            continue
                        conv = fast_convolve(h)
                        val = abs(2.0 * N * np.max(conv) / (s * s))
                        if val < best_value:
                            candidates.append((val, h.copy()))
                        if time.time() - start_time > DEADLINE:
                            return best_h

    # Sort and keep top candidates
    candidates.sort(key=lambda x: x[0])
    for val_c, h_c in candidates[:50]:
        if val_c < best_value:
            best_value = val_c
            best_h = h_c.copy()

    if time.time() - start_time > DEADLINE - 5:
        return best_h

    # --- Local refinement from best candidates ---
    def refine(h_init, max_iter=200):
        """Nelder-Mead refinement."""
        if time.time() - start_time > DEADLINE:
            return h_init, evaluate(h_init)[0]

        h = h_init.copy()
        s = np.sum(h)
        if abs(s) < 1e-10:
            h = h + 0.1
            s = np.sum(h)

        # Nelder-Mead on subset of parameters
        # Use coordinate descent as fallback
        best_local_val = evaluate(h)[0]
        best_local_h = h.copy()

        # Coordinate descent with adaptive step
        lr = 0.1
        improved = True
        iters = 0
        while improved and iters < max_iter and time.time() - start_time < DEADLINE:
            improved = False
            iters += 1
            for coord in np.random.permutation(N):
                for delta in [lr, -lr]:
                    h_try = best_local_h.copy()
                    h_try[coord] += delta
                    val_try, valid = evaluate(h_try)
                    if valid and val_try < best_local_val:
                        best_local_val = val_try
                        best_local_h = h_try
                        improved = True
                        break
            if not improved:
                lr *= 0.8
                if lr < 1e-4:
                    break

        return best_local_h, best_local_val

    # Refine top candidates
    for val_c, h_c in candidates[:20]:
        if time.time() - start_time > DEADLINE - 2:
            break
        h_ref, val_ref = refine(h_c)
        if val_ref < best_value:
            best_value = val_ref
            best_h = h_ref.copy()

    # --- Additional structured patterns ---
    rng = np.random.RandomState(123)

    # Random Fourier synthesis with emphasis on high frequencies
    for n_comp in [3, 5, 8, 12]:
        for _ in range(10):
            if time.time() - start_time > DEADLINE - 2:
                break
            freqs = rng.randint(3, 50, n_comp)
            phases = rng.uniform(0, 2 * np.pi, n_comp)
            amps = rng.uniform(0.5, 2.0, n_comp)
            h_fourier = np.zeros(N)
            for f, p, a in zip(freqs, phases, amps):
                h_fourier += a * np.cos(2 * np.pi * f * t + p)
            # Add negative DC to push peak down
            dc = -np.sum(h_fourier) / N * rng.uniform(0.5, 2.0)
            h_fourier += dc
            val_f, valid = evaluate(h_fourier)
            if valid and val_f < best_value:
                best_value = val_f
                best_h = h_fourier.copy()
            # Refine
            if val_f < 2.0:
                h_ref, val_ref = refine(h_fourier, max_iter=100)
                if val_ref < best_value:
                    best_value = val_ref
                    best_h = h_ref.copy()

    # Bimodal patterns with negative bias
    for _ in range(20):
        if time.time() - start_time > DEADLINE - 2:
            break
        h = np.zeros(N)
        k = rng.randint(50, 300)
        idx = rng.choice(N, size=k, replace=False)
        h[idx] = rng.choice([-1, 1], size=k)
        dc = -np.sum(h) / N * rng.uniform(0.3, 1.5)
        h += dc
        val_b, valid = evaluate(h)
        if valid and val_b < best_value:
            best_value = val_b
            best_h = h.copy()

    # Smooth random with negative DC
    for _ in range(15):
        if time.time() - start_time > DEADLINE - 2:
            break
        h_smooth = rng.uniform(-1, 1, N)
        # Apply Gaussian-like smoothing
        kernel_size = rng.randint(3, 21) | 1
        kernel = np.exp(-np.arange(kernel_size) ** 2 / (2 * (kernel_size / 4) ** 2))
        kernel /= np.sum(kernel)
        h_smooth = np.convolve(h_smooth, kernel, mode='same')
        dc = -np.sum(h_smooth) / N * rng.uniform(0.5, 2.0)
        h_smooth += dc
        val_s, valid = evaluate(h_smooth)
        if valid and val_s < best_value:
            best_value = val_s
            best_h = h_smooth.copy()
        if val_s < 2.0:
            h_ref, val_ref = refine(h_smooth, max_iter=100)
            if val_ref < best_value:
                best_value = val_ref
                best_h = h_ref.copy()

    return best_h
