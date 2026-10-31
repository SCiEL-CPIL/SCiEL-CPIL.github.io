import time
import numpy as np

N = 600

def compute_score_fast(h):
    """Compute normalized max of autoconvolution efficiently."""
    h = np.asarray(h, dtype=np.float64)
    h = np.maximum(h, 0)
    s = np.sum(h)
    if s < 1e-15:
        return 1e10
    n = len(h)
    nfft = 1
    while nfft < 2 * n:
        nfft <<= 1
    H = np.fft.rfft(h, nfft)
    autoconv = np.fft.irfft(H * H.conjugate(), nfft)[:2*n-1]
    return np.max(autoconv) / (s * s)

def construct():
    start = time.time()
    deadline = start + 145

    x = np.linspace(-0.25, 0.25, N, endpoint=False)

    best_val = 1e10
    best_h = None

    # Evaluate parametric families
    candidates = []

    # Uniform
    candidates.append(np.ones(N))

    # Cosine modulations
    for freq in [1, 2, 3]:
        for a in np.linspace(-0.95, 0.95, 25):
            h = 1.0 + a * np.cos(2 * np.pi * freq * x)
            candidates.append(np.maximum(h, 0))

    # Power-law shapes
    for p in np.linspace(0.3, 5.0, 30):
        candidates.append(np.abs(x / 0.25) ** p)
        candidates.append((1.0 - np.abs(x / 0.25)) ** p)

    # Two-lobe (edge-weighted)
    for width in np.linspace(0.1, 0.4, 15):
        h = np.zeros(N)
        mask = np.abs(x) > (0.25 - width)
        h[mask] = 1.0
        candidates.append(h)

    for h in candidates:
        if np.sum(h) < 1e-10:
            continue
        val = compute_score_fast(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()

    # Local coordinate descent refinement
    if best_h is not None and time.time() < deadline - 10:
        h_cur = best_h.copy()
        val_cur = best_val

        for _ in range(50):
            if time.time() > deadline - 5:
                break
            improved = False
            for i in range(0, N, 5):
                if time.time() > deadline - 5:
                    break
                for delta in [0.02, -0.02, 0.05, -0.05]:
                    h_try = h_cur.copy()
                    h_try[i] = max(0.0, h_cur[i] + delta)
                    val_try = compute_score_fast(h_try)
                    if val_try < val_cur:
                        h_cur = h_try
                        val_cur = val_try
                        improved = True
                        break
            if not improved:
                break

        if val_cur < best_val:
            best_val = val_cur
            best_h = h_cur

    if best_h is None:
        return [1.0] * N

    return list(best_h)
