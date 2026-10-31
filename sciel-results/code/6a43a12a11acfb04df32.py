import numpy as np
import time

N = 600
DEADLINE = 140  # safety margin before 150s hard limit


def construct():
    start_time = time.time()

    def elapsed():
        return time.time() - start_time

    # FFT size for exact linear convolution of two length-N signals
    nfft = 2 * N

    def conv(h):
        """Exact linear autoconvolution via FFT (length 2N-1)."""
        H = np.fft.rfft(h, n=nfft)
        C = np.fft.irfft(H * H, n=nfft)
        return C[:2 * N - 1]

    def metric(h):
        """Normalized metric: max_conv / (sum_h)^2, scale-invariant."""
        s = np.sum(h)
        if s < 1e-15:
            return 1e10
        return np.max(conv(h)) / (s * s)

    def smooth_max(c, temp=0.01):
        """Smooth (log-sum-exp) approximation of max(c)."""
        cs = c / temp
        cs = cs - np.max(cs)
        return temp * (np.max(cs) + np.log(np.sum(np.exp(cs))))

    def smooth_metric(h, temp=0.01):
        s = np.sum(h)
        if s < 1e-15:
            return 1e10
        return smooth_max(conv(h), temp) / (s * s)

    # ---------------- Phase 1: structured candidates ----------------
    candidates = []
    i = np.arange(N)

    candidates.append(np.ones(N))

    # Cosine taper
    candidates.append(0.5 * (1 + np.cos(np.pi * i / (N - 1))))

    # Sinc-like
    center = N / 2 - 0.5
    sinc_h = np.sinc((i - center) / (N * 0.3))
    sinc_h = np.clip(sinc_h, 0, None)
    if sinc_h.sum() > 0:
        candidates.append(sinc_h)

    # Gaussian
    xg = (i - N / 2) / (N * 0.3)
    candidates.append(np.exp(-0.5 * xg**2))

    # Blackman
    blackman = 0.42 - 0.5 * np.cos(2 * np.pi * i / (N - 1)) + 0.08 * np.cos(4 * np.pi * i / (N - 1))
    candidates.append(np.clip(blackman, 0, None))

    # Hamming
    hamming = 0.54 - 0.46 * np.cos(2 * np.pi * i / (N - 1))
    candidates.append(np.clip(hamming, 0, None))

    # Hanning
    hanning = 0.5 - 0.5 * np.cos(2 * np.pi * i / (N - 1))
    candidates.append(np.clip(hanning, 0, None))

    # Flat top
    flat_top = (0.21557895 + 0.41663158 * np.cos(2 * np.pi * i / (N - 1)) +
                0.277263158 * np.cos(4 * np.pi * i / (N - 1)) +
                0.083578947 * np.cos(6 * np.pi * i / (N - 1)) +
                0.006947368 * np.cos(8 * np.pi * i / (N - 1)))
    candidates.append(np.clip(flat_top, 0, None))

    # Two-plateau: high at edges, low in center
    two_plat = np.ones(N)
    two_plat[N // 4:3 * N // 4] = 0.5
    candidates.append(two_plat)

    # Inverse: low at edges, high in center
    inv_plat = np.ones(N)
    inv_plat[:N // 4] = 0.3
    inv_plat[3 * N // 4:] = 0.3
    candidates.append(inv_plat)

    # Broad low-center plateaus at several widths (reduce center peak)
    for w in (0.2, 0.3, 0.4, 0.5):
        pl = np.ones(N)
        lo = int(N * w)
        hi = int(N * (1 - w))
        if hi > lo:
            pl[lo:hi] = 0.4
            candidates.append(pl)

    best_val = 1e10
    best_h = np.ones(N)
    for h in candidates:
        if np.sum(h) < 1e-10:
            continue
        v = metric(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()

    # ---------------- Phase 2: cosine-series Remez-like equioscillation ----------------
    # Parameterize h via a cosine series and iteratively reduce the dominant
    # convolution peak by moving against its gradient (flattening the peak).
    K = 16

    def cheb_to_h(coeffs):
        h = np.full(N, coeffs[0])
        for k in range(1, K + 1):
            h += coeffs[k] * np.cos(2 * np.pi * k * i / N)
        return np.maximum(h, 0.0)

    def fit_cosine(h):
        A = np.zeros((N, K + 1))
        A[:, 0] = 1.0
        for k in range(1, K + 1):
            A[:, k] = np.cos(2 * np.pi * k * i / N)
        coeffs, _, _, _ = np.linalg.lstsq(A, h, rcond=None)
        return coeffs

    coeffs = fit_cosine(best_h)

    for remez_iter in range(40):
        if elapsed() > DEADLINE - 45:
            break
        h = cheb_to_h(coeffs)
        s = np.sum(h)
        if s < 1e-15:
            break
        c = conv(h)
        max_idx = int(np.argmax(c))
        max_val = c[max_idx]
        k = max_idx if max_idx < N else (2 * N - 1 - max_idx)

        # Gradient of normalized conv at lag k w.r.t. each cosine coefficient.
        grad = np.zeros(K + 1)
        # c[k] = sum_j h[j] h[j+k]  (0<=j<N, j+k<N) ; by symmetry also j-k
        j_idx = np.arange(N - k) if k < N else np.arange(N)
        if k < N:
            a = h[:N - k]
            b = h[k:]
        else:
            a = h
            b = h
        for jj in range(K + 1):
            basis = np.cos(2 * np.pi * jj * i / N)
            # d/dc_j of sum a*b = sum (db_j * a + b * da_j)
            g = 0.0
            if k < N:
                da = basis[:N - k]
                db = basis[k:]
                g = np.sum(db * a) + np.sum(b * da)
            grad[jj] = g / (s * s)

        gnorm = np.max(np.abs(grad))
        if gnorm < 1e-15:
            break
        lr = 0.02 * max_val / gnorm
        lr = min(lr, 0.1)
        improved = False
        for shrink in (1.0, 0.5, 0.2, 0.05):
            new_coeffs = coeffs - lr * shrink * grad
            new_h = cheb_to_h(new_coeffs)
            if np.sum(new_h) > 1e-15:
                nv = metric(new_h)
                if nv < best_val - 1e-12:
                    best_val = nv
                    best_h = new_h.copy()
                    coeffs = new_coeffs
                    improved = True
                    break
        if not improved:
            # no coefficient move helps; stop the Remez loop
            break

    # ---------------- Phase 3: coordinate descent (fast FFT metric) ----------------
    current_h = best_h.copy()
    current_val = best_val
    np.random.seed(42)

    no_improve = 0
    max_no_improve = 20000

    while elapsed() < DEADLINE - 12 and no_improve < max_no_improve:
        idx = int(np.random.randint(N))
        orig = current_h[idx]
        mean_h = np.mean(current_h)
        step_factors = np.array([0.05, 0.1, 0.2, 0.5, 1.0])
        step_sizes = step_factors * mean_h
        cand_vals = np.array([max(0.0, v) for v in
                              list(orig + step_sizes) + list(orig - step_sizes) + [0.0]])
        best_local_val = current_val
        best_local_v = orig
        for cv in cand_vals:
            if abs(cv - orig) < 1e-10:
                continue
            current_h[idx] = cv
            v = metric(current_h)
            if v < best_local_val:
                best_local_val = v
                best_local_v = cv
        if best_local_val < current_val - 1e-12:
            current_h[idx] = best_local_v
            current_val = best_local_val
            no_improve = 0
            if current_val < best_val:
                best_val = current_val
                best_h = current_h.copy()
        else:
            current_h[idx] = orig
            no_improve += 1

        # Periodic random restart toward a fresh smooth shape
        if no_improve % 1500 == 0 and no_improve > 0:
            pert = current_h.copy()
            pert += 0.05 * np.random.randn(N) * np.mean(current_h)
            pert = np.clip(pert, 0, None)
            if pert.sum() > 0:
                pv = metric(pert)
                if pv < current_val:
                    current_h = pert
                    current_val = pv
                    no_improve = 0
                    if pv < best_val:
                        best_val = pv
                        best_h = current_h.copy()

    # ---------------- Phase 4: high-precision polish (small exchange moves) ----------------
    if elapsed() < DEADLINE - 3:
        mean_h = np.mean(best_h)
        for step_frac in (0.01, 0.003, 0.001):
            if elapsed() > DEADLINE - 3:
                break
            for _ in range(3000):
                if elapsed() > DEADLINE - 2:
                    break
                a, b = np.random.choice(N, 2, replace=False)
                delta = step_frac * np.random.uniform(-1, 1) * mean_h
                new_h = best_h.copy()
                new_h[a] += delta
                new_h[b] -= delta
                if new_h[a] >= 0 and new_h[b] >= 0:
                    v = metric(new_h)
                    if v < best_val - 1e-13:
                        best_val = v
                        best_h = new_h

    # ---------------- Final: ensure validity ----------------
    best_h = np.maximum(best_h, 0)
    if np.sum(best_h) < 1e-15:
        best_h = np.ones(N)

    return best_h.tolist()
