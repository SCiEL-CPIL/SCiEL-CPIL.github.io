import numpy as np
import time
import scipy.optimize as opt

N = 600
W = 1.0 / 1200.0
X = np.linspace(-0.25, 0.25, N, endpoint=False)

# Precompute fine grid for fast convolution ratio
FINE_N = 8000
X_FINE = np.linspace(-0.25, 0.25, FINE_N, endpoint=False)
DX = 0.5 / FINE_N
CELL = np.clip(((X_FINE + 0.25) / W).astype(np.int64), 0, N - 1)
N_FFT = 1 << (2 * FINE_N - 1).bit_length()

def ratio_from_h(h):
    """Compute max(autoconv) / ||h||_2^2 using FFT on fine grid."""
    h = np.asarray(h, dtype=np.float64)
    if h.size != N:
        h = np.resize(h, N)
    h = np.clip(h, 0.0, None)
    s = h.sum()
    if s < 1e-15:
        return float('inf')
    f = h[CELL]
    F = np.fft.rfft(f, N_FFT)
    conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1] * DX
    l2 = np.sum(f * f) * DX
    if l2 < 1e-15:
        return float('inf')
    return float(np.max(conv) / l2)

def smooth_ratio_from_h(h, beta=80.0):
    """Compute smooth approximation of max(autoconv) / ||h||_2^2 using log-sum-exp."""
    h = np.asarray(h, dtype=np.float64)
    if h.size != N:
        h = np.resize(h, N)
    h = np.clip(h, 0.0, None)
    s = h.sum()
    if s < 1e-15:
        return float('inf')
    f = h[CELL]
    F = np.fft.rfft(f, N_FFT)
    conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1] * DX
    l2 = np.sum(f * f) * DX
    if l2 < 1e-15:
        return float('inf')
    conv_max = np.max(conv)
    if conv_max <= 0:
        return 0.0
    log_sum = conv_max + np.log(np.sum(np.exp((conv - conv_max) * beta)) / len(conv)) / beta
    return float(log_sum / l2)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    best_h = np.ones(N)
    best_val = ratio_from_h(best_h)

    def update_best(h, val):
        nonlocal best_h, best_val
        if val < best_val:
            best_val = val
            best_h = h.copy()

    rng = np.random.default_rng(12345)

    # === Phase 1: Fourier-domain optimization ===
    N_HARM = 35
    n_idx = np.arange(N)
    cos_bases = [np.cos(2 * np.pi * k * n_idx / N) for k in range(1, N_HARM + 1)]
    sin_bases = [np.sin(2 * np.pi * k * n_idx / N) for k in range(1, N_HARM + 1)]

    def fourier_to_h(coeffs):
        h = np.full(N, coeffs[0])
        for k in range(1, N_HARM + 1):
            a_k = coeffs[2 * k - 1]
            b_k = coeffs[2 * k]
            if abs(a_k) > 1e-10:
                h += a_k * cos_bases[k - 1]
            if abs(b_k) > 1e-10:
                h += b_k * sin_bases[k - 1]
        return np.clip(h, 0.0, None)

    def fourier_smooth_ratio(coeffs, beta=60.0):
        h = fourier_to_h(coeffs)
        return smooth_ratio_from_h(h, beta)

    def fourier_ratio(coeffs):
        h = fourier_to_h(coeffs)
        return ratio_from_h(h)

    n_fourier_starts = 40

    for start in range(n_fourier_starts):
        if time.time() > deadline - 50.0:
            break

        coeffs = np.zeros(2 * N_HARM + 1)
        coeffs[0] = float(rng.uniform(0.3, 1.5))
        for k in range(1, N_HARM + 1):
            scale = 1.0 / (1.0 + 0.15 * k)
            coeffs[2 * k - 1] = float(rng.uniform(-0.35 * scale, 0.35 * scale))
            coeffs[2 * k] = float(rng.uniform(-0.35 * scale, 0.35 * scale))

        try:
            result = opt.minimize(
                fourier_smooth_ratio,
                coeffs,
                method='Nelder-Mead',
                options={'maxiter': 350, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
            )
            h_opt = fourier_to_h(result.x)
            v = ratio_from_h(h_opt)
            if v < best_val:
                update_best(h_opt, v)
        except Exception:
            pass

    # === Phase 2: Gaussian superposition model ===
    def model_to_h(params):
        h = np.full(N, params[0])
        n_g = (len(params) - 1) // 3
        for k in range(n_g):
            amp = params[1 + 3 * k]
            mu = params[2 + 3 * k]
            sig = params[3 + 3 * k]
            if sig < 1e-6:
                sig = 1e-6
            h += amp * np.exp(-0.5 * ((X - mu) / sig) ** 2)
        return np.clip(h, 0.0, None)

    def nll_smooth(params, beta=60.0):
        h = model_to_h(params)
        return smooth_ratio_from_h(h, beta)

    n_gaussians_options = [1, 2, 3, 4]

    for n_g in n_gaussians_options:
        n_starts = 25
        for start in range(n_starts):
            if time.time() > deadline - 35.0:
                break

            base = float(rng.uniform(0.0, 0.6))
            params = np.array([base])
            for _ in range(n_g):
                amp = float(rng.uniform(0.1, 3.0))
                mu = float(rng.uniform(-0.22, 0.22))
                sig = float(rng.uniform(0.015, 0.18))
                params = np.append(params, [amp, mu, sig])

            try:
                result = opt.minimize(
                    nll_smooth,
                    params,
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
                )
                h_opt = model_to_h(result.x)
                v = ratio_from_h(h_opt)
                if v < best_val:
                    update_best(h_opt, v)
            except Exception:
                pass

    # === Phase 3: Spatial-domain coordinate descent refinement ===
    if time.time() < deadline - 20.0:
        rng2 = np.random.default_rng(999)
        cur_h = best_h.copy()
        cur_val = best_val

        def quick_ratio(h):
            f = h[CELL]
            F = np.fft.rfft(f, N_FFT)
            conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1] * DX
            l2 = np.sum(f * f) * DX
            if l2 < 1e-15:
                return float('inf')
            return float(np.max(conv) / l2)

        n_iter = 15000
        for it in range(n_iter):
            if time.time() > deadline - 10.0:
                break

            n_coords = int(rng2.integers(1, 20))
            idxs = rng2.choice(N, size=n_coords, replace=False)

            h_try = cur_h.copy()
            for idx in idxs:
                factor = float(rng2.uniform(0.82, 1.18))
                h_try[idx] = max(0.0, cur_h[idx] * factor)

            v = quick_ratio(h_try)
            if v < cur_val:
                cur_val = v
                cur_h = h_try
                update_best(cur_h, cur_val)

    # === Phase 4: Hand-crafted shapes ===
    if time.time() < deadline - 5.0:
        x_norm = np.linspace(-1.0, 1.0, N)
        crafted = []
        crafted.append(np.exp(-2.0 * x_norm**2))
        crafted.append(np.exp(-0.5 * x_norm**2))
        crafted.append(0.5 + 0.5 * np.exp(-8.0 * x_norm**2))
        crafted.append(np.exp(-20.0 * (x_norm - 0.3)**2) + np.exp(-20.0 * (x_norm + 0.3)**2))
        crafted.append(0.1 + np.exp(-30.0 * (x_norm - 0.4)**2) + np.exp(-30.0 * (x_norm + 0.4)**2))
        crafted.append(1.0 - 0.3 * np.exp(-5.0 * x_norm**2))
        crafted.append(0.5 + 0.5 * x_norm**2)
        crafted.append(0.5 + 0.5 * np.cos(np.pi * x_norm))
        crafted.append(np.exp(-4.0 * x_norm**2) * (1 + 0.3 * np.cos(4 * np.pi * x_norm)))
        crafted.append(0.3 + 0.7 * np.exp(-6.0 * x_norm**2))
        crafted.append(np.exp(-3.0 * x_norm**2) + 0.2 * np.exp(-15.0 * (x_norm - 0.5)**2) + 0.2 * np.exp(-15.0 * (x_norm + 0.5)**2))
        # Additional shapes
        crafted.append(0.2 + 0.8 * np.exp(-10.0 * x_norm**2))
        crafted.append(np.exp(-5.0 * (x_norm - 0.15)**2) + np.exp(-5.0 * (x_norm + 0.15)**2) + 0.1 * np.exp(-50.0 * x_norm**2))
        crafted.append(0.4 + 0.6 * np.exp(-4.0 * x_norm**2) * (1 + 0.2 * np.cos(2 * np.pi * x_norm)))

        for h in crafted:
            if time.time() > deadline - 3.0:
                break
            h = np.clip(h, 0, None)
            if h.sum() < 1e-12:
                continue
            v = ratio_from_h(h)
            update_best(h, v)

    # === Phase 5: Final polish ===
    if time.time() < deadline - 3.0:
        h_best = best_h
        A = np.ones((N, 1))
        for k in range(1, N_HARM + 1):
            A = np.column_stack([A, cos_bases[k-1], sin_bases[k-1]])
        coeffs_init, _, _, _ = np.linalg.lstsq(A, h_best, rcond=None)

        def polish_smooth_ratio(coeffs, beta=100.0):
            h = fourier_to_h(coeffs)
            return smooth_ratio_from_h(h, beta)

        try:
            result = opt.minimize(
                polish_smooth_ratio,
                coeffs_init,
                method='Nelder-Mead',
                options={'maxiter': 800, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True}
            )
            h_opt = fourier_to_h(result.x)
            v = ratio_from_h(h_opt)
            if v < best_val:
                update_best(h_opt, v)
        except Exception:
            pass

        # Also try a local spatial polish
        if time.time() < deadline - 1.0:
            cur_h = best_h.copy()
            cur_val = best_val
            rng3 = np.random.default_rng(555)
            for it in range(5000):
                if time.time() > deadline - 0.5:
                    break
                idx = int(rng3.integers(0, N))
                factor = float(rng3.uniform(0.9, 1.1))
                h_try = cur_h.copy()
                h_try[idx] = max(0.0, cur_h[idx] * factor)
                v = quick_ratio(h_try)
                if v < cur_val:
                    cur_val = v
                    cur_h = h_try
                    update_best(cur_h, cur_val)

    return list(best_h)
