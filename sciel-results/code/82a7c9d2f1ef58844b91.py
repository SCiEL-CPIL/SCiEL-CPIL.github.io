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
    # Use log-sum-exp for smooth max
    # Shift by conv_max to avoid overflow
    shifted = (conv - conv_max) * beta
    # Clip to prevent underflow/overflow in exp
    shifted = np.clip(shifted, -50, 50)
    log_sum = conv_max + np.log(np.sum(np.exp(shifted)) / len(conv)) / beta
    return float(log_sum / l2)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_h = np.ones(N)
    best_val = ratio_from_h(best_h)
    
    def update_best(h, val):
        nonlocal best_h, best_val
        if val < best_val:
            best_val = val
            best_h = h.copy()

    rng = np.random.default_rng(42)

    # Phase 1: Fourier optimization with L-BFGS-B
    N_HARM = 30
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

    def fourier_smooth_ratio(coeffs, beta=50.0):
        h = fourier_to_h(coeffs)
        return smooth_ratio_from_h(h, beta)

    n_fourier_starts = 20
    for start in range(n_fourier_starts):
        if time.time() > deadline - 60.0:
            break

        coeffs = np.zeros(2 * N_HARM + 1)
        coeffs[0] = float(rng.uniform(0.5, 1.0))
        for k in range(1, N_HARM + 1):
            scale = 1.0 / (1.0 + 0.2 * k)
            coeffs[2 * k - 1] = float(rng.uniform(-0.3 * scale, 0.3 * scale))
            coeffs[2 * k] = float(rng.uniform(-0.3 * scale, 0.3 * scale))

        try:
            result = opt.minimize(
                fourier_smooth_ratio,
                coeffs,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-6, 'gtol': 1e-4}
            )
            h_opt = fourier_to_h(result.x)
            v = ratio_from_h(h_opt)
            if v < best_val:
                update_best(h_opt, v)
        except Exception:
            pass

    # Phase 2: Full-space optimization using softmax parameterization
    # h_i = softmax(z_i)
    # Objective: smooth_ratio_from_h(h)
    
    def z_to_h(z):
        # Stable softmax
        z_max = np.max(z)
        exp_z = np.exp(z - z_max)
        h = exp_z / np.sum(exp_z)
        return h

    def z_smooth_ratio(z, beta=50.0):
        h = z_to_h(z)
        return smooth_ratio_from_h(h, beta)

    # Initialize z from best_h
    # h_best is positive, so log is defined. Add small epsilon to avoid -inf
    h_init = np.clip(best_h, 1e-6, None)
    z_init = np.log(h_init)
    
    # Run L-BFGS-B on z
    # This is high dim (600), so limit iterations
    if time.time() < deadline - 20.0:
        try:
            result = opt.minimize(
                z_smooth_ratio,
                z_init,
                method='L-BFGS-B',
                options={'maxiter': 50, 'ftol': 1e-5, 'gtol': 1e-3}
            )
            h_opt = z_to_h(result.x)
            v = ratio_from_h(h_opt)
            if v < best_val:
                update_best(h_opt, v)
        except Exception:
            pass

    # Phase 3: Hand-crafted shapes
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

        for h in crafted:
            if time.time() > deadline - 3.0:
                break
            h = np.clip(h, 0, None)
            if h.sum() < 1e-12:
                continue
            v = ratio_from_h(h)
            update_best(h, v)

    # Phase 4: Final polish
    if time.time() < deadline - 3.0:
        # Refine the best h using L-BFGS-B on z again, starting from the current best
        h_init = np.clip(best_h, 1e-6, None)
        z_init = np.log(h_init)
        
        def z_smooth_ratio_polish(z, beta=80.0):
            h = z_to_h(z)
            return smooth_ratio_from_h(h, beta)

        try:
            result = opt.minimize(
                z_smooth_ratio_polish,
                z_init,
                method='L-BFGS-B',
                options={'maxiter': 50, 'ftol': 1e-6, 'gtol': 1e-4}
            )
            h_opt = z_to_h(result.x)
            v = ratio_from_h(h_opt)
            if v < best_val:
                update_best(h_opt, v)
        except Exception:
            pass

    return list(best_h)
