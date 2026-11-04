import numpy as np
import time
from scipy.optimize import minimize

N = 600
W = 1.0 / 1200.0
BIN_CENTERS = -0.25 + (np.arange(N) + 0.5) * W

# Fine grid for accurate autoconvolution evaluation
FINE_N = 12000
X_FINE = np.linspace(-0.25, 0.25, FINE_N, endpoint=False)
DX = 0.5 / FINE_N
CELL = np.clip(((X_FINE + 0.25) / W).astype(np.int64), 0, N - 1)
N_FFT = 1 << (2 * FINE_N - 1).bit_length()

def _compute_ratio_fast(h):
    """
    Compute max(autoconv) / (integral f)^2 for step function h.
    Uses FFT on a fine grid for accurate peak detection.
    """
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
    return float(max_conv / integral_sq)

def _smooth_objective(h, beta=200.0):
    """Smooth surrogate using soft-max for the direct h optimization."""
    h = np.clip(h, 0.0, None)
    s = h.sum()
    if s < 1e-15:
        return 1e10
    integral_sq = (s * W) ** 2
    
    f_fine = h[CELL]
    F = np.fft.rfft(f_fine, N_FFT)
    conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1]
    conv *= DX
    
    m = np.max(conv)
    if m < 1e-15:
        return 1e10
    # Use a stable log-sum-exp implementation
    shifted = conv - m
    # Clip to avoid underflow issues in exp, though shifted <= 0
    # np.log(sum(exp)) is fine if m is the max
    soft_max = m + np.log(np.sum(np.exp(shifted))) / beta
    return soft_max / integral_sq

def _chirp_to_h(params):
    """
    Convert chirp parameters to step heights.
    Model: f(x) = A * exp(-x^2 / (2*w^2)) * (1 + alpha * cos(2*pi*mu*x + phi))
    This represents the magnitude squared of a chirped Gaussian, ensuring non-negativity
    if 1 + alpha*cos(...) >= 0. We enforce alpha in [0, 1) via softplus/sigmoid mapping.
    
    params: [log_A, log_w, alpha_raw, mu, phi]
    """
    log_A = params[0]
    log_w = params[1]
    alpha_raw = params[2]
    mu = params[3]
    phi = params[4]
    
    A = np.exp(np.clip(log_A, -5, 5))
    w = np.exp(np.clip(log_w, -5, 2))
    # Map alpha_raw to [0, 0.99] to ensure non-negativity of the modulation term
    # sigmoid maps R to (0,1). Scale to [0, 0.99]
    alpha = 0.99 * (1.0 / (1.0 + np.exp(-alpha_raw)))
    
    x = BIN_CENTERS
    gaussian = A * np.exp(-0.5 * (x / w) ** 2)
    modulation = 1.0 + alpha * np.cos(2.0 * np.pi * mu * x + phi)
    
    h = gaussian * modulation
    return np.clip(h, 0.0, None)

def _chirp_objective(params):
    h = _chirp_to_h(params)
    val = _compute_ratio_fast(h)
    if not np.isfinite(val):
        return 1e10
    return val

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # Initial best: uniform
    best_h = np.ones(N, dtype=np.float64)
    best_val = _compute_ratio_fast(best_h)
    
    def update_best(h, val):
        nonlocal best_h, best_val
        if val < best_val and np.isfinite(val):
            best_val = val
            best_h = h.copy()

    rng = np.random.default_rng(42)

    # === Phase 1: Chirp-based optimization ===
    # We optimize the parameters of the chirp model directly.
    # This provides a structured, low-dimensional search space that leverages
    # low-autocorrelation properties of chirps.
    
    n_chirp_starts = 20
    for i in range(n_chirp_starts):
        if time.time() > deadline - 60.0:
            break
        
        # Generate diverse initializations for chirp parameters
        # [log_A, log_w, alpha_raw, mu, phi]
        if i == 0:
            # Near uniform (small alpha)
            p0 = np.array([0.0, 0.0, -5.0, 0.0, 0.0])
        elif i == 1:
            # Moderate chirp
            p0 = np.array([0.0, 0.5, 0.0, 5.0, 0.0])
        elif i == 2:
            # Strong chirp, narrower
            p0 = np.array([0.0, -0.5, 2.0, 10.0, 0.0])
        else:
            # Random
            p0 = np.array([
                rng.uniform(-1, 1),      # log_A
                rng.uniform(-2, 1),      # log_w
                rng.uniform(-5, 5),      # alpha_raw
                rng.uniform(-20, 20),    # mu
                rng.uniform(0, 2*np.pi)  # phi
            ])
        
        try:
            res = minimize(
                _chirp_objective,
                p0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6}
            )
            if res.success or res.fun < best_val:
                h_candidate = _chirp_to_h(res.x)
                val = _compute_ratio_fast(h_candidate)
                update_best(h_candidate, val)
        except Exception:
            pass

    # === Phase 2: Direct bin-level optimization with L-BFGS-B using soft-max surrogate ===
    # Start from the best result so far (which might be a chirp or uniform)
    if time.time() < deadline - 30.0:
        bounds = [(0.0, None)] * N
        try:
            res_direct = minimize(
                _smooth_objective,
                best_h,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            h_candidate = np.clip(res_direct.x, 0.0, None)
            val = _compute_ratio_fast(h_candidate)
            update_best(h_candidate, val)
        except Exception:
            pass
    
    # === Phase 3: High-precision polish with tighter tolerances ===
    if time.time() < deadline - 10.0:
        # Second round of direct optimization with tighter settings
        bounds = [(0.0, None)] * N
        try:
            res_polish = minimize(
                _smooth_objective,
                best_h,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 40}
            )
            h_candidate = np.clip(res_polish.x, 0.0, None)
            val = _compute_ratio_fast(h_candidate)
            update_best(h_candidate, val)
        except Exception:
            pass
    
    # === Phase 4: Local search polish with small perturbations ===
    if time.time() < deadline - 3.0:
        rng2 = np.random.default_rng(777)
        for it in range(3000):
            if time.time() > deadline - 1.0:
                break
            
            n_perturb = int(rng2.integers(3, 15))
            idx = rng2.choice(N, size=n_perturb, replace=False)
            h_try = best_h.copy()
            h_try[idx] += rng2.normal(0, 0.015, size=n_perturb)
            h_try = np.clip(h_try, 0, None)
            
            val = _compute_ratio_fast(h_try)
            update_best(h_try, val)
    
    # Final evaluation
    final_val = _compute_ratio_fast(best_h)
    if final_val < best_val:
        best_val = final_val
    
    return best_h.tolist()
