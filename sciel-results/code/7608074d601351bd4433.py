import numpy as np
import time
import scipy.optimize as opt

N = 600
W = 1.0 / 1200.0
X = np.linspace(-0.25, 0.25, N, endpoint=False)

# Precompute fine grid for fast convolution ratio
FINE_N = 12000
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

def complex_exp_to_h(params):
    """
    Convert complex exponential parameters to step heights.
    params = [theta_1, r_1, phi_1, theta_2, r_2, phi_2, ..., r_base]
    h(x) = r_base^2 + |sum_k r_k * exp(i*(theta_k*x + phi_k))|^2
    """
    n_terms = (len(params) - 1) // 3
    r_base = params[-1]
    
    z = np.zeros(N, dtype=np.complex128)
    for k in range(n_terms):
        theta = params[3*k]
        r = params[3*k + 1]
        phi = params[3*k + 2]
        z += r * np.exp(1j * (theta * X + phi))
    
    h = r_base**2 + np.real(z * np.conj(z))
    return h

def gaussian_to_h(params):
    """
    Convert Gaussian superposition parameters to step heights.
    params = [amp0, mu0, sigma0, amp1, mu1, sigma1, ...]
    """
    M = len(params) // 3
    bin_centers = -0.25 + (np.arange(N) + 0.5) * W
    heights = np.zeros(N, dtype=np.float64)
    for m in range(M):
        amp = params[3*m]
        mu = params[3*m+1]
        sigma = params[3*m+2]
        if sigma < 1e-6:
            sigma = 1e-6
        heights += amp * np.exp(-0.5 * ((bin_centers - mu) / sigma) ** 2)
    return np.clip(heights, 0.0, None)

def tent_to_h(params):
    """
    Symmetric triangular pulse centered at 0.
    params: [base_half_width, apex_height]
    """
    half_w = params[0]
    apex = params[1]
    actual_half_w = half_w * 0.25
    bin_centers = -0.25 + (np.arange(N) + 0.5) * W
    heights = apex * np.maximum(0.0, 1.0 - np.abs(bin_centers) / actual_half_w)
    return np.clip(heights, 0.0, None)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    best_h = np.ones(N)
    best_val = ratio_from_h(best_h)

    def update_best(h, val):
        nonlocal best_h, best_val
        if val < best_val:
            best_val = val
            best_h = h.copy()

    rng = np.random.default_rng(42)

    # === Phase 1: Quick Tent Scan ===
    if time.time() < deadline - 100.0:
        for hw in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]:
            for amp in [1.0, 1.5, 2.0, 2.5, 3.0]:
                h_cand = tent_to_h([hw, amp])
                val = ratio_from_h(h_cand)
                if val < best_val:
                    update_best(h_cand, val)

    # === Phase 2: Complex exponential superposition with L-BFGS-B ===
    for n_terms in [2, 3, 4]:
        if time.time() > deadline - 60.0:
            break
        
        n_params = 3 * n_terms + 1
        bounds = []
        for _ in range(n_terms):
            bounds.append((-50.0, 50.0))
            bounds.append((0.0, 5.0))
            bounds.append((-np.pi, np.pi))
        bounds.append((0.0, 5.0))
        
        n_restarts = 10
        for i in range(n_restarts):
            if time.time() > deadline - 30.0:
                break
            
            x0 = np.array([rng.uniform(b[0], b[1]) for b in bounds])
            
            try:
                def objective_complex(params):
                    h = complex_exp_to_h(params)
                    h = np.clip(h, 0.0, None)
                    return ratio_from_h(h)
                
                result = opt.minimize(
                    objective_complex,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                h_opt = complex_exp_to_h(result.x)
                h_clipped = np.clip(h_opt, 0, None)
                true_val = ratio_from_h(h_clipped)
                update_best(h_clipped, true_val)
                
            except Exception:
                pass

    # === Phase 3: Gaussian superposition with L-BFGS-B ===
    if time.time() < deadline - 30.0:
        M = 5
        g_bounds = []
        for _ in range(M):
            g_bounds.append((0.0, 5.0))
            g_bounds.append((-0.25, 0.25))
            g_bounds.append((0.01, 0.25))
        
        n_restarts = 5
        for i in range(n_restarts):
            if time.time() > deadline - 15.0:
                break
            
            x0 = []
            for _ in range(M):
                x0.append(rng.uniform(0.5, 2.0))
                x0.append(rng.uniform(-0.1, 0.1))
                x0.append(rng.uniform(0.05, 0.15))
            
            try:
                def objective_gauss(params):
                    h = gaussian_to_h(params)
                    return ratio_from_h(h)
                
                result = opt.minimize(
                    objective_gauss,
                    x0,
                    method='L-BFGS-B',
                    bounds=g_bounds,
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                h_opt = gaussian_to_h(result.x)
                true_val = ratio_from_h(h_opt)
                update_best(h_opt, true_val)
                
            except Exception:
                pass

    # === Phase 4: Local refinement in spatial domain ===
    if time.time() < deadline - 10.0:
        for restart in range(3):
            if time.time() > deadline - 5.0:
                break
            
            h_cur = best_h.copy()
            val_cur = best_val
            
            noise = rng.normal(0, 0.05, N)
            h_cur = np.clip(h_cur + noise, 0, None)
            val_cur = ratio_from_h(h_cur)
            
            for it in range(200):
                if time.time() > deadline - 2.0:
                    break
                
                idx = rng.integers(0, N)
                current_val = h_cur[idx]
                candidates = [
                    current_val * 0.5,
                    current_val * 0.8,
                    current_val * 1.2,
                    current_val * 1.5,
                    max(0.0, current_val - 0.1),
                    max(0.0, current_val + 0.1),
                    0.0,
                    1.0,
                ]
                
                best_candidate = h_cur[idx]
                best_candidate_val = val_cur
                
                for cand in candidates:
                    h_try = h_cur.copy()
                    h_try[idx] = cand
                    v_try = ratio_from_h(h_try)
                    if v_try < best_candidate_val:
                        best_candidate_val = v_try
                        best_candidate = cand
                
                if best_candidate != h_cur[idx]:
                    h_cur[idx] = best_candidate
                    val_cur = best_candidate_val
            
            if val_cur < best_val:
                update_best(h_cur, val_cur)

    return list(best_h)
