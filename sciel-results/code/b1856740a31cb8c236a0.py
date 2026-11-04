import numpy as np
import time
import scipy.optimize as opt

N = 600
W = 0.5 / N
X_GRID = -0.25 + (np.arange(N) + 0.5) * W  # centers of intervals

# Precompute fine grid for accurate peak evaluation
FINE_N = 20000
X_FINE = np.linspace(-0.25, 0.25, FINE_N, endpoint=False)
DX = 0.5 / FINE_N
CELL = np.clip(((X_FINE + 0.25) / W).astype(np.int64), 0, N - 1)
N_FFT_FINE = 1 << (2 * FINE_N - 1).bit_length()

def ratio_from_h(h):
    """Compute normalized max of autoconvolution: peak(g) / sum(h^2)."""
    h = np.asarray(h, dtype=np.float64)
    if h.size != N:
        h = np.resize(h, N)
    h = np.clip(h, 0.0, None)
    
    l2_sq = np.sum(h * h)
    if l2_sq < 1e-15:
        return float('inf')
    
    f = h[CELL]
    F = np.fft.rfft(f, N_FFT_FINE)
    conv = np.fft.irfft(F * F, N_FFT_FINE)[:2 * FINE_N - 1] * DX
    peak = np.max(conv)
    return float(peak / l2_sq)

def gaussians_to_h(params):
    """
    Convert Gaussian parameters to step heights.
    params = [c0, (mu1, sigma1, amp1), (mu2, sigma2, amp2), ...]
    """
    c0 = params[0]
    h = np.full(N, c0, dtype=np.float64)
    n_gauss = (len(params) - 1) // 3
    for i in range(n_gauss):
        mu = params[1 + 3*i]
        sigma = params[2 + 3*i]
        amp = params[3 + 3*i]
        sigma = max(abs(sigma), 1e-4)
        h += amp * np.exp(-0.5 * ((X_GRID - mu) / sigma) ** 2)
    return np.clip(h, 0.0, None)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s safety margin for return overhead
    
    # Number of Gaussian bumps
    N_GAUSS = 6
    
    # Objective function
    def objective(params):
        h = gaussians_to_h(params)
        if h.sum() < 1e-12:
            return 1e6
        return ratio_from_h(h)
    
    # Initial best: uniform
    best_h = np.ones(N)
    best_val = ratio_from_h(best_h)
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Optimize Gaussian parameters with L-BFGS-B
    n_starts = 10
    for i in range(n_starts):
        if time.time() > deadline - 40.0:
            break
        
        # Random initialization
        c0_init = rng.uniform(0.3, 0.8)
        params_init = [c0_init]
        for j in range(N_GAUSS):
            mu_j = rng.uniform(-0.2, 0.2)
            sigma_j = rng.uniform(0.02, 0.15)
            amp_j = rng.uniform(-0.5, 0.5)
            params_init.extend([mu_j, sigma_j, amp_j])
        
        x0 = np.array(params_init)
        
        try:
            res = opt.minimize(
                objective, x0, method='L-BFGS-B',
                bounds=[(-1.0, 2.0)] + [(-0.25, 0.25), (0.01, 0.25), (-1.0, 1.0)] * N_GAUSS,
                options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            h_opt = gaussians_to_h(res.x)
            v_opt = ratio_from_h(h_opt)
            if v_opt < best_val:
                best_val = v_opt
                best_h = h_opt
        except Exception:
            pass
    
    # Phase 2: Local refinement in spatial domain with adaptive step size
    if time.time() < deadline - 10.0:
        rng2 = np.random.default_rng(12345)
        current_h = best_h.copy()
        current_val = best_val
        
        # Adaptive step size for perturbations
        base_amp = 0.1
        step_scale = 1.0
        
        n_iter = 3000
        for it in range(n_iter):
            if time.time() > deadline - 2.0:
                break
            
            mode = int(rng2.integers(0, 5))
            h_try = current_h.copy()
            
            # Adaptive amplitude based on recent success/failure
            amp = base_amp * step_scale
            
            if mode == 0:
                # Additive Gaussian bump
                c = int(rng2.integers(0, N))
                w = int(rng2.integers(10, 60))
                lo = max(0, c - w)
                hi = min(N, c + w)
                if hi - lo > 5:
                    xs = np.linspace(-1, 1, hi - lo)
                    h_try[lo:hi] += amp * np.exp(-xs**2)
            elif mode == 1:
                # Subtractive dip
                c = int(rng2.integers(0, N))
                w = int(rng2.integers(10, 60))
                lo = max(0, c - w)
                hi = min(N, c + w)
                if hi - lo > 5:
                    xs = np.linspace(-1, 1, hi - lo)
                    h_try[lo:hi] -= amp * np.exp(-xs**2)
            elif mode == 2:
                # Multiplicative modulation
                c = int(rng2.integers(0, N))
                w = int(rng2.integers(20, 80))
                lo = max(0, c - w)
                hi = min(N, c + w)
                if hi - lo > 5:
                    f = 1.0 + amp * rng2.normal(0, 1)
                    f = max(0.5, min(1.5, f))
                    h_try[lo:hi] *= f
            elif mode == 3:
                # Smoothing
                c = int(rng2.integers(0, N))
                w = int(rng2.integers(5, 20))
                lo = max(0, c - w)
                hi = min(N, c + w)
                if hi - lo > 4:
                    kernel = np.ones(5) / 5.0
                    h_try[lo:hi] = np.convolve(h_try[lo:hi], kernel, mode='same')
            else:
                # Local averaging / blurring
                c = int(rng2.integers(0, N))
                w = int(rng2.integers(3, 10))
                lo = max(0, c - w)
                hi = min(N, c + w)
                if hi - lo > 2:
                    local = h_try[lo:hi]
                    mean_val = np.mean(local)
                    h_try[lo:hi] = (1 - amp) * local + amp * mean_val
            
            h_try = np.clip(h_try, 0.0, None)
            v = ratio_from_h(h_try)
            
            if v < current_val:
                current_val = v
                current_h = h_try
                if current_val < best_val:
                    best_val = current_val
                    best_h = current_h.copy()
                # Increase step size on success
                step_scale = min(2.0, step_scale * 1.2)
            else:
                # Decrease step size on failure
                step_scale = max(0.3, step_scale * 0.9)
    
    return list(best_h)
