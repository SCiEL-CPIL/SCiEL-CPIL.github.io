import numpy as np
import time
from scipy.optimize import minimize

N = 600
W = 1.0 / 1200.0
L = 0.5

def compute_ratio(h, n_grid=2048):
    """Compute max(autoconv) / l2_sq for step function h."""
    h = np.asarray(h, dtype=np.float64)
    h = np.clip(h, 0.0, None)
    if np.sum(h) < 1e-15:
        return float('inf')

    # Upsample to a fine grid for accurate convolution peak detection
    x_fine = np.linspace(-0.25, 0.25, n_grid, endpoint=False)
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

def fourier_to_heights(coeffs, n_grid, K):
    """
    Reconstruct function from Fourier coefficients of log(f).
    coeffs: [a0, a1, b1, a2, b2, ..., aK, bK]
    """
    x = np.linspace(-0.25, 0.25, n_grid, endpoint=False)
    g = np.full(n_grid, coeffs[0] / 2.0)
    
    for k in range(1, K + 1):
        idx_a = 1 + 2 * (k - 1)
        idx_b = 2 + 2 * (k - 1)
        a_k = coeffs[idx_a]
        b_k = coeffs[idx_b]
        g += a_k * np.cos(2 * np.pi * k * x / L) + b_k * np.sin(2 * np.pi * k * x / L)
    
    # f(x) = exp(g(x))
    g = np.clip(g, -50, 50)
    f = np.exp(g)
    return f

def objective_smooth(coeffs, K, n_grid, beta):
    f = fourier_to_heights(coeffs, n_grid, K)
    
    n_fft = 1 << (2 * n_grid).bit_length()
    F = np.fft.rfft(f, n_fft)
    conv = np.fft.irfft(F * F, n_fft)[:2 * n_grid - 1]
    dx = 0.5 / n_grid
    conv *= dx
    
    c_max = np.max(conv)
    if c_max < 1e-15:
        return 1e15
    
    log_sum_exp = c_max + (1.0 / beta) * np.log(np.sum(np.exp(beta * (conv - c_max))))
    smoothed_max = log_sum_exp
    
    l2_sq = np.sum(f ** 2) * dx
    if l2_sq < 1e-15:
        return 1e15
        
    return smoothed_max / l2_sq

def construct():
    deadline = time.time() + 145.0
    rng = np.random.default_rng(42)
    
    best_h = np.ones(N)
    best_ratio = compute_ratio(best_h)
    
    # Fourier mode optimization
    # Try different numbers of modes
    for K in [10, 15, 20]:
        if time.time() > deadline - 10:
            break
            
        dim = 1 + 2 * K
        n_grid = 2048
        beta = 50.0
        
        # Multiple restarts
        for _ in range(15):
            if time.time() > deadline - 5:
                break
                
            coeffs0 = rng.standard_normal(dim) * 0.5
            
            def obj(c):
                return objective_smooth(c, K, n_grid, beta)
                
            try:
                res = minimize(
                    obj,
                    coeffs0,
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6}
                )
                
                # Evaluate the actual score on the 600-step grid
                f_opt = fourier_to_heights(res.x, n_grid, K)
                x_fine = np.linspace(-0.25, 0.25, n_grid, endpoint=False)
                cell_idx = np.clip(((x_fine + 0.25) / W).astype(np.int32), 0, N - 1)
                
                step_heights = np.zeros(N)
                counts = np.zeros(N)
                for idx, val in zip(cell_idx, f_opt):
                    step_heights[idx] += val
                    counts[idx] += 1
                step_heights /= counts
                
                r = compute_ratio(step_heights)
                if r < best_ratio:
                    best_ratio = r
                    best_h = step_heights.copy()
            except Exception:
                pass

    return best_h.tolist()
