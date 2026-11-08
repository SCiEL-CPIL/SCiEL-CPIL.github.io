"""Optimized step function to minimize normalized max autoconvolution."""
import time
import numpy as np
from scipy.optimize import minimize

N = 600
# Interval length
dx = (1/4 - (-1/4)) / N  # 0.5/600 = 1/1200

# Precompute fine grid parameters
M = 8  # sub-intervals per step for accurate convolution
n_fine = N * M
dx_fine = dx / M

def compute_autoconvolution_max(h):
    """
    Compute the normalized maximum of the autoconvolution of the step function.
    Uses FFT-based convolution on a fine grid for speed and accuracy.
    """
    h = np.asarray(h, dtype=np.float64)
    if h.shape[0] != N:
        return 1.0
    
    norm2 = np.sum(h**2) * dx
    if norm2 < 1e-15:
        return 1.0
    
    # Use FFT for fast convolution
    # Need at least 2*N-1 points for full convolution
    nfft = 1 << (2 * N - 1).bit_length()
    
    h_pad = np.zeros(nfft)
    h_pad[:N] = h
    
    H = np.fft.fft(h_pad)
    conv = np.fft.ifft(H * H).real * dx
    
    max_conv = np.max(conv)
    return max_conv / norm2

def objective(h):
    """Objective function to minimize."""
    return compute_autoconvolution_max(h)

def soft_max_objective(h, beta=50.0):
    """
    Smooth surrogate using log-sum-exp (soft max) of the autoconvolution values.
    This provides smooth gradients even when the max is achieved at multiple points.
    """
    h = np.asarray(h, dtype=np.float64)
    if h.shape[0] != N:
        return 1.0
    
    norm2 = np.sum(h**2) * dx
    if norm2 < 1e-15:
        return 1.0
    
    # Compute autoconvolution via FFT
    nfft = 1 << (2 * N - 1).bit_length()
    h_pad = np.zeros(nfft)
    h_pad[:N] = h
    
    H = np.fft.fft(h_pad)
    conv = np.fft.ifft(H * H).real * dx
    
    # Soft max using log-sum-exp for numerical stability
    # (1/beta) * log(sum(exp(beta * conv)))
    # Shift for numerical stability
    conv_shifted = conv - np.max(conv)
    log_sum_exp = np.log(np.sum(np.exp(beta * conv_shifted)))
    soft_max = (1.0 / beta) * log_sum_exp + np.max(conv)
    
    return soft_max / norm2

def construct():
    start_time = time.time()
    deadline = 130.0  # Leave 20s margin for safety
    
    # Initial guess: uniform
    h0 = np.ones(N)
    
    best_h = h0.copy()
    best_val = objective(h0)
    
    # Phase 1: Use soft-max surrogate to get out of the flat uniform region
    # The soft-max provides smoother gradients that can guide away from uniform
    
    def sym_soft_objective(h_sym, beta=30.0):
        """Symmetric soft-max objective."""
        h = np.empty(N)
        h[:N//2] = np.maximum(h_sym, 0)
        h[N//2:] = h[:N//2][::-1]
        return soft_max_objective(h, beta=beta)
    
    h_sym0 = np.ones(N//2)
    
    try:
        res = minimize(
            sym_soft_objective,
            h_sym0,
            method='L-BFGS-B',
            bounds=[(0, None)] * (N//2),
            options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 100}
        )
        
        elapsed = time.time() - start_time
        if elapsed < deadline:
            h_sym_opt = np.maximum(res.x, 0)
            h_opt = np.empty(N)
            h_opt[:N//2] = h_sym_opt
            h_opt[N//2:] = h_sym_opt[::-1]
            
            val = objective(h_opt)
            if val < best_val:
                best_val = val
                best_h = h_opt
    except Exception:
        pass
    
    # Phase 2: Try asymmetric soft-max if time permits
    elapsed = time.time() - start_time
    if elapsed < deadline - 20:
        try:
            def asym_soft_objective(h, beta=30.0):
                return soft_max_objective(h, beta=beta)
            
            res = minimize(
                asym_soft_objective,
                best_h.copy(),
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 1500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 100}
            )
            
            h_opt = np.maximum(res.x, 0)
            val = objective(h_opt)
            if val < best_val:
                best_val = val
                best_h = h_opt
        except Exception:
            pass
    
    # Phase 3: Polish with the exact (nonsmooth) objective
    elapsed = time.time() - start_time
    if elapsed < deadline - 10:
        try:
            def sym_exact_objective(h_sym):
                h = np.empty(N)
                h[:N//2] = np.maximum(h_sym, 0)
                h[N//2:] = h[:N//2][::-1]
                return compute_autoconvolution_max(h)
            
            h_sym_init = best_h[:N//2].copy()
            
            res = minimize(
                sym_exact_objective,
                h_sym_init,
                method='L-BFGS-B',
                bounds=[(0, None)] * (N//2),
                options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 200}
            )
            
            h_sym_opt = np.maximum(res.x, 0)
            h_opt = np.empty(N)
            h_opt[:N//2] = h_sym_opt
            h_opt[N//2:] = h_sym_opt[::-1]
            
            val = objective(h_opt)
            if val < best_val:
                best_val = val
                best_h = h_opt
        except Exception:
            pass
    
    # Phase 4: Final asymmetric polish if time permits
    elapsed = time.time() - start_time
    if elapsed < deadline - 5:
        try:
            res = minimize(
                objective,
                best_h.copy(),
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 200}
            )
            
            h_opt = np.maximum(res.x, 0)
            val = objective(h_opt)
            if val < best_val:
                best_val = val
                best_h = h_opt
        except Exception:
            pass
    
    # Ensure nonnegativity
    best_h = np.maximum(best_h, 0)
    
    return list(best_h)

# Test
if __name__ == "__main__":
    h = construct()
    print(f"Length: {len(h)}")
    print(f"Score: {compute_autoconvolution_max(np.array(h))}")
