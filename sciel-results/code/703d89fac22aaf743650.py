"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy import optimize

N = 600

# EVOLVE-BLOCK-START
def _compute_score_and_grad(H):
    """Compute score = max(autoconv) / (sum(H))^2 and its gradient."""
    # Ensure non-negative
    H = np.maximum(H, 0)
    S = np.sum(H)
    if S < 1e-12:
        return 0.0, np.zeros_like(H)
    
    # Compute autoconvolution using FFT
    n_fft = 2 * N
    H_fft = np.fft.rfft(H, n=n_fft)
    C_fft = H_fft * H_fft
    C = np.fft.irfft(C_fft, n=n_fft)
    # C[k] = sum_i H[i] * H[i+k] for k in 0..N-1 (linear convolution)
    C_lin = C[:N]  # lags 0 to N-1
    
    max_c = np.max(C_lin)
    score = max_c / (S * S)
    
    # Gradient: d(score)/dH[j]
    # score = max_k C(k) / S^2
    # Only the argmax lag contributes to gradient (subgradient)
    m = np.argmax(C_lin)
    
    # C(m) = sum_i H[i] * H[i+m]
    # dC(m)/dH[j] = H[j+m] (if j+m < N) + H[j-m] (if j-m >= 0)
    # More precisely: C(m) = sum_{i} H[i] H[i+m] where both indices in [0,N-1]
    # For fixed m: C(m) = sum_{i=0}^{N-1-m} H[i] H[i+m]
    # dC(m)/dH[j] = H[j+m] (for j <= N-1-m, from term where i=j) 
    #              + H[j-m] (for j >= m, from term where i+m=j, i=j-m)
    
    grad_Cm = np.zeros(N)
    if m == 0:
        grad_Cm = 2.0 * H
    else:
        grad_Cm[:N-m] += H[m:]      # d/dH[j] of H[j]*H[j+m] for j < N-m
        grad_Cm[m:] += H[:N-m]      # d/dH[j] of H[j-m]*H[j] for j >= m
    
    # d(score)/dH[j] = grad_Cm[j] / S^2 - 2 * max_c * H[j] / S^3
    grad_score = grad_Cm / (S * S) - 2.0 * max_c * H / (S * S * S)
    
    return score, grad_score


def _local_search(H_init, max_iters=2000, deadline=None):
    """Gradient descent with projection onto non-negative orthant."""
    H = H_init.copy()
    S = np.sum(H)
    if S < 1e-12:
        H = np.random.rand(N)
    
    best_score, _ = _compute_score_and_grad(H)
    best_H = H.copy()
    
    lr = 0.01
    patience = 0
    prev_score = best_score
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        score, grad = _compute_score_and_grad(H)
        
        if score < best_score - 1e-12:
            best_score = score
            best_H = H.copy()
            patience = 0
        else:
            patience += 1
        
        # Adaptive learning rate
        if patience > 50:
            lr *= 0.9
            patience = 0
        if lr < 1e-6:
            break
        
        # Gradient step
        H = H - lr * grad
        
        # Project onto non-negative orthant
        H = np.maximum(H, 0)
        
        # Optional: normalize to keep scale reasonable
        S = np.sum(H)
        if S < 1e-12:
            break
        
    return best_H, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    np.random.seed(42)
    
    best_H = None
    best_score = float('inf')
    
    # Multiple initializations
    n_starts = 20
    
    for s in range(n_starts):
        if time.time() > deadline:
            break
        
        # Different initialization strategies
        if s == 0:
            H_init = np.ones(N)
        elif s == 1:
            # Raised cosine
            t = np.linspace(0, 1, N)
            H_init = 0.5 * (1 + np.cos(2 * np.pi * t))
        elif s == 2:
            # Gaussian centered
            x = np.linspace(-1, 1, N)
            H_init = np.exp(-x**2 / (2 * 0.3**2))
        elif s == 3:
            # Triangular
            t = np.linspace(0, 1, N)
            H_init = np.maximum(0, 1 - 2 * np.abs(t - 0.5))
        elif s == 4:
            # Uniform on middle half
            H_init = np.zeros(N)
            H_init[N//4:3*N//4] = 1.0
        else:
            # Random with some structure
            H_init = np.random.rand(N)
            # Apply a smooth filter
            from scipy.ndimage import gaussian_filter1d
            H_init = gaussian_filter1d(H_init, sigma=10)
        
        H_init = np.maximum(H_init, 0)
        S = np.sum(H_init)
        if S > 0:
            H_init = H_init / S * np.sqrt(N)  # normalize
        
        H_opt, score = _local_search(H_init, max_iters=500, deadline=deadline)
        
        if score < best_score:
            best_score = score
            best_H = H_opt.copy()
    
    # Final check
    if best_H is None:
        best_H = np.ones(N)
    
    # Ensure non-negative
    best_H = np.maximum(best_H, 0)
    
    return list(best_H)
# EVOLVE-BLOCK-END
