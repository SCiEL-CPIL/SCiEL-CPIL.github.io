"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_overlap_max(h):
    """Compute the maximum overlap of h with shifted 1-h."""
    # h is array of length N, sum is N/2.
    # Overlap I(s) = sum_i h_i * (1 - h_{i-s}) * (2/N)
    # We want max over s.
    # Use FFT for efficiency.
    # 1-h is the complement.
    # Correlation of h and 1-h.
    # R(s) = sum h_i (1-h_{i-s})
    # This is cross-correlation.
    
    # Let's compute all shifts efficiently using FFT.
    # corr = ifft(fft(h) * conj(fft(1-h)))
    # But we need circular shifts.
    
    h_comp = 1.0 - h
    # Cross correlation of h and h_comp
    # C[k] = sum_i h[i] * h_comp[(i-k) % N]
    # This is ifft(fft(h) * conj(fft(h_comp)))
    
    H = np.fft.fft(h)
    Hc = np.fft.fft(h_comp)
    corr = np.fft.ifft(H * np.conj(Hc)).real
    
    # The overlap integral for shift k is corr[k] * (2/N)
    # We want the maximum of this.
    max_overlap = np.max(corr) * (2.0 / N)
    return max_overlap

def _soft_min(x, beta=10.0):
    """Soft minimum surrogate."""
    return -1.0 / beta * np.log(np.sum(np.exp(-beta * x)))

def _objective(params):
    """Objective to minimize: negative of the soft-min of the autocorrelation of u."""
    # We want to maximize min_s R_u(s).
    # So we minimize -min_s R_u(s).
    # Using soft-min: minimize -soft_min(R_u)
    
    # Reconstruct u from params
    # params: [a1, b1, a2, b2, ...]
    # u_i = sum (a_k cos(2pi k i/N) + b_k sin(2pi k i/N))
    
    M = len(params) // 2
    i = np.arange(N)
    u = np.zeros(N)
    for k in range(1, M + 1):
        angle = 2.0 * np.pi * k * i / N
        a = params[2*(k-1)]
        b = params[2*(k-1)+1]
        u += a * np.cos(angle) + b * np.sin(angle)
        
    # Project u to [-0.5, 0.5]
    u = np.clip(u, -0.5, 0.5)
    
    # Compute autocorrelation of u
    # R_u(s) = sum_i u_i u_{i-s}
    # This is ifft(fft(u) * conj(fft(u)))
    U = np.fft.fft(u)
    Ru = np.fft.ifft(U * np.conj(U)).real
    
    # We want to maximize min(Ru).
    # So we minimize -soft_min(Ru).
    return -_soft_min(Ru, beta=5.0)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    # Initialize with small random spectral coefficients
    np.random.seed(42)
    M = 10 # Number of harmonics
    x0 = np.random.uniform(-0.1, 0.1, size=2*M)
    
    best_score = 1.0 # Baseline
    best_h = np.full(N, 0.5)
    
    # Try a few restarts
    for restart in range(5):
        if time.time() > deadline:
            break
            
        if restart == 0:
            x_curr = x0
        else:
            x_curr = np.random.uniform(-0.2, 0.2, size=2*M)
        
        # Use Nelder-Mead for non-smooth objective (soft-min is smooth but landscape is complex)
        # Or L-BFGS-B if we trust the gradient of soft-min.
        # Let's try L-BFGS-B with numerical gradient or analytic if possible.
        # Analytic gradient is complex due to projection.
        # Nelder-Mead is robust to non-differentiability.
        
        res = minimize(_objective, x_curr, method='Nelder-Mead', 
                       options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
        
        if time.time() > deadline:
            break
            
        # Reconstruct h from best params
        params = res.x
        M_curr = len(params) // 2
        i = np.arange(N)
        u = np.zeros(N)
        for k in range(1, M_curr + 1):
            angle = 2.0 * np.pi * k * i / N
            a = params[2*(k-1)]
            b = params[2*(k-1)+1]
            u += a * np.cos(angle) + b * np.sin(angle)
            
        u = np.clip(u, -0.5, 0.5)
        h = 0.5 + u
        
        # Ensure sum is exactly N/2
        # The projection might break the sum constraint slightly.
        # We need to adjust h to satisfy sum(h) = N/2.
        # Simple adjustment: shift all values by delta, then clip.
        # This is an iterative projection.
        for _ in range(10):
            target_sum = N / 2.0
            current_sum = np.sum(h)
            diff = target_sum - current_sum
            h += diff / N
            h = np.clip(h, 0.0, 1.0)
            
        # Final check
        if np.isclose(np.sum(h), N/2.0, rtol=1e-5, atol=1e-5):
            score = _compute_overlap_max(h)
            # Score in task is 1/score? No, "score is the upper bound on C5"
            # "search score is seed/value"
            # If value is lower, search score is higher.
            # We want to minimize the raw value (overlap).
            if score < best_score:
                best_score = score
                best_h = h.copy()
                
    return best_h
# EVOLVE-BLOCK-END
