"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy: Discrete spectral search on the real-symmetric subspace.
Parameterize the step function by its real, even DFT coefficients.
Enforce non-negativity via a soft-plus projection in the optimization loop.
Minimize the maximum value of the autoconvolution using coordinate descent
with efficient 1D scalar optimization (minimize_scalar) on each frequency bin.
"""

import numpy as np
import time
from scipy.optimize import minimize_scalar

N = 600
TIME_LIMIT = 115  # safety margin before 120s kill

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    f_fft = np.fft.fft(f, n=2*N)
    conv_fft = f_fft * f_fft
    conv = np.real(np.fft.ifft(conv_fft))
    max_val = np.max(conv[:2*N-1])
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    return max_val / (s * s)

def score_soft_from_symmetric_F(F_sym):
    """
    Compute the score using a soft-plus projection for non-negativity.
    F_sym: array of length N//2 + 1 containing real DFT coefficients.
    """
    # Construct full complex DFT from symmetric real coefficients
    F_complex = np.zeros(N, dtype=np.complex128)
    F_complex[0] = F_sym[0]
    if N % 2 == 0:
        F_complex[N//2] = F_sym[N//2]
    for k in range(1, N//2):
        F_complex[k] = F_sym[k]
        F_complex[N-k] = F_sym[k]
        
    f_raw = np.real(np.fft.ifft(F_complex))
    
    # Soft-plus projection: log(1 + exp(x))
    # This ensures f_soft > 0 and is smooth
    f_soft = np.log1p(np.exp(f_raw))
    
    # Compute score of the soft function
    f_fft = np.fft.fft(f_soft, n=2*N)
    conv_fft = f_fft * f_fft
    conv = np.real(np.fft.ifft(conv_fft))
    max_val = np.max(conv[:2*N-1])
    s = np.sum(f_soft)
    if s < 1e-15:
        return np.inf
    return max_val / (s * s)

def construct():
    start_time = time.time()
    
    # Initial best: constant function
    best_f = np.ones(N)
    best_score = compute_score(best_f)
    
    half = N // 2
    n_restarts = 10
    max_time_per_restart = (TIME_LIMIT - 5) / n_restarts
    
    for r in range(n_restarts):
        if time.time() - start_time > TIME_LIMIT - 5:
            break
        
        # Initialize F_sym
        if r == 0:
            # Constant function in time -> Delta in freq
            F_sym = np.zeros(half + 1)
            F_sym[0] = N # Sum of ones
        elif r == 1:
            # Box in freq -> Sinc in time
            F_sym = np.zeros(half + 1)
            cutoff = N // 4
            F_sym[:cutoff] = 1.0
            F_sym[0] = 1.0 # DC
        elif r == 2:
            # Gaussian in freq
            k = np.arange(half + 1)
            F_sym = np.exp(-k**2 / (N/10)**2)
        else:
            # Random with positive DC
            F_sym = np.random.randn(half + 1)
            F_sym[0] = np.abs(F_sym[0]) + 1.0
            
        # Coordinate Descent Loop
        n_cd_iters = 15
        
        for it in range(n_cd_iters):
            if time.time() - start_time > TIME_LIMIT - 5:
                break
            
            improved = False
            for k in range(half + 1):
                if time.time() - start_time > TIME_LIMIT - 5:
                    break
                
                current_val = F_sym[k]
                
                # Define search bounds
                # Heuristic: Allow 50% change relative to current, or absolute min 0.5
                if abs(current_val) < 1e-6:
                    lower = -1.0
                    upper = 1.0
                else:
                    delta = 0.5 * abs(current_val)
                    lower = current_val - delta
                    upper = current_val + delta
                    
                # Special case for DC term (k=0): keep it positive
                if k == 0:
                    lower = max(0.0, lower)
                    
                def obj(x):
                    new_F = F_sym.copy()
                    new_F[k] = x
                    return score_soft_from_symmetric_F(new_F)
                    
                try:
                    res = minimize_scalar(obj, bounds=(lower, upper), method='bounded', options={'xatol': 1e-4})
                    if res.success:
                        new_val = res.x
                        F_sym[k] = new_val
                        improved = True
                except Exception:
                    pass
            
            if not improved:
                break
                
        # Convert back to time domain and clip for final evaluation
        F_complex = np.zeros(N, dtype=np.complex128)
        F_complex[0] = F_sym[0]
        if N % 2 == 0:
            F_complex[N//2] = F_sym[N//2]
        for k in range(1, N//2):
            F_complex[k] = F_sym[k]
            F_complex[N-k] = F_sym[k]
            
        f_candidate = np.real(np.fft.ifft(F_complex))
        f_candidate = np.maximum(f_candidate, 0.0)
        
        score_candidate = compute_score(f_candidate)
        
        if score_candidate < best_score:
            best_score = score_candidate
            best_f = f_candidate.copy()
            
    # Ensure nonnegative
    best_f = np.maximum(best_f, 0.0)
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
