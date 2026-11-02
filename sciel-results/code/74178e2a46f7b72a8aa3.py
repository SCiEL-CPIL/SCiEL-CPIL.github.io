import time
import numpy as np
from scipy.optimize import minimize

N = 600

def compute_score(h):
    """
    Compute the normalized maximum of the autoconvolution.
    h: array of nonnegative values.
    Returns max_k (h*h)[k] / (sum h)^2
    """
    h = np.asarray(h, dtype=np.float64)
    s = np.sum(h)
    if s < 1e-12:
        return 1e10
    # Use FFT for fast convolution
    n_fft = 1
    while n_fft < 2 * N - 1:
        n_fft *= 2
    h_padded = np.zeros(n_fft)
    h_padded[:N] = h
    conv = np.real(np.fft.ifft(np.fft.fft(h_padded) ** 2))[:2 * N - 1]
    max_conv = np.max(conv)
    return max_conv / (s * s)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Strategy: Spectral Shaping
    # Parameterize h via its DFT coefficients.
    # We focus on low-frequency components to ensure h is smooth and non-negative.
    # h = IDFT(S)
    # We constrain the imaginary part of S to 0 (real symmetric h) and optimize the real part.
    # To ensure h >= 0, we use a softplus transform or projection.
    # Given the risk of non-negativity violation with direct spectral optimization,
    # we will use a hybrid: 
    # 1. Start with uniform (optimal baseline).
    # 2. Define a parametric family of deviations from uniform in the frequency domain.
    #    h = uniform + IDFT(delta_S)
    # 3. Optimize the parameters of delta_S to minimize the score.
    
    # Baseline: Uniform
    h_uniform = np.ones(N) / N
    best_h = h_uniform.copy()
    best_score = compute_score(h_uniform)
    
    # If we have time, try to improve via spectral perturbations.
    # The uniform spectrum is a delta at 0.
    # We add small low-frequency ripples.
    
    # Number of frequency bins to perturb (excluding DC)
    num_freqs = 10
    
    # Parameters: amplitudes and phases of the first few frequency bins
    # To keep h real and symmetric, we only need amplitudes (phase 0 or pi).
    # Let h = (1/N) + sum_{k=1}^{K} a_k * cos(2*pi*k*n/N)
    # This ensures h is real and symmetric.
    # We optimize a_k.
    
    def score_from_params(params):
        # params: array of length num_freqs
        # Construct h
        n = np.arange(N)
        h = np.ones(N) / N
        for k in range(1, num_freqs + 1):
            # Use a_k as amplitude
            # Scale down to keep h positive
            amp = params[k-1] * 0.01 # Small perturbation
            h += amp * np.cos(2 * np.pi * k * n / N)
        
        # Enforce non-negativity
        h = np.maximum(h, 0)
        
        if np.sum(h) < 1e-12:
            return 1e10
            
        return compute_score(h)
    
    # Initial params: zeros
    x0 = np.zeros(num_freqs)
    
    # Run optimization
    try:
        res = minimize(
            score_from_params, 
            x0, 
            method='Nelder-Mead', 
            options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        
        if res.fun < best_score:
            best_score = res.fun
            # Reconstruct best_h from res.x
            n = np.arange(N)
            h_opt = np.ones(N) / N
            for k in range(1, num_freqs + 1):
                amp = res.x[k-1] * 0.01
                h_opt += amp * np.cos(2 * np.pi * k * n / N)
            h_opt = np.maximum(h_opt, 0)
            best_h = h_opt
    except Exception:
        pass

    # Also try a few random spectral initializations if time permits
    if time.time() - start_time < deadline - 20:
        rng = np.random.RandomState(42)
        for _ in range(5):
            x_init = rng.randn(num_freqs) * 0.1
            try:
                res = minimize(
                    score_from_params, 
                    x_init, 
                    method='Nelder-Mead', 
                    options={'maxiter': 200}
                )
                if res.fun < best_score:
                    best_score = res.fun
                    n = np.arange(N)
                    h_opt = np.ones(N) / N
                    for k in range(1, num_freqs + 1):
                        amp = res.x[k-1] * 0.01
                        h_opt += amp * np.cos(2 * np.pi * k * n / N)
                    h_opt = np.maximum(h_opt, 0)
                    best_h = h_opt
            except Exception:
                pass

    # Final check: ensure non-negativity and return
    best_h = np.maximum(best_h, 0)
    
    return best_h.tolist()
# EVOLVE-BLOCK-END
