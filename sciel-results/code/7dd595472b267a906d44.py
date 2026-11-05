# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = 145.0  # wall-clock limit with safety margin
N = 400

def _evaluate_fast(h):
    """Compute the task value using FFT for speed."""
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return np.inf
    # Use FFT for convolution
    fft_h = np.fft.fft(h, n=2*N)
    conv = np.real(np.fft.ifft(fft_h * fft_h))
    # The full linear convolution is in the first 2N-1 elements
    # The last element (index 2N-1) is a wrap-around artifact
    max_conv = np.max(conv[:2*N-1])
    val = abs(2.0 * N * max_conv / (s * s))
    if not np.isfinite(val):
        return np.inf
    return val

def _build_zadoff_chu_windowed(params):
    """
    Build h from parameters:
    h[n] = scale * w[n] * Re( (1/sqrt(N)) * exp(j * pi * u * n * (n+1) / N) )
    where w[n] is a smooth window (Hann or Gaussian)
    
    params: [scale, window_width, center_offset, u]
    - scale: overall amplitude
    - window_width: width of the window (for Gaussian, sigma; for Hann, fraction of N)
    - center_offset: center of the window
    - u: Zadoff-Chu parameter (odd integer)
    """
    scale, window_width, center_offset, u = params
    
    n = np.arange(N)
    
    # Zadoff-Chu sequence
    zc = np.exp(1j * np.pi * u * n * (n + 1) / N)
    h_base = np.real(zc) / np.sqrt(N)
    
    # Window function
    if window_width > 0:
        # Gaussian window
        w = np.exp(-0.5 * ((n - center_offset) / window_width) ** 2)
    else:
        # No window (full sequence)
        w = np.ones(N)
    
    h = scale * w * h_base
    return h

def _objective(params):
    """
    Objective function for optimization.
    Returns the task value.
    """
    h = _build_zadoff_chu_windowed(params)
    return _evaluate_fast(h)

def _local_refine(h, deadline, max_iter=50000):
    """
    Local refinement using direct perturbation.
    """
    best_val = _evaluate_fast(h)
    best_h = h.copy()
    rng = np.random.default_rng(42)
    
    step_size = 0.1
    for step in range(max_iter):
        if time.time() > deadline:
            break
        
        n_perturb = rng.integers(1, 6)
        idx = rng.choice(N, size=n_perturb, replace=False)
        h_new = best_h.copy()
        h_new[idx] += rng.normal(0, step_size, n_perturb)
        
        v_new = _evaluate_fast(h_new)
        if np.isfinite(v_new) and v_new < best_val - 1e-10:
            best_val = v_new
            best_h = h_new
            step_size *= 0.99
        elif step % 5000 == 0:
            step_size *= 0.95
    
    return best_h, best_val

def construct():
    t0 = time.time()
    deadline = t0 + _DEADLINE
    
    rng = np.random.default_rng(123)
    
    best_val = 2.0
    best_h = np.ones(N)
    
    # Phase 1: Optimize Zadoff-Chu with window parameters
    # Try different u values (odd integers)
    u_values = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    
    for u in u_values:
        if time.time() > deadline - 30:
            break
        
        # Try different initial window parameters
        for window_width in [50, 100, 150, 200, 250, 300, 350, 400]:
            if time.time() > deadline - 25:
                break
            
            for center_offset in [100, 150, 200, 250, 300]:
                if time.time() > deadline - 20:
                    break
                
                # Initial parameters
                init_params = np.array([1.0, float(window_width), float(center_offset), float(u)])
                
                try:
                    res = minimize(
                        _objective,
                        init_params,
                        method='Nelder-Mead',
                        options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8}
                    )
                    h_opt = _build_zadoff_chu_windowed(res.x)
                    v_opt = _evaluate_fast(h_opt)
                    
                    if np.isfinite(v_opt) and v_opt < best_val:
                        best_val = v_opt
                        best_h = h_opt.copy()
                except Exception:
                    pass
    
    # Phase 2: Also try direct optimization of h with structured initialization
    # Use the best Zadoff-Chu as initial guess for local refinement
    if time.time() < deadline - 10:
        best_h, best_val = _local_refine(best_h, deadline - 5)
    
    # Phase 3: Try more random restarts with direct optimization if time permits
    while time.time() < deadline - 15:
        # Random restart with low-frequency structure
        n = np.arange(N)
        h_rand = np.zeros(N)
        for k in range(5):
            freq = rng.uniform(0.1, 2.0)
            phase = rng.uniform(0, 2*np.pi)
            amp = rng.uniform(0.5, 2.0)
            h_rand += amp * np.cos(2 * np.pi * freq * n / N + phase)
        
        # Quick local refinement
        try:
            h_opt, v_opt = _local_refine(h_rand, deadline - 10, max_iter=5000)
            if np.isfinite(v_opt) and v_opt < best_val:
                best_val = v_opt
                best_h = h_opt.copy()
        except Exception:
            pass
    
    # Final check
    final_val = _evaluate_fast(best_h)
    if not np.isfinite(final_val) or final_val >= 2.0:
        best_h = np.ones(N)
        
    return best_h
# EVOLVE-BLOCK-END
