"""
Combined approach: Spectral design via Golay-like construction + L-BFGS-B refinement.

Strategy:
1. Use a deterministic algebraic/chaotic sequence (logistic map) as a robust initial guess
   that inherently has good spectral properties.
2. Refine using L-BFGS-B with a soft-max objective on the off-diagonal autoconvolution,
   which is much more efficient and robust than manual gradient descent.
3. The metric is max|c_k| / (sum(x))^2 for k != 0 (off-diagonal).
"""
import numpy as np
import time
import scipy.optimize

N = 400

def _autoconv_lin(x):
    """Compute linear autoconvolution using FFT."""
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    fx = np.fft.rfft(x, size)
    return np.fft.irfft(fx * fx, size)[:2 * n - 1]

def _metric_offdiag(x):
    """Metric: max |c_k| for k != 0 / S^2"""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    c = _autoconv_lin(x)
    if len(c) > 1:
        return float(np.max(np.abs(c[1:]))) / (s * s)
    return 0.0

def _objective_soft(x, beta=50.0):
    """
    Smoothed objective for off-diagonal metric using log-sum-exp for soft max.
    This provides a smooth landscape for L-BFGS-B.
    """
    s = np.sum(x)
    if abs(s) < 1e-12:
        return 1e10
    c = _autoconv_lin(x)
    if len(c) <= 1:
        return 0.0
    abs_c = np.abs(c[1:])
    m = np.max(abs_c)
    if m < 1e-15:
        return 0.0
    # Soft max approximation
    soft_max = m + np.log(np.sum(np.exp(beta * (abs_c - m)))) / beta
    return soft_max / (s * s)

def _generate_logistic_sequence(n, r=3.99, x0=0.5, seed=42):
    """
    Generate a deterministic sequence using the logistic map.
    The logistic map at r near 4 produces chaotic sequences with good
    low-discrepancy properties and flat spectral density.
    """
    rng = np.random.default_rng(seed)
    # Use a slightly perturbed initial condition for variety
    x_val = x0 + rng.uniform(-0.01, 0.01)
    seq = np.zeros(n)
    for i in range(n):
        x_val = r * x_val * (1.0 - x_val)
        # Map to [-1, 1] range to create signed heights
        seq[i] = 2.0 * x_val - 1.0
    return seq

def construct():
    """
    Construct a signed step function minimizing max|autoconv| / (sum(x))^2.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # Phase 1: Generate multiple candidate initial sequences
    candidates = []
    
    # Candidate 1: All ones (baseline)
    x_const = np.ones(N)
    candidates.append(x_const)
    
    # Candidate 2: Logistic map sequence
    x_log = _generate_logistic_sequence(N, r=3.99, x0=0.5, seed=42)
    candidates.append(x_log)
    
    # Candidate 3: Logistic map with different parameters
    x_log2 = _generate_logistic_sequence(N, r=3.95, x0=0.3, seed=123)
    candidates.append(x_log2)
    
    # Candidate 4: Hanning window (known to have low sidelobes)
    x_hann = np.hanning(N)
    candidates.append(x_hann)
    
    # Candidate 5: Random sequence with zero mean (for comparison)
    rng = np.random.default_rng(999)
    x_rand = rng.normal(0, 1, N)
    candidates.append(x_rand)
    
    # Evaluate all candidates
    best_x = None
    best_score = float('inf')
    
    for x in candidates:
        s = np.sum(x)
        if abs(s) < 1e-10:
            continue
        sc = _metric_offdiag(x)
        if sc < best_score:
            best_score = sc
            best_x = x.copy()
    
    if best_x is None:
        return np.ones(N)
    
    # Phase 2: Refine the best candidate using L-BFGS-B
    # We optimize the soft-max objective, which is a smooth approximation
    # of the actual metric
    
    x_init = best_x.copy()
    
    # Ensure sum is not too small by adding a small DC offset if needed
    s_init = np.sum(x_init)
    if abs(s_init) < 1.0:
        x_init += (1.0 - s_init) / N
    
    try:
        # Use L-BFGS-B for efficient optimization
        # The objective is smooth due to soft-max
        res = scipy.optimize.minimize(
            _objective_soft,
            x_init,
            method='L-BFGS-B',
            options={
                'maxiter': 500,
                'ftol': 1e-15,
                'gtol': 1e-10,
                'maxls': 50
            },
            tol=1e-12
        )
        
        x_opt = res.x
        sc_opt = _metric_offdiag(x_opt)
        
        if sc_opt < best_score:
            best_score = sc_opt
            best_x = x_opt.copy()
            
    except Exception:
        pass
    
    # Phase 3: Local refinement with random perturbations
    x = best_x.copy()
    current_score = best_score
    rng = np.random.default_rng(777)
    
    max_iters = 500
    for it in range(max_iters):
        if time.time() > deadline - 5.0:
            break
        
        # Random perturbation with decreasing scale
        scale = 0.1 * (1.0 - it / max_iters)
        mask = rng.random(N) < 0.05
        perturb = rng.normal(0, scale, N)
        x_new = x + mask * perturb
        
        s_new = np.sum(x_new)
        if abs(s_new) < 1e-10:
            continue
        
        sc_new = _metric_offdiag(x_new)
        if sc_new < current_score:
            x = x_new
            current_score = sc_new
            if current_score < best_score:
                best_score = current_score
                best_x = x.copy()
    
    # Phase 4: Try thresholding to ±1
    x_thresh = np.sign(best_x)
    # Ensure not all same sign
    if np.all(x_thresh == 1):
        x_thresh[::2] = -1
    elif np.all(x_thresh == -1):
        x_thresh[::2] = 1
    
    s_thresh = np.sum(x_thresh)
    if abs(s_thresh) > 1e-10:
        score_thresh = _metric_offdiag(x_thresh)
        if score_thresh < best_score:
            best_score = score_thresh
            best_x = x_thresh.copy()
    
    return best_x
