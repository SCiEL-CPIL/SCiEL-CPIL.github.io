"""Rewritten program for third autocorrelation inequality optimization."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.fft import fft, ifft
from scipy.optimize import minimize

def _eval_value(h):
    """Compute the objective value."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _convolve_fft(h):
    """Compute full autoconvolution using FFT."""
    n_conv = 2 * len(h) - 1
    n_fft = 1
    while n_fft < n_conv:
        n_fft *= 2
    H = fft(h, n=n_fft)
    conv = ifft(H * H, n=n_fft).real[:n_conv]
    return conv

def _soft_max(x, beta):
    """Numerically stable soft maximum."""
    m = np.max(x)
    if not np.isfinite(m):
        return m
    exp_vals = np.exp(beta * (x - m))
    sm = m + np.log(np.sum(exp_vals)) / beta
    return sm

def _objective(h, beta):
    """
    Objective function for optimization.
    Minimizes the soft-max of the autoconvolution peak, normalized by sum^2.
    """
    if not np.all(np.isfinite(h)):
        return 1e10
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = _convolve_fft(h)
    sm = _soft_max(conv, beta)
    return 2.0 * N * sm / (s * s)

def _simulated_annealing_binary(rng, start_time, deadline, max_steps_per_trial=5000):
    """
    Simulated annealing on binary sequences {-1, +1} to minimize max autoconvolution.
    Returns the best binary sequence found and its value.
    """
    best_seq = None
    best_val = np.inf
    
    # Try multiple restarts
    n_restarts = 3
    for restart in range(n_restarts):
        if time.time() > deadline - 30:
            break
        
        # Initialize binary sequence
        if restart == 0:
            # Start with all +1
            seq = np.ones(N, dtype=np.int8)
        else:
            # Random initialization
            seq = rng.choice([-1, 1], size=N).astype(np.int8)
        
        # Ensure sum is not zero (for binary, sum must be even, so avoid sum=0)
        # If sum is 0, flip one element
        while np.sum(seq) == 0:
            idx = rng.randint(N)
            seq[idx] *= -1
        
        current_val = _eval_value(seq.astype(np.float64))
        current_seq = seq.copy()
        
        # Annealing schedule
        T_start = 1.0
        T_end = 0.01
        n_steps = max_steps_per_trial
        cooling = (T_end / T_start) ** (1.0 / n_steps)
        
        T = T_start
        for step in range(n_steps):
            if time.time() > deadline - 15:
                break
            
            # Pick a random index to flip
            idx = rng.randint(N)
            old_val = seq[idx]
            seq[idx] *= -1
            
            # Check if sum is still valid
            new_sum = np.sum(seq)
            if new_sum == 0:
                # Revert if sum becomes zero
                seq[idx] = old_val
                continue
            
            new_val = _eval_value(seq.astype(np.float64))
            
            # Acceptance criterion
            if new_val < current_val:
                # Always accept improvement
                current_val = new_val
                current_seq = seq.copy()
                if new_val < best_val:
                    best_val = new_val
                    best_seq = seq.copy()
            else:
                # Accept with probability exp(-delta/T)
                delta = new_val - current_val
                if delta > 0:
                    prob = np.exp(-delta / max(T, 1e-10))
                    if rng.random() < prob:
                        current_val = new_val
                        current_seq = seq.copy()
                    else:
                        # Revert
                        seq[idx] = old_val
            
            # Cool down
            T *= cooling
        
        # End of restart
        if current_val < best_val:
            best_val = current_val
            best_seq = current_seq.copy()
    
    if best_seq is None:
        return None, np.inf
    return best_seq, best_val

def _gradient_refine(h_init, start_time, deadline):
    """
    Gradient-based refinement using soft-max surrogate.
    Returns the best refined sequence and its value.
    """
    best_h = h_init.copy()
    best_val = _eval_value(h_init)
    
    betas = [5.0, 20.0, 50.0, 100.0, 200.0]
    current_h = h_init.copy()
    
    for beta in betas:
        if time.time() > deadline - 10:
            break
        
        def obj(h):
            return _objective(h, beta)
        
        # Try a few restarts
        for trial in range(2):
            if time.time() > deadline - 5:
                break
            
            if trial == 0:
                h_try = current_h.copy()
            else:
                noise_scale = 0.1 / (beta + 1.0)
                h_try = current_h + noise_scale * np.random.randn(N)
                h_try -= np.mean(h_try)
                h_try += np.mean(current_h)
            
            try:
                result = minimize(
                    obj,
                    h_try,
                    method='L-BFGS-B',
                    options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 30}
                )
                
                h_opt = result.x
                if np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-10:
                    v_opt = _eval_value(h_opt)
                    if v_opt < best_val:
                        best_val = v_opt
                        best_h = h_opt.copy()
                        current_h = h_opt.copy()
            except Exception:
                continue
    
    return best_h, best_val

def construct():
    """
    Returns a sequence of 400 signed step heights.
    Uses simulated annealing on binary sequences followed by gradient refinement.
    """
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    rng = np.random.RandomState(42)
    
    # Phase 1: Simulated annealing on binary sequences
    binary_seq, binary_val = _simulated_annealing_binary(rng, start_time, deadline)
    
    if binary_seq is not None:
        best_h = binary_seq.astype(np.float64)
        best_val = binary_val
    else:
        best_h = np.ones(N)
        best_val = 2.0
    
    # Phase 2: Gradient-based refinement
    if time.time() < deadline - 20:
        refined_h, refined_val = _gradient_refine(best_h, start_time, deadline)
        if refined_val < best_val:
            best_h = refined_h
            best_val = refined_val
    
    # Ensure finite and sum != 0
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        return np.ones(N)
    
    return best_h
# EVOLVE-BLOCK-END
