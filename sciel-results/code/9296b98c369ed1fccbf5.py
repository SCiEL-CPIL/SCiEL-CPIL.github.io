import numpy as np
import time
from scipy.optimize import minimize

N = 400

def _compute_autocorr(h):
    """
    Compute linear aperiodic autocorrelation using FFT.
    Returns array of length 2*N.
    """
    n_fft = 2 * N
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    ac = np.fft.irfft(power, n=n_fft)
    return ac

def _max_sidelobe(ac):
    """
    Compute max absolute value of sidelobes.
    """
    # Due to symmetry, max |ac[1:N]| is sufficient
    return np.max(np.abs(ac[1:N]))

def _objective_and_gradient(x, beta=50.0):
    """
    Objective: Smooth approximation of max sidelobe (soft-max).
    Gradient: Analytic.
    """
    h = x
    ac = _compute_autocorr(h)
    sidelobes_abs = np.abs(ac[1:N])
    
    # Soft max value
    max_val = np.max(sidelobes_abs)
    if max_val < 1e-10:
        return max_val, np.zeros(N)
        
    shifted = beta * (sidelobes_abs - max_val)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    obj = max_val + np.log(sum_exp) / beta
    
    # Gradient of soft max w.r.t. sidelobes
    # d/dx_i (soft_max) = sum_k w_k * d|ac[k]|/dx_i
    # where w_k = exp(beta * (ac[k] - max)) / sum_exp = beta * exp_shifted[k] / sum_exp
    # Note: d|ac[k]|/dx_i = sign(ac[k]) * d(ac[k])/dx_i
    # d(ac[k])/dx_i = h[i+k] + h[i-k] (if valid)
    
    weights = beta * exp_shifted / sum_exp
    
    grad = np.zeros(N)
    
    for k in range(1, N):
        w_k = weights[k-1]
        if w_k == 0:
            continue
        
        # ac[k] = sum_j h[j] h[j+k]
        # d(ac[k])/dx_i = h[i+k] + h[i-k] (if valid)
        # sign(ac[k]) * d(ac[k])/dx_i
        
        sign_k = np.sign(ac[k])
        c_k = sign_k * w_k
        
        if c_k == 0:
            continue
            
        # Vectorized update
        # Indices i such that i+k < N => i < N-k
        # grad[0:N-k] += c_k * h[k:N]
        # Indices i such that i-k >= 0 => i >= k
        # grad[k:N] += c_k * h[0:N-k]
        
        grad[0:N-k] += c_k * h[k:N]
        grad[k:N] += c_k * h[0:N-k]
        
    return obj, grad

def _local_search(h, max_iterations=10000, time_limit=5.0, T0=None):
    """
    Fast local search using FFT for evaluation with simulated annealing.
    """
    start_time = time.time()
    ac = _compute_autocorr(h)
    current_val = _max_sidelobe(ac)
    
    if T0 is None:
        T0 = current_val * 0.05 if current_val > 1e-10 else 0.01
    
    # Simulated annealing parameters
    T = T0
    cooling_rate = 0.999
    
    for _ in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        # Try flipping a random bit
        i = np.random.randint(N)
        old_h_i = h[i]
        h[i] = -old_h_i
        
        ac_new = _compute_autocorr(h)
        new_val = _max_sidelobe(ac_new)
        
        delta = new_val - current_val
        
        if delta < 0:
            # Improvement
            current_val = new_val
            ac = ac_new
        else:
            # Deterioration: accept with probability exp(-delta/T)
            if T > 1e-10 and np.random.rand() < np.exp(-delta / T):
                current_val = new_val
                ac = ac_new
            else:
                # Revert
                h[i] = old_h_i
                # ac remains unchanged (old ac)
        
        T *= cooling_rate
        
    return h, current_val

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_h = None
    best_val = float('inf')
    
    # Phase 1: Continuous Relaxation with Smooth Objective
    # Try a few random initializations
    num_continuous_starts = 8
    betas = [20.0, 50.0, 100.0, 200.0, 50.0, 100.0, 20.0, 50.0]
    
    for i in range(num_continuous_starts):
        if time.time() >= deadline - 30.0:  # Reserve 30s for final polish
            break
            
        # Initialize x in [-1, 1]
        # Start near binary to help rounding
        x0 = np.random.choice([-1.0, 1.0], size=N)
        
        # Use a beta that increases over iterations or pick one
        # Let's try different betas for different starts
        beta = betas[i % len(betas)]
        
        # Define objective with fixed beta
        def obj_grad(x, beta=beta):
            return _objective_and_gradient(x, beta)
        
        # Use L-BFGS-B
        try:
            res = minimize(obj_grad, x0, method='L-BFGS-B', 
                           bounds=[(-1.0, 1.0)] * N, 
                           options={'maxiter': 150, 'ftol': 1e-8, 'gtol': 1e-6, 'maxls': 30})
        except Exception:
            continue
        
        # Round to binary
        h_bin = np.sign(res.x)
        # Handle zeros
        zero_mask = (h_bin == 0)
        if np.any(zero_mask):
            h_bin[zero_mask] = np.random.choice([-1.0, 1.0], size=np.sum(zero_mask))
            
        # Evaluate binary solution
        ac = _compute_autocorr(h_bin)
        val = _max_sidelobe(ac)
        
        if val < best_val:
            best_val = val
            best_h = h_bin.copy()
            
        # Quick local search refinement on the new candidate
        if time.time() < deadline - 10.0:
            h_ref, val_ref = _local_search(h_bin.copy(), max_iterations=2000, time_limit=2.0)
            if val_ref < best_val:
                best_val = val_ref
                best_h = h_ref
    
    # Phase 2: Refine best solution with local search
    if best_h is not None and time.time() < deadline - 10.0:
        h_ref, val_ref = _local_search(best_h.copy(), max_iterations=5000, time_limit=5.0)
        if val_ref < best_val:
            best_val = val_ref
            best_h = h_ref
    
    # Phase 3: Final Discrete Search
    remaining_time = deadline - time.time()
    if remaining_time > 2.0 and best_h is not None:
        # Use a higher initial temperature for the final pass to escape local minima
        T_final = best_val * 0.05 if best_val > 1e-10 else 0.01
        h_final, val_final = _local_search(best_h.copy(), max_iterations=50000, time_limit=remaining_time - 1.0, T0=T_final)
        if val_final < best_val:
            best_val = val_final
            best_h = h_final
            
    if best_h is None:
        return np.ones(N)
        
    return best_h
