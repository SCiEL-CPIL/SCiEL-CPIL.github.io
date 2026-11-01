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

def _soft_max_sidelobe(ac, beta=10.0):
    """
    Smooth approximation to max |sidelobe| using log-sum-exp.
    """
    sidelobes = np.abs(ac[1:N])
    max_val = np.max(sidelobes)
    if max_val < 1e-10:
        return max_val
    # log-sum-exp trick for numerical stability
    shifted = beta * (sidelobes - max_val)
    log_sum = np.log(np.sum(np.exp(shifted))) / beta
    return max_val + log_sum

def _objective_and_gradient(x, beta=10.0):
    """
    Objective: Soft-max of sidelobes (smooth surrogate for max).
    Gradient: Analytic via chain rule.
    """
    h = x
    ac = _compute_autocorr(h)
    sidelobes_abs = np.abs(ac[1:N])
    
    # Soft-max objective
    max_val = np.max(sidelobes_abs)
    if max_val < 1e-10:
        return max_val, np.zeros(N)
    
    shifted = beta * (sidelobes_abs - max_val)
    exp_shifted = np.exp(shifted)
    log_sum = np.log(np.sum(exp_shifted))
    obj = max_val + log_sum / beta
    
    # Weights for each sidelobe in the soft-max
    weights = exp_shifted / np.sum(exp_shifted)
    
    # Gradient computation
    # d(obj)/d(h_i) = sum_k weights[k] * sign(ac[k]) * d(ac[k])/d(h_i)
    # d(ac[k])/d(h_i) = h[i+k] + h[i-k] (if valid)
    
    grad = np.zeros(N)
    signs = np.sign(ac[1:N])
    
    for k in range(1, N):
        w_k = weights[k-1]
        if w_k < 1e-15:
            continue
        s_k = signs[k-1]
        coeff = w_k * s_k
        
        # Contribution to indices i where i+k < N: h[i+k]
        # Contribution to indices i where i-k >= 0: h[i-k]
        grad[0:N-k] += coeff * h[k:N]
        grad[k:N] += coeff * h[0:N-k]
        
    return obj, grad

def _local_search(h, max_iterations=10000, time_limit=5.0):
    """
    Fast local search using FFT for evaluation with simulated annealing.
    """
    start_time = time.time()
    ac = _compute_autocorr(h)
    current_val = _max_sidelobe(ac)
    
    # Simulated annealing parameters
    T0 = current_val * 0.1 if current_val > 1e-10 else 0.01
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
            # Accept improvement
            current_val = new_val
            ac = ac_new
        else:
            # Simulated annealing: accept with probability exp(-delta/T)
            T = T0 * (cooling_rate ** _)
            if T > 1e-15 and np.random.random() < np.exp(-delta / T):
                current_val = new_val
                ac = ac_new
            else:
                # Revert
                h[i] = old_h_i
                # ac remains unchanged (old ac)
                
    return h, current_val

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_h = None
    best_val = float('inf')
    
    # Phase 1: Continuous Relaxation with soft-max objective
    # Try multiple random initializations
    num_continuous_starts = 8
    
    for i in range(num_continuous_starts):
        if time.time() >= deadline - 20.0:
            break
            
        # Initialize x in [-1, 1]
        # Start near binary to help rounding
        x0 = np.random.choice([-1.0, 1.0], size=N) * 0.85
        
        # Use L-BFGS-B with soft-max objective
        try:
            res = minimize(_objective_and_gradient, x0, method='L-BFGS-B', 
                           bounds=[(-1.0, 1.0)] * N, 
                           options={'maxiter': 100, 'ftol': 1e-6, 'gtol': 1e-6, 'maxls': 20})
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
            
        # Refine the best binary solution found so far using local search
        if best_h is not None:
            if time.time() < deadline - 15.0:
                h_ref, val_ref = _local_search(best_h.copy(), max_iterations=3000, time_limit=3.0)
                if val_ref < best_val:
                    best_val = val_ref
                    best_h = h_ref
    
    # Phase 2: Final Discrete Search with simulated annealing
    remaining_time = deadline - time.time()
    if remaining_time > 5.0 and best_h is not None:
        # Use the best_h as starting point for a longer local search
        h_final, val_final = _local_search(best_h.copy(), max_iterations=200000, time_limit=remaining_time - 2.0)
        if val_final < best_val:
            best_val = val_final
            best_h = h_final
            
    if best_h is None:
        return np.ones(N)
        
    return best_h
