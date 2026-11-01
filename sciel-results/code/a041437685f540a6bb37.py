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

def _objective_and_gradient(x, beta=12.0):
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

def _local_search(h, max_iterations=10000, time_limit=5.0, multi_flip_prob=0.3):
    """
    Fast local search using FFT for evaluation with simulated annealing.
    Supports single and multi-bit flips for better exploration.
    """
    start_time = time.time()
    ac = _compute_autocorr(h)
    current_val = _max_sidelobe(ac)
    
    # Simulated annealing parameters
    T0 = current_val * 0.1 if current_val > 1e-10 else 0.01
    cooling_rate = 0.9995
    
    for it in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        # Decide between single-bit flip and multi-bit flip
        use_multi = np.random.random() < multi_flip_prob
        if use_multi:
            # Flip 2-3 random bits
            num_flips = np.random.randint(2, 4)
            indices = np.random.choice(N, size=num_flips, replace=False)
            old_vals = h[indices].copy()
            h[indices] = -h[indices]
        else:
            # Flip a single random bit
            indices = np.array([np.random.randint(N)])
            old_vals = h[indices].copy()
            h[indices] = -h[indices]
        
        ac_new = _compute_autocorr(h)
        new_val = _max_sidelobe(ac_new)
        
        delta = new_val - current_val
        if delta < 0:
            # Accept improvement
            current_val = new_val
            ac = ac_new
        else:
            # Simulated annealing: accept with probability exp(-delta/T)
            T = T0 * (cooling_rate ** it)
            if T > 1e-15 and np.random.random() < np.exp(-delta / T):
                current_val = new_val
                ac = ac_new
            else:
                # Revert
                h[indices] = old_vals
                # ac remains unchanged (old ac)
                
    return h, current_val

def _polish_binary(h, time_limit=10.0):
    """
    High-precision polish: try all single-bit flips and accept only strict improvements.
    Repeat until no improvement found or time limit reached.
    """
    start_time = time.time()
    ac = _compute_autocorr(h)
    current_val = _max_sidelobe(ac)
    improved = True
    
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for i in range(N):
            if (time.time() - start_time) > time_limit:
                break
            old_val = h[i]
            h[i] = -old_val
            ac_new = _compute_autocorr(h)
            new_val = _max_sidelobe(ac_new)
            if new_val < current_val:
                current_val = new_val
                ac = ac_new
                improved = True
            else:
                h[i] = old_val
                # ac remains unchanged
    
    return h, current_val

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    best_h = None
    best_val = float('inf')
    
    # Phase 1: Continuous Relaxation with soft-max objective
    # Try multiple random initializations with varying beta
    num_continuous_starts = 12
    betas = [8.0, 10.0, 12.0, 15.0, 20.0, 8.0, 10.0, 12.0, 15.0, 20.0, 10.0, 12.0]
    
    for i in range(num_continuous_starts):
        if time.time() >= deadline - 30.0:
            break
            
        beta = betas[i]
        
        # Initialize x in [-1, 1]
        # Start near binary to help rounding
        x0 = np.random.choice([-1.0, 1.0], size=N) * 0.9
        
        # Use L-BFGS-B with soft-max objective
        try:
            res = minimize(lambda x: _objective_and_gradient(x, beta=beta)[0], 
                         x0, 
                         jac=lambda x: _objective_and_gradient(x, beta=beta)[1],
                         method='L-BFGS-B', 
                         bounds=[(-1.0, 1.0)] * N, 
                         options={'maxiter': 150, 'ftol': 1e-7, 'gtol': 1e-7, 'maxls': 30, 'disp': False})
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
            if time.time() < deadline - 25.0:
                h_ref, val_ref = _local_search(best_h.copy(), max_iterations=5000, time_limit=3.0)
                if val_ref < best_val:
                    best_val = val_ref
                    best_h = h_ref
    
    # Phase 2: Additional discrete restarts from best found
    remaining_time = deadline - time.time()
    if remaining_time > 15.0 and best_h is not None:
        # Do a few more restarts with local search
        for restart in range(3):
            if time.time() >= deadline - 10.0:
                break
            # Perturb best_h
            h_pert = best_h.copy()
            num_pert = np.random.randint(5, 15)
            pert_indices = np.random.choice(N, size=num_pert, replace=False)
            h_pert[pert_indices] = -h_pert[pert_indices]
            
            h_ref, val_ref = _local_search(h_pert, max_iterations=10000, time_limit=5.0)
            if val_ref < best_val:
                best_val = val_ref
                best_h = h_ref
    
    # Phase 3: Final Discrete Search with simulated annealing
    remaining_time = deadline - time.time()
    if remaining_time > 10.0 and best_h is not None:
        h_final, val_final = _local_search(best_h.copy(), max_iterations=500000, time_limit=remaining_time - 5.0)
        if val_final < best_val:
            best_val = val_final
            best_h = h_final
            
    # Phase 4: High-precision polish
    remaining_time = deadline - time.time()
    if remaining_time > 3.0 and best_h is not None:
        h_polished, val_polished = _polish_binary(best_h.copy(), time_limit=remaining_time - 2.0)
        if val_polished < best_val:
            best_val = val_polished
            best_h = h_polished
            
    if best_h is None:
        return np.ones(N)
        
    return best_h
