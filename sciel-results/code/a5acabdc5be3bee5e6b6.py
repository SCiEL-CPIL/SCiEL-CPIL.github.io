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
    return np.max(np.abs(ac[1:N]))

def _soft_max_sidelobe(ac, beta=50.0):
    """
    Smooth approximation of max |ac[1:N]| using log-sum-exp.
    """
    sidelobes = np.abs(ac[1:N])
    max_val = np.max(sidelobes)
    if max_val < 1e-10:
        return max_val
    shifted = beta * (sidelobes - max_val)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return max_val + log_sum / beta

def _objective_and_gradient(x, beta=50.0):
    """
    Objective: Smooth approximation of max sidelobe.
    Gradient: Analytic.
    """
    h = x
    ac = _compute_autocorr(h)
    sidelobes_abs = np.abs(ac[1:N])
    
    max_val = np.max(sidelobes_abs)
    if max_val < 1e-10:
        return max_val, np.zeros(N)
    
    shifted = beta * (sidelobes_abs - max_val)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    obj = max_val + np.log(sum_exp) / beta
    
    weights = beta * exp_shifted / sum_exp
    
    grad = np.zeros(N)
    signs = np.sign(ac[1:N])
    
    for k in range(1, N):
        w_k = weights[k-1]
        if w_k < 1e-15:
            continue
        
        c_k = signs[k-1] * w_k
        
        grad[0:N-k] += c_k * h[k:N]
        grad[k:N] += c_k * h[0:N-k]
        
    return obj, grad

def _local_search(h, max_iterations=10000, time_limit=5.0, use_pair_flips=True):
    """
    Fast local search using FFT for evaluation with simulated annealing and pair flips.
    """
    start_time = time.time()
    ac = _compute_autocorr(h)
    current_val = _max_sidelobe(ac)
    
    T0 = max(current_val * 0.15, 0.01)
    cooling_rate = 0.9995
    
    for it in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        # Choose move type: single flip or pair flip
        if use_pair_flips and it % 3 == 0:
            # Pair flip: flip two random bits
            i, j = np.random.choice(N, size=2, replace=False)
            old_i, old_j = h[i], h[j]
            h[i] = -h[i]
            h[j] = -h[j]
            
            ac_new = _compute_autocorr(h)
            new_val = _max_sidelobe(ac_new)
            
            delta = new_val - current_val
            T = T0 * (cooling_rate ** it)
            if delta < 0 or (T > 1e-15 and np.random.random() < np.exp(-delta / T)):
                current_val = new_val
                ac = ac_new
            else:
                h[i] = old_i
                h[j] = old_j
        else:
            # Single flip
            i = np.random.randint(N)
            old_h_i = h[i]
            h[i] = -h[i]
            
            ac_new = _compute_autocorr(h)
            new_val = _max_sidelobe(ac_new)
            
            delta = new_val - current_val
            T = T0 * (cooling_rate ** it)
            if delta < 0 or (T > 1e-15 and np.random.random() < np.exp(-delta / T)):
                current_val = new_val
                ac = ac_new
            else:
                h[i] = old_h_i
                
    return h, current_val

def _refine_discrete(h, time_limit=2.0):
    """
    Quick discrete refinement using greedy improvement.
    """
    start_time = time.time()
    ac = _compute_autocorr(h)
    current_val = _max_sidelobe(ac)
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for i in range(N):
            h[i] = -h[i]
            ac_new = _compute_autocorr(h)
            new_val = _max_sidelobe(ac_new)
            if new_val < current_val:
                current_val = new_val
                ac = ac_new
                improved = True
            else:
                h[i] = -h[i]
                
    return h, current_val

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_h = None
    best_val = float('inf')
    
    # Phase 1: Continuous Relaxation with soft-max objective
    num_continuous_starts = 10
    
    for i in range(num_continuous_starts):
        if time.time() >= deadline - 30.0:
            break
        
        # Initialize x in [-1, 1], start near binary
        x0 = np.random.choice([-1.0, 1.0], size=N) * 0.9
        
        try:
            res = minimize(_objective_and_gradient, x0, method='L-BFGS-B', 
                           bounds=[(-1.0, 1.0)] * N, 
                           options={'maxiter': 150, 'ftol': 1e-7, 'gtol': 1e-7, 'maxls': 30})
        except Exception:
            continue
        
        # Round to binary
        h_bin = np.sign(res.x)
        zero_mask = (h_bin == 0)
        if np.any(zero_mask):
            h_bin[zero_mask] = np.random.choice([-1.0, 1.0], size=np.sum(zero_mask))
            
        # Evaluate binary solution
        ac = _compute_autocorr(h_bin)
        val = _max_sidelobe(ac)
        
        if val < best_val:
            best_val = val
            best_h = h_bin.copy()
            
        # Quick greedy refinement
        if best_h is not None and time.time() < deadline - 25.0:
            h_ref, val_ref = _refine_discrete(best_h.copy(), time_limit=1.0)
            if val_ref < best_val:
                best_val = val_ref
                best_h = h_ref
    
    # Phase 2: Local search with simulated annealing and pair flips
    if best_h is not None:
        remaining_time = deadline - time.time()
        if remaining_time > 10.0:
            h_final, val_final = _local_search(best_h.copy(), max_iterations=500000, time_limit=remaining_time - 5.0, use_pair_flips=True)
            if val_final < best_val:
                best_val = val_final
                best_h = h_final
    
    # Phase 3: Final greedy polish
    if best_h is not None and time.time() < deadline - 2.0:
        h_polish, val_polish = _refine_discrete(best_h.copy(), time_limit=1.5)
        if val_polish < best_val:
            best_val = val_polish
            best_h = h_polish
    
    if best_h is None:
        return np.ones(N)
        
    return best_h
