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

def _soft_max_sidelobe(ac, beta=50.0):
    """
    Smooth approximation of max |ac[1:N]| using log-sum-exp.
    """
    sidelobes = np.abs(ac[1:N])
    max_val = np.max(sidelobes)
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
    
    # Soft max value
    max_val = np.max(sidelobes_abs)
    shifted = beta * (sidelobes_abs - max_val)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    obj = max_val + np.log(sum_exp) / beta
    
    # Gradient of soft max w.r.t. sidelobes
    weights = beta * exp_shifted / sum_exp
    
    grad = np.zeros(N)
    
    for k in range(1, N):
        w_k = weights[k-1]
        if w_k == 0:
            continue
        
        sign_k = np.sign(ac[k])
        c_k = sign_k * w_k
        
        if c_k == 0:
            continue
            
        grad[0:N-k] += c_k * h[k:N]
        grad[k:N] += c_k * h[0:N-k]
        
    return obj, grad

def _local_search(h, max_iterations=10000, time_limit=5.0):
    """
    Fast local search using FFT for evaluation.
    """
    start_time = time.time()
    ac = _compute_autocorr(h)
    current_val = _max_sidelobe(ac)
    
    for _ in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        # Try flipping a random bit
        i = np.random.randint(N)
        old_h_i = h[i]
        h[i] = -old_h_i
        
        ac_new = _compute_autocorr(h)
        new_val = _max_sidelobe(ac_new)
        
        if new_val < current_val:
            current_val = new_val
            ac = ac_new
        else:
            # Revert
            h[i] = old_h_i
            
    return h, current_val

def _simulated_annealing(h, start_temp=1.0, end_temp=0.01, iterations=5000, time_limit=10.0):
    """
    Simulated annealing to escape local minima.
    """
    start_time = time.time()
    ac = _compute_autocorr(h)
    current_val = _max_sidelobe(ac)
    best_h = h.copy()
    best_val = current_val
    
    for i in range(iterations):
        if time.time() - start_time > time_limit:
            break
        
        # Temperature schedule
        progress = i / iterations
        temp = start_temp * (end_temp / start_temp) ** progress
        
        # Try flipping a random bit
        idx = np.random.randint(N)
        old_h_i = h[idx]
        h[idx] = -old_h_i
        
        ac_new = _compute_autocorr(h)
        new_val = _max_sidelobe(ac_new)
        
        delta = new_val - current_val
        
        if delta < 0 or np.random.random() < np.exp(-delta / max(temp, 1e-10)):
            current_val = new_val
            ac = ac_new
            if new_val < best_val:
                best_val = new_val
                best_h = h.copy()
        else:
            # Revert
            h[idx] = old_h_i
            
    return best_h, best_val

def _gradient_descent_step(h, lr=0.1, beta=50.0, steps=100):
    """
    Perform gradient descent on the continuous relaxation, then round.
    """
    x = h.astype(float)
    for _ in range(steps):
        obj, grad = _objective_and_gradient(x, beta)
        x = x - lr * grad
        # Project to [-1, 1]
        x = np.clip(x, -1.0, 1.0)
    
    # Round to binary
    h_bin = np.sign(x)
    zero_mask = (h_bin == 0)
    if np.any(zero_mask):
        h_bin[zero_mask] = np.random.choice([-1.0, 1.0], size=np.sum(zero_mask))
    
    return h_bin

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_h = None
    best_val = float('inf')
    
    # Phase 1: Continuous Relaxation with Smooth Objective
    # Try a few random initializations
    num_continuous_starts = 12
    betas = [10.0, 20.0, 50.0, 100.0, 200.0]
    
    for i in range(num_continuous_starts):
        if time.time() >= deadline - 30.0:  # Reserve 30s for final polish
            break
            
        # Initialize x in [-1, 1]
        x0 = np.random.choice([-1.0, 1.0], size=N)
        
        beta = betas[i % len(betas)]
        
        # Define objective with fixed beta
        def obj_grad(x, beta=beta):
            return _objective_and_gradient(x, beta)
        
        # Use L-BFGS-B
        try:
            res = minimize(obj_grad, x0, method='L-BFGS-B', 
                           bounds=[(-1.0, 1.0)] * N, 
                           options={'maxiter': 200, 'ftol': 1e-9, 'gtol': 1e-7, 'maxls': 30})
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
    
    # Phase 2: Refine best solution with local search
    if best_h is not None and time.time() < deadline - 20.0:
        h_ref, val_ref = _local_search(best_h.copy(), max_iterations=5000, time_limit=5.0)
        if val_ref < best_val:
            best_val = val_ref
            best_h = h_ref
    
    # Phase 3: Simulated Annealing to escape local minima
    if best_h is not None and time.time() < deadline - 15.0:
        h_sa, val_sa = _simulated_annealing(best_h.copy(), start_temp=0.5, end_temp=0.01, 
                                            iterations=3000, time_limit=8.0)
        if val_sa < best_val:
            best_val = val_sa
            best_h = h_sa
    
    # Phase 4: Final Discrete Search
    remaining_time = deadline - time.time()
    if remaining_time > 2.0 and best_h is not None:
        h_final, val_final = _local_search(best_h.copy(), max_iterations=50000, time_limit=remaining_time - 1.0)
        if val_final < best_val:
            best_val = val_final
            best_h = h_final
            
    if best_h is None:
        return np.ones(N)
        
    return best_h
