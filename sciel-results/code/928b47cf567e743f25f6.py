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

def _c3_objective_continuous(h):
    """
    Compute the C3 upper bound for continuous h.
    Metric: max_k |conv(h, h)[k]| / (sum_i |h_i|)^2
    """
    n_fft = 2 * N
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    ac = np.fft.irfft(power, n=n_fft)
    
    sum_abs_h = np.sum(np.abs(h))
    if sum_abs_h < 1e-12:
        return 1e10
        
    return np.max(np.abs(ac)) / (sum_abs_h ** 2)

def _soft_c3_objective_continuous(h, beta=50.0):
    """
    Smooth approximation of C3 upper bound using soft-max.
    """
    n_fft = 2 * N
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    ac = np.fft.irfft(power, n=n_fft)
    
    sum_abs_h = np.sum(np.abs(h))
    if sum_abs_h < 1e-12:
        return 1e10
    
    sidelobes = np.abs(ac[1:N])
    max_val = np.max(sidelobes)
    if max_val < 1e-10:
        return max_val / (sum_abs_h ** 2)
    
    shifted = beta * (sidelobes - max_val)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_max = max_val + log_sum / beta
    
    return soft_max / (sum_abs_h ** 2)

def _objective_and_gradient(x, beta=50.0):
    """
    Objective function with gradient for L-BFGS-B.
    x is the raw variable, h = tanh(x) to bound in (-1, 1).
    """
    h = np.tanh(x)
    val = _soft_c3_objective_continuous(h, beta=beta)
    
    # Numerical gradient
    grad = np.zeros_like(x)
    eps = 1e-5
    for i in range(N):
        x_plus = x.copy()
        x_plus[i] += eps
        h_plus = np.tanh(x_plus)
        val_plus = _soft_c3_objective_continuous(h_plus, beta=beta)
        
        x_minus = x.copy()
        x_minus[i] -= eps
        h_minus = np.tanh(x_minus)
        val_minus = _soft_c3_objective_continuous(h_minus, beta=beta)
        
        grad[i] = (val_plus - val_minus) / (2 * eps)
    
    return val, grad

def _local_search(h, max_iterations=100000, time_limit=5.0, use_pair_flips=True):
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
    
    # Phase 1: Continuous optimization with L-BFGS-B
    # Try multiple initializations
    initializations = [
        np.full(N, 3.0),  # All positive
        np.array([3.0 if i % 2 == 0 else -3.0 for i in range(N)]),  # Alternating
        np.random.randn(N) * 2.0,  # Random
        np.repeat(np.random.choice([1, -1], size=N//10) * 3, 10),  # Blocky
    ]
    
    for x0 in initializations:
        if time.time() >= deadline - 30.0:
            break
        
        try:
            res = minimize(_objective_and_gradient, x0, method='L-BFGS-B',
                           jac=True, options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            
            h_opt = np.tanh(res.x)
            # Threshold to binary
            h_bin = np.sign(h_opt)
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
        
        except Exception:
            continue
    
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
