import numpy as np
import time
from scipy.optimize import minimize

N = 400
DEADLINE = 118.0  # Safety margin before 120s/150s kill

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

def _objective_sum_sq(x):
    """
    Objective: Sum of squares of sidelobes.
    This is a smooth surrogate for the max sidelobe problem.
    """
    h = x
    ac = _compute_autocorr(h)
    sidelobes = ac[1:N]
    obj = np.sum(sidelobes**2)
    
    # Gradient calculation
    # d/dx_i (sum_k C[k]^2) = 2 * sum_k C[k] * dC[k]/dx_i
    # dC[k]/dx_i = h[i+k] + h[i-k] (if valid)
    
    grad = np.zeros(N)
    for k in range(1, N):
        c_k = sidelobes[k-1]
        if c_k == 0:
            continue
        
        # Vectorized update
        grad[0:N-k] += 2.0 * c_k * h[k:N]
        grad[k:N] += 2.0 * c_k * h[0:N-k]
        
    return obj, grad

def _local_search_sa(h, time_limit=5.0, T0=None, cooling_rate=0.9995):
    """
    Simulated Annealing local search.
    """
    start_time = time.time()
    ac = _compute_autocorr(h)
    current_val = _max_sidelobe(ac)
    
    if T0 is None:
        T0 = max(current_val * 0.1, 0.01)
    
    # Adaptive cooling: start hot, cool down
    # We will run for a fixed number of steps or until time limit
    # To be efficient, we estimate steps per second
    # For N=400, one FFT is ~10-20us. Python loop overhead is ~10-50us.
    # So ~10k-50k steps per second.
    
    h = h.copy()
    
    for it in range(1000000): # Large cap, time limit will break
        if time.time() - start_time > time_limit:
            break
        
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

def _greedy_polish(h, time_limit=2.0):
    """
    Strict greedy search: flip only if it strictly improves the max sidelobe.
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
    deadline = start_time + DEADLINE
    
    best_h = None
    best_val = float('inf')
    
    # Phase 1: Continuous Relaxation
    # We try a few different initializations.
    num_continuous_starts = 5
    
    for i in range(num_continuous_starts):
        if time.time() >= deadline - 20.0:
            break
        
        # Initialize x in [-1, 1], start near binary
        # Using random signs gives good basins
        x0 = np.random.choice([-1.0, 1.0], size=N)
        
        try:
            # L-BFGS-B is good for this.
            # We use a low maxiter to keep it fast, as it's just for initialization
            res = minimize(_objective_sum_sq, x0, method='L-BFGS-B', 
                           bounds=[(-1.0, 1.0)] * N, 
                           options={'maxiter': 50, 'ftol': 1e-6, 'gtol': 1e-6, 'maxls': 20})
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
            
        # Phase 2: Local Search from this start
        if best_h is not None and time.time() < deadline - 10.0:
            # Allocate remaining time / remaining starts for SA
            # Simple heuristic: give each start 2-4 seconds of SA time
            time_for_sa = min(3.0, max(1.0, (deadline - 10.0 - time.time()) / max(1, num_continuous_starts - i)))
            
            h_sa, val_sa = _local_search_sa(best_h.copy(), time_limit=time_for_sa)
            
            if val_sa < best_val:
                best_val = val_sa
                best_h = h_sa
    
    # Phase 3: If we haven't found a good solution, or just to be safe,
    # run a final global SA from the best candidate with more time
    if best_h is not None and time.time() < deadline - 5.0:
        remaining_time = deadline - 5.0 - time.time()
        if remaining_time > 2.0:
            h_final, val_final = _local_search_sa(best_h.copy(), time_limit=remaining_time)
            if val_final < best_val:
                best_val = val_final
                best_h = h_final
    
    # Phase 4: Final Greedy Polish
    if best_h is not None and time.time() < deadline - 1.0:
        h_polish, val_polish = _greedy_polish(best_h.copy(), time_limit=min(2.0, deadline - 1.0 - time.time()))
        if val_polish < best_val:
            best_val = val_polish
            best_h = h_polish
    
    if best_h is None:
        # Fallback
        return np.ones(N)
        
    return best_h
