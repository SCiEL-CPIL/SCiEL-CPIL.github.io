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

def _local_search(h, max_iterations=100000, time_limit=2.0, use_pair_flips=True):
    """
    Fast local search using FFT for evaluation with simulated annealing and pair flips.
    """
    start_time = time.time()
    ac = _compute_autocorr(h)
    current_val = _max_sidelobe(ac)
    
    T0 = max(current_val * 0.2, 0.05)
    cooling_rate = 0.999
    
    for it in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        if use_pair_flips:
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

def _refine_discrete(h, time_limit=1.0):
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
    deadline = start_time + 140.0
    
    best_h = None
    best_val = float('inf')
    
    while time.time() < deadline - 10.0:
        h_init = np.random.choice([-1.0, 1.0], size=N)
        
        h_sa, val_sa = _local_search(h_init.copy(), max_iterations=50000, time_limit=2.0, use_pair_flips=True)
        
        h_ref, val_ref = _refine_discrete(h_sa.copy(), time_limit=1.0)
        
        if val_ref < best_val:
            best_val = val_ref
            best_h = h_ref.copy()
            
    if best_h is not None and time.time() < deadline - 2.0:
        h_polish, val_polish = _refine_discrete(best_h.copy(), time_limit=5.0)
        if val_polish < best_val:
            best_val = val_polish
            best_h = h_polish
            
    if best_h is None:
        return np.ones(N)
        
    return best_h
