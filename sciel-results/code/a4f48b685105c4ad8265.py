import numpy as np
import time
from scipy.optimize import minimize

N = 400
DEADLINE = 115.0  # Safety margin

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
    if len(ac) < 2:
        return 0.0
    return np.max(np.abs(ac[1:N]))

def _objective_soft_max(x, p=4):
    """
    Objective: Sum of p-th power of sidelobes.
    p=4 emphasizes the largest sidelobes more than p=2.
    """
    h = x
    ac = _compute_autocorr(h)
    sidelobes = ac[1:N]
    return np.sum(sidelobes**p)

def _objective_and_gradient(x, p=4):
    """
    Objective and gradient for L-BFGS-B.
    """
    h = x
    ac = _compute_autocorr(h)
    sidelobes = ac[1:N]
    obj = np.sum(sidelobes**p)
    
    # Gradient calculation for sum(sidelobes^p)
    # d/dx_i sum_k C[k]^p = p * sum_k C[k]^(p-1) * dC[k]/dx_i
    # dC[k]/dx_i = h[i+k] + h[i-k]
    
    grad = np.zeros(N)
    # We can vectorize this partially, but a loop over k is safer for correctness in gradient
    # For N=400, loop is 400 iterations. Each iteration is vectorized over N.
    # 400 * 400 = 160,000 ops. Very fast.
    
    if p == 2:
        weights = sidelobes
    else:
        weights = p * (sidelobes ** (p-1))
        
    for k in range(1, N):
        w = weights[k-1]
        if w == 0:
            continue
        # dC[k]/dx_i for i in 0..N-k-1 is h[i+k]
        # dC[k]/dx_i for i in k..N-1 is h[i-k]
        grad[0:N-k] += w * h[k:N]
        grad[k:N] += w * h[0:N-k]
        
    return obj, grad

def _balanced_random_init():
    """
    Generate a random balanced sequence of +1 and -1.
    """
    h = np.ones(N)
    # Flip N//2 random indices to -1
    indices = np.random.choice(N, N//2, replace=False)
    h[indices] = -1.0
    return h

def _local_search_pair_swap(h, time_limit=5.0, T0=None, cooling_rate=0.999):
    """
    Simulated Annealing with Pair Swaps (flipping two bits).
    This maintains the balance of the sequence (sum=0).
    """
    start_time = time.time()
    h = h.copy()
    
    # Ensure h is balanced
    # If not, we could balance it, but we assume input is balanced
    
    ac = _compute_autocorr(h)
    current_val = _max_sidelobe(ac)
    current_obj = _objective_soft_max(h, p=4)
    
    if T0 is None:
        T0 = max(current_obj * 0.1, 100.0) # Scale T to objective magnitude
    
    best_h = h.copy()
    best_val = current_val
    
    for it in range(1000000):
        if time.time() - start_time > time_limit:
            break
        
        # Pick two distinct indices to flip
        i, j = np.random.choice(N, 2, replace=False)
        
        h[i] = -h[i]
        h[j] = -h[j]
        
        ac_new = _compute_autocorr(h)
        new_val = _max_sidelobe(ac_new)
        new_obj = _objective_soft_max(h, p=4)
        
        delta_obj = new_obj - current_obj
        T = T0 * (cooling_rate ** it)
        
        # Accept if better objective, or with probability exp(-delta/T)
        if delta_obj < 0 or (T > 1e-15 and np.random.random() < np.exp(-delta_obj / T)):
            current_val = new_val
            current_obj = new_obj
            ac = ac_new
            
            if new_val < best_val:
                best_val = new_val
                best_h = h.copy()
        else:
            # Revert
            h[i] = -h[i]
            h[j] = -h[j]
            
    return best_h, best_val

def _greedy_polish_pair_swap(h, time_limit=2.0):
    """
    Greedy search: Try all pair swaps, keep the best one.
    Repeat until no improvement or time limit.
    """
    start_time = time.time()
    h = h.copy()
    ac = _compute_autocorr(h)
    current_val = _max_sidelobe(ac)
    current_obj = _objective_soft_max(h, p=4)
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        best_delta = 0
        best_i, best_j = -1, -1
        
        # To speed up, we don't check ALL pairs (N^2/2 = 80,000 pairs).
        # We check a random subset of pairs, or just all pairs if N is small.
        # For N=400, 80k pairs * FFT cost is too slow for a full sweep in 2s.
        # We will do a "Randomized Greedy": check 1000 random pairs per iteration.
        
        num_trials = 2000
        for _ in range(num_trials):
            i, j = np.random.choice(N, 2, replace=False)
            
            h[i] = -h[i]
            h[j] = -h[j]
            
            ac_new = _compute_autocorr(h)
            new_val = _max_sidelobe(ac_new)
            new_obj = _objective_soft_max(h, p=4)
            
            delta_obj = new_obj - current_obj
            
            if delta_obj < best_delta:
                best_delta = delta_obj
                best_i, best_j = i, j
                best_ac = ac_new
                best_val = new_val
                
            h[i] = -h[i]
            h[j] = -h[j]
            
        if best_delta < 0 and best_i != -1:
            # Apply the best move
            h[best_i] = -h[best_i]
            h[best_j] = -h[best_j]
            current_val = best_val
            current_obj += best_delta # Approximation, or recompute
            ac = best_ac
            improved = True
            
    return h, current_val

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_h = None
    best_val = float('inf')
    
    num_restarts = 8
    
    for r in range(num_restarts):
        if time.time() >= deadline - 15.0:
            break
        
        # Phase 1: Initialization
        # Mix of random balanced and continuous relaxation starts
        
        if r % 2 == 0:
            # Continuous relaxation
            x0 = _balanced_random_init()
            try:
                res = minimize(_objective_and_gradient, x0, method='L-BFGS-B', 
                               bounds=[(-1.0, 1.0)] * N, 
                               options={'maxiter': 30, 'ftol': 1e-5, 'gtol': 1e-5, 'maxls': 20})
                x_cont = res.x
                # Round to balanced binary
                # Sort by magnitude? No, just sign.
                h_bin = np.sign(x_cont)
                zero_mask = (h_bin == 0)
                if np.any(zero_mask):
                    h_bin[zero_mask] = np.random.choice([-1.0, 1.0], size=np.sum(zero_mask))
                
                # Balance the sequence
                num_pos = np.sum(h_bin == 1.0)
                num_neg = np.sum(h_bin == -1.0)
                if num_pos != num_neg:
                    # Flip some to balance
                    if num_pos > num_neg:
                        # Flip (num_pos - num_neg)//2 positive to negative
                        pos_indices = np.where(h_bin == 1.0)[0]
                        to_flip = np.random.choice(pos_indices, (num_pos - num_neg)//2, replace=False)
                        h_bin[to_flip] = -1.0
                    else:
                        neg_indices = np.where(h_bin == -1.0)[0]
                        to_flip = np.random.choice(neg_indices, (num_neg - num_pos)//2, replace=False)
                        h_bin[to_flip] = 1.0
                        
            except Exception:
                h_bin = _balanced_random_init()
        else:
            # Random balanced
            h_bin = _balanced_random_init()
            
        # Evaluate
        ac = _compute_autocorr(h_bin)
        val = _max_sidelobe(ac)
        
        if val < best_val:
            best_val = val
            best_h = h_bin.copy()
            
        # Phase 2: Local Search (SA with Pair Swaps)
        if best_h is not None and time.time() < deadline - 10.0:
            # Allocate time
            time_for_sa = min(4.0, max(1.0, (deadline - 10.0 - time.time()) / max(1, num_restarts - r)))
            h_sa, val_sa = _local_search_pair_swap(best_h.copy(), time_limit=time_for_sa)
            
            if val_sa < best_val:
                best_val = val_sa
                best_h = h_sa
    
    # Phase 3: Final Polish
    if best_h is not None and time.time() < deadline - 2.0:
        remaining_time = deadline - 2.0 - time.time()
        if remaining_time > 1.0:
            h_polish, val_polish = _greedy_polish_pair_swap(best_h.copy(), time_limit=remaining_time)
            if val_polish < best_val:
                best_val = val_polish
                best_h = h_polish
                
    if best_h is None:
        return _balanced_random_init()
        
    return best_h
