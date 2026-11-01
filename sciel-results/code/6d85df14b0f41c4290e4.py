import numpy as np
import time
from scipy.optimize import minimize

N = 400

def _max_sidelobe(h):
    """
    Compute the maximum absolute value of the aperiodic autocorrelation sidelobes.
    h: numpy array of length N containing 1s and -1s.
    """
    n_fft = 2 * N
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    ac = np.fft.irfft(power, n=n_fft)
    
    pos_lags = ac[1:N]
    neg_lags = ac[N+1:2*N]
    
    return np.max(np.concatenate([np.abs(pos_lags), np.abs(neg_lags)]))

def _compute_ac(h):
    """
    Compute the full aperiodic autocorrelation sidelobes.
    Returns array of length 2*(N-1) containing ac[1..N-1] and ac[-(N-1)..-1]
    """
    n_fft = 2 * N
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    ac = np.fft.irfft(power, n=n_fft)
    
    pos_lags = ac[1:N]
    neg_lags = ac[N+1:2*N]
    
    return np.concatenate([pos_lags, neg_lags])

def _smooth_max(x, gamma):
    """
    Smooth maximum approximation: (1/gamma) * log(sum(exp(gamma * x)))
    """
    # Numerical stability: subtract max
    x_max = np.max(x)
    return x_max + np.log(np.sum(np.exp(gamma * (x - x_max)))) / gamma

def _qubo_objective(h, gamma=10.0):
    """
    Objective function for continuous optimization.
    h is a continuous vector; we use soft constraint to push toward ±1.
    Returns (objective_value, gradient)
    """
    # Soft max of squared sidelobes
    ac = _compute_ac(h)
    ac_sq = ac ** 2
    obj = _smooth_max(ac_sq, gamma)
    
    # Add penalty for deviation from ±1
    # Use (h^2 - 1)^2 as penalty
    penalty = np.sum((h**2 - 1.0)**2)
    
    total = obj + 0.1 * penalty
    
    # Gradient computation
    # Gradient of smooth max: weighted average
    weights = np.exp(gamma * (ac_sq - np.max(ac_sq)))
    weights = weights / np.sum(weights)
    d_obj_d_ac_sq = weights * gamma * np.exp(gamma * (ac_sq - np.max(ac_sq))) / np.sum(np.exp(gamma * (ac_sq - np.max(ac_sq))))
    # Actually simpler: d/dx log(sum(exp(gamma*x))) = gamma * softmax(gamma*x)
    softmax_vals = np.exp(gamma * (ac_sq - np.max(ac_sq)))
    softmax_vals = softmax_vals / np.sum(softmax_vals)
    d_obj_d_ac_sq = gamma * softmax_vals
    
    # d ac_sq / d h: need Jacobian of ac w.r.t. h
    # ac[k] = sum_{i} h[i] * h[i+k] (for positive lags)
    # This is complex. Let's use numerical gradient for the penalty part
    # and approximate the main part.
    
    # For efficiency, let's just use numerical gradient via finite differences
    # if analytical is too complex. But that's slow.
    
    # Alternative: use scipy's minimize with BFGS and finite differences
    # This might be slow but let's try a hybrid approach.
    
    return total

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_h = None
    best_val = float('inf')
    
    # Phase 1: Random restarts with local search using delta updates
    # Phase 2: Smooth optimization from best local solution
    
    # First, do some random restarts to get a good starting point
    max_restarts = 50
    
    for restart in range(max_restarts):
        if time.time() > deadline - 10.0:
            break
        
        h = np.random.choice([-1.0, 1.0], size=N)
        current_val = _max_sidelobe(h)
        
        # Local search: flip one bit at a time
        # Precompute autocorrelation for efficient updates
        ac_full = _compute_ac(h)
        current_max = np.max(np.abs(ac_full))
        
        no_improve = 0
        max_no_improve = 500
        
        while no_improve < max_no_improve:
            if time.time() > deadline - 10.0:
                break
            
            # Pick random index to flip
            i = np.random.randint(N)
            
            # Compute delta in autocorrelation when flipping h[i]
            # h[i] changes from h[i] to -h[i], so delta = -2*h[i]
            delta = -2.0 * h[i]
            
            # Update autocorrelation
            # For lag k: ac[k] = sum_j h[j]*h[j+k]
            # Flipping h[i] affects:
            # ac[k] where i is in [0, N-1-k]: term h[i]*h[i+k] changes by delta*h[i+k]
            # ac[k] where i+k is in [0, N-1]: term h[i-k]*h[i] changes by h[i-k]*delta
            # For negative lags: ac[-k] = ac[k] by symmetry
            
            new_ac = ac_full.copy()
            
            # For positive lags k = 1 to N-1:
            # h[i] appears in ac[k] as h[i]*h[i+k] if i+k < N, and as h[i-k]*h[i] if i-k >= 0
            # So ac[k] changes by: delta*h[i+k] (if i+k < N) + h[i-k]*delta (if i-k >= 0)
            
            # Vectorized update
            # For each lag k (1..N-1):
            # contribution from h[i]*h[i+k]: if i+k < N, ac[k] += delta * h[i+k]
            # contribution from h[i-k]*h[i]: if i-k >= 0, ac[k] += h[i-k] * delta
            
            # Let's compute this more carefully
            # ac[k] = sum_{j=0}^{N-1-k} h[j]*h[j+k]
            # Flipping h[i]: 
            #   If i < N-k, then term j=i is present: change = delta * h[i+k]
            #   If i-k >= 0, then term j=i-k is present: change = h[i-k] * delta
            
            # So for lag k:
            # delta_ac[k] = delta * h[i+k] (if i < N-k) + delta * h[i-k] (if i-k >= 0)
            
            # Let's compute for all k at once
            # k ranges from 1 to N-1
            # For k where i < N-k i.e. k < N-i: contribution delta*h[i+k]
            # For k where i-k >= 0 i.e. k <= i: contribution delta*h[i-k]
            
            # This is getting complex. Let's just recompute if it's not too slow.
            # With N=400, FFT is O(N log N) ~ 400*9 = 3600 ops, which is fast.
            
            h[i] = -h[i]
            new_ac = _compute_ac(h)
            new_max = np.max(np.abs(new_ac))
            
            if new_max < current_max:
                ac_full = new_ac
                current_max = new_max
                no_improve = 0
                if current_max < best_val:
                    best_val = current_max
                    best_h = h.copy()
            else:
                h[i] = -h[i]
                no_improve += 1
        
        # After local search, try smooth optimization from this point
        if time.time() > deadline - 5.0:
            break
        
        # Use scipy minimize with Nelder-Mead on a smoothed objective
        # But we need to handle the discrete constraint. Let's use a penalty approach.
        
        # Define smoothed objective
        gamma = 5.0
        
        def smooth_obj(x):
            # x is continuous
            ac = _compute_ac(x)
            ac_sq = ac ** 2
            obj = _smooth_max(ac_sq, gamma)
            penalty = np.sum((x**2 - 1.0)**2)
            return obj + 0.05 * penalty
        
        # Start from current best
        x0 = h.copy()
        
        # Use Nelder-Mead for small number of iterations
        try:
            result = minimize(smooth_obj, x0, method='Nelder-Mead', 
                             options={'maxiter': 100, 'xatol': 0.01, 'fatol': 0.01})
            
            if result.fun < smooth_obj(x0):
                # Round to nearest ±1
                h_rounded = np.sign(result.x)
                h_rounded[h_rounded == 0] = 1.0
                val_rounded = _max_sidelobe(h_rounded)
                
                if val_rounded < best_val:
                    best_val = val_rounded
                    best_h = h_rounded.copy()
        except Exception:
            pass
    
    if best_h is None:
        return np.ones(N)
    
    return best_h
