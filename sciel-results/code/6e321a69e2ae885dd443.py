"""
Combined program: Fast discrete local search with incremental autocorrelation updates.
Uses Simulated Annealing to escape local minima.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 400

# EVOLVE-BLOCK-START

def _compute_autocorr(h):
    """
    Compute linear aperiodic autocorrelation using FFT.
    Returns array of length 2*N (padded).
    ac[k] for k=0..N-1 is the standard autocorrelation.
    """
    n_fft = 2 * N
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    ac = np.fft.irfft(power, n=n_fft)
    return ac

def _max_sidelobe(ac):
    """
    Compute max absolute value of sidelobes (indices 1 to N-1).
    """
    return np.max(np.abs(ac[1:N]))

def _objective_and_gradient(x, beta=50.0):
    """
    Objective: Smooth approximation of max sidelobe for continuous relaxation.
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
    weights = beta * exp_shifted / sum_exp
    
    grad = np.zeros(N)
    
    # Only process weights that are significant
    for k in range(1, N):
        w_k = weights[k-1]
        if w_k < 1e-12:
            continue
        
        sign_k = np.sign(ac[k])
        c_k = sign_k * w_k
        
        if c_k == 0:
            continue
            
        # Vectorized update
        grad[0:N-k] += c_k * h[k:N]
        grad[k:N] += c_k * h[0:N-k]
        
    return obj, grad

def _fast_local_search(h, start_time, deadline, use_sa=True, T0=0.05):
    """
    Fast local search using incremental autocorrelation updates.
    Uses Simulated Annealing if use_sa is True.
    """
    h = h.copy()
    ac = _compute_autocorr(h)
    current_val = _max_sidelobe(ac)
    best_val = current_val
    best_h = h.copy()
    
    T = T0 if use_sa else 0.0
    cooling_rate = 0.9995
    
    # Precompute indices for faster access? 
    # The loop below is O(N) per flip. N=400 is small enough.
    
    while time.time() < deadline - 5.0:  # Stop 5s before hard deadline to allow return
        # Try flipping a random bit
        i = np.random.randint(N)
        old_h_i = h[i]
        
        # Calculate delta for all k
        # Delta R[k] = -2 * old_h_i * (h[i+k] + h[i-k])
        # We update ac in place? No, we need to compare first.
        # Let's compute the new max efficiently.
        
        # To avoid full O(N) check for max every time, we can just update ac 
        # and check the max. Since N=400, np.max is fast.
        
        h[i] = -old_h_i
        
        # Incremental update of ac
        # ac[k] changes by -2 * old_h_i * (h[i+k] + h[i-k])
        # Note: h[i] is already flipped, so h[i+k] and h[i-k] are current values.
        # The term involving h[i] itself in the sum is handled by the factorization.
        
        # Let's derive carefully.
        # R[k] = sum_j h[j] h[j+k]
        # Terms involving index i:
        # 1. j=i: h[i] h[i+k] -> (-h[i]) h[i+k] = -h[i] h[i+k]. Change: -2 h[i] h[i+k]
        # 2. j=i-k: h[i-k] h[i] -> h[i-k] (-h[i]) = -h[i-k] h[i]. Change: -2 h[i-k] h[i]
        # So Delta R[k] = -2 * old_h_i * (h[i+k] + h[i-k]) (with bounds)
        
        # We update ac[1:N-1] only, as ac[0] is constant N.
        
        # Vectorized update is tricky due to bounds.
        # Let's do a loop for k=1 to N-1.
        # This is O(N) per flip.
        
        # Optimization: Only update k where h[i+k] or h[i-k] exists.
        
        # For k in 1..N-1:
        # term1 = h[i+k] if i+k < N else 0
        # term2 = h[i-k] if i-k >= 0 else 0
        # delta = -2 * old_h_i * (term1 + term2)
        
        # We can compute this vectorized:
        # k = np.arange(1, N)
        # term1_idx = i + k
        # valid1 = term1_idx < N
        # term1 = np.where(valid1, h[term1_idx[valid1]], 0) ... padding needed.
        
        # Simpler:
        k = np.arange(1, N)
        
        # Term 1: h[i+k]
        idx1 = i + k
        valid1 = idx1 < N
        term1 = np.zeros(N-1)
        term1[valid1] = h[idx1[valid1]]
        
        # Term 2: h[i-k]
        idx2 = i - k
        valid2 = idx2 >= 0
        term2 = np.zeros(N-1)
        term2[valid2] = h[idx2[valid2]]
        
        deltas = -2.0 * old_h_i * (term1 + term2)
        
        # Update ac[1:N]
        ac[1:N] += deltas
        
        new_val = _max_sidelobe(ac)
        
        if new_val < current_val:
            # Accept
            current_val = new_val
            if new_val < best_val:
                best_val = new_val
                best_h = h.copy()
        else:
            # Check SA acceptance
            accepted = False
            if T > 1e-9:
                delta_val = new_val - current_val
                if delta_val < 0:
                    accepted = True
                else:
                    # Accept with probability exp(-delta / T)
                    # Clip delta to avoid overflow in exp
                    prob = np.exp(-min(delta_val / T, 50.0))
                    if np.random.rand() < prob:
                        accepted = True
            
            if accepted:
                current_val = new_val
                # ac is already updated
            else:
                # Revert
                h[i] = old_h_i
                ac[1:N] -= deltas
        
        if T > 1e-9:
            T *= cooling_rate

    return best_h, best_val

def _pair_flip_search(h, start_time, deadline):
    """
    Local search trying to flip pairs of bits.
    """
    h = h.copy()
    ac = _compute_autocorr(h)
    current_val = _max_sidelobe(ac)
    best_val = current_val
    best_h = h.copy()
    
    while time.time() < deadline - 5.0:
        # Pick two random distinct indices
        i, j = np.random.choice(N, 2, replace=False)
        
        old_hi = h[i]
        old_hj = h[j]
        
        h[i] = -old_hi
        h[j] = -old_hj
        
        # Update ac
        # Delta from i: -2*old_hi*(h[i+k] + h[i-k])
        # Delta from j: -2*old_hj*(h[j+k] + h[j-k])
        # Note: h is already updated.
        
        k = np.arange(1, N)
        
        # From i
        idx1_i = i + k
        valid1_i = idx1_i < N
        term1_i = np.zeros(N-1)
        term1_i[valid1_i] = h[idx1_i[valid1_i]]
        
        idx2_i = i - k
        valid2_i = idx2_i >= 0
        term2_i = np.zeros(N-1)
        term2_i[valid2_i] = h[idx2_i[valid2_i]]
        
        delta_i = -2.0 * old_hi * (term1_i + term2_i)
        
        # From j
        idx1_j = j + k
        valid1_j = idx1_j < N
        term1_j = np.zeros(N-1)
        term1_j[valid1_j] = h[idx1_j[valid1_j]]
        
        idx2_j = j - k
        valid2_j = idx2_j >= 0
        term2_j = np.zeros(N-1)
        term2_j[valid2_j] = h[idx2_j[valid2_j]]
        
        delta_j = -2.0 * old_hj * (term1_j + term2_j)
        
        total_delta = delta_i + delta_j
        ac[1:N] += total_delta
        
        new_val = _max_sidelobe(ac)
        
        if new_val < current_val:
            current_val = new_val
            if new_val < best_val:
                best_val = new_val
                best_h = h.copy()
        else:
            # Revert
            h[i] = old_hi
            h[j] = old_hj
            ac[1:N] -= total_delta

    return best_h, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_h = None
    best_val = float('inf')
    
    # Phase 1: Generate initial candidates
    # 1. Random
    np.random.seed(42)
    candidates = []
    
    # A few random starts
    for _ in range(3):
        x = np.random.choice([-1.0, 1.0], size=N)
        candidates.append(x)
        
    # 2. L-BFGS start (if time permits, usually fast enough for 1-2 runs)
    # We'll use one L-BFGS run to try to find a better basin.
    try:
        x0 = np.random.choice([-1.0, 1.0], size=N)
        beta = 50.0
        
        def obj_grad(x):
            return _objective_and_gradient(x, beta)
            
        res = minimize(obj_grad, x0, method='L-BFGS-B', 
                       bounds=[(-1.0, 1.0)] * N, 
                       options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6, 'maxls': 20})
        
        if res.success or (hasattr(res, 'fun') and res.fun < float('inf')):
            x_cont = res.x
            x_bin = np.sign(x_cont)
            x_bin[x_bin == 0] = 1.0
            candidates.append(x_bin)
    except Exception:
        pass
    
    # Evaluate candidates to pick the best start for local search
    for x in candidates:
        ac = _compute_autocorr(x)
        val = _max_sidelobe(ac)
        if val < best_val:
            best_val = val
            best_h = x.copy()
            
    if best_h is None:
        # Fallback
        best_h = np.ones(N)
        best_val = _max_sidelobe(_compute_autocorr(best_h))
        
    # Phase 2: Fast Local Search with SA
    # Use most of the time here
    local_deadline = deadline - 20.0  # Reserve 20s for pair search and final checks
    best_h, best_val = _fast_local_search(best_h, start_time, local_deadline, use_sa=True, T0=0.05)
    
    # Phase 3: Pair Flip Search
    if time.time() < deadline - 5.0:
        best_h, best_val = _pair_flip_search(best_h, start_time, deadline)
        
    return best_h

# EVOLVE-BLOCK-END
