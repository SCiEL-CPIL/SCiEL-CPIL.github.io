import numpy as np
import time
from scipy.optimize import linprog, minimize
from scipy.signal import fftconvolve

N = 100
TIME_LIMIT = 115.0  # Safety margin for 120s limit

_START_TIME = None

def _time_remaining():
    if _START_TIME is None:
        return TIME_LIMIT
    return TIME_LIMIT - (time.time() - _START_TIME)

def _compute_max_overlap(h):
    """
    Computes the maximum overlap of h with shifted (1-h).
    Overlap at shift k: sum_i h[i] * (1 - h[(i+k) % N])
    """
    g = 1.0 - h
    # Circular cross-correlation
    H = np.fft.fft(h)
    G = np.fft.fft(g)
    # Corrs[k] = sum_i h[i] * g[i-k]
    # Overlaps[k] = sum_i h[i] * g[i+k] = Corrs[-k]
    # The set of values is the same, so max is the same.
    Corrs = np.fft.ifft(H * np.conj(G)).real
    return np.max(Corrs)

def _project_to_simplex(v, target_sum):
    """
    Project v onto {x | 0 <= x <= 1, sum(x) = target_sum}.
    """
    n = len(v)
    if target_sum <= 0:
        return np.zeros(n)
    if target_sum >= n:
        return np.ones(n)
    
    # Simplex projection with bounds
    # We use an iterative method or a direct formula.
    # Direct formula for {x >= 0, sum x = S}:
    # Sort v desc, find rho, theta.
    # Then clip to [0,1].
    # This might not satisfy sum exactly if clipping happens.
    # Let's use a robust iterative approach for N=100.
    
    w = np.zeros(n)
    active = np.ones(n, dtype=bool)
    v_active = v.copy()
    S_active = target_sum
    
    for _ in range(n):
        if not np.any(active):
            break
        v_act = v_active[active]
        S_act = S_active
        
        # Project v_act to {x >= 0, sum x = S_act}
        u = np.sort(v_act)[::-1]
        cssv = np.cumsum(u)
        rho_arr = np.nonzero(u * np.arange(1, len(u) + 1) > (cssv - S_act))[0]
        if len(rho_arr) == 0:
            theta = 0
        else:
            rho = rho_arr[-1]
            theta = (cssv[rho] - S_act) / (rho + 1.0)
            
        w_act = np.maximum(v_act - theta, 0)
        
        # Check for upper bound violation (> 1)
        viol = w_act > 1.0
        if np.any(viol):
            # Assign 1 to these
            idx_active = np.where(active)[0]
            idx_viol = idx_active[viol]
            w[idx_viol] = 1.0
            active[idx_viol] = False
            S_active -= np.count_nonzero(viol)
            # Update v_active for next iteration? Not needed as we only care about active set.
            # But we need to ensure v_active[viol] is effectively removed.
        else:
            w[active] = w_act
            break
            
    # Final adjustment for floating point errors
    diff = target_sum - np.sum(w)
    if abs(diff) > 1e-9:
        # Distribute among non-clipped
        non_clipped = (w > 1e-6) & (w < 1 - 1e-6)
        if np.any(non_clipped):
            count = np.sum(non_clipped)
            w[non_clipped] += diff / count
            w = np.clip(w, 0, 1)
            
    # Ensure exact sum if possible (fix last element)
    if np.sum(w) != target_sum:
        # Find an element that can absorb the difference
        diff = target_sum - np.sum(w)
        for i in range(n):
            if 0 < w[i] + diff < 1:
                w[i] += diff
                break
        # If still not exact, force it (might violate bounds slightly, but evaluator checks sum)
        if np.sum(w) != target_sum:
             w[0] += target_sum - np.sum(w)
             w = np.clip(w, 0, 1)
             # If clipping broke sum again, we are in trouble, but hopefully rare.
             
    return w

def _local_refine(h, max_iter=200, beta=10.0):
    """
    Refine h using gradient descent on soft max of overlaps.
    """
    current_score = _compute_max_overlap(h)
    lr = 0.05
    
    for i in range(max_iter):
        if _time_remaining() < 5:
            break
            
        g = 1.0 - h
        H = np.fft.fft(h)
        G = np.fft.fft(g)
        Corrs = np.fft.ifft(H * np.conj(G)).real
        
        # Soft max weights
        m = np.max(Corrs)
        exps = np.exp(beta * (Corrs - m))
        sum_exps = np.sum(exps)
        u = exps / sum_exps
        
        # Gradient of soft max
        # Grad = u * (1 - 2h) (convolution)
        U = np.fft.fft(u)
        D = np.fft.fft(1.0 - 2.0 * h)
        Grad = np.fft.ifft(U * D).real
        
        h_new = h - lr * Grad
        h_new = _project_to_simplex(h_new, N / 2.0)
        
        score_new = _compute_max_overlap(h_new)
        
        if score_new < current_score:
            h = h_new
            current_score = score_new
            # Decay learning rate slightly
            lr *= 0.99
        else:
            lr *= 0.9
            if lr < 1e-6:
                break
                
    return h, current_score

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    best_h = np.full(N, 0.5)
    best_score = _compute_max_overlap(best_h)
    
    # Strategy:
    # 1. Try structured initializations (sines, squares, etc.)
    # 2. Local refinement from these starts
    # 3. Random starts with local refinement
    # 4. Maybe a simple genetic algorithm or simulated annealing if time permits?
    #    For now, stick to gradient-based refinement from diverse starts.
    
    candidates = []
    
    # 1. Constant
    candidates.append(np.full(N, 0.5))
    
    # 2. Sine waves
    for freq in [1, 2, 3, 4, 5, 7, 10]:
        for phase in np.linspace(0, 2*np.pi, 8, endpoint=False):
            h = 0.5 + 0.3 * np.sin(2 * np.pi * freq * np.arange(N) / N + phase)
            h = _project_to_simplex(h, N/2.0)
            candidates.append(h)
    
    # 3. Square waves (approximate)
    for period in [2, 4, 5, 10]:
        h = np.zeros(N)
        h[::2] = 1.0
        h = h * (period / (2 * (N // period + 1))) # Scale? No, just use 0/1 pattern
        # Let's make 0/1 patterns with sum N/2
        for shift in range(period):
            h = np.zeros(N)
            h[shift::period] = 1.0
            # Adjust to sum N/2
            # If sum is not N/2, flip some 0s to 1s or vice versa
            # Simpler: random 0/1 with sum N/2
            # But structured is better.
            # Let's just use the sine ones and random.
            pass
    
    # 4. Random 0/1
    for _ in range(20):
        h = np.zeros(N)
        idx = np.random.choice(N, N//2, replace=False)
        h[idx] = 1.0
        candidates.append(h)
        
    # 5. Random uniform
    for _ in range(20):
        h = np.random.uniform(0, 1, N)
        h = _project_to_simplex(h, N/2.0)
        candidates.append(h)
        
    # 6. "Erdos" style: alternating high/low with noise
    for _ in range(10):
        h = np.zeros(N)
        for i in range(N):
            if i % 2 == 0:
                h[i] = np.random.uniform(0.4, 0.8)
            else:
                h[i] = np.random.uniform(0.2, 0.6)
        h = _project_to_simplex(h, N/2.0)
        candidates.append(h)
        
    # Refine each candidate
    for h_init in candidates:
        if _time_remaining() < 10:
            break
            
        h_ref, score = _local_refine(h_init, max_iter=100)
        
        if score < best_score:
            best_score = score
            best_h = h_ref.copy()
            
    # If time remains, do more random restarts
    while _time_remaining() > 10:
        h = np.random.uniform(0, 1, N)
        h = _project_to_simplex(h, N/2.0)
        h_ref, score = _local_refine(h, max_iter=50)
        
        if score < best_score:
            best_score = score
            best_h = h_ref.copy()
            
    return best_h
