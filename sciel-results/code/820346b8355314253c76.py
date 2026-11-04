"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def _max_overlap(x):
    """
    Compute max over all circular shifts k of sum_i x[i]*(1-x[(i+k)%N]).
    Uses FFT-based autocorrelation for O(N log N) evaluation.
    """
    X = np.fft.fft(x)
    power = X * np.conj(X)
    autocorr = np.fft.ifft(power).real
    min_autocorr = np.min(autocorr)
    S = np.sum(x)
    max_overlap = S - min_autocorr
    return max_overlap

def _project(x):
    """Project x onto the feasible set: x in [0,1]^N, sum(x) = N/2."""
    x = np.clip(x, 0.0, 1.0)
    s = x.sum()
    target = N / 2.0
    if abs(s - target) < 1e-14:
        return x
    # Water-filling style projection
    for _ in range(200):
        s = x.sum()
        need = target - s
        if abs(need) < 1e-14:
            break
        if need > 0:
            mask = x < 1.0
            if not mask.any():
                break
            x[mask] += need / mask.sum()
        else:
            mask = x > 0.0
            if not mask.any():
                break
            x[mask] += need / mask.sum()
        x = np.clip(x, 0.0, 1.0)
    return x

def _autocorr(x):
    """Compute circular autocorrelation of x."""
    X = np.fft.fft(x)
    power = X * np.conj(X)
    return np.fft.ifft(power).real

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin
    
    # Theoretical lower bound: max_overlap >= N/4 = 25
    # Uniform achieves exactly 25.
    # We'll use a spectral approach: maximize the minimum autocorrelation coefficient.
    # This is equivalent to minimizing max overlap.
    
    best_h = np.full(N, 0.5)
    best_val = _max_overlap(best_h)
    
    # Try multiple random initializations with projected gradient ascent on min autocorr
    rng = np.random.default_rng(123)
    
    n_trials = 0
    while time.time() < deadline:
        n_trials += 1
        # Random initialization
        h = rng.uniform(0, 1, N)
        h = _project(h)
        
        # Projected gradient ascent to maximize min_k autocorr(k)
        # = minimize max_k overlap(k)
        lr = 0.01
        for it in range(200):
            if time.time() > deadline:
                break
            ac = _autocorr(h)
            min_ac = np.min(ac)
            # Gradient of min ac w.r.t. h is tricky; use subgradient
            # ac[k] = sum_i h[i]*h[(i+k)%N]
            # d(ac[k])/d(h[j]) = h[(j-k)%N] + h[(j+k)%N] for k != 0
            # For k=0: d(ac[0])/d(h[j]) = 2*h[j]
            # We want to increase the minimum, so gradient points in direction of the argmin k
            k_min = np.argmin(ac)
            if k_min == 0:
                grad = 2.0 * h
            else:
                grad = np.roll(h, -k_min) + np.roll(h, k_min)
            
            # Projected gradient ascent
            h_new = h + lr * grad
            h_new = _project(h_new)
            
            # Check if improvement
            ac_new = _autocorr(h_new)
            min_ac_new = np.min(ac_new)
            if min_ac_new >= min_ac - 1e-10:
                h = h_new
            else:
                lr *= 0.5
                if lr < 1e-8:
                    break
        
        val = _max_overlap(h)
        if val < best_val - 1e-10:
            best_val = val
            best_h = h.copy()
        
        # Early exit if we've hit the theoretical bound
        if best_val <= 25.0 + 1e-10:
            break
    
    return best_h
# EVOLVE-BLOCK-END
