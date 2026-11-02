import numpy as np
import time

N = 100
DEADLINE = 110.0  # safety margin before 120s kill

def _max_overlap_fft(h):
    """
    Compute max overlap using FFT-based circular autocorrelation.
    O_k = sum_i h_i * (1 - h_{(i+k)%N}) = N/2 - C_k
    where C_k = sum_i h_i * h_{(i+k)%N} is circular autocorrelation.
    """
    H = np.fft.fft(h)
    autocorr = np.fft.ifft(H * np.conj(H)).real
    # autocorr[k] = sum_i h[i] * h[(i-k) % N]
    # By symmetry of circular autocorrelation for real signals, C_k = C_{N-k}
    # So autocorr[k] gives us C_k directly (up to indexing convention)
    # O_k = N/2 - C_k for all k
    overlaps = N / 2.0 - autocorr
    return float(np.max(overlaps))

def _grad_max_overlap_soft(h, T=0.1):
    """
    Compute smoothed gradient of max overlap using soft-max weighting.
    dO_k/dh_j = 1 - h_{(j-k)%N} - h_{(j+k)%N}
    """
    H = np.fft.fft(h)
    autocorr = np.fft.ifft(H * np.conj(H)).real
    overlaps = N / 2.0 - autocorr
    
    # Soft-max weights over shifts
    weights = np.exp(overlaps / T)
    weights /= np.sum(weights)
    
    grad = np.zeros(N)
    for k in range(N):
        if weights[k] < 1e-7:
            continue
        # dO_k/dh_j = 1 - h_{(j-k)%N} - h_{(j+k)%N}
        # np.roll(h, k)[j] = h[(j-k)%N]
        # np.roll(h, -k)[j] = h[(j+k)%N]
        grad += weights[k] * (1.0 - np.roll(h, k) - np.roll(h, -k))
    return grad

def _project(h):
    """Project h onto the feasible set: sum = N/2, each in [0,1]."""
    h = h.copy()
    for _ in range(20):
        diff = N / 2.0 - np.sum(h)
        if abs(diff) < 1e-12:
            break
        mask = (h > 1e-10) & (h < 1.0 - 1e-10)
        if np.sum(mask) == 0:
            # All at bounds; try adjusting any non-extreme
            mask = (h > 0) & (h < 1)
            if np.sum(mask) == 0:
                break
        h[mask] += diff / np.sum(mask)
        h = np.clip(h, 0.0, 1.0)
    return h

def _random_feasible(rng):
    """Generate a random feasible starting point (non-uniform)."""
    h = rng.random(N)
    h = np.sort(h)
    # Shift to have mean 0.5
    h = h - np.mean(h) + 0.5
    h = np.clip(h, 0.0, 1.0)
    h = _project(h)
    return h

def construct():
    t0 = time.time()
    rng = np.random.default_rng(12345)
    
    # Start from a non-uniform random feasible point
    h = _random_feasible(rng)
    cur_val = _max_overlap_fft(h)
    best_h = h.copy()
    best_val = cur_val
    
    # Gradient descent with adaptive learning rate
    lr = 0.05
    stagnation = 0
    
    while time.time() - t0 < DEADLINE:
        grad = _grad_max_overlap_soft(h)
        
        # Project gradient to sum-zero subspace
        d = grad - np.mean(grad)
        
        # Take step
        h_new = h - lr * d
        h_new = _project(h_new)
        
        new_val = _max_overlap_fft(h_new)
        
        if new_val < best_val:
            best_val = new_val
            best_h = h_new.copy()
            stagnation = 0
        else:
            stagnation += 1
        
        # Adaptive learning rate
        if new_val < cur_val - 1e-8:
            lr = min(lr * 1.05, 0.3)
        else:
            lr *= 0.92
            h_new = h  # revert
        
        if stagnation > 200:
            # Perturb to escape local structure
            h_new = h + 0.02 * (rng.random(N) - 0.5)
            h_new = _project(h_new)
            new_val = _max_overlap_fft(h_new)
            if new_val < best_val:
                best_val = new_val
                best_h = h_new.copy()
            h = h_new
            cur_val = new_val
            stagnation = 0
            lr = 0.05
            continue
        
        cur_val = new_val if new_val < cur_val else cur_val
        h = h_new
        
        # Early termination if we've essentially reached the analytical optimum
        # The theoretical minimum is N/4 = 25.0
        if best_val <= N / 4.0 + 1e-6:
            break
    
    # The analytical optimum is N/4, achieved by h=0.5
    # If optimization converged near there, return the optimized result
    # If not (due to time), return best found
    # But also check: if best_val is very close to N/4, the uniform vector
    # is provably optimal, so return it for exactness
    if best_val <= N / 4.0 + 1e-4:
        # We're at the theoretical minimum; return exact uniform for cleanliness
        return np.full(N, 0.5)
    
    return best_h
