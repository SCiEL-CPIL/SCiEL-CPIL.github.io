"""Spectral convex optimization for Erdos minimum overlap."""
import numpy as np
from scipy.optimize import minimize
import time

N = 100

def _compute_overlaps(h):
    """Compute overlap(k) = sum_i h[i]*(1-h[(i+k)%N]) for all k."""
    # overlap(k) = N/2 - autocorr(k) where autocorr(k) = sum_i h[i]*h[(i+k)%N]
    # Use FFT for autocorrelation
    H = np.fft.fft(h)
    autocorr = np.fft.ifft(H * np.conj(H)).real
    # autocorr[k] = sum_i h[i]*h[(i+k)%N] (circular)
    return N / 2.0 - autocorr

def _max_overlap(h):
    overlaps = _compute_overlaps(h)
    return np.max(overlaps)

def _objective(x):
    """Smooth max of overlaps using log-sum-exp."""
    h = np.clip(x, 0, 1)
    overlaps = _compute_overlaps(h)
    # Log-sum-exp for smooth max
    m = np.max(overlaps)
    return m + np.log(np.sum(np.exp(overlaps - m))) / N

def _project(x):
    """Project to sum(x) = N/2, 0 <= x <= 1."""
    x = np.clip(x, 0, 1)
    # Adjust to meet sum constraint
    current_sum = np.sum(x)
    target = N / 2.0
    diff = target - current_sum
    if abs(diff) < 1e-12:
        return x
    # Distribute adjustment
    if diff > 0:
        # Need to increase sum: add to elements below 1
        mask = x < 1 - 1e-12
        if np.sum(mask) == 0:
            return x
        per = diff / np.sum(mask)
        x[mask] = np.minimum(x[mask] + per, 1)
    else:
        # Need to decrease sum: subtract from elements above 0
        mask = x > 1e-12
        if np.sum(mask) == 0:
            return x
        per = (-diff) / np.sum(mask)
        x[mask] = np.maximum(x[mask] - per, 0)
    return x

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    # Start from uniform
    h_best = np.full(N, 0.5)
    best_score = _max_overlap(h_best)
    
    # Try multiple random initializations with projected gradient
    np.random.seed(42)
    
    for trial in range(200):
        if time.time() - start_time > deadline:
            break
        
        # Random initialization
        h = np.random.rand(N)
        h = _project(h)
        
        # Projected gradient descent
        lr = 0.1
        for step in range(500):
            if time.time() - start_time > deadline:
                break
            
            h = np.clip(h, 0, 1)
            current_score = _max_overlap(h)
            if current_score < best_score:
                best_score = current_score
                h_best = h.copy()
            
            # Subgradient of max overlap
            overlaps = _compute_overlaps(h)
            max_idx = np.argmax(overlaps)
            
            # Gradient of overlap(k) w.r.t. h[i]
            # overlap(k) = sum_i h[i]*(1-h[(i+k)%N])
            # d/dh[j] overlap(k) = (1-h[(j+k)%N]) - h[j-k mod N] (from the second term)
            # Actually: overlap(k) = sum_i h[i] - sum_i h[i]*h[(i+k)%N]
            # d/dh[j] = 1 - h[(j+k)%N] - sum_i delta_{i,j} h[(i+k)%N] - sum_i h[i] delta_{(i+k)%N,j}
            # = 1 - h[(j+k)%N] - h[(j+k)%N] - h[(j-k)%N]  ... let me redo
            # overlap(k) = N/2 - autocorr(k)
            # autocorr(k) = sum_i h[i]*h[(i+k)%N]
            # d/dh[j] autocorr(k) = h[(j+k)%N] + h[(j-k)%N]
            # So d/dh[j] overlap(k) = -(h[(j+k)%N] + h[(j-k)%N])
            grad = -(h[(max_idx + np.arange(N)) % N] + h[(max_idx - np.arange(N)) % N])
            
            # Projected gradient step
            h = h - lr * grad
            h = _project(h)
        
        if time.time() - start_time > deadline:
            break
    
    # Final projection to ensure exact constraint
    h_best = _project(h_best)
    
    # Verify sum is exactly N/2
    current_sum = np.sum(h_best)
    if abs(current_sum - N/2.0) > 1e-10:
        # Fine-tune
        diff = N/2.0 - current_sum
        mask = (h_best > 1e-12) & (h_best < 1 - 1e-12)
        if np.sum(mask) > 0:
            h_best[mask] += diff / np.sum(mask)
            h_best = np.clip(h_best, 0, 1)
    
    return h_best
