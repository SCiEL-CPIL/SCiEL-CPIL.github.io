"""Optimize nonnegative step function to minimize normalized autoconvolution peak."""
import time
import numpy as np
from scipy.optimize import minimize

N = 600

def _autoconvolve(h):
    """Compute linear autoconvolution of sequence h."""
    n = len(h)
    # Full linear autoconvolution
    R = np.correlate(h, h, mode='full')
    return R

def _normalized_max_conv(h):
    """Compute normalized maximum of autoconvolution."""
    R = _autoconvolve(h)
    s = np.sum(h)
    if s < 1e-12:
        return 1.0
    return np.max(R) / (s * s)

def _smooth_max_conv(h, temperature=10.0):
    """Smooth approximation of max autoconvolution for differentiable optimization."""
    R = _autoconvolve(h)
    s = np.sum(h)
    if s < 1e-12:
        return 1.0, np.zeros_like(h)
    
    # Use log-sum-exp for smooth max
    R_clipped = np.clip(R, 1e-12, None)
    log_R = np.log(R_clipped)
    max_log = np.max(log_R)
    weights = np.exp(log_R - max_log)
    weights /= np.sum(weights)
    
    smooth_max = max_log + np.log(np.sum(weights))
    value = smooth_max - 2 * np.log(s)
    
    # Gradient with respect to h
    # d/dh_i [smooth_max(R)] where R[k] = sum_j h[j] h[j+k]
    # This is complex; use numerical gradient or simplify
    # For now, use the weights to guide descent
    grad = np.zeros_like(h)
    # Approximate: the gradient of smooth max w.r.t. R is weights
    # dR[k]/dh[i] = h[i-k] + h[i+k] (with bounds)
    for k in range(len(R)):
        if weights[k] > 1e-10:
            offset = k - (len(R) - 1)
            # R[k] = sum_j h[j] * h[j + offset]
            for j in range(max(0, -offset), min(N, N - offset)):
                # dR[k]/dh[j] = h[j + offset] if j + offset in range
                if 0 <= j + offset < N:
                    grad[j] += weights[k] * h[j + offset]
                # dR[k]/dh[j+offset] = h[j]
                if 0 <= j < N:
                    grad[j + offset] += weights[k] * h[j]
    
    # Normalize by (sum h)^2
    grad /= (s * s)
    
    return value, grad

# EVOLVE-BLOCK-START
def construct():
    """
    Minimize the normalized maximum of the autoconvolution of a nonnegative
    step function with N equal intervals on [-1/4, 1/4].
    
    Uses gradient-based optimization with smooth max approximation.
    """
    start_time = time.time()
    deadline = 140.0  # safety margin
    
    def elapsed():
        return time.time() - start_time
    
    # Start with uniform
    best_h = np.ones(N)
    best_score = _normalized_max_conv(best_h)
    
    # Try different initializations and optimize
    rng = np.random.default_rng(42)
    
    # Gradient descent with smooth max
    h = np.ones(N)
    lr = 0.01
    temperature = 5.0
    
    for iteration in range(5000):
        if elapsed() > deadline - 5:
            break
        
        # Compute smooth max and gradient
        value, grad = _smooth_max_conv(h, temperature)
        
        # Update
        h_new = h - lr * grad
        h_new = np.clip(h_new, 0, None)  # Project to nonnegative
        
        # Normalize to keep sum constant (scale invariant)
        s = np.sum(h_new)
        if s > 1e-12:
            h_new = h_new / s * N  # Keep mean ~1
        
        # Check if improved
        actual_score = _normalized_max_conv(h_new)
        if actual_score < best_score:
            best_h = h_new.copy()
            best_score = actual_score
            h = h_new.copy()
        else:
            h = h_new.copy()
        
        # Reduce learning rate
        lr *= 0.999
        temperature *= 0.9995
    
    # Try a few random restarts if time permits
    for restart in range(5):
        if elapsed() > deadline - 10:
            break
        
        h = np.ones(N)
        # Add random perturbation
        h += rng.normal(0, 0.3, N)
        h = np.clip(h, 0, None)
        s = np.sum(h)
        if s > 1e-12:
            h = h / s * N
        
        lr = 0.005
        temperature = 3.0
        
        for iteration in range(3000):
            if elapsed() > deadline - 5:
                break
            
            value, grad = _smooth_max_conv(h, temperature)
            h_new = h - lr * grad
            h_new = np.clip(h_new, 0, None)
            s = np.sum(h_new)
            if s > 1e-12:
                h_new = h_new / s * N
            
            actual_score = _normalized_max_conv(h_new)
            if actual_score < best_score:
                best_h = h_new.copy()
                best_score = actual_score
            h = h_new.copy()
            
            lr *= 0.999
            temperature *= 0.9995
    
    # Final cleanup: ensure nonnegative
    best_h = np.clip(best_h, 0, None)
    
    if elapsed() > deadline:
        return best_h.tolist()
    
    return best_h.tolist()
# EVOLVE-BLOCK-END
