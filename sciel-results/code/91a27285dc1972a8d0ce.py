import numpy as np
import time
import sys

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]

def _autoconv_max_continuous(h):
    """
    Calculate the supremum norm of the continuous autoconvolution of the step function defined by h.
    h: array of heights.
    The step function f(t) = h[i] for t in [x_i, x_{i+1}).
    """
    # Use FFT for efficiency if N is large, but N=50 is small.
    # Direct calculation of convolution of step functions is equivalent to 
    # convolving the sequence h with itself, but scaled by Delta?
    # Let's derive.
    # f(t) = sum h_i 1_{[x_i, x_{i+1})}(t).
    # (f*f)(t) = sum_i sum_j h_i h_j (1_{[x_i, x_{i+1})} * 1_{[x_j, x_{j+1})})(t).
    # The convolution of two unit boxes of width Delta is a triangle of width 2*Delta and peak Delta.
    # So (f*f)(t) = Delta * (h * h)_discrete(t) ??
    # No, the discrete convolution sum k_i = sum_j h_j h_{j+i} approximates the integral.
    # The continuous convolution peak is roughly Delta * max(sum_j h_j h_{j+k}).
    # Let's verify with uniform h=1.
    # Discrete conv peak = N.
    # Continuous conv peak should be L = N*Delta.
    # Formula: Delta * N = N*Delta. Matches.
    # So max_cont = DELTA * max_discrete_conv(h).
    
    h_conv = np.convolve(h, h, mode='full')
    max_conv = np.max(h_conv)
    return DELTA * max_conv

def _score_raw(h):
    l1 = np.sum(h)
    l2_sq = np.sum(h**2)
    if l1 == 0:
        return 0.0
    autoconv_max = _autoconv_max_continuous(h)
    if autoconv_max == 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max))

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # Initialize with uniform
    h = np.ones(N)
    
    # Normalize? The score is scale invariant in the mixed metric?
    # V(c*h) = (c^2 L2^2) / (c L1 * c^2 AutoMax) = (1/c) V(h).
    # So we want small c?
    # But h must be non-negative.
    # If I scale h down, V goes up.
    # Is there a lower bound on h?
    # "nonnegative".
    # If I can scale to 0, V -> inf.
    # This suggests the evaluator normalizes h to L1=1 or similar BEFORE scoring.
    # Assuming L1=1 normalization.
    
    def normalized_score(h):
        if np.sum(h) == 0: return 0
        h_norm = h / np.sum(h)
        return _score_raw(h_norm)

    best_h = h.copy()
    best_score = normalized_score(h)
    
    # Simple coordinate descent
    # We want to maximize the score.
    # Since uniform is a local max in some metrics, we try to perturb it.
    # Try Gaussian shapes.
    
    best_h = np.ones(N)
    best_score = normalized_score(best_h)
    
    # Try a few parametric families
    centers = np.linspace(0, N-1, 10)
    widths = np.linspace(1, N, 10)
    
    for c in centers:
        for w in widths:
            # Gaussian
            x = np.arange(N)
            h_gauss = np.exp(-0.5 * ((x - c) / w)**2)
            # Ensure non-negative (it is)
            s = normalized_score(h_gauss)
            if s > best_score:
                best_score = s
                best_h = h_gauss.copy()
                
            # Triangular
            h_tri = np.maximum(0, 1 - np.abs(x - c) / w)
            s = normalized_score(h_tri)
            if s > best_score:
                best_score = s
                best_h = h_tri.copy()
                
            # Ramp
            h_ramp = np.maximum(0, 1 - np.abs((x - c) / w - 0.5) * 2) # Rough triangle
            # Actually just use the triangular one above.
            
    # Coordinate descent on best_h
    step = 0.1
    for i in range(N):
        for _ in range(50):
            if time.time() > deadline:
                break
            h_trial = best_h.copy()
            h_trial[i] += step
            if h_trial[i] < 0: h_trial[i] = 0
            s = normalized_score(h_trial)
            if s > best_score:
                best_score = s
                best_h = h_trial
            else:
                h_trial = best_h.copy()
                h_trial[i] -= step
                if h_trial[i] < 0: h_trial[i] = 0
                s = normalized_score(h_trial)
                if s > best_score:
                    best_score = s
                    best_h = h_trial
                else:
                    break
            
    # Final check
    if time.time() > deadline:
        return list(best_h)
        
    return list(best_h)

# EVOLVE-BLOCK-START
def construct():
    # Re-define inside to keep scope clean if needed, or use globals
    start_time = time.time()
    deadline = 110.0
    
    h = np.ones(N)
    
    def normalized_score(h):
        s = np.sum(h)
        if s == 0: return 0
        h_norm = h / s
        l1 = np.sum(h_norm)
        l2_sq = np.sum(h_norm**2)
        autoconv_max = DELTA * np.max(np.convolve(h_norm, h_norm, mode='full'))
        if autoconv_max == 0: return 0
        return (l2_sq / (l1 * autoconv_max))

    best_h = h.copy()
    best_score = normalized_score(h)
    
    # Parametric search
    centers = np.linspace(0, N-1, 20)
    widths = np.linspace(1, N, 20)
    
    for c in centers:
        for w in widths:
            x = np.arange(N)
            h_gauss = np.exp(-0.5 * ((x - c) / w)**2)
            s = normalized_score(h_gauss)
            if s > best_score:
                best_score = s
                best_h = h_gauss.copy()
                
            h_tri = np.maximum(0, 1 - np.abs(x - c) / w)
            s = normalized_score(h_tri)
            if s > best_score:
                best_score = s
                best_h = h_tri.copy()
                
    # Coordinate Descent
    step = 0.1
    for _ in range(100): # Max iterations
        if time.time() > deadline: break
        improved = False
        for i in range(N):
            if time.time() > deadline: break
            # Try increase
            h_trial = best_h.copy()
            h_trial[i] += step
            s = normalized_score(h_trial)
            if s > best_score:
                best_score = s
                best_h = h_trial
                improved = True
            else:
                # Try decrease
                h_trial = best_h.copy()
                h_trial[i] -= step
                if h_trial[i] < 0: h_trial[i] = 0
                s = normalized_score(h_trial)
                if s > best_score:
                    best_score = s
                    best_h = h_trial
                    improved = True
        if not improved:
            break
            
    return list(best_h)
# EVOLVE-BLOCK-END
