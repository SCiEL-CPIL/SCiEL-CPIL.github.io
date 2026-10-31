"""Rewritten program for Erdos minimum overlap task."""
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 100

def _max_overlap(h):
    """Compute the maximum overlap of h with shifted 1-h over all shifts."""
    # h is a sequence of length N with values in [0,1]
    one_minus_h = 1.0 - h
    
    max_ov = 0.0
    for s in range(N):
        shifted = np.roll(one_minus_h, s)
        ov = np.dot(h, shifted)
        if ov > max_ov:
            max_ov = ov
    return max_ov

def _project_to_sum(h, target_sum):
    """
    Project h onto the set {x in [0,1]^N : sum(x) = target_sum}.
    Uses a water-filling / iterative clipping method.
    """
    h = h.copy()
    # Iterative adjustment
    for _ in range(100):
        current_sum = np.sum(h)
        diff = target_sum - current_sum
        if abs(diff) < 1e-12:
            break
        
        if diff > 0:
            # Need to increase sum. Increase elements that are < 1.
            mask = h < 1.0
            if not np.any(mask):
                break
            # To minimize disturbance, we should distribute the increase evenly among those that can take it,
            # or just add to all that can take it.
            # Simple approach: add diff / count to all that can take it.
            # This might push some to 1.0, which is fine, we clip and iterate.
            step = diff / np.sum(mask)
            h[mask] += step
            h = np.clip(h, 0.0, 1.0)
        else:
            # Need to decrease sum. Decrease elements that are > 0.
            mask = h > 0.0
            if not np.any(mask):
                break
            step = diff / np.sum(mask)
            h[mask] += step
            h = np.clip(h, 0.0, 1.0)
    
    return h

def _objective(params):
    """
    params: [c, w, L, R]
    c: transition center
    w: transition width
    L: left value
    R: right value
    """
    c, w, L, R = params
    i = np.arange(N)
    
    # Avoid division by zero
    if w < 1e-6:
        w = 1e-6
    
    # Sigmoid transition
    # We want a smooth step from L to R centered at c.
    # Standard sigmoid: 1 / (1 + exp(-x))
    # Let x = (i - c) / w
    # sig = 1 / (1 + exp(-x))
    # h = L + (R - L) * sig
    
    # To avoid overflow in exp, we can use np.tanh or careful sigmoid.
    # np.tanh((i-c)/(2*w)) goes from -1 to 1.
    # Let's use tanh for stability.
    # h = L + (R-L) * (tanh((i-c)/(2*w)) + 1) / 2
    
    arg = (i - c) / (2.0 * w)
    # Clip arg to prevent tanh saturation issues (though tanh is stable)
    # tanh is fine.
    t = np.tanh(arg)
    h = L + (R - L) * (t + 1.0) / 2.0
    
    # Clamp to [0,1] just in case of numerical issues
    h = np.clip(h, 0.0, 1.0)
    
    # Project to satisfy sum constraint exactly
    h = _project_to_sum(h, N / 2.0)
    
    # Compute max overlap
    ov = _max_overlap(h)
    
    return ov

def construct():
    """
    Construct the sequence of step heights.
    """
    # Define bounds for parameters
    # c: transition center, 0 to N
    # w: transition width, 1 to N
    # L: left value, 0 to 1
    # R: right value, 0 to 1
    
    bounds = [(0, N), (1, N), (0, 1), (0, 1)]
    
    # Use differential evolution for global search
    # Increased iterations and polish for better convergence
    result = differential_evolution(
        _objective, 
        bounds, 
        seed=42, 
        maxiter=200, 
        tol=1e-8, 
        polish=True,
        init='sobol'
    )
    
    best_params = result.x
    c, w, L, R = best_params
    
    i = np.arange(N)
    if w < 1e-6:
        w = 1e-6
    
    arg = (i - c) / (2.0 * w)
    t = np.tanh(arg)
    h = L + (R - L) * (t + 1.0) / 2.0
    h = np.clip(h, 0.0, 1.0)
    
    # Final projection to ensure exact sum
    h = _project_to_sum(h, N / 2.0)
    
    return h

# EVOLVE-BLOCK-START
# (The construct function is defined above, but we need to keep the interface)
# The task says "Keep the function names and signatures of the seed program."
# The seed program has construct() -> sequence of step heights.
# So we just need to make sure construct() is defined and returns the right thing.
# EVOLVE-BLOCK-END
