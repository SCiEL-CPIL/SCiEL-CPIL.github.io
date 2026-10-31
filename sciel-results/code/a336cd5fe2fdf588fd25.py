"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 600
W = 1.0 / N  # interval width on [-1/4, 1/4], total width = 1/2, so w = (1/2)/N = 1/(2N)
# Actually [-1/4, 1/4] has width 1/2, so w = (1/2)/600 = 1/1200
W = 0.5 / N

def _compute_c1_bound(h):
    """Compute upper bound on C1 for step function with heights h."""
    h = np.asarray(h, dtype=np.float64)
    integral = W * np.sum(h)
    if integral <= 0:
        return np.inf
    # Autoconvolution: (f*f)(t) for step functions
    # Discrete autoconv: ac[k] = sum_i h[i]*h[i+k] for k in 0..N-1
    # The continuous autoconvolution max is approx W^2 * max(ac)
    # But more precisely, for step functions, f*f is piecewise linear
    # The max of the piecewise linear autoconv occurs at interval edges
    # Approximation: max(f*f) ≈ W^2 * max(ac[k])
    ac = np.correlate(h, h, mode='full')
    # ac has length 2N-1, center at N-1
    max_ac = np.max(ac)
    max_ff = W * W * max_ac
    c1 = max_ff / (integral * integral)
    return c1

def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s limit
    
    # Start with all ones (the seed)
    h = np.ones(N, dtype=np.float64)
    best_h = h.copy()
    best_score = _compute_c1_bound(best_h)
    
    # Try a few different initial patterns
    # Pattern 1: all ones (baseline)
    # Pattern 2: alternating 1,0,1,0,... (if N even)
    # Pattern 3: blocks of ones separated by gaps
    
    # Let's try gradient descent on log-heights
    # Minimize max(autoconv) subject to sum(h) = constant (or just minimize the ratio)
    
    # Use iterative refinement: 
    # 1. Start with a reasonable pattern
    # 2. Use coordinate descent or gradient-based method
    
    # Try binary patterns first
    patterns = []
    
    # All ones
    patterns.append(np.ones(N))
    
    # Alternating
    if N % 2 == 0:
        alt = np.ones(N)
        alt[1::2] = 0.0
        patterns.append(alt)
    
    # Blocks: e.g., 30 ones, 30 zeros, repeat
    block_size = 30
    for bs in [10, 20, 30, 50, 100]:
        pat = np.zeros(N)
        for start in range(0, N, 2*bs):
            end = min(start + bs, N)
            pat[start:end] = 1.0
        if np.sum(pat) > 0:
            patterns.append(pat)
    
    # Evaluate all patterns
    for pat in patterns:
        s = _compute_c1_bound(pat)
        if s < best_score:
            best_score = s
            best_h = pat.copy()
    
    # Now do coordinate descent on the best pattern
    # For each position, try to adjust the height to reduce the max autoconv
    # Use a simple approach: for each i, try h[i] in a range and pick the best
    
    # Actually, let's use a smarter approach: 
    # The max autoconv is dominated by the peak. 
    # We can use subgradient-like methods or just coordinate descent.
    
    # Coordinate descent: for each index i, find the optimal h[i] given others fixed
    # This is hard analytically, so use numerical search per coordinate
    
    num_cd_iters = 5
    for it in range(num_cd_iters):
        if time.time() > deadline:
            break
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            old_h = h[i]
            # Try a few values around current
            candidates = [old_h * 0.5, old_h * 0.75, old_h * 1.25, old_h * 1.5, 0.0, 1.0]
            if old_h > 0:
                candidates.append(old_h * 0.1)
            candidates = np.array([max(0, c) for c in candidates])
            best_val = old_h
            best_s = _compute_c1_bound(h)
            for cv in candidates:
                h[i] = cv
                s = _compute_c1_bound(h)
                if s < best_s:
                    best_s = s
                    best_val = cv
            h[i] = best_val
            if best_val != old_h:
                improved = True
        if not improved:
            break
    
    s = _compute_c1_bound(h)
    if s < best_score:
        best_score = s
        best_h = h.copy()
    
    # Return as list of floats
    return [float(x) for x in best_h]

# EVOLVE-BLOCK-START
# (construct is defined above)
# EVOLVE-BLOCK-END
