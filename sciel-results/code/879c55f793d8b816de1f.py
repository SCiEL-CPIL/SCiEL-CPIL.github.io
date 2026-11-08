"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def compute_c3_bound(signs):
    """
    Compute the upper bound on C3 for a given signed step function.
    The autoconvolution of a step function h(x) = sum_i s_i * 1_{[x_i, x_{i+1})}(x)
    is (h * h)(t) = sum_{i,j} s_i s_j * length of intersection of intervals shifted by t.
    
    For equal intervals of width w = 1/200 on [-1/4, 1/4], the autoconvolution is a piecewise
    linear function. The upper bound on C3 (third autocorrelation inequality) relates to
    the L-infinity norm or similar functional of the autoconvolution.
    
    Given the task description: "improve an autoconvolution upper bound" and "score is the
    upper bound on C3", with "raw value (lower is better)", we need to minimize some
    measure of the autoconvolution.
    
    The autoconvolution (h*h)(t) for a step function with N equal intervals can be computed
    efficiently. For a binary/signed step function, the autoconvolution peak is related to
    the sum of squares and cross terms.
    
    A common upper bound involves ||h*h||_inf or similar. Let's compute the autoconvolution
    and take its maximum absolute value or L1 norm, etc.
    
    Actually, for the third autocorrelation inequality, a relevant quantity is often
    related to the integral of |h*h| or the sup norm. Let's compute the autoconvolution
    values at the midpoints of intervals and take the max absolute value as a proxy.
    
    More precisely: if h(x) = sum_{i=0}^{N-1} s_i * 1_{[x_i, x_{i+1})}(x) where each interval
    has width w = 1/200, then (h*h)(t) = sum_{i,j} s_i s_j * max(0, w - |t - (x_i + x_j - x_0 - x_0)|) ... 
    
    Actually, let's think more carefully. The autoconvolution (h*h)(t) = integral h(x)h(x-t) dx.
    For a step function on N equal intervals of width w, this is a piecewise linear function
    with breakpoints at t = k*w for k = -(N-1), ..., N-1.
    
    We can compute (h*h)(t) at t = m*w for m = 0, 1, ..., N-1 (by symmetry, we only need non-negative).
    (h*h)(m*w) = w * sum_{i,j: |i-j| = m} s_i s_j  ... no wait.
    
    Let me be more precise. Let the intervals be I_i = [i*w, (i+1)*w) for i = 0,...,N-1,
    but shifted to center on [-1/4, 1/4]. The actual positions don't matter for the autoconvolution
    shape, only the relative positions.
    
    (h*h)(t) = sum_{i,j} s_i s_j * measure({x: x in I_i and x-t in I_j})
             = sum_{i,j} s_i s_j * max(0, w - |t - (i-j)*w|)  [if intervals are contiguous]
    
    For t = m*w where m is an integer:
    (h*h)(m*w) = w * sum_{i,j: |i-j| = m} s_i s_j  ... this isn't quite right either.
    
    Let's just compute it numerically for efficiency. For each lag m = 0, 1, ..., N-1,
    (h*h)(m*w) = w * sum_{i=0}^{N-1-m} s_i * s_{i+m}
    
    Wait, that's for the discrete case. Let me verify: if h is constant s_i on [i*w, (i+1)*w),
    then for t = m*w, the overlap between I_i and I_j shifted by t is:
    - If j = i - m, the overlap is w (full overlap when shifted by exactly m*w? No...)
    
    Actually, for a step function that is s_i on [i*w, (i+1)*w), the autoconvolution at lag m*w is:
    (h*h)(m*w) = sum_{i} s_i * s_{i-m} * w  (with s_k = 0 if k outside [0, N-1])
    
    So (h*h)(m*w) = w * sum_{i=m}^{N-1} s_i * s_{i-m} for m >= 0.
    
    The upper bound on C3 might be related to the maximum of |(h*h)(t)| or the L1 norm.
    Given "lower is better", let's use the maximum absolute value of the autoconvolution
    as the score, or perhaps the L-infinity norm.
    
    Let's use: score = max_m |(h*h)(m*w)| = w * max_m |sum_{i=m}^{N-1} s_i s_{i-m}|
    
    Since w = 1/200 is a constant factor, we can just minimize max_m |sum_{i=m}^{N-1} s_i s_{i-m}|.
    """
    w = 1.0 / N  # width of each interval, since total length is 1/2 = N * w, so w = 1/(2N)? 
    # Wait, [-1/4, 1/4] has length 1/2. N = 400 intervals, so w = (1/2)/400 = 1/800.
    w = 0.5 / N
    
    # Compute autoconvolution at lags m = 0, 1, ..., N-1
    max_abs = 0.0
    for m in range(N):
        s_val = 0.0
        for i in range(m, N):
            s_val += signs[i] * signs[i - m]
        abs_val = abs(s_val) * w
        if abs_val > max_abs:
            max_abs = abs_val
    return max_abs

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Initialize with random signs
    signs = np.random.choice([-1, 1], size=N)
    best_signs = signs.copy()
    best_score = compute_c3_bound(best_signs)
    
    # Simulated annealing
    T_start = 1.0
    T_end = 1e-4
    n_steps = 200000
    
    for step in range(n_steps):
        if time.time() > deadline:
            break
        
        # Cooling schedule
        alpha = step / n_steps
        T = T_start * (T_end / T_start) ** alpha
        
        # Pick a random index to flip
        idx = random.randint(0, N - 1)
        old_sign = signs[idx]
        signs[idx] = -old_sign
        
        new_score = compute_c3_bound(signs)
        delta = new_score - best_score
        
        if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-10)):
            # Accept the move
            if new_score < best_score:
                best_score = new_score
                best_signs = signs.copy()
        else:
            # Reject the move
            signs[idx] = old_sign
    
    return best_signs
# EVOLVE-BLOCK-END
