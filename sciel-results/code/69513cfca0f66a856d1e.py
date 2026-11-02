"""
Strategy: Two-Block Disjoint Rectangles
Represents the step function as two disjoint rectangular blocks.
The autoconvolution of such a function is a sum of trapezoids.
We optimize the positions, widths, and amplitudes of these blocks to minimize the normalized max autoconvolution.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 600
h = 0.5 / N  # Interval length

def _compute_score(a):
    """
    Compute the normalized maximum autoconvolution.
    Input a: array of heights.
    Returns: normalized max value.
    """
    if np.sum(a) == 0:
        return float('inf')
    n = len(a)
    nfft = 1 << (2 * n - 1).bit_length()
    A = np.fft.rfft(a, n=nfft)
    conv = np.fft.irfft(A * A, n=nfft)[:2*n-1]
    max_conv = np.max(conv)
    norm = h**2 * (np.sum(a))**2
    return max_conv / norm

def _construct_from_blocks(params):
    """
    Construct the step function array from block parameters.
    params: [start1, width1, amp1, start2, width2, amp2]
    Returns: array of length N
    """
    a = np.zeros(N)
    # Block 1
    s1, w1, a1 = params[0], params[1], params[2]
    # Block 2
    s2, w2, a2 = params[3], params[4], params[5]
    
    # Convert continuous coordinates to discrete indices
    # Domain is [-0.25, 0.25]
    # Index i corresponds to x_i = -0.25 + i*h
    # Block covers [s, s+w]
    
    # Discretize block 1
    idx_start1 = int(np.floor((s1 + 0.25) / h))
    idx_end1 = int(np.ceil((s1 + w1 + 0.25) / h))
    idx_start1 = max(0, min(N, idx_start1))
    idx_end1 = max(0, min(N, idx_end1))
    if idx_end1 > idx_start1:
        a[idx_start1:idx_end1] += a1
        
    # Discretize block 2
    idx_start2 = int(np.floor((s2 + 0.25) / h))
    idx_end2 = int(np.ceil((s2 + w2 + 0.25) / h))
    idx_start2 = max(0, min(N, idx_start2))
    idx_end2 = max(0, min(N, idx_end2))
    if idx_end2 > idx_start2:
        a[idx_start2:idx_end2] += a2
        
    return a

def _objective_blocks(params):
    """
    Objective function for block parameters.
    Penalize overlap to ensure disjointness (or just allow it, but disjoint is the strategy).
    Actually, the strategy specifies disjoint. We can enforce this by parameterizing
    the gap between them.
    """
    s1, w1, a1, s2, w2, a2 = params
    
    # Enforce disjointness: s2 >= s1 + w1 + gap
    # For simplicity in optimization, let's just compute the score.
    # If they overlap, the "destructive interference" might not work as intended for simple rectangles,
    # but the optimizer will find non-overlapping if it's beneficial.
    
    a = _construct_from_blocks(params)
    score = _compute_score(a)
    
    # Add penalty if blocks are not disjoint or out of bounds
    # Domain: [-0.25, 0.25]
    if s1 < -0.25 or s1 + w1 > 0.25 or s2 < -0.25 or s2 + w2 > 0.25:
        return score + 100.0
        
    # Penalize overlap to encourage the "two disjoint" structure
    if s1 + w1 > s2 and s2 + w2 > s1:
        overlap = min(s1 + w1, s2 + w2) - max(s1, s2)
        if overlap > 0:
            return score + 10.0 * overlap
            
    return score

def construct():
    deadline = time.time() + 140.0  # Safety margin
    
    best_score = float('inf')
    best_a = np.ones(N) / N  # Fallback
    
    # Initial best: Uniform
    a_uni = np.ones(N) / N
    best_score = _compute_score(a_uni)
    best_a = a_uni.copy()
    
    # Strategy: Search over two disjoint blocks
    # Parameters: [start1, width1, amp1, start2, width2, amp2]
    # Normalization: The total mass is sum(a). The score is scale-invariant?
    # Score = max(conv) / (h^2 * (sum a)^2)
    # conv is quadratic in a. sum a is linear in a.
    # So score is scale-invariant. We can fix the sum of amplitudes * widths to 1,
    # or just let the optimizer find the relative amplitudes.
    # Let's fix the total "mass" to 1 for numerical stability.
    # a1*w1 + a2*w2 = 1 => a2 = (1 - a1*w1) / w2
    
    def _obj_normalized(params):
        s1, w1, a1, s2, w2, _ = params # a2 is derived
        if w1 <= 0 or w2 <= 0:
            return 1e6
        a2 = (1.0 - a1 * w1) / w2
        if a2 < 0:
            return 1e6 # Ensure non-negative
        params_full = [s1, w1, a1, s2, w2, a2]
        return _objective_blocks(params_full)
    
    # Grid search for initial positions and widths to provide good starting points
    # Positions in [-0.25, 0.25]
    # Widths in [0.01, 0.25]
    
    start_times = time.time()
    
    # Define a set of candidate configurations to optimize
    candidates = []
    
    # Symmetric configurations: Block 1 at left, Block 2 at right
    # Let center of mass be 0.
    # s1 = -d - w1/2, s2 = d - w2/2
    # Try different separations d and widths
    
    for w1 in [0.05, 0.1, 0.15, 0.2]:
        for w2 in [0.05, 0.1, 0.15, 0.2]:
            for d in [0.05, 0.1, 0.15, 0.2]:
                # Ensure within bounds
                s1 = -d - w1/2
                s2 = d - w2/2
                if s1 < -0.25 or s1 + w1 > 0.25: continue
                if s2 < -0.25 or s2 + w2 > 0.25: continue
                
                # Initial amplitudes: equal mass
                a1 = 0.5 / w1
                a2 = 0.5 / w2
                params_init = [s1, w1, a1, s2, w2, a2]
                candidates.append(params_init)
                
                # Also try asymmetric amplitudes
                for ratio in [0.2, 0.5, 0.8]:
                    mass1 = ratio
                    mass2 = 1.0 - ratio
                    a1 = mass1 / w1
                    a2 = mass2 / w2
                    params_init = [s1, w1, a1, s2, w2, a2]
                    candidates.append(params_init)

    # Also try some random initializations
    np.random.seed(42)
    for _ in range(20):
        w1 = np.random.uniform(0.05, 0.25)
        w2 = np.random.uniform(0.05, 0.25)
        s1 = np.random.uniform(-0.25, 0.25 - w1)
        s2 = np.random.uniform(-0.25, 0.25 - w2)
        # Ensure disjoint
        if s1 + w1 > s2:
            s2 = s1 + w1 + np.random.uniform(0.01, 0.1)
            if s2 + w2 > 0.25:
                s1 = s2 - w2 - np.random.uniform(0.01, 0.1)
                if s1 < -0.25:
                    continue
        mass1 = np.random.uniform(0.1, 0.9)
        a1 = mass1 / w1
        a2 = (1 - mass1) / w2
        candidates.append([s1, w1, a1, s2, w2, a2])
        
    # Optimize each candidate
    for p_init in candidates:
        if time.time() > deadline - 5:
            break
            
        # Convert to optimization variables where a2 is derived?
        # Or just optimize all 6 and penalize sum != 1?
        # Let's optimize s1, w1, s2, w2, and log(a1), log(a2) to ensure positivity.
        # And add a penalty for sum(a) != 1 to handle scale invariance naturally.
        
        def _obj_opt(x):
            s1, w1, s2, w2, log_a1, log_a2 = x
            a1 = np.exp(log_a1)
            a2 = np.exp(log_a2)
            
            # Penalty for bounds
            pen = 0
            if s1 < -0.25: pen += 100 * (-0.25 - s1)
            if s1 + w1 > 0.25: pen += 100 * (s1 + w1 - 0.25)
            if s2 < -0.25: pen += 100 * (-0.25 - s2)
            if s2 + w2 > 0.25: pen += 100 * (s2 + w2 - 0.25)
            if w1 < 0.01: pen += 100 * (0.01 - w1)
            if w2 < 0.01: pen += 100 * (0.02 - w2)
            
            # Construct
            a = _construct_from_blocks([s1, w1, a1, s2, w2, a2])
            score = _compute_score(a)
            
            # Penalty for overlap
            if s1 + w1 > s2 and s2 + w2 > s1:
                overlap = min(s1 + w1, s2 + w2) - max(s1, s2)
                if overlap > 0:
                    pen += 10.0 * overlap
            
            return score + pen
        
        # Initial guess for log amplitudes
        a1_init = p_init[2]
        a2_init = p_init[5]
        x0 = [p_init[0], p_init[1], p_init[3], p_init[4], np.log(max(a1_init, 1e-6)), np.log(max(a2_init, 1e-6))]
        
        try:
            res = minimize(_obj_opt, x0, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
            if res.success or res.fun < best_score:
                x_opt = res.x
                s1, w1, s2, w2, log_a1, log_a2 = x_opt
                a1 = np.exp(log_a1)
                a2 = np.exp(log_a2)
                a_opt = _construct_from_blocks([s1, w1, a1, s2, w2, a2])
                
                # Normalize to ensure non-negative and consistent
                if np.any(a_opt < 0):
                    a_opt = np.maximum(a_opt, 0)
                
                score_opt = _compute_score(a_opt)
                if score_opt < best_score:
                    best_score = score_opt
                    best_a = a_opt.copy()
        except Exception:
            pass

    # Final check: is the two-block solution better than uniform?
    # If not, return uniform.
    # Also, try a single block (just in case, though uniform is usually better than single block for this metric? 
    # Actually, single block autoconv is a triangle. Max is at peak. 
    # Uniform is flat. 
    # Two blocks can create a "dip" in the middle of the autoconvolution if separated enough.
    
    return list(best_a)

if __name__ == "__main__":
    a = construct()
    print(f"Score: {_compute_score(np.array(a))}")
