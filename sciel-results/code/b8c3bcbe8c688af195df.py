"""
Strategy: Randomized Block-Coordinate Descent with Analytic DC Decoupling

This strategy abandons the low-rank Fourier parameterization used in the previous 
program. Instead, it operates directly on the 400-dimensional height vector h.

Key Structural Changes:
1. DC Decoupling: 
   The objective V(h) = |2N * max(conv(h,h)) / sum(h)^2| is scale-invariant 
   except for the DC component's effect on the sum and the convolution peak.
   For a fixed zero-mean shape u (where sum(u)=0), let h = u + c*1.
   sum(h) = N*c.
   conv(h,h) = conv(u,u) + c*(conv(u,1)+conv(1,u)) + c^2*conv(1,1).
   Since sum(u)=0, conv(u,1) and conv(1,u) are zero.
   So conv(h,h) = conv(u,u) + c^2 * (N * ones(N)).
   The peak of conv(h,h) is max(max(conv(u,u)), c^2*N + max(conv(u,u))_non_peak?) 
   Actually, conv(1,1) is a triangle. 
   Let's re-evaluate.
   If h = u + c, sum(h) = N*c (assuming sum(u)=0).
   conv(h,h) = conv(u,u) + 2c*conv(u,1) + c^2*conv(1,1).
   conv(u,1) at lag k is sum_{i} u[i] * 1[i+k]. This is not zero unless u is orthogonal to 1 at all shifts, which is impossible for a finite sequence unless u=0.
   Wait, conv(u,1) is the cumulative sum of u. It is not zero.
   
   However, the "inspiration" note says: "Decoupling the DC offset optimization 
   from the global search allows the optimizer to focus on the signal shape while 
   always operating at the optimal DC level for that shape."
   
   Let's use a simpler heuristic: 
   For any vector h, we can optimize the DC offset c given the AC component u.
   Let h = u + c.
   We want to minimize |2N * max(conv(u+c, u+c)) / (sum(u+c))^2|.
   Since sum(u) is fixed (let's say S_u), sum(h) = S_u + N*c.
   conv(h,h) = conv(u,u) + c*(conv(u,1)+conv(1,u)) + c^2*conv(1,1).
   This is a quadratic function of c in the numerator's max term, but max is not differentiable.
   However, we can simply scan a small range of c values for each shape u to find the best c.
   This reduces the effective search dimension by 1 and ensures we are always at a 
   local optimum for the DC level.

2. Randomized Block Coordinate Descent:
   Instead of optimizing all 400 variables at once (which is expensive and prone 
   to local minima in the full space), we partition h into blocks. 
   In each iteration, we select a random subset of indices (a block) and optimize 
   those variables via a local gradient step or small random perturbation + line search,
   while keeping the rest fixed. 
   This allows the algorithm to explore "rougher" shapes that low-rank Fourier 
   methods might miss, while the DC decoupling keeps the objective well-behaved.
   
   To answer the open question about 'diff':
   The random block updates act as a 'diff' operator in the sense that they introduce 
   stochastic changes to the signal structure. We will use a mix of smooth and 
   sharp updates to see if structural roughness helps lower the peak autocorrelation.
"""

import numpy as np
import time

N = 400
_DEADLINE = 145.0  # Wall-clock limit with safety margin


def _evaluate(h):
    """Compute the task value."""
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))


def _optimize_dc(u):
    """
    Given a zero-mean (or arbitrary) shape u, find the optimal DC offset c 
    such that h = u + c minimizes the objective.
    We do this by scanning a small range of c values.
    """
    # Sum of u
    sum_u = np.sum(u)
    # We want sum(h) != 0.
    # Let's try a range of c values.
    # The optimal c is likely related to the magnitude of u.
    max_abs_u = np.max(np.abs(u))
    if max_abs_u == 0:
        return u + 1.0
    
    # Try a grid of c values
    # c should be large enough to make sum(h) significant.
    # Let's sample c in [-2*max_abs_u, 2*max_abs_u]
    # But sum(h) = sum_u + N*c.
    # If sum_u is large, c might need to be large to balance it?
    # Actually, the objective is scale invariant if we scale h.
    # V(kh) = V(h).
    # So we can normalize u first?
    # Let's just scan c.
    
    best_c = 0
    best_val = np.inf
    
    # Heuristic: c should be around -sum_u/N to make sum(h) small? No, sum(h) in denominator.
    # We want sum(h)^2 large relative to max(conv).
    # Let's just try a few values.
    
    # Normalize u to have unit norm for stability
    norm_u = np.linalg.norm(u)
    if norm_u < 1e-10:
        return np.ones(N)
    
    u_norm = u / norm_u
    
    # Try c in a range
    c_vals = np.linspace(-10, 10, 200)
    
    for c in c_vals:
        h = u_norm + c
        s = np.sum(h)
        if abs(s) < 1e-10:
            continue
        conv = np.convolve(h, h)
        val = abs(2.0 * N * np.max(conv) / (s * s))
        if val < best_val:
            best_val = val
            best_c = c
            
    return u_norm + best_c


def construct():
    """
    Construct step heights to minimize the autocorrelation inequality value.
    """
    t0 = time.time()
    deadline = t0 + _DEADLINE
    
    # Initialize with a random signal
    rng = np.random.default_rng(42)
    best_h = np.ones(N)
    best_val = 2.0
    
    # Initial candidates:
    # 1. Random noise
    # 2. Chirp
    # 3. Window functions
    candidates = []
    
    # Random
    candidates.append(rng.normal(size=N))
    
    # Chirp
    n = np.arange(N)
    candidates.append(np.cos(2 * np.pi * (n**2) / (2 * N)))
    
    # Hamming window
    candidates.append(np.hamming(N))
    
    # Blackman window
    candidates.append(np.blackman(N))
    
    # Sinc-like
    x = np.linspace(-2, 2, N)
    candidates.append(np.sinc(x))
    
    # Evaluate initial candidates with DC optimization
    for u in candidates:
        if time.time() > deadline - 10:
            break
        h_opt = _optimize_dc(u)
        val = _evaluate(h_opt)
        if val < best_val:
            best_val = val
            best_h = h_opt.copy()
    
    # Randomized Block Coordinate Descent
    # We will perform iterations. In each iteration, we pick a random block of indices
    # and perturb them, then optimize DC.
    
    num_iterations = 2000
    block_size = 20
    
    for i in range(num_iterations):
        if time.time() > deadline - 5:
            break
            
        # Select a random block
        start = rng.integers(0, N - block_size)
        end = start + block_size
        
        # Create a copy of current best
        h_curr = best_h.copy()
        
        # Perturb the block
        # Mix of smooth and sharp perturbations
        if rng.random() < 0.5:
            # Smooth perturbation: low freq
            t = np.linspace(0, 1, block_size)
            pert = rng.normal(size=block_size) * 0.1 * np.cos(2 * np.pi * t * rng.integers(1, 5))
        else:
            # Sharp perturbation: random
            pert = rng.normal(size=block_size) * 0.1
        
        h_curr[start:end] += pert
        
        # Optimize DC for this new shape
        # To speed up, we can do a quick local search for c
        u = h_curr
        # Re-use _optimize_dc but it might be slow due to scanning 200 values.
        # Let's do a faster version: try c = best_c + delta
        
        # Actually, let's just call _optimize_dc. 200 evals * 2000 iters = 400k evals.
        # Each eval is O(N^2) = 160k ops. 400k * 160k = 64e9 ops. Too slow.
        
        # We need a faster DC optimization.
        # Let's approximate the optimal c.
        # V(c) = |2N * max(conv(u+c, u+c)) / (sum(u+c))^2|
        # Let's just try c = 0, c = 1, c = -1, c = current_c + 0.1, c = current_c - 0.1
        
        # Better: Use the fact that V is smooth in c.
        # Let's do a small line search.
        
        c_curr = np.mean(h_curr) - np.mean(u) # This is 0 if u=h_curr.
        # Let's just try a few c values around the mean of h_curr.
        
        base_u = h_curr - np.mean(h_curr)
        
        best_c_local = np.mean(h_curr)
        best_val_local = _evaluate(h_curr)
        
        # Try a few offsets
        for delta_c in [-1.0, -0.5, -0.1, 0.1, 0.5, 1.0]:
            h_test = base_u + (np.mean(h_curr) + delta_c)
            val_test = _evaluate(h_test)
            if val_test < best_val_local:
                best_val_local = val_test
                best_c_local = np.mean(h_curr) + delta_c
                h_curr = h_test
        
        if best_val_local < best_val:
            best_val = best_val_local
            best_h = h_curr.copy()
            
    # Final polish: Local gradient descent on full vector?
    # Or just return best_h
    
    final_val = _evaluate(best_h)
    if not np.isfinite(final_val) or final_val >= 2.0:
        best_h = np.ones(N)
        
    return best_h
