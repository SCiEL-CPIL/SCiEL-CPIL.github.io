"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 600
W = 1.0 / 1200.0  # Width of each interval

# EVOLVE-BLOCK-START

def _compute_ratio_fast(h):
    """
    Compute max(autoconv) / l2_sq for step function h.
    Uses FFT on a fine grid for accurate peak detection of the continuous convolution.
    """
    h = np.asarray(h, dtype=np.float64)
    h = np.clip(h, 0.0, None)
    total_energy = np.sum(h)
    if total_energy < 1e-15:
        return float('inf')
    
    # We normalize the metric by (sum h)^2 * dx? 
    # The task says "normalized maximum". 
    # Insp 1 uses max_conv / l2_sq. 
    # Let's check the scaling.
    # If h is scaled by c, max_conv scales by c^2, l2_sq scales by c^2. Ratio is invariant.
    # The current program uses max_conv / (total^2 * dx).
    # Let's stick to the Insp 1 metric as it was slightly better (1.0056 vs 1.0000).
    # Wait, Insp 1 score was 1.0056, Current was 1.0000. 
    # The prompt says "search score is seed/value". 
    # If seed=1.0, and value is the upper bound on C1.
    # Lower value is better.
    # Insp 1 value was lower than Current? 
    # Score 1.0056 means Value = 1.0 / 1.0056 ≈ 0.994.
    # Score 1.0000 means Value = 1.0.
    # So Insp 1 found a better (lower) value.
    
    n_grid = 4096
    # Map fine grid points to step function intervals
    # x_fine is in [-0.25, 0.25)
    x_fine = np.linspace(-0.25, 0.25, n_grid, endpoint=False)
    # Index = floor((x + 0.25) / W)
    cell_idx = np.clip(((x_fine + 0.25) / W).astype(np.int32), 0, N - 1)
    f_fine = h[cell_idx]
    
    n_fft = 2 * n_grid
    F = np.fft.rfft(f_fine, n_fft)
    conv = np.fft.irfft(F * F, n_fft)[:2 * n_grid - 1]
    dx = 0.5 / n_grid
    conv *= dx
    max_conv = np.max(conv)
    
    l2_sq = np.sum(f_fine ** 2) * dx
    if l2_sq < 1e-15:
        return float('inf')
    return max_conv / l2_sq

def _objective_with_reg(h, lam=0.01):
    """
    Objective function for optimization: 
    Ratio + lambda * smoothness_penalty.
    """
    h_pos = np.maximum(h, 0.0)
    if np.sum(h_pos) < 1e-15:
        return 1e6
    ratio = _compute_ratio_fast(h_pos)
    # Smoothness penalty: sum of squared differences
    if len(h_pos) > 1:
        diff = np.diff(h_pos)
        penalty = lam * np.sum(diff**2)
    else:
        penalty = 0.0
    return ratio + penalty

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_h = np.ones(N)
    best_score = _compute_ratio_fast(best_h)
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Uniform baseline
    candidates.append(np.ones(N))
    
    # 2. Cosine bell (smooth, zero at edges)
    t = np.linspace(-0.5, 0.5, N)
    candidates.append(0.5 * (1 + np.cos(2 * np.pi * t)))
    candidates.append(np.maximum(0, np.cos(np.pi * t)))
    
    # 3. Gaussian-like (various widths)
    for sigma in [0.1, 0.2, 0.3, 0.4]:
        candidates.append(np.exp(-0.5 * (t / sigma) ** 2))
    
    # 4. Sinc-like
    candidates.append(np.sinc(t))
    
    # 5. Power law shapes
    base = 1 - np.abs(2 * t)
    for p in [0.5, 1.0, 2.0]:
        candidates.append(np.maximum(0, base ** p))
        
    # 6. Random smooth candidates
    rng = np.random.RandomState(42)
    for _ in range(20):
        h_rand = rng.rand(N)
        # Smooth it more aggressively to avoid noise
        kernel = np.ones(10) / 10
        h_smooth = np.convolve(h_rand, kernel, mode='same')
        h_smooth = np.clip(h_smooth, 0, None)
        if np.sum(h_smooth) > 1e-15:
            candidates.append(h_smooth)
            
    # Evaluate all candidates
    for h in candidates:
        h = np.clip(h, 0.0, None)
        if np.sum(h) < 1e-15:
            continue
        r = _compute_ratio_fast(h)
        if r < best_score:
            best_score = r
            best_h = h.copy()
            
    # Local refinement using L-BFGS-B on heights
    # We will run this on the top few candidates or just the best one?
    # To save time, let's try the best one and a few diverse ones.
    top_candidates = [best_h]
    # Add a few other good ones if time permits
    if time.time() < deadline - 20.0:
        # Sort candidates by score to pick the next best ones
        scored_cands = []
        for h in candidates:
            h = np.clip(h, 0.0, None)
            if np.sum(h) < 1e-15: continue
            r = _compute_ratio_fast(h)
            scored_cands.append((r, h))
        scored_cands.sort(key=lambda x: x[0])
        for r, h in scored_cands[:3]:
            if h is not best_h:
                top_candidates.append(h)
                
    for h0 in top_candidates:
        if time.time() > deadline - 10.0:
            break
            
        h0 = np.array(h0, dtype=np.float64)
        
        try:
            # Use L-BFGS-B with bounds
            bounds = [(0.0, None)] * N
            # We need to be careful with the number of iterations
            result = minimize(
                _objective_with_reg,
                h0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-5}
            )
            
            h_opt = np.maximum(result.x, 0.0)
            score = _compute_ratio_fast(h_opt)
            if score < best_score:
                best_score = score
                best_h = h_opt.copy()
        except Exception:
            pass

    # Final polish: Coordinate descent (from Inspiration 1)
    if time.time() < deadline - 5.0:
        h = best_h.copy()
        step_size = max(0.01, np.max(h) * 0.01)
        
        for iteration in range(200):
            if time.time() > deadline - 2.0:
                break
            
            improved = False
            # Try perturbing a subset of coordinates
            n_trials = 50
            idxs = np.random.choice(N, size=n_trials, replace=False)
            
            for idx in idxs:
                h_old = h[idx]
                
                # Try increasing
                h[idx] = h_old + step_size
                r_plus = _compute_ratio_fast(h)
                
                # Try decreasing
                h[idx] = h_old - step_size
                r_minus = _compute_ratio_fast(h)
                
                h[idx] = h_old
                
                if r_plus < best_score:
                    h[idx] = h_old + step_size
                    best_score = r_plus
                    improved = True
                elif r_minus < best_score and h[idx] > 0:
                    h[idx] = h_old - step_size
                    best_score = r_minus
                    improved = True
            
            if not improved:
                step_size *= 0.8
                if step_size < 1e-6:
                    break
        best_h = h

    return best_h.tolist()

# EVOLVE-BLOCK-END
