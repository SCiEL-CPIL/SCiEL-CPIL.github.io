"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import math

def _compute_penalty(x, W, H):
    """
    Compute penalty for a configuration x.
    x is of shape (2N,): first N are x-coords, next N are y-coords, last N are radii.
    Penalty is sum of constraint violations.
    """
    cx = x[:N]
    cy = x[N:2*N]
    r = x[2*N:]
    
    penalty = 0.0
    
    # Boundary constraints: circle must be inside [0, W] x [0, H]
    # cx - r >= 0, cx + r <= W, cy - r >= 0, cy + r <= H
    # Violations:
    pen_x_left = np.maximum(0, r - cx)
    pen_x_right = np.maximum(0, cx + r - W)
    pen_y_bottom = np.maximum(0, r - cy)
    pen_y_top = np.maximum(0, cy + r - H)
    
    penalty += np.sum(pen_x_left) + np.sum(pen_x_right) + np.sum(pen_y_bottom) + np.sum(pen_y_top)
    
    # Overlap constraints: dist(i, j) >= r_i + r_j
    # dist(i, j) = sqrt((cx_i - cx_j)^2 + (cy_i - cy_j)^2)
    # We want dist(i, j) - r_i - r_j >= 0
    # Violation is max(0, r_i + r_j - dist(i, j))
    
    # Vectorized distance calculation
    # cx shape (N,), cx[:, None] shape (N, 1), cx[None, :] shape (1, N)
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    
    # r_i + r_j
    r_sum = r[:, None] + r[None, :]
    
    # Violation matrix
    violations = np.maximum(0, r_sum - dist)
    
    # Sum over all pairs (upper triangle to avoid double counting, but sum is fine for penalty)
    # Diagonal is 0 because dist=0 and r_i+r_i=2r_i, wait.
    # For i=j, dist=0, r_sum=2r_i. Violation = 2r_i.
    # This is wrong. We only care about i != j.
    # Set diagonal to 0.
    np.fill_diagonal(violations, 0.0)
    
    penalty += np.sum(violations)
    
    return penalty

def _objective(x, W, H):
    """
    Objective function to minimize: -sum(r) + lambda * penalty
    """
    r = x[2*N:]
    sum_r = np.sum(r)
    pen = _compute_penalty(x, W, H)
    
    # Penalty weight
    # Start with a moderate weight, maybe increase if needed, but L-BFGS-B handles it via constraints?
    # Actually, we are using penalty method.
    # If penalty > 0, we want to minimize it strongly.
    # Let's use a large penalty factor.
    lambda_pen = 10.0
    return -sum_r + lambda_pen * pen

def _construct_config(W, H, seed=0):
    """
    Generate an initial valid-ish configuration or random configuration.
    """
    rng = np.random.RandomState(seed)
    
    # Initialize with small radii and random positions
    # Or try a grid initialization to give it a head start
    # Let's try a grid initialization for better starting point
    # 5x5 grid = 25 spots, we need 21.
    # W, H.
    # Spacing in x: W/5, in y: H/5.
    # Radius max ~ min(W, H)/10
    
    # Let's use a simple random initialization with small radii to avoid immediate huge penalties
    # But random might get stuck in local minima.
    # Let's try a few different initializations.
    
    cx = rng.uniform(0.1*W, 0.9*W, N)
    cy = rng.uniform(0.1*H, 0.9*H, N)
    r = np.full(N, min(W, H) * 0.05) # Small initial radii
    
    x0 = np.concatenate([cx, cy, r])
    
    # Bounds for L-BFGS-B
    # cx in [0, W], cy in [0, H], r in [0, min(W, H)/2]
    # But we want to encourage larger r.
    # The penalty will push them to be valid.
    # If we bound r to be small, we limit the solution.
    # Let's bound r to [0, min(W, H)]
    r_max = min(W, H)
    
    bounds = []
    for i in range(N):
        bounds.append((0.0, W))
    for i in range(N):
        bounds.append((0.0, H))
    for i in range(N):
        bounds.append((0.0, r_max))
        
    return x0, bounds

def _optimize_pack(W, H, seed=0, max_iter=1000):
    """
    Run L-BFGS-B optimization for a given rectangle W x H.
    """
    x0, bounds = _construct_config(W, H, seed)
    
    # We might want to do a multi-start or restart with different seeds
    best_x = x0
    best_val = _objective(x0, W, H)
    
    # Try a few different random starts to escape local minima
    for s in range(seed, seed + 5):
        x0_s, bounds_s = _construct_config(W, H, s)
        try:
            res = minimize(
                _objective,
                x0_s,
                args=(W, H),
                method='L-BFGS-B',
                bounds=bounds_s,
                options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            val = res.fun
            if val < best_val:
                best_val = val
                best_x = res.x
        except Exception:
            pass
            
    return best_x

def _validate_and_fix(x, W, H):
    """
    Check if the configuration is valid (penalty ~ 0).
    If not, try to repair by shrinking radii or moving circles?
    For now, if penalty is high, we might discard or try to fix.
    The penalty method usually converges to low penalty.
    """
    pen = _compute_penalty(x, W, H)
    if pen > 1e-6:
        # Not valid. Try to fix by scaling down radii slightly?
        # Or just return as is and let the evaluator decide?
        # The evaluator measures raw value. If invalid, it might score 0 or penalize.
        # The contract says "disjoint interiors".
        # We should ensure validity.
        # Simple fix: if overlap, reduce radii?
        # This is complex. Let's hope the optimizer works.
        # If penalty is non-zero, the objective includes it, so the optimizer tries to minimize it.
        # If it fails, the score might be low.
        pass
    return x

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    best_score = -1
    best_result = None
    
    # Perimeter 4 => W + H = 2
    # Try different aspect ratios
    # Square (1x1) is often good for packing.
    # Also try slightly rectangular.
    
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.2]
    
    for w in aspect_ratios:
        if time.time() > deadline:
            break
        h = 2.0 - w
        if h <= 0:
            continue
        
        # Run optimization with a few different seeds
        for seed in range(3):
            if time.time() > deadline:
                break
            x_opt = _optimize_pack(w, h, seed=seed, max_iter=500)
            x_valid = _validate_and_fix(x_opt, w, h)
            
            # Calculate score
            r = x_valid[2*N:]
            score = np.sum(r)
            
            # Check validity again for safety
            pen = _compute_penalty(x_valid, w, h)
            if pen < 1e-5:
                if score > best_score:
                    best_score = score
                    # Format result: (n, 3) array [cx, cy, r]
                    cx = x_valid[:N]
                    cy = x_valid[N:2*N]
                    res = np.column_stack((cx, cy, r))
                    best_result = res
            else:
                # If invalid, try to salvage by reducing radii?
                # Or just skip.
                # Let's try a simple repair: scale down all radii until valid?
                # This is expensive.
                # Maybe just skip this result.
                pass
    
    # Fallback: simple grid if optimization failed
    if best_result is None:
        # 5x5 grid in 1x1 square
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                cx = (i + 0.5) / side
                cy = (j + 0.5) / side
                r = 0.5 / side - 1e-4
                cells.append((cx, cy, r))
        best_result = np.array(cells[:N])
    
    return best_result
# EVOLVE-BLOCK-END
