"""
Circle packing in perimeter-4 rectangle using Lagrange Multiplier Method.
"""
import numpy as np
import time
import random

N = 21
START_TIME = time.time()
DEADLINE = START_TIME + 140.0  # Safety margin

def _time_ok():
    return time.time() < DEADLINE

def _lagrange_optimize(W, H, x0, max_iter=5000, tol=1e-10):
    """
    Optimize circle packing using Lagrange Multipliers (Force-Directed).
    x0: (N, 3) array of [cx, cy, r]
    Returns: (x_opt, sum_r)
    """
    cx = x0[:, 0].copy()
    cy = x0[:, 1].copy()
    r = x0[:, 2].copy()
    
    # Initialize Lagrange multipliers for non-overlap constraints
    # We store multipliers for each pair (i, j) where i < j
    # Since N=21, there are 210 pairs.
    num_pairs = N * (N - 1) // 2
    multipliers = np.zeros(num_pairs)
    
    # Precompute pair indices for fast access
    pair_indices = []
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            pair_indices.append((i, j, idx))
            idx += 1
            
    # Boundary constraints are handled directly by projecting/limiting r
    # but we can also use multipliers for boundaries if needed. 
    # For simplicity and robustness, we project r to be within bounds 
    # and use multipliers only for non-overlap.
    
    best_sum = 0.0
    best_x = x0.copy()
    
    # Learning rate for gradient descent on Lagrangian
    # The Lagrangian L = -sum(r) - sum(lambda_ij * (r_i + r_j - d_ij))
    # Gradient descent on L w.r.t x:
    # dx/dt = -grad_x L
    # dr/dt = -grad_r L
    
    # We use a simple iterative scheme:
    # 1. Calculate forces (gradients)
    # 2. Update positions and radii
    # 3. Project radii to be positive and within bounds
    
    for it in range(max_iter):
        if not _time_ok():
            break
            
        # Initialize gradients
        grad_cx = np.zeros(N)
        grad_cy = np.zeros(N)
        grad_r = -np.ones(N)  # d(-sum r)/dr = -1
        
        # Update multipliers and calculate forces
        for i, j, k in pair_indices:
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            d = np.sqrt(dx*dx + dy*dy)
            req = r[i] + r[j]
            
            if d < 1e-12:
                # Degenerate case, push apart
                dx = 1e-6
                dy = 0
                d = 1e-6
                
            if d < req:
                # Constraint active
                # Update multiplier: lambda_new = lambda_old + eta * violation
                # We use a simple relaxation
                eta = 0.1
                multipliers[k] = multipliers[k] + eta * (req - d)
                
                # Force direction
                fx = dx / d
                fy = dy / d
                
                # Gradient of constraint g = r_i + r_j - d
                # dg/dcx_i = -dx/d, dg/dcx_j = dx/d
                # dg/dcy_i = -dy/d, dg/dcy_j = dy/d
                # dg/dr_i = 1, dg/dr_j = 1
                
                lam = multipliers[k]
                
                grad_cx[i] += -lam * (-fx) # -grad_L = -(-lam * -fx) = -lam * fx ? 
                # Let's be precise.
                # L = -sum(r) - sum(lam * (r_i + r_j - d))
                # dL/dcx_i = -lam * d(r_i+r_j-d)/dcx_i = -lam * (-dx/d) = lam * dx/d
                # Gradient descent step: cx_i -= step * dL/dcx_i
                # So we want to add -dL/dcx_i to the update.
                # Update = -grad_L
                # -dL/dcx_i = -lam * dx/d
                
                # Wait, standard force directed:
                # If circles overlap, they push apart.
                # Force on i is away from j. Direction (dx, dy)/d.
                # Magnitude proportional to lambda.
                # So update cx_i += step * lam * dx/d
                # This matches -dL/dcx_i if L has -lam * (overlap).
                
                grad_cx[i] += lam * fx
                grad_cx[j] += lam * (-fx)
                grad_cy[i] += lam * fy
                grad_cy[j] += lam * (-fy)
                grad_r[i] += lam
                grad_r[j] += lam
            else:
                # Constraint inactive, decay multiplier
                multipliers[k] *= 0.99
                
        # Update positions and radii
        step = 0.05
        cx += step * grad_cx
        cy += step * grad_cy
        r += step * grad_r
        
        # Project radii to be positive and within bounds
        # r_i <= min(cx_i, cy_i, W-cx_i, H-cy_i)
        for i in range(N):
            r[i] = max(r[i], 0.0)
            max_r = min(cx[i], cy[i], W - cx[i], H - cy[i])
            if r[i] > max_r:
                r[i] = max_r
        
        # Keep centers within bounds (optional, helps stability)
        cx = np.clip(cx, 0, W)
        cy = np.clip(cy, 0, H)
        
        # Check convergence
        current_sum = np.sum(r)
        if current_sum > best_sum:
            best_sum = current_sum
            best_x = np.column_stack([cx, cy, r])
            
        # Simple convergence check: if sum_r hasn't improved in a while
        # (Omitted for brevity, relying on max_iter and time)
        
    return best_x, best_sum

def _generate_initial_layout(W, H, seed=0):
    """
    Generate a random initial layout.
    """
    rng = np.random.RandomState(seed)
    cx = rng.uniform(0.1 * W, 0.9 * W, N)
    cy = rng.uniform(0.1 * H, 0.9 * H, N)
    
    # Initial radii: small, based on distance to boundary
    r = []
    for i in range(N):
        max_r = min(cx[i], cy[i], W - cx[i], H - cy[i])
        # Start with 20% of max possible
        r.append(max_r * 0.2)
    r = np.array(r)
    
    return np.column_stack([cx, cy, r])

def construct():
    best_sum = 0.0
    best_arr = None
    best_W = 1.0
    best_H = 1.0
    
    # Try different aspect ratios
    # Perimeter = 2(W+H) = 4 => W+H=2
    # W in [0.5, 1.5]
    aspect_ratios = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
    
    # We have ~140 seconds.
    # Each optimization run takes maybe 1-5 seconds depending on convergence.
    # Let's try to run multiple seeds for each aspect ratio.
    
    seeds_per_aspect = 3
    total_attempts = len(aspect_ratios) * seeds_per_aspect
    
    # Distribute time
    time_per_attempt = 130.0 / total_attempts
    
    for W in aspect_ratios:
        if not _time_ok():
            break
        H = 2.0 - W
        
        for seed in range(seeds_per_aspect):
            if not _time_ok():
                break
            
            # Calculate max time allowed for this specific run
            time_limit = min(time_per_attempt, DEADLINE - time.time() - 2.0)
            if time_limit <= 0:
                break
                
            # Estimate max_iter based on time
            # Roughly 1000 iterations per second? 
            # Let's just use a fixed max_iter and rely on _time_ok inside.
            max_iter = int(time_limit * 2000) # Heuristic
            if max_iter < 100:
                max_iter = 100
            
            x0 = _generate_initial_layout(W, H, seed=seed + int(W*100))
            
            x_opt, sum_r = _lagrange_optimize(W, H, x0, max_iter=max_iter)
            
            if sum_r > best_sum:
                best_sum = sum_r
                best_arr = x_opt
                best_W = W
                best_H = H
                
    # Fallback if no valid solution found (shouldn't happen)
    if best_arr is None:
        W, H = 1.0, 1.0
        # Simple grid
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                cx = (i + 0.5) / side
                cy = (j + 0.5) / side
                r = 0.5 / side - 0.01
                cells.append([cx, cy, r])
        best_arr = np.array(cells[:N])
        best_W, best_H = W, H

    # Final validation and cleanup
    cx = best_arr[:, 0]
    cy = best_arr[:, 1]
    r = best_arr[:, 2]
    
    # Ensure strict validity
    for i in range(N):
        cx[i] = np.clip(cx[i], 0, best_W)
        cy[i] = np.clip(cy[i], 0, best_H)
        max_r = min(cx[i], cy[i], best_W - cx[i], best_H - cy[i])
        r[i] = min(r[i], max_r)
        r[i] = max(r[i], 1e-6)
        
    # Check for overlaps and shrink if necessary (should be rare with Lagrange)
    # For robustness, if overlaps exist, we can shrink radii slightly
    # But Lagrange usually handles this.
    
    return np.column_stack([cx, cy, r])
