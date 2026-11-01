"""Circle packing in perimeter-4 rectangle: Delaunay-based Contact Graph Optimization.

Strategy:
1. Initialize with a hexagonal-like grid pattern inside a rectangle of perimeter 4.
2. Use a smooth objective function that maximizes the sum of radii while penalizing
   overlaps quadratically. This allows the optimizer to "push" circles against each
   other and the walls without hard constraints that might cause SLSQP to fail or
   get stuck.
3. Use SLSQP to jointly optimize centers (x, y), radii (r), and the rectangle width (W).
   The height H is derived from W (H = 2 - W) to maintain perimeter 4.
4. The "soft" nature of the objective allows the topology (which circles are neighbors)
   to change naturally as the optimizer progresses, effectively performing a Delaunay
   triangulation-based relaxation.
5. After optimization, apply a strict validity check and a final "growth" polish to
   ensure maximum radii within the valid constraints.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def _objective(z, W, H):
    """
    Objective function for SLSQP.
    z = [x_0..x_{N-1}, y_0..y_{N-1}, r_0..r_{N-1}]
    Returns negative (sum r - penalty).
    """
    x = z[:N]
    y = z[N:2*N]
    r = z[2*N:]
    
    # Penalty for circles going out of bounds
    # We want x >= r, x <= W-r, y >= r, y <= H-r
    # Soft penalty: sum of violations^2
    pen_bound = 0.0
    for i in range(N):
        # Left
        if x[i] < r[i]:
            pen_bound += (r[i] - x[i])**2
        # Right
        if x[i] > W - r[i]:
            pen_bound += (x[i] - (W - r[i]))**2
        # Bottom
        if y[i] < r[i]:
            pen_bound += (r[i] - y[i])**2
        # Top
        if y[i] > H - r[i]:
            pen_bound += (y[i] - (H - r[i]))**2
            
    # Penalty for overlaps
    # We want dist(i,j) >= r_i + r_j
    pen_overlap = 0.0
    for i in range(N):
        for j in range(i+1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx*dx + dy*dy
            min_dist = r[i] + r[j]
            if dist_sq < min_dist*min_dist:
                # Approximate penalty: (min_dist - dist)^2
                # To avoid sqrt, we can use a differentiable approximation or just use sqrt
                # sqrt is fine for SLSQP
                dist = np.sqrt(dist_sq + 1e-12)
                overlap = min_dist - dist
                pen_overlap += overlap*overlap
                
    # Objective: Maximize sum(r) - Penalty
    # We return negative for minimization
    val = np.sum(r) - 50.0 * (pen_bound + pen_overlap)
    return -val


def _constraints(z, W, H):
    """
    Hard constraints for SLSQP to keep the solution in a sane region.
    We don't want to rely solely on the penalty, but SLSQP handles
    linear/quadratic constraints well.
    """
    x = z[:N]
    y = z[N:2*N]
    r = z[2*N:]
    
    cons = []
    
    # Bounds: r > 0
    for i in range(N):
        cons.append({'type': 'ineq', 'fun': lambda z, i=i: z[2*N+i]})
        
    # Bounds: x within [0, W], y within [0, H] (loose, penalty handles tightness)
    # Actually, let's let the penalty handle the tight "touching" and just keep them
    # inside the rectangle to prevent wild divergence.
    for i in range(N):
        cons.append({'type': 'ineq', 'fun': lambda z, i=i: z[i]}) # x >= 0
        cons.append({'type': 'ineq', 'fun': lambda z, i=i: W - z[i]}) # x <= W
        cons.append({'type': 'ineq', 'fun': lambda z, i=i: z[N+i]}) # y >= 0
        cons.append({'type': 'ineq', 'fun': lambda z, i=i: H - z[N+i]}) # y <= H
        
    return cons


def _is_valid(cx, cy, r, W, H, tol=1e-8):
    """Strict validity check."""
    if W <= 0 or H <= 0:
        return False
    if np.any(cx - r < -tol) or np.any(cy - r < -tol):
        return False
    if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - tol:
                return False
    return True


def _generate_initial_grid(rows, cols, W, H):
    """Generate initial positions in a grid pattern."""
    x = np.zeros(N)
    y = np.zeros(N)
    r = np.zeros(N)
    
    # Calculate cell dimensions
    cell_w = W / cols
    cell_h = H / rows
    
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= N:
                break
            # Center of cell
            cx = (j + 0.5) * cell_w
            cy = (i + 0.5) * cell_h
            # Initial radius: half of min dimension, with a safety factor
            rad = 0.45 * min(cell_w, cell_h)
            
            x[idx] = cx
            y[idx] = cy
            r[idx] = rad
            idx += 1
            
    # If we didn't fill all N, place remaining in a corner with tiny radius
    if idx < N:
        x[idx:] = 0.1
        y[idx:] = 0.1
        r[idx:] = 0.01
        
    return x, y, r


def _polish_and_validate(x, y, r, W, H, deadline):
    """
    Take a potentially invalid or suboptimal configuration and try to fix it
    by shrinking radii and relaxing positions, then growing them back.
    """
    # First, ensure validity by shrinking if necessary
    if not _is_valid(x, y, r, W, H):
        # Binary search for a valid scale
        scale_lo = 0.0
        scale_hi = 1.0
        for _ in range(20):
            mid = (scale_lo + scale_hi) / 2
            r_test = r * mid
            if _is_valid(x, y, r_test, W, H):
                scale_lo = mid
            else:
                scale_hi = mid
        r = r * scale_lo
        
    if not _is_valid(x, y, r, W, H):
        return None

    # Now, try to grow radii uniformly
    # Use a simple gradient-free growth: try to increase all r by a small factor
    # and check validity. If invalid, use force relaxation to move centers.
    
    # Force relaxation to push circles into corners/walls
    dt = 0.005
    for _ in range(100):
        if time.time() > deadline:
            break
        fx = np.zeros(N)
        fy = np.zeros(N)
        
        # Wall forces
        for i in range(N):
            if x[i] < r[i]:
                fx[i] += 10.0
            if x[i] > W - r[i]:
                fx[i] -= 10.0
            if y[i] < r[i]:
                fy[i] += 10.0
            if y[i] > H - r[i]:
                fy[i] -= 10.0
                
        # Overlap forces
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy + 1e-12)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    overlap = min_dist - dist
                    force = overlap * 5.0
                    fx[i] += (dx/dist) * force
                    fy[i] += (dy/dist) * force
                    fx[j] -= (dx/dist) * force
                    fy[j] -= (dy/dist) * force
                    
        x += dt * fx
        y += dt * fy
        
        # Clip to bounds
        x = np.clip(x, r, W - r)
        y = np.clip(y, r, H - r)
        
        if not _is_valid(x, y, r, W, H):
             # If we broke validity, revert or shrink slightly
             # Simplest: revert last step
             x -= dt * fx
             y -= dt * fy
             x = np.clip(x, r, W - r)
             y = np.clip(y, r, H - r)
             # If still invalid, break
             if not _is_valid(x, y, r, W, H):
                 break

    # Try to grow radii
    for _ in range(50):
        if time.time() > deadline:
            break
        r_trial = r * 1.001
        if _is_valid(x, y, r_trial, W, H):
            r = r_trial
        else:
            break
            
    if _is_valid(x, y, r, W, H):
        return x, y, r
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_value = 0.0
    best_result = None
    
    # Candidate aspect ratios (W, where H = 2 - W)
    # Perimeter 2(W+H) = 4 => W+H=2.
    # Square is W=1, H=1.
    # Rectangles: 0.8x1.2, 0.7x1.3, etc.
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.42, 0.6, 1.66, 0.9, 1.11]
    
    # Grid topologies for initialization
    # 21 circles.
    # 4x6 = 24 (drop 3)
    # 3x7 = 21 (exact)
    # 5x5 = 25 (drop 4)
    # 2x11 = 22 (drop 1)
    grid_configs = [
        (3, 7),
        (4, 6),
        (5, 5),
        (2, 11),
        (7, 3),
        (6, 4),
        (1, 21) # Degenerate, but might work for very wide/short
    ]
    
    # De-duplicate
    seen = set()
    unique_configs = []
    for cfg in grid_configs:
        if cfg not in seen and cfg[0] * cfg[1] >= N:
            seen.add(cfg)
            unique_configs.append(cfg)
            
    seed_counter = 0
    for W in aspect_ratios:
        if time.time() > deadline - 20:
            break
        H = 2.0 - W
        if H < 0.2:
            continue
            
        for (rows, cols) in unique_configs:
            if time.time() > deadline - 15:
                break
                
            # Skip if aspect ratio is too extreme for the grid
            cell_w = W / cols
            cell_h = H / rows
            if min(cell_w, cell_h) < 0.05:
                continue
                
            x0, y0, r0 = _generate_initial_grid(rows, cols, W, H)
            
            # Add small jitter to break symmetry
            rng = np.random.default_rng(seed_counter)
            x0 = x0 + rng.uniform(-0.01, 0.01, size=N)
            y0 = y0 + rng.uniform(-0.01, 0.01, size=N)
            r0 = r0 * 0.9 # Start slightly smaller to allow growth
            
            # Ensure initial validity
            if not _is_valid(x0, y0, r0, W, H):
                 # Shrink to valid
                 scale = 0.5
                 while not _is_valid(x0, y0, r0*scale, W, H) and scale > 0.1:
                     scale *= 0.9
                 r0 = r0 * scale
                 if not _is_valid(x0, y0, r0, W, H):
                     continue
                     
            z0 = np.concatenate([x0, y0, r0])
            
            # Bounds for SLSQP
            bounds = []
            for i in range(N):
                bounds.append((0, W)) # x
            for i in range(N):
                bounds.append((0, H)) # y
            for i in range(N):
                bounds.append((0.01, min(W, H)/2)) # r
                
            # Constraints
            cons = _constraints(z0, W, H)
            
            start_opt = time.time()
            remaining = deadline - start_opt - 5 # Leave 5s for polish
            if remaining < 2:
                continue
                
            maxiter = int(min(200, max(50, remaining * 20)))
            
            try:
                res = minimize(
                    _objective, 
                    z0, 
                    args=(W, H),
                    method='SLSQP', 
                    bounds=bounds, 
                    constraints=cons,
                    options={'maxiter': maxiter, 'ftol': 1e-12}
                )
            except Exception:
                continue
                
            z_opt = res.x
            x_opt = z_opt[:N]
            y_opt = z_opt[N:2*N]
            r_opt = z_opt[2*N:]
            
            # Clip to ensure strict bounds
            x_opt = np.clip(x_opt, 0, W)
            y_opt = np.clip(y_opt, 0, H)
            r_opt = np.clip(r_opt, 0.01, min(W, H)/2)
            
            # Polish and validate
            polished = _polish_and_validate(x_opt, y_opt, r_opt, W, H, deadline - 5)
            
            if polished is not None:
                xp, yp, rp = polished
                val = float(np.sum(rp))
                if val > best_value:
                    best_value = val
                    best_result = (xp, yp, rp, W, H)
                    
            seed_counter += 1
            
    if best_result is None:
        # Fallback: uniform grid
        W, H = 1.0, 1.0
        x, y, r = _generate_initial_grid(3, 7, W, H)
        # Shrink to ensure validity
        scale = 1.0
        while not _is_valid(x, y, r*scale, W, H) and scale > 0.1:
            scale *= 0.95
        r = r * scale
        return np.column_stack((x, y, r))
        
    x, y, r, W, H = best_result
    
    # Final safety check
    if not _is_valid(x, y, r, W, H):
         # Shrink
         scale = 1.0
         while not _is_valid(x, y, r*scale, W, H) and scale > 0.1:
             scale *= 0.95
         r = r * scale
         
    return np.column_stack((x, y, r))
