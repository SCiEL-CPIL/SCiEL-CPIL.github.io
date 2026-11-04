"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import random

def _eval_packing(xy, r):
    """
    Evaluate a packing: returns (total_radius, is_valid).
    Valid means all circles inside [0,1]^2 and disjoint interiors.
    """
    n = len(xy)
    # Check walls
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        if x - ri < -1e-9 or 1.0 - x - ri < -1e-9 or y - ri < -1e-9 or 1.0 - y - ri < -1e-9:
            return 0.0, False
    # Check pairwise distances
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-9:
                return 0.0, False
    return np.sum(r), True

def _init_hexagonal(n):
    """
    Initialize circles in a hexagonal-like pattern to maximize initial packing density.
    Returns xy (n, 2) and r (n,).
    """
    # Approximate hexagonal packing density is pi / (2 * sqrt(3)) ≈ 0.9069
    # Area of square = 1. Total circle area = sum(pi * r_i^2).
    # If all r equal: n * pi * r^2 = density * 1 => r = sqrt(density / (n * pi))
    density = 0.9069
    r0 = np.sqrt(density / (n * np.pi))
    
    # Create a grid of candidate positions in a hex lattice
    # We need to fit n circles.
    # Side length of hex unit cell is 2*r0.
    # Let's generate a large grid and pick the best n positions that fit in [0,1]x[0,1]
    
    positions = []
    # Hex lattice spacing
    dx = 2 * r0
    dy = dx * np.sqrt(3) / 2
    
    # Start from center to keep them inside
    cx, cy = 0.5, 0.5
    
    # Generate grid indices
    max_range = int(np.ceil(1.0 / dx)) + 2
    
    candidates = []
    for i in range(-max_range, max_range + 1):
        for j in range(-max_range, max_range + 1):
            x = cx + i * dx + (j % 2) * (dx / 2)
            y = cy + j * dy
            # Check if inside with margin
            if r0 <= x <= 1.0 - r0 and r0 <= y <= 1.0 - r0:
                candidates.append([x, y])
                
    # If we don't have enough, try denser or different offset
    if len(candidates) < n:
        # Fallback to square grid
        side = int(np.ceil(np.sqrt(n)))
        for i in range(side):
            for j in range(side):
                if len(candidates) < n:
                    x = (i + 0.5) / side
                    y = (j + 0.5) / side
                    candidates.append([x, y])
                    
    # Select n candidates that are most central (to allow growth)
    # Sort by distance from center
    candidates = np.array(candidates)
    dists = np.sqrt((candidates[:, 0] - 0.5)**2 + (candidates[:, 1] - 0.5)**2)
    indices = np.argsort(dists)
    selected = candidates[indices[:n]]
    
    xy = selected
    r = np.full(n, r0)
    return xy, r

def _repulsion_force(xy, r, wall_margin=0.0, strength=1.0):
    """
    Calculate repulsive forces on circles due to overlaps and wall violations.
    Returns forces (n, 2).
    """
    n = len(xy)
    forces = np.zeros((n, 2))
    
    # Pairwise repulsion
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = r[i] + r[j]
            
            if dist < min_dist:
                # Overlap exists
                # Force magnitude proportional to overlap depth
                overlap = min_dist - dist
                if dist > 1e-6:
                    # Normalize direction
                    fx = (dx / dist) * overlap * strength
                    fy = (dy / dist) * overlap * strength
                else:
                    # Random direction if coincident
                    fx = random.uniform(-1, 1) * overlap * strength
                    fy = random.uniform(-1, 1) * overlap * strength
                
                forces[i, 0] += fx
                forces[i, 1] += fy
                forces[j, 0] -= fx
                forces[j, 1] -= fy
                
    # Wall repulsion
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        
        # Left wall
        if x < ri + wall_margin:
            forces[i, 0] += (ri + wall_margin - x) * strength
        # Right wall
        if x > 1.0 - ri - wall_margin:
            forces[i, 0] -= (x - (1.0 - ri - wall_margin)) * strength
        # Bottom wall
        if y < ri + wall_margin:
            forces[i, 1] += (ri + wall_margin - y) * strength
        # Top wall
        if y > 1.0 - ri - wall_margin:
            forces[i, 1] -= (y - (1.0 - ri - wall_margin)) * strength
            
    return forces

def _solve_jammed(xy, r, deadline):
    """
    Iteratively adjust positions to resolve overlaps and maximize radii.
    This is a form of iterative geometric relaxation.
    """
    n = len(xy)
    current_xy = xy.copy()
    current_r = r.copy()
    
    start_time = time.time()
    # Adaptive step size
    step_size = 0.05
    damping = 0.9
    
    max_iter = 2000
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
            
        # Calculate forces
        forces = _repulsion_force(current_xy, current_r, wall_margin=0.0, strength=1.0)
        
        # Update positions
        # Limit step size to avoid instability
        max_force = np.max(np.linalg.norm(forces, axis=1))
        if max_force > 1e-6:
            # Normalize forces to unit vectors, then scale by step_size
            # Actually, using the force magnitude directly with a small step is better
            # to handle varying overlap depths.
            moves = forces * step_size
            
            # Limit individual move magnitude to prevent jumping over
            # other circles or walls
            move_mags = np.linalg.norm(moves, axis=1)
            max_move = 0.02
            for i in range(n):
                if move_mags[i] > max_move:
                    moves[i] *= (max_move / move_mags[i])
                    
            current_xy += moves
            
            # Clamp to bounds (with safety margin for radius)
            for i in range(n):
                ri = current_r[i]
                current_xy[i, 0] = np.clip(current_xy[i, 0], ri, 1.0 - ri)
                current_xy[i, 1] = np.clip(current_xy[i, 1], ri, 1.0 - ri)
                
        # Check convergence
        if np.max(np.linalg.norm(forces, axis=1)) < 1e-4:
            break
            
        # Decay step size slightly
        step_size *= damping
        
    # After jamming, try to expand radii
    # We can do a few iterations of "expand radius, resolve overlaps"
    for _ in range(20):
        if time.time() > deadline:
            break
            
        # Calculate max possible radius for each circle given current positions
        # and other circles fixed? No, we want to expand all.
        # Simple approach: increase all radii by a small epsilon, then run force solver again.
        epsilon = 0.001
        current_r += epsilon
        
        # Run a short force solver to fix overlaps
        short_deadline = time.time() + 0.5
        for _ in range(50):
            if time.time() > short_deadline:
                break
            forces = _repulsion_force(current_xy, current_r, wall_margin=0.0, strength=1.0)
            moves = forces * 0.05
            current_xy += moves
            for i in range(n):
                ri = current_r[i]
                current_xy[i, 0] = np.clip(current_xy[i, 0], ri, 1.0 - ri)
                current_xy[i, 1] = np.clip(current_xy[i, 1], ri, 1.0 - ri)
                
    return current_xy, current_r

def _smooth_objective(params, n, temperature=0.01, penalty_weight=1000.0):
    """
    Smooth objective using log-sum-exp for soft min/max.
    """
    params = np.asarray(params)
    xy = params[0::3].reshape(-1, 2)
    r = params[1::3].reshape(-1)
    
    r = np.maximum(r, 0.001)
    
    # Wall violations
    wall_violations = np.zeros(n)
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        vals = np.array([ri - x, ri - (1.0 - x), ri - y, ri - (1.0 - y), 0.0])
        m = np.max(vals)
        wall_violations[i] = m + temperature * np.log(np.sum(np.exp((vals - m) / temperature)))
    
    # Overlap violations
    overlap_violations = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = r[i] + r[j]
            vals = np.array([min_dist - dist, 0.0])
            m = np.max(vals)
            sv = m + temperature * np.log(np.sum(np.exp((vals - m) / temperature)))
            overlap_violations.append(sv)
    
    overlap_violations = np.array(overlap_violations)
    
    penalty = penalty_weight * (np.sum(wall_violations**2) + np.sum(overlap_violations**2))
    total_radius = np.sum(r)
    
    return -total_radius + penalty

def _gradient_polish(xy, r, deadline):
    """
    Use L-BFGS-B with smooth penalty terms to polish the packing.
    """
    n = len(xy)
    
    params = np.zeros(3 * n)
    params[0::3] = xy[:, 0]
    params[1::3] = xy[:, 1]
    params[2::3] = r
    
    bounds = []
    for i in range(n):
        bounds.append((0.001, 0.999))
        bounds.append((0.001, 0.999))
        bounds.append((0.001, 0.4))
    
    best_params = params.copy()
    best_total, best_valid = _eval_packing(xy, r)
    
    # Run L-BFGS-B
    try:
        result = minimize(
            _smooth_objective,
            params,
            args=(n, 0.01, 1000.0),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'maxfun': 3000, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        opt_params = result.x
        opt_xy = opt_params[0::3].reshape(-1, 2)
        opt_r = np.maximum(opt_params[1::3], 0.001)
        
        total, valid = _eval_packing(opt_xy, opt_r)
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_params = opt_params.copy()
            
    except Exception:
        pass
    
    final_xy = best_params[0::3].reshape(-1, 2)
    final_r = np.maximum(best_params[1::3], 0.001)
    
    return final_xy, final_r, best_total, best_valid

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_xy = None
    best_r = None
    best_total = 0.0
    best_valid = False
    
    # 1. Hexagonal Initialization + Jamming
    xy, r = _init_hexagonal(N)
    
    # Run jamming solver
    jam_deadline = deadline - 30.0
    j_xy, j_r = _solve_jammed(xy, r, jam_deadline)
    
    total, valid = _eval_packing(j_xy, j_r)
    if valid:
        best_total = total
        best_valid = True
        best_xy = j_xy
        best_r = j_r
    else:
        # If jamming failed to produce a valid packing (shouldn't happen with clamping, but check)
        # Fallback to simple grid
        side = int(np.ceil(np.sqrt(N)))
        xy = np.zeros((N, 2))
        r = np.zeros(N)
        for i in range(N):
            gx = i % side
            gy = i // side
            xy[i, 0] = (gx + 0.5) / side
            xy[i, 1] = (gy + 0.5) / side
            r[i] = 0.4 / side
        total, valid = _eval_packing(xy, r)
        if valid:
            best_total = total
            best_valid = True
            best_xy = xy
            best_r = r
    
    # 2. Gradient Polish
    if time.time() < deadline - 10.0:
        g_xy, g_r, g_total, g_valid = _gradient_polish(best_xy, best_r, deadline - 5.0)
        if g_valid and g_total > best_total:
            best_total = g_total
            best_valid = True
            best_xy = g_xy
            best_r = g_r
    
    # Final validation and clipping
    n = len(best_xy)
    result = np.zeros((n, 3))
    for i in range(n):
        x, y, ri = best_xy[i, 0], best_xy[i, 1], best_r[i]
        ri = min(ri, x, 1.0 - x, y, 1.0 - y)
        ri = max(ri, 0.001)
        result[i, 2] = ri
        result[i, 0] = np.clip(x, ri, 1.0 - ri)
        result[i, 1] = np.clip(y, ri, 1.0 - ri)
    
    # Final check
    final_xy = result[:, :2]
    final_r = result[:, 2]
    total, valid = _eval_packing(final_xy, final_r)
    if not valid:
        # Fallback to simple grid
        side = int(np.ceil(np.sqrt(N)))
        result = np.zeros((N, 3))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = 0.4 / side
    
    return result
# EVOLVE-BLOCK-END
