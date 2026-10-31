"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
import random
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 26

# EVOLVE-BLOCK-START

def _objective(x, penalty_weight=500.0):
    """
    Objective function: minimize negative sum of radii + penalty for violations.
    x is a flattened array of [x_1, y_1, ..., x_N, y_N, r_1, ..., r_N]
    """
    pos = x[:2*N].reshape(N, 2)
    radii = x[2*N:]
    
    # Ensure radii are positive
    radii = np.maximum(radii, 1e-6)
    
    # 1. Boundary Violations (Smooth Squared Penalty)
    left_viol = np.maximum(0.0, radii - pos[:, 0])
    right_viol = np.maximum(0.0, pos[:, 0] - (1.0 - radii))
    bottom_viol = np.maximum(0.0, radii - pos[:, 1])
    top_viol = np.maximum(0.0, pos[:, 1] - (1.0 - radii))
    
    boundary_penalty = np.sum(left_viol**2) + np.sum(right_viol**2) + \
                       np.sum(bottom_viol**2) + np.sum(top_viol**2)
    
    # 2. Pairwise Overlap Violations (Vectorized)
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    min_dist_sq = min_dist * min_dist
    
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    # Use a smoother penalty: (violation)^2 / (min_dist^2 + epsilon)
    violation = np.maximum(0.0, min_dist_sq - dist_sq)
    denom = min_dist_sq + 1e-8
    smooth_overlap = (violation ** 2) / denom
    
    overlap_penalty = np.sum(smooth_overlap[mask])
    
    total_penalty = boundary_penalty + overlap_penalty
    
    return -np.sum(radii) + penalty_weight * total_penalty

def _boundary_constraints(x):
    """
    Boundary constraints: each circle must be inside [0,1]^2.
    Returns array of constraint values (all must be >= 0).
    """
    pos = x[:2*N].reshape(N, 2)
    radii = x[2*N:]
    
    # x_i - r_i >= 0
    left = pos[:, 0] - radii
    # 1 - x_i - r_i >= 0
    right = 1.0 - pos[:, 0] - radii
    # y_i - r_i >= 0
    bottom = pos[:, 1] - radii
    # 1 - y_i - r_i >= 0
    top = 1.0 - pos[:, 1] - radii
    
    return np.concatenate([left, right, bottom, top])

def _overlap_constraints(x):
    """
    Overlap constraints: no two circles may overlap.
    Returns array of constraint values (all must be >= 0).
    For each pair (i,j) with i<j: dist(i,j) - r_i - r_j >= 0
    """
    pos = x[:2*N].reshape(N, 2)
    radii = x[2*N:]
    
    # Compute all pairwise distances
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist = np.sqrt(dx*dx + dy*dy)
    
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    
    # Get upper triangle (i < j)
    iu = np.triu_indices(N, k=1)
    constraints = dist[iu] - min_dist[iu]
    
    return constraints

def _all_constraints(x):
    """Combine all constraints into a single array."""
    bnd = _boundary_constraints(x)
    ovl = _overlap_constraints(x)
    return np.concatenate([bnd, ovl])

def _is_valid(pos, radii, tol=1e-7):
    """Strict validity check for final result."""
    n = len(radii)
    # Check boundaries
    if np.any(pos[:, 0] - radii < -tol) or np.any(pos[:, 0] + radii > 1.0 + tol):
        return False
    if np.any(pos[:, 1] - radii < -tol) or np.any(pos[:, 1] + radii > 1.0 + tol):
        return False
        
    # Check overlaps
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist = np.sqrt(dx*dx + dy*dy)
    
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    iu = np.triu_indices(n, k=1)
    
    # dist < min_dist
    if np.any(dist[iu] < min_dist[iu] - tol):
        return False
        
    return True

def _post_process(pos, radii):
    """
    Shrink radii slightly to ensure strict validity if the optimizer
    left them on the boundary or slightly overlapping due to numerical noise.
    """
    pos = pos.copy()
    radii = radii.copy()
    
    for _ in range(100):
        valid = True
        
        # 1. Boundary fix
        for i in range(N):
            r = radii[i]
            x, y = pos[i]
            max_r = min(x, 1.0-x, y, 1.0-y)
            if r > max_r:
                radii[i] = max(max_r * 0.99999, 1e-5)
                valid = False
        
        # 2. Overlap fix
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r:
                    # Scale both radii to make them just touch with a small gap
                    k = dist / (sum_r + 1e-12) * 0.99999
                    if k < 1.0:
                        radii[i] = max(radii[i] * k, 1e-5)
                        radii[j] = max(radii[j] * k, 1e-5)
                        valid = False
                        
        if valid:
            break
            
    return pos, radii

def _initial_best_fit():
    """
    Generate a tight initial packing using a greedy best-fit algorithm.
    This creates a valid, dense configuration that serves as a strong starting point.
    Uses a combination of grid sampling and random sampling for better coverage.
    """
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # Start with a large central circle
    pos[0] = [0.5, 0.5]
    radii[0] = 0.25
    
    for i in range(1, N):
        # Try to find a position for the next circle
        # We will try to maximize the radius of the new circle at a given position
        # by checking distance to all existing circles and walls.
        
        best_x, best_y, best_r = 0.5, 0.5, 0.0
        found = False
        
        rng = np.random.RandomState(i * 1000 + 42)
        
        # Sample 3000 random points for better coverage
        for _ in range(3000):
            x = rng.uniform(0.02, 0.98)
            y = rng.uniform(0.02, 0.98)
            
            # Max radius allowed by walls
            max_r_wall = min(x, 1.0-x, y, 1.0-y)
            
            # Max radius allowed by existing circles
            max_r_circ = max_r_wall
            for j in range(i):
                dx = x - pos[j, 0]
                dy = y - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                # If we place circle i at (x,y) with radius r, and circle j has radius r_j:
                # dist >= r + r_j  =>  r <= dist - r_j
                possible_r = dist - radii[j]
                if possible_r < max_r_circ:
                    max_r_circ = possible_r
            
            if max_r_circ > best_r and max_r_circ > 0.005:
                best_r = max_r_circ
                best_x = x
                best_y = y
                found = True
        
        if not found or best_r < 0.005:
            # Fallback: place tiny circle at center
            best_x, best_y = 0.5, 0.5
            best_r = 0.005
            
        pos[i] = [best_x, best_y]
        radii[i] = best_r * 0.98  # Slight shrink to ensure no overlap in initial state
        
    return pos, radii

def _initial_hexagonal():
    """
    Generate an initial packing based on a hexagonal/lattice arrangement.
    This provides a good structural starting point for optimization.
    """
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # Create a grid of positions with hexagonal offset
    # Start with larger circles and fill in smaller ones
    
    # Define a set of candidate positions on a grid
    positions = []
    for i in range(5):
        for j in range(5):
            x = 0.1 + i * 0.2
            y = 0.1 + j * 0.2
            positions.append((x, y))
    
    # Sort by some heuristic (e.g., distance from center) to place larger circles first
    positions.sort(key=lambda p: (p[0]-0.5)**2 + (p[1]-0.5)**2)
    
    for i in range(min(N, len(positions))):
        pos[i] = positions[i]
        radii[i] = 0.15
    
    # Fill remaining with random positions
    if N > len(positions):
        rng = np.random.RandomState(42)
        for i in range(len(positions), N):
            pos[i] = rng.uniform(0.1, 0.9, 2)
            radii[i] = 0.08
    
    # Post-process to ensure validity
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _initial_random(seed):
    """Generate a random valid-ish initial state."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, (N, 2))
    radii = rng.uniform(0.04, 0.12, N)
    # Force validity by shrinking
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _vectorized_refine(pos, radii, n_iter=50, step_size=0.01):
    """
    Vectorized local refinement: try small perturbations and accept if sum of radii increases.
    This helps escape shallow local minima.
    """
    pos = pos.copy()
    radii = radii.copy()
    current_score = np.sum(radii)
    
    for it in range(n_iter):
        # Generate random perturbations for positions
        delta_pos = np.random.randn(N, 2) * step_size
        new_pos = pos + delta_pos
        
        # Try to increase radii slightly
        new_radii = radii * (1.0 + np.random.rand(N) * 0.02)
        
        # Post-process to ensure validity
        new_pos, new_radii = _post_process(new_pos, new_radii)
        
        if _is_valid(new_pos, new_radii):
            new_score = np.sum(new_radii)
            if new_score > current_score:
                pos = new_pos
                radii = new_radii
                current_score = new_score
                step_size *= 0.9  # Reduce step size on success
    
    return pos, radii

def _high_precision_polish(pos, radii, time_budget=10.0):
    """
    High-precision polish using SLSQP with tight tolerances.
    """
    start_time = time.time()
    x0 = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
    for i in range(N):
        bounds.append((1e-4, 0.5))
    
    # Use SLSQP with explicit constraints for high precision
    try:
        result = minimize(
            lambda x: -np.sum(x[2*N:]),  # Maximize sum of radii
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=[
                {'type': 'ineq', 'fun': _boundary_constraints},
                {'type': 'ineq', 'fun': _overlap_constraints}
            ],
            options={'maxiter': 1000, 'ftol': 1e-12, 'disp': False}
        )
        
        if result.success or result.x is not None:
            x_opt = result.x
            pos_opt = x_opt[:2*N].reshape(N, 2)
            radii_opt = np.maximum(x_opt[2*N:], 1e-4)
            
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
            if _is_valid(pos_pp, radii_pp):
                return pos_pp, radii_pp
    except Exception:
        pass
    
    return pos, radii

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Strategy: Multi-start with different initializations using SLSQP with constraints
    inits = []
    
    # 1. Best Fit greedy
    pos_bf, rad_bf = _initial_best_fit()
    inits.append((pos_bf, rad_bf))
    
    # 2. Hexagonal lattice
    pos_hex, rad_hex = _initial_hexagonal()
    inits.append((pos_hex, rad_hex))
    
    # 3. Random perturbations
    for i in range(3):
        pos_r, rad_r = _initial_random(100 + i)
        inits.append((pos_r, rad_r))
    
    best_pos = pos_bf.copy()
    best_radii = rad_bf.copy()
    best_score = np.sum(best_radii)
    
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
    for i in range(N):
        bounds.append((1e-4, 0.5))
    
    # Multi-stage optimization
    for idx, (pos_init, rad_init) in enumerate(inits):
        if time.time() > deadline - 20:
            break
        
        x0 = np.concatenate([pos_init.flatten(), rad_init])
        
        # Stage 1: SLSQP with constraints (primary method)
        try:
            result = minimize(
                lambda x: -np.sum(x[2*N:]),
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=[
                    {'type': 'ineq', 'fun': _boundary_constraints},
                    {'type': 'ineq', 'fun': _overlap_constraints}
                ],
                options={'maxiter': 500, 'ftol': 1e-8, 'disp': False}
            )
            x0 = result.x
        except Exception:
            pass
        
        # Stage 2: L-BFGS-B with penalty for refinement
        if time.time() < deadline - 25:
            try:
                result = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds,
                                  args=(1000.0,),
                                  options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-6})
                x0 = result.x
            except Exception:
                pass
        
        # Extract and validate
        x_opt = x0
        pos_opt = x_opt[:2*N].reshape(N, 2)
        radii_opt = np.maximum(x_opt[2*N:], 1e-4)
        
        pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
        score = np.sum(radii_pp)
        
        if _is_valid(pos_pp, radii_pp) and score > best_score:
            best_score = score
            best_pos = pos_pp.copy()
            best_radii = radii_pp.copy()
    
    # Vectorized refinement on best result
    if time.time() < deadline - 30:
        best_pos, best_radii = _vectorized_refine(best_pos, best_radii, n_iter=30, step_size=0.01)
        best_score = np.sum(best_radii)
    
    # High-precision polish
    if time.time() < deadline - 15:
        best_pos, best_radii = _high_precision_polish(best_pos, best_radii)
        best_score = np.sum(best_radii)
    
    # Final validation and post-processing
    best_pos, best_radii = _post_process(best_pos, best_radii)
    
    # Fallback if invalid
    if not _is_valid(best_pos, best_radii):
         pos = np.zeros((N, 2))
         radii = np.zeros(N)
         k = 0
         for i in range(5):
             for j in range(5):
                 if k < N-1:
                     pos[k] = [(i+0.5)/5.0, (j+0.5)/5.0]
                     radii[k] = 0.095
                     k += 1
         pos[k] = [0.5, 0.02]
         radii[k] = 0.01
         best_pos, best_radii = pos, radii
         best_pos, best_radii = _post_process(best_pos, best_radii)

    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
