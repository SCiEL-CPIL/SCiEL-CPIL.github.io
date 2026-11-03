"""Optimize 50 equal circles in a unit square via SLSQP with epigraph formulation."""
import numpy as np
import time
from scipy.optimize import minimize
import itertools

N = 50

# EVOLVE-BLOCK-START

def _compute_radius(centers):
    """Compute the minimum radius for given centers (all constraints)."""
    centers = np.asarray(centers)
    # Distance to walls: min over all points and both axes of (0.5 - |coord|)
    wall_margin = np.min(0.5 - np.abs(centers))
    # Pairwise distances
    # For 50 points, 1225 pairs - manageable
    dists = []
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(centers[i] - centers[j])
            dists.append(d)
    min_pair_dist = min(dists) if dists else 1.0
    return min(wall_margin, min_pair_dist / 2.0)


def _objective_and_grad(x, t):
    """
    Objective: maximize t (the radius).
    We minimize -t subject to constraints.
    x = [centers (2N), t]
    """
    centers = x[:2 * N].reshape(N, 2)
    t = x[2 * N]
    
    # Compute all pairwise distances and wall margins
    # For SLSQP, we need smooth constraints. Use the actual min as constraint.
    # The objective is -t, and we have constraints that all distances >= 2t and wall margins >= t.
    
    # Actually, for SLSQP we formulate:
    # Minimize: -t
    # Constraints:
    #   For each pair (i,j): |c_i - c_j| - 2*t >= 0
    #   For each point i and axis k: 0.5 - |c_ik| - t >= 0
    
    # But this is non-smooth at c_ik = 0. Let's use a different approach.
    # We'll use a penalty-free approach: minimize -t with inequality constraints.
    
    # For the gradient, we need to be careful. Let's just return the objective value.
    # SLSQP can handle the constraints separately.
    return -t


def _make_constraints():
    """Create constraint functions for SLSQP."""
    constraints = []
    
    # Pairwise distance constraints: |c_i - c_j| >= 2t
    for i in range(N):
        for j in range(i + 1, N):
            def pair_constraint(x, i=i, j=j):
                ci = x[2 * i:2 * i + 2]
                cj = x[2 * j:2 * j + 2]
                t = x[2 * N]
                return np.linalg.norm(ci - cj) - 2 * t
            constraints.append({'type': 'ineq', 'fun': pair_constraint})
    
    # Wall constraints: 0.5 - |c_ik| >= t
    for i in range(N):
        for k in range(2):
            def wall_constraint(x, i=i, k=k):
                c_ik = x[2 * i + k]
                t = x[2 * N]
                return 0.5 - abs(c_ik) - t
            constraints.append({'type': 'ineq', 'fun': wall_constraint})
    
    return constraints


def _initial_grid():
    """Create a good initial grid configuration."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    grid = grid[:N]
    # Estimate initial radius
    r = _compute_radius(grid)
    return grid, r


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Initial grid
    centers, r_init = _initial_grid()
    best_centers = centers.copy()
    best_radius = r_init
    
    # Flatten: [x1, y1, x2, y2, ..., xN, yN, t]
    x0 = np.concatenate([centers.flatten(), [r_init]])
    
    # Create constraints
    constraints = _make_constraints()
    
    # Bounds: centers in [-0.5, 0.5], t in [0, 0.5]
    bounds = [(-0.5, 0.5)] * (2 * N) + [(0.0, 0.5)]
    
    # Try SLSQP optimization
    try:
        result = minimize(
            _objective_and_grad,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 500, 'ftol': 1e-12}
        )
        
        elapsed = time.time() - start_time
        if result.success and elapsed < deadline:
            new_centers = result.x[:2 * N].reshape(N, 2)
            new_t = result.x[2 * N]
            # Verify validity
            new_radius = _compute_radius(new_centers)
            if new_radius > best_radius:
                best_centers = new_centers.copy()
                best_radius = new_radius
        
        # If we have time, do a second refinement pass with tighter tolerance
        elapsed = time.time() - start_time
        if elapsed < deadline - 10:
            # Use the best result as new starting point
            x1 = np.concatenate([best_centers.flatten(), [best_radius]])
            try:
                result2 = minimize(
                    _objective_and_grad,
                    x1,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 300, 'ftol': 1e-14}
                )
                if result2.success:
                    new_centers2 = result2.x[:2 * N].reshape(N, 2)
                    new_radius2 = _compute_radius(new_centers2)
                    if new_radius2 > best_radius:
                        best_centers = new_centers2.copy()
                        best_radius = new_radius2
            except Exception:
                pass
    except Exception:
        # If SLSQP fails, fall back to the grid
        pass
    
    # Final validation
    elapsed = time.time() - start_time
    if elapsed < deadline:
        # Ensure all centers are strictly inside
        for i in range(N):
            for k in range(2):
                if best_centers[i, k] <= -0.5:
                    best_centers[i, k] = -0.499999
                elif best_centers[i, k] >= 0.5:
                    best_centers[i, k] = 0.499999
    
    return best_centers

# EVOLVE-BLOCK-END
