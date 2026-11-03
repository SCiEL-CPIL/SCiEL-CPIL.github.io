"""Differentiable optimization for packing 50 equal circles in a unit circle."""
import numpy as np
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import time


def _hexagonal_initialization(n_points=50):
    """Generate initial positions on a hexagonal lattice inside the unit circle."""
    points = []
    # Generate a dense hexagonal lattice
    spacing = 0.15
    max_coord = 1.5
    x_vals = np.arange(-max_coord, max_coord + spacing, spacing)
    y_spacing = spacing * np.sqrt(3) / 2
    y_vals = np.arange(-max_coord, max_coord + y_spacing, y_spacing)
    
    for i, x in enumerate(x_vals):
        y_offset = (i % 2) * y_spacing
        for y in y_vals:
            y_pos = y + y_offset
            r = np.hypot(x, y_pos)
            if r < 0.9:  # Keep well inside
                points.append([x, y_pos])
    
    points = np.array(points)
    if len(points) < n_points:
        # Fallback: random points
        return np.random.randn(n_points, 2) * 0.5
    
    # Select the n_points closest to center, but with some diversity
    dists = np.linalg.norm(points, axis=1)
    # Sort by distance and take first n_points
    idx = np.argsort(dists)[:n_points]
    return points[idx]


def _packing_objective(params, n=N):
    """
    Objective function to minimize: negative of the minimum radius.
    We want to maximize r = min(min_pair_dist/2, min_boundary_dist)
    So we minimize -r.
    
    To make this differentiable, we use a soft minimum.
    """
    centers = params.reshape(n, 2)
    
    # Pairwise distances
    dists = cdist(centers, centers)
    np.fill_diagonal(dists, np.inf)
    
    # Boundary distances: 1 - |c_i|
    boundary_dists = 1.0 - np.linalg.norm(centers, axis=1)
    
    # We want r = min(min_pair_dist / 2, min_boundary_dist)
    # Let's compute all constraint values:
    # For each pair: dist_ij / 2
    # For each point: 1 - |c_i|
    
    # Get all pairwise distances (upper triangle)
    triu_indices = np.triu_indices(n, k=1)
    pair_dists = dists[triu_indices] / 2.0
    
    # Combine all constraints
    all_constraints = np.concatenate([pair_dists, boundary_dists])
    
    # Use soft minimum with temperature
    # soft_min(x, t) = -t * log(sum(exp(-x_i / t)))
    t = 0.05  # temperature parameter
    # For numerical stability
    min_val = np.min(all_constraints)
    exp_vals = np.exp(-(all_constraints - min_val) / t)
    soft_min = min_val - t * np.log(np.sum(exp_vals))
    
    return -soft_min  # We want to maximize, so minimize negative


def _packing_objective_grad(params, n=N):
    """Gradient of the objective function using finite differences or analytical."""
    # Use numerical gradient for simplicity and reliability
    eps = 1e-7
    grad = np.zeros_like(params)
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += eps
        f_plus = _packing_objective(params_plus, n)
        
        params_minus = params.copy()
        params_minus[i] -= eps
        f_minus = _packing_objective(params_minus, n)
        
        grad[i] = (f_plus - f_minus) / (2 * eps)
    return grad


def construct():
    """
    Construct centers of 50 equal circles inside the unit circle
    to maximize the common radius.
    """
    np.random.seed(42)
    
    # Initial positions from hexagonal lattice
    x0 = _hexagonal_initialization(N)
    
    # Flatten for optimizer
    x0_flat = x0.flatten()
    
    # Use L-BFGS-B with bounds to keep points inside the unit circle
    # Bounds: each coordinate in [-1, 1], but we'll rely on the objective
    # to keep them inside. Actually, let's add soft bounds.
    
    best_x = x0_flat.copy()
    best_obj = _packing_objective(best_x, N)
    
    # Try multiple initializations and local optimizations
    n_restarts = 3
    for restart in range(n_restarts):
        if restart > 0:
            # Perturb the initial solution
            x0_flat = best_x + np.random.randn(len(best_x)) * 0.1
            # Project back: ensure roughly inside
            centers = x0_flat.reshape(N, 2)
            norms = np.linalg.norm(centers, axis=1)
            mask = norms > 0.95
            centers[mask] *= 0.95 / norms[mask]
            x0_flat = centers.flatten()
        
        result = minimize(
            _packing_objective,
            x0_flat,
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if result.fun < best_obj:
            best_obj = result.fun
            best_x = result.x
    
    centers = best_x.reshape(N, 2)
    
    # Final cleanup: ensure all centers are strictly inside the unit circle
    norms = np.linalg.norm(centers, axis=1)
    mask = norms >= 1.0
    if np.any(mask):
        centers[mask] *= 0.999 / norms[mask]
    
    # Verify validity
    # Check no coincident centers
    dists = cdist(centers, centers)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Check all inside
    min_boundary = np.min(1.0 - np.linalg.norm(centers, axis=1))
    
    # The radius is min(min_pair_dist/2, min_boundary)
    # If invalid, fall back to hexagonal
    if min_pair_dist <= 0 or min_boundary <= 0 or not np.all(np.isfinite(centers)):
        # Fallback to simple hexagonal
        return _hexagonal_initialization(N)
    
    return centers
# EVOLVE-BLOCK-END
