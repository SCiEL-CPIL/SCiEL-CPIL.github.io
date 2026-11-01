"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_score(points):
    """Compute the packing score efficiently."""
    n = len(points)
    if n < 2:
        return 0.0
    
    # Use broadcasting for pairwise distances
    dx = points[:, np.newaxis, 0] - points[np.newaxis, :, 0]
    dy = points[:, np.newaxis, 1] - points[np.newaxis, :, 1]
    dists_sq = dx**2 + dy**2
    np.fill_diagonal(dists_sq, np.inf)
    min_pair = np.sqrt(np.min(dists_sq)) / 2.0
    
    radii = np.hypot(points[:, 0], points[:, 1])
    min_boundary = 1.0 - np.max(radii)
    
    return min(min_pair, min_boundary)


def _optimize(points, max_time=60.0, lr_init=0.002, n_iter_max=3000):
    """Efficient repulsion-based optimization."""
    start = time.time()
    best_score = _compute_score(points)
    best_points = points.copy()
    n = len(points)
    lr = lr_init
    
    for it in range(n_iter_max):
        if time.time() - start > max_time:
            break
        
        # Compute forces efficiently
        dx = points[:, np.newaxis, 0] - points[np.newaxis, :, 0]
        dy = points[:, np.newaxis, 1] - points[np.newaxis, :, 1]
        dists_sq = dx**2 + dy**2
        np.fill_diagonal(dists_sq, 1.0)
        dists = np.sqrt(dists_sq)
        
        # Repulsion: force ~ 1/d^2 in direction away
        inv_dists_sq = 1.0 / dists_sq
        np.fill_diagonal(inv_dists_sq, 0.0)
        
        fx = np.sum(dx * inv_dists_sq, axis=1)
        fy = np.sum(dy * inv_dists_sq, axis=1)
        
        # Boundary force
        radii = np.hypot(points[:, 0], points[:, 1])
        boundary_mask = radii > 0.85
        if np.any(boundary_mask):
            b_force = (radii[boundary_mask] - 0.85) * 5.0
            fx[boundary_mask] -= points[boundary_mask, 0] / radii[boundary_mask] * b_force
            fy[boundary_mask] -= points[boundary_mask, 1] / radii[boundary_mask] * b_force
        
        # Update
        points += lr * np.column_stack([fx, fy])
        
        # Project inside circle
        radii = np.hypot(points[:, 0], points[:, 1])
        too_far = radii > 0.999
        if np.any(too_far):
            points[too_far] *= 0.999 / radii[too_far, np.newaxis]
        
        # Adaptive learning rate
        lr *= 0.999
        
        score = _compute_score(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()
    
    return best_points, best_score


def _hex_lattice_init(spacing):
    """Generate hexagonal lattice points inside unit disk."""
    k = int(1.0 / spacing) + 3
    rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
            for i in range(-k, k + 1) for j in range(-k, k + 1)]
    points = np.array(rows)
    radius = np.hypot(points[:, 0], points[:, 1])
    mask = radius <= 1.0 - spacing / 2.0
    points = points[mask]
    
    if len(points) < N:
        return None
    
    order = np.argsort(radius, kind="stable")
    return points[order[:N]]


def construct():
    start = time.time()
    time_limit = 105.0  # Safety margin for 150s limit
    
    best_points = None
    best_score = 0.0
    
    # Try multiple initial spacings
    base_spacing = 2.0 / (np.sqrt(N) + 1.0)
    spacings = [base_spacing * f for f in [1.0, 0.92, 1.08, 0.85, 1.15]]
    
    for spacing in spacings:
        elapsed = time.time() - start
        if elapsed > time_limit - 10.0:
            break
        
        points = _hex_lattice_init(spacing)
        if points is None:
            continue
        
        # Small perturbation to break symmetry
        np.random.seed(42)
        points = points + np.random.normal(0, spacing * 0.005, size=points.shape)
        
        remaining = time_limit - (time.time() - start) - 5.0
        if remaining < 5.0:
            break
        
        opt_points, score = _optimize(points, max_time=remaining)
        
        if score > best_score:
            best_score = score
            best_points = opt_points
    
    if best_points is None:
        # Fallback
        spacing = base_spacing
        while True:
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            points = points[radius <= 1.0 - spacing / 2.0]
            if len(points) >= N:
                return points[np.argsort(radius, kind="stable")[:N]]
            spacing *= 0.98
    
    return best_points
# EVOLVE-BLOCK-END

