"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.distance import pdist
import time


def _compute_radius(centres):
    """Compute the maximum common radius for a set of centres."""
    n = centres.shape[0]
    if n < 2:
        return 1.0 - np.linalg.norm(centres[0])
    dists = pdist(centres)
    min_pair = np.min(dists) / 2.0
    boundary = 1.0 - np.linalg.norm(centres, axis=1)
    min_boundary = np.min(boundary)
    return min(min_pair, min_boundary)


def _generate_initial_config(n, seed=0):
    """Generate a diverse initial configuration inside the unit disk."""
    rng = np.random.default_rng(seed)
    
    # Strategy 1: Hexagonal lattice with random offset
    spacing = 2.0 / (np.sqrt(n) + 2.0)
    k = int(1.0 / spacing) + 3
    rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
            for i in range(-k, k + 1) for j in range(-k, k + 1)]
    points = np.array(rows)
    radius = np.hypot(points[:, 0], points[:, 1])
    points = points[radius <= 0.98]
    
    if len(points) >= n:
        # Add small random perturbation
        points = points + rng.normal(0, spacing * 0.1, size=points.shape)
        # Keep only points inside disk
        radius = np.hypot(points[:, 0], points[:, 1])
        points = points[radius <= 0.99]
        if len(points) >= n:
            return points[:n]
    
    # Strategy 2: Concentric rings with random angular offsets
    points = []
    # Center point
    points.append([0.0, 0.0])
    
    # Add rings
    ring_radii = np.linspace(0.1, 0.95, 12)
    for r_ring in ring_radii:
        n_points = int(2 * np.pi * r_ring / spacing)
        if n_points < 3:
            n_points = 3
        angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False) + rng.uniform(0, 2 * np.pi)
        for angle in angles:
            points.append([r_ring * np.cos(angle), r_ring * np.sin(angle)])
    
    points = np.array(points)
    radius = np.hypot(points[:, 0], points[:, 1])
    points = points[radius <= 0.99]
    
    if len(points) >= n:
        return points[:n]
    
    # Fallback: Random points inside disk
    while len(points) < n:
        r = rng.uniform(0, 1)
        theta = rng.uniform(0, 2 * np.pi)
        points.append([r * np.cos(theta), r * np.sin(theta)])
    
    return np.array(points[:n])


def _residuals_fixed_r(x, n, r_target):
    """
    Residuals for fixed target radius r_target.
    Minimize sum of squared shortfalls.
    """
    centres = x[:2*n].reshape(n, 2)
    
    # Pairwise distances
    dists = pdist(centres)
    pair_residuals = np.maximum(0, 2 * r_target - dists)
    
    # Boundary distances
    boundary_dists = 1.0 - np.linalg.norm(centres, axis=1)
    boundary_residuals = np.maximum(0, r_target - boundary_dists)
    
    return np.concatenate([pair_residuals, boundary_residuals])


def construct():
    """
    Construct centres of 300 equal circles inside the unit circle.
    Uses multi-start least_squares with increasing target radius.
    """
    n = N
    best_centres = None
    best_radius = 0.0
    
    # Try multiple initial configurations
    for start_seed in range(3):
        # Generate initial configuration
        x0 = _generate_initial_config(n, seed=start_seed)
        x0 = x0.flatten()
        
        # Compute initial radius
        current_radius = _compute_radius(x0.reshape(n, 2))
        
        if current_radius > best_radius:
            best_radius = current_radius
            best_centres = x0.reshape(n, 2).copy()
        
        # If initial radius is already very small, skip optimization
        if current_radius < 0.01:
            continue
        
        # Increase target radius gradually
        r_target = current_radius * 0.9
        max_iter = 15
        for _ in range(max_iter):
            # Try to optimize for current r_target
            result = least_squares(
                _residuals_fixed_r, x0, args=(n, r_target),
                method='trf', max_nfev=5000,
                xtol=1e-10, ftol=1e-10, gtol=1e-10
            )
            
            new_centres = result.x[:2*n].reshape(n, 2)
            new_radius = _compute_radius(new_centres)
            
            if new_radius > best_radius:
                best_radius = new_radius
                best_centres = new_centres.copy()
            
            # Check if we can increase r_target
            if result.cost < 1e-8:
                # Feasible, try to increase r_target
                r_target *= 1.02
            else:
                # Not feasible, decrease r_target slightly
                r_target *= 0.99
            
            # Update x0 for next iteration
            x0 = new_centres.flatten()
        
        # Final polish at best radius found
        r_polish = best_radius * 1.001
        result = least_squares(
            _residuals_fixed_r, x0, args=(n, r_polish),
            method='trf', max_nfev=10000,
            xtol=1e-12, ftol=1e-12, gtol=1e-12
        )
        
        polished_centres = result.x[:2*n].reshape(n, 2)
        polished_radius = _compute_radius(polished_centres)
        
        if polished_radius > best_radius:
            best_radius = polished_radius
            best_centres = polished_centres.copy()
    
    # Ensure valid output
    if best_centres is None or best_radius <= 0 or not np.isfinite(best_radius):
        # Fallback: hexagonal lattice
        spacing = 2.0 / (np.sqrt(n) + 1.0)
        while True:
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
            if len(points) >= n:
                return points[np.argsort(radius, kind="stable")[:n]]
            spacing *= 0.98
    
    return best_centres
# EVOLVE-BLOCK-END
