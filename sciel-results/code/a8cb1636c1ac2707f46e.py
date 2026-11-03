"""Improved program for sphere covering with 50 points using Gram matrix optimization."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import math


def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms


def _covering_radius(pts):
    """Compute the covering radius in degrees using the convex hull approach."""
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 180.0

    max_circumradius = 0.0
    facets = hull.simplices

    for simplex in facets:
        p1 = pts[simplex[0]]
        p2 = pts[simplex[1]]
        p3 = pts[simplex[2]]

        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm < 1e-15:
            continue
        normal = normal / norm

        for p in [p1, p2, p3]:
            cos_angle = np.dot(normal, p)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            if angle > max_circumradius:
                max_circumradius = angle

    return np.degrees(max_circumradius)


def _fibonacci_points(n):
    """Generate points using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(points)


def _random_points(n, seed=None):
    """Generate random points on the sphere."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    vecs = rng.randn(n, 3)
    return _normalize(vecs)


def _icosahedron_based_points(n):
    """Generate points starting from icosahedron vertices and adding more."""
    # Icosahedron vertices
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    verts = _normalize(verts)
    
    if n <= 12:
        return verts[:n]
    
    # Add random points to fill up to n
    remaining = n - 12
    extra = _random_points(remaining, seed=42)
    points = np.vstack([verts, extra])
    return _normalize(points)


def _spherical_lloyd_iteration(points, n_iters=100, tolerance=1e-10):
    """Perform Lloyd's algorithm on the sphere."""
    points = _normalize(points.copy())
    
    for _ in range(n_iters):
        # Compute Voronoi regions by finding nearest neighbor for grid points
        # Use a fine grid to approximate centroids
        n_grid = 500
        # Generate grid points on sphere
        theta = np.linspace(0, np.pi, int(np.sqrt(n_grid)))
        phi = np.linspace(0, 2 * np.pi, int(np.sqrt(n_grid)))
        theta, phi = np.meshgrid(theta, phi)
        theta_flat = theta.flatten()
        phi_flat = phi.flatten()
        
        grid_x = np.sin(theta_flat) * np.cos(phi_flat)
        grid_y = np.sin(theta_flat) * np.sin(phi_flat)
        grid_z = np.cos(theta_flat)
        grid_points = np.column_stack([grid_x, grid_y, grid_z])
        
        # Find nearest point for each grid point
        dots = grid_points @ points.T
        nearest = np.argmax(dots, axis=1)
        
        # Compute centroids
        new_points = np.zeros_like(points)
        for i in range(len(points)):
            mask = nearest == i
            if np.sum(mask) > 0:
                centroid = np.mean(grid_points[mask], axis=0)
                norm = np.linalg.norm(centroid)
                if norm > 1e-10:
                    new_points[i] = centroid / norm
                else:
                    new_points[i] = points[i]
            else:
                new_points[i] = points[i]
        
        new_points = _normalize(new_points)
        
        # Check convergence
        if np.max(np.linalg.norm(new_points - points, axis=1)) < tolerance:
            points = new_points
            break
        points = new_points
    
    return points


def _smooth_surrogate(points, temperature=0.05):
    """
    Smooth surrogate for covering radius using log-sum-exp over grid points.
    For each grid point, compute min angular distance to any configuration point,
    then use log-sum-exp to get a smooth maximum.
    """
    points = _normalize(points)
    
    # Generate a moderate grid
    n_theta = 30
    n_phi = 60
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    theta, phi = np.meshgrid(theta, phi)
    theta_flat = theta.flatten()
    phi_flat = phi.flatten()
    
    grid_x = np.sin(theta_flat) * np.cos(phi_flat)
    grid_y = np.sin(theta_flat) * np.sin(phi_flat)
    grid_z = np.cos(theta_flat)
    grid_points = np.column_stack([grid_x, grid_y, grid_z])
    
    # Compute angular distances from each grid point to each configuration point
    dots = grid_points @ points.T
    dots = np.clip(dots, -1.0, 1.0)
    angles = np.arccos(dots)  # shape: (n_grid, N)
    
    # For each grid point, find the minimum angle (nearest neighbor)
    min_angles = np.min(angles, axis=1)  # shape: (n_grid,)
    
    # Use log-sum-exp to approximate the maximum
    max_angle = np.max(min_angles)
    scaled = min_angles / temperature
    scaled_max = np.max(scaled)
    log_sum_exp = scaled_max + np.log(np.sum(np.exp(scaled - scaled_max)))
    smooth_max = temperature * log_sum_exp
    
    return np.degrees(smooth_max)


def _gram_matrix_objective(x, temperature=0.05):
    """
    Objective function working on Gram matrix entries.
    x is a flattened 3N vector representing N points.
    """
    pts = x.reshape(N, 3)
    pts = _normalize(pts)
    return _smooth_surrogate(pts, temperature)


def _nelder_mead_optimize(points, maxiter=2000, xatol=1e-8, fatol=1e-8):
    """Run Nelder-Mead optimization."""
    def objective(x):
        return _gram_matrix_objective(x)
    
    try:
        result = minimize(
            objective,
            points.flatten(),
            method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': xatol, 'fatol': fatol, 'adaptive': True}
        )
        return _normalize(result.x.reshape(N, 3)), result.fun
    except Exception:
        return points, _smooth_surrogate(points)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    # Try multiple initializations
    initializations = [
        _fibonacci_points(N),
        _random_points(N, seed=42),
        _icosahedron_based_points(N),
    ]
    
    # Add perturbed versions
    np.random.seed(123)
    for base in initializations[:2]:
        noise = np.random.randn(N, 3) * 0.2
        perturbed = _normalize(base + noise)
        initializations.append(perturbed)
    
    best_points = None
    best_radius = float('inf')
    
    for init in initializations:
        if time.time() > deadline:
            break
        
        # First, apply Lloyd's iteration to improve the initial configuration
        lloyd_points = _spherical_lloyd_iteration(init, n_iters=50, tolerance=1e-8)
        
        # Then apply Nelder-Mead optimization
        opt_points, opt_radius = _nelder_mead_optimize(
            lloyd_points, maxiter=1500, xatol=1e-9, fatol=1e-9
        )
        
        # Compute actual covering radius
        actual_radius = _covering_radius(opt_points)
        
        if actual_radius < best_radius:
            best_radius = actual_radius
            best_points = opt_points.copy()
    
    # If we have time, do a final polish on the best points
    if time.time() < deadline - 20.0 and best_points is not None:
        # Apply another round of Lloyd's iteration
        lloyd_points = _spherical_lloyd_iteration(best_points, n_iters=30, tolerance=1e-9)
        
        # Then Nelder-Mead
        opt_points, opt_radius = _nelder_mead_optimize(
            lloyd_points, maxiter=1000, xatol=1e-10, fatol=1e-10
        )
        
        actual_radius = _covering_radius(opt_points)
        if actual_radius < best_radius:
            best_radius = actual_radius
            best_points = opt_points.copy()
    
    # Final normalization
    if best_points is None:
        best_points = _fibonacci_points(N)
    
    best_points = _normalize(best_points)
    return best_points
# EVOLVE-BLOCK-END
