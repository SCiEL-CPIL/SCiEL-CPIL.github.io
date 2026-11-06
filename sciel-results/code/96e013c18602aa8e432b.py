"""
Covering the sphere with 130 equal caps.

Strategy:
1. Start from a Fibonacci spiral (deterministic, well-distributed).
2. Use a repulsive-energy minimization approach: each point feels a force
   pushing it away from its nearest neighbors (to increase the minimum
   separation), while also being pushed away from the "holes" (Voronoi
   cell circumcenters) to reduce the covering radius.
3. Use projected gradient descent / L-BFGS on the sphere.

The key insight for covering radius: the covering radius is determined by
the largest circumradius of the Delaunay triangulation facets. To minimize
it, we want to equalize the circumradii of all facets. This is achieved by
a force that pushes each point toward the center of its largest adjacent
Voronoi cell.

A simpler but effective approach: minimize the sum of squared distances
from each point to its k-nearest neighbors' Voronoi vertices, or use a
repulsive kernel that penalizes large gaps.

We'll use a combination of:
- Repulsive forces between close points (to prevent clustering)
- Attractive forces toward the "weakest" directions (to fill gaps)

Actually, the most effective known approach for small N is:
1. Start from a good initial configuration
2. Use simulated annealing or gradient-based optimization of the covering radius

For the covering radius, we can approximate it by evaluating the maximum
over a dense set of test points on the sphere of the distance to the
nearest configuration point. Then we can use gradient descent to minimize
this maximum distance.

However, this is non-smooth. A better approach: minimize the sum of
exp(-d^2/sigma^2) for all pairs, which penalizes both too-close and too-far
configurations. Or use the "repulsive energy" E = sum_{i<j} 1/d_ij^2 which
maximizes the minimum distance (packing), but we want covering.

For covering, a good objective is: for each point i, compute the distance
to its nearest neighbor d_i. The covering radius is approximately the
maximum over the sphere of dist to nearest point. A proxy is the maximum
of d_i/2 for each point (since the Voronoi cell of point i has circumradius
roughly d_i/2 in the worst case).

Better: compute the actual covering radius by finding, for each point, the
farthest point in its Voronoi cell. This is the circumradius of the largest
Delaunay triangle containing that point.

Let me use a practical approach:
1. Initialize with Fibonacci spiral
2. Optimize using L-BFGS-B on the objective: max over a grid of sphere points
   of the angular distance to the nearest configuration point
3. Use multiple restarts if time permits

Actually, for 130 points, a simpler and very effective method is:
- Use the "repulsive" method where each point is repelled from others
  with a force proportional to 1/r^2, and also attracted toward the
  center of mass of its neighbors' Voronoi cell.

Let me implement a clean gradient-based optimizer for the covering radius.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull, Delaunay
import time

N = 130


def _normalize(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return points / norms


def _covering_radius(points, n_test=5000):
    """
    Estimate covering radius by sampling test points on the sphere.
    Returns the maximum angular distance (in radians) from any test point
    to its nearest configuration point.
    """
    points = _normalize(points)
    # Generate test points using Fibonacci spiral
    i = np.arange(n_test) + 0.5
    z = 1.0 - 2.0 * i / n_test
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    test_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Compute distances from each test point to all configuration points
    # Using dot product: cos(theta) = dot(p, q), theta = arccos(dot)
    dots = test_pts @ points.T  # (n_test, N)
    # Clamp for numerical stability
    dots = np.clip(dots, -1.0, 1.0)
    # Angular distance
    angles = np.arccos(dots)  # (n_test, N)
    # Minimum distance from each test point to any configuration point
    min_angles = np.min(angles, axis=1)
    # Covering radius is the maximum of these minimum distances
    return np.max(min_angles)


def _covering_radius_smooth(points, n_test=2000, sigma=0.1):
    """
    Smooth approximation of covering radius for gradient-based optimization.
    Uses a soft-min: -sigma * log(sum(exp(-dist/sigma)))
    """
    points = _normalize(points)
    i = np.arange(n_test) + 0.5
    z = 1.0 - 2.0 * i / n_test
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    test_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    dots = test_pts @ points.T
    dots = np.clip(dots, -1.0, 1.0)
    angles = np.arccos(dots)
    
    # Soft max over test points of min distance
    # min over config points for each test point
    min_angles = np.min(angles, axis=1)
    # Soft max
    m = np.max(min_angles)
    log_sum = m + np.log(np.sum(np.exp(min_angles - m)))
    return sigma * log_sum


def _repulsive_energy(points, power=2.0):
    """
    Repulsive energy: sum over pairs of 1/d^power.
    Maximizing this helps with packing, but for covering we want a different
    objective. However, minimizing a combination can help.
    """
    points = _normalize(points)
    dots = points @ points.T
    np.fill_diagonal(dots, 0.0)
    dots = np.clip(dots, -1.0, 1.0)
    angles = np.arccos(dots)
    # Avoid division by zero
    angles = np.maximum(angles, 1e-10)
    energy = np.sum(1.0 / (angles ** power))
    return energy


def _objective(x, n_test=2000):
    """Objective function for optimization: covering radius estimate."""
    points = x.reshape(N, 3)
    points = _normalize(points)
    # Use smooth covering radius
    i = np.arange(n_test) + 0.5
    z = 1.0 - 2.0 * i / n_test
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    test_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    dots = test_pts @ points.T
    dots = np.clip(dots, -1.0, 1.0)
    angles = np.arccos(dots)
    min_angles = np.min(angles, axis=1)
    
    # Use a soft-max to make it differentiable
    # Temperature for soft-max
    T = 0.05
    m = np.max(min_angles)
    soft_max = m + T * np.log(np.sum(np.exp((min_angles - m) / T)))
    
    # Also add a small repulsive term to avoid coincident points
    dots2 = points @ points.T
    np.fill_diagonal(dots2, 0.0)
    dots2 = np.clip(dots2, -1.0, 1.0)
    angles2 = np.arccos(dots2)
    angles2 = np.maximum(angles2, 1e-8)
    repulsive = np.sum(1.0 / (angles2 ** 2)) * 1e-6
    
    return soft_max + repulsive


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin
    
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    points = _normalize(points)
    
    best_points = points.copy()
    best_radius = _covering_radius(points, n_test=10000)
    
    # Try multiple optimization strategies
    strategies = [
        ('nelder-mead', {'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10}),
        ('L-BFGS-B', {'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}),
    ]
    
    for method, options in strategies:
        if time.time() > deadline - 30:
            break
        
        x0 = points.reshape(-1)
        
        try:
            if method == 'nelder-mead':
                res = minimize(_objective, x0, method='Nelder-Mead', 
                             options={'maxiter': options['maxiter'], 
                                     'xatol': options['xatol'], 
                                     'fatol': options['fatol']})
            elif method == 'L-BFGS-B':
                res = minimize(_objective, x0, method='L-BFGS-B',
                             options={'maxiter': options['maxiter'],
                                     'ftol': options['ftol'],
                                     'gtol': options['gtol']})
            
            if res.success or hasattr(res, 'x'):
                new_points = _normalize(res.x.reshape(N, 3))
                new_radius = _covering_radius(new_points, n_test=10000)
                if new_radius < best_radius:
                    best_radius = new_radius
                    best_points = new_points.copy()
                    points = new_points.copy()
                    
        except Exception:
            pass
    
    # Additional refinement: local search with random perturbations
    if time.time() < deadline - 10:
        rng = np.random.default_rng(42)
        for _ in range(200):
            if time.time() > deadline - 5:
                break
            
            # Random perturbation
            noise = rng.normal(0, 0.01, size=(N, 3))
            perturbed = _normalize(points + noise)
            
            # Quick evaluation
            radius = _covering_radius(perturbed, n_test=3000)
            if radius < best_radius:
                best_radius = radius
                best_points = perturbed.copy()
                points = perturbed.copy()
    
    # Final polish with a few more Nelder-Mead iterations
    if time.time() < deadline - 5:
        try:
            x0 = points.reshape(-1)
            res = minimize(_objective, x0, method='Nelder-Mead',
                         options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12})
            if hasattr(res, 'x'):
                new_points = _normalize(res.x.reshape(N, 3))
                new_radius = _covering_radius(new_points, n_test=10000)
                if new_radius < best_radius:
                    best_radius = new_radius
                    best_points = new_points.copy()
        except Exception:
            pass
    
    return _normalize(best_points)
