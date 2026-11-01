"""Improved packing: hexagonal lattice with local optimization via simulated annealing."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_radius(points):
    """Compute the maximum common radius for given points."""
    # Distance to boundary
    dist_boundary = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = np.min(dist_boundary)
    
    # Minimum pairwise distance / 2
    # For N=300, O(N^2) is 90000 pairs, manageable
    min_pair = np.inf
    n = len(points)
    for i in range(n):
        diffs = points[i+1:] - points[i]
        dists = np.linalg.norm(diffs, axis=1)
        if len(dists) > 0:
            min_pair = min(min_pair, np.min(dists))
    min_pair /= 2.0
    return min(min_pair, min_boundary)

def _hexagonal_lattice_init(n_points, target_radius):
    """Generate initial hexagonal lattice points that fit in unit circle."""
    points = []
    spacing = 2.0 * target_radius
    
    # Generate hexagonal grid
    k = int(1.0 / spacing) + 3
    rows = []
    for j in range(-k, k + 1):
        y = j * spacing * np.sqrt(3.0) / 2.0
        offset = (j % 2) * spacing / 2.0
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            r = np.hypot(x, y)
            if r <= 1.0 - target_radius * 1.01:
                rows.append((x, y))
    
    points = np.array(rows) if rows else np.array([[0, 0]])
    
    # If we have more than needed, select closest to center
    if len(points) > n_points:
        radii = np.linalg.norm(points, axis=1)
        idx = np.argsort(radii, kind='stable')[:n_points]
        points = points[idx]
    elif len(points) < n_points:
        # Pad with small perturbations around origin
        extra = n_points - len(points)
        extra_pts = np.random.randn(extra, 2) * 0.01
        points = np.vstack([points, extra_pts])
    
    return points

def _local_optimize(points, r0, max_iter=5000, time_limit=100):
    """Local optimization using gradient-free method to maximize minimum distance."""
    start_time = time.time()
    best_points = points.copy()
    best_radius = _compute_radius(points)
    
    # Use a simple hill-climbing / simulated annealing approach
    current_points = points.copy()
    current_radius = best_radius
    
    # Temperature schedule
    T0 = 0.001
    T_end = 1e-8
    cooling = (T_end / T0) ** (1.0 / max_iter)
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        T = T0 * (cooling ** it)
        
        # Pick a random point to move
        idx = np.random.randint(len(current_points))
        
        # Random perturbation
        delta = np.random.randn(2) * (T * 10 + 0.0001)
        new_point = current_points[idx] + delta
        
        # Check if still in circle (with margin for radius)
        if np.linalg.norm(new_point) > 1.0 - 0.01:
            continue
        
        trial_points = current_points.copy()
        trial_points[idx] = new_point
        
        trial_radius = _compute_radius(trial_points)
        
        if trial_radius >= current_radius or np.random.random() < np.exp((trial_radius - current_radius) / max(T, 1e-15)):
            current_points = trial_points
            current_radius = trial_radius
            if current_radius > best_radius:
                best_radius = current_radius
                best_points = current_points.copy()
    
    return best_points, best_radius

def construct():
    """Construct centers of 300 equal circles inside unit circle."""
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    # Initial estimate: best known is ~0.05315
    r_init = 0.053
    
    # Generate initial configuration
    points = _hexagonal_lattice_init(N, r_init)
    
    # Ensure we have exactly N points
    if len(points) != N:
        points = _hexagonal_lattice_init(N, r_init * 0.95)
    
    # Local optimization
    best_points, best_radius = _local_optimize(points, r_init, max_iter=8000, time_limit=120)
    
    # Try a few more restarts if time permits
    n_restarts = 0
    while time.time() - start_time < deadline - 10:
        # Slightly perturb initial config
        perturb = np.random.randn(N, 2) * 0.002
        trial_points = best_points + perturb
        # Project back to feasible region
        norms = np.linalg.norm(trial_points, axis=1)
        mask = norms > 1.0 - 0.01
        trial_points[mask] = trial_points[mask] * (1.0 - 0.01) / norms[mask, np.newaxis]
        
        p, r = _local_optimize(trial_points, best_radius, max_iter=3000, time_limit=20)
        if r > best_radius:
            best_points = p
            best_radius = r
        n_restarts += 1
    
    # Final check: ensure all points are strictly inside
    norms = np.linalg.norm(best_points, axis=1)
    # If any point is on or outside, shrink slightly
    if np.any(norms >= 1.0):
        best_points = best_points * (1.0 - 1e-6)
    
    return best_points
# EVOLVE-BLOCK-END
