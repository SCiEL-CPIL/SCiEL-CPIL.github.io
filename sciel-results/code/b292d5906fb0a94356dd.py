# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist

def _project_to_sphere(pts, R=1.0):
    """Project points onto sphere of radius R."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1e-12, norms)
    return pts / norms * R

def _score(points):
    """Compute (d_min / d_max)^2 for a set of points."""
    if points.shape != (14, 3):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    dists = pdist(points)
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max == 0:
        return 0.0
    return (d_min / d_max) ** 2

def _fibonacci_sphere(n, R=1.0):
    """Generate n points on a sphere using fibonacci lattice."""
    golden = (1 + np.sqrt(5)) / 2
    i = np.arange(n)
    # Avoid exact poles
    z = 1 - (2 * i + 1) / n
    r = np.sqrt(1 - z**2)
    theta = 2 * np.pi * i / golden
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    pts = np.column_stack([x, y, z])
    return pts * R

def _force_optimize(pts, R=1.0, iters=50000, lr=0.01):
    """
    Force-directed optimization on sphere.
    Points repel each other; force is projected onto tangent plane.
    """
    pts = _project_to_sphere(pts.copy(), R)
    n = len(pts)
    
    for it in range(iters):
        # Compute forces: repulsion between all pairs
        forces = np.zeros_like(pts)
        for i in range(n):
            for j in range(i + 1, n):
                diff = pts[i] - pts[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    # Random small perturbation
                    diff = np.random.randn(3) * 1e-3
                    dist = np.linalg.norm(diff)
                # Repulsive force ~ 1/dist^2, direction from j to i
                f = diff / (dist ** 3) * 1.0  # proportional to 1/dist^2 in magnitude
                forces[i] += f
                forces[j] -= f
        
        # Project forces onto tangent plane of sphere
        for i in range(n):
            normal = pts[i] / np.linalg.norm(pts[i])
            # Tangential component
            f_tan = forces[i] - np.dot(forces[i], normal) * normal
            pts[i] += lr * f_tan
        
        # Project back to sphere
        pts = _project_to_sphere(pts, R)
        
        # Adaptive learning rate: reduce over time
        if it % 1000 == 0 and it > 0:
            lr *= 0.95
    
    return pts

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the coordinates of the 14 points.
    """
    start_time = time.time()
    deadline = start_time + 590  # 10s safety margin
    
    n = 14
    R = 1.0
    
    # Start with fibonacci sphere
    best_pts = _fibonacci_sphere(n, R)
    best_score = _score(best_pts)
    
    # Multiple restarts with different initializations
    np.random.seed(42)
    
    for restart in range(50):
        if time.time() > deadline - 30:
            break
        
        if restart == 0:
            pts = _fibonacci_sphere(n, R)
        else:
            # Perturb the fibonacci sphere or use random
            if restart % 2 == 0:
                pts = _fibonacci_sphere(n, R) + np.random.randn(n, 3) * 0.1
            else:
                pts = np.random.randn(n, 3)
            
            pts = _project_to_sphere(pts, R)
        
        # Force optimization
        iters = 30000 if restart < 10 else 15000
        pts = _force_optimize(pts, R, iters=iters, lr=0.02)
        
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
        
        # If we're doing well, do a finer polish
        if s > 0.3 and time.time() < deadline - 60:
            pts_fine = _force_optimize(pts, R, iters=20000, lr=0.005)
            s_fine = _score(pts_fine)
            if s_fine > best_score:
                best_score = s_fine
                best_pts = pts_fine.copy()
    
    return best_pts

# EVOLVE-BLOCK-END
