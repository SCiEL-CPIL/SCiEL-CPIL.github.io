# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist

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
    z = 1 - (2 * i + 1) / n
    r = np.sqrt(np.maximum(0, 1 - z**2))
    theta = 2 * np.pi * i / golden
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    pts = np.column_stack([x, y, z])
    return pts * R

def _repulsive_optimize(points, n_iter=20000, learning_rate=0.01, power=12):
    """Optimize points on sphere using repulsive forces to maximize min distance."""
    n = points.shape[0]
    pts = points.copy().astype(np.float64)
    # Project to sphere
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1e-12, norms)
    pts = pts / norms * 1.0
    
    best_pts = pts.copy()
    best_score = _score(pts)
    
    for it in range(n_iter):
        # Compute pairwise distances
        dists = pdist(pts)
        d_min = np.min(dists)
        
        # Compute repulsive forces (higher power = stronger penalty for close pairs)
        # Force on point i from point j is proportional to (1/d^power) * direction
        n_pairs = n * (n - 1) // 2
        # Reconstruct full distance matrix for force computation
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                d = dists[i*(n-1) - i*(i-1)//2 + j - i - 1]
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d
        
        forces = np.zeros_like(pts)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                d = dist_matrix[i, j]
                if d < 1e-10:
                    # Random perturbation
                    forces[i] += np.random.randn(3) * 0.01
                    continue
                # Repulsive force: push away, magnitude ~ 1/d^power
                force_mag = 1.0 / (d ** power)
                direction = (pts[i] - pts[j]) / d
                forces[i] += force_mag * direction
        
        # Project force to tangent plane
        for i in range(n):
            norm_i = np.linalg.norm(pts[i])
            if norm_i < 1e-12:
                continue
            radial = (np.dot(forces[i], pts[i]) / (norm_i**2)) * pts[i]
            tangent_force = forces[i] - radial
            forces[i] = tangent_force
        
        # Normalize forces for stable updates
        force_norms = np.linalg.norm(forces, axis=1, keepdims=True)
        force_norms = np.where(force_norms < 1e-12, 1.0, force_norms)
        forces = forces / force_norms
        
        # Update positions
        lr = learning_rate * (1.0 - it / n_iter)  # Decay learning rate
        pts = pts + lr * forces
        
        # Project back to sphere
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1e-12, norms)
        pts = pts / norms * 1.0
        
        # Track best
        current_score = _score(pts)
        if current_score > best_score:
            best_score = current_score
            best_pts = pts.copy()
    
    return best_pts

def _local_optimize(points, n_iter=5000, power=15):
    """Fine-tune using steepest ascent on the min-distance objective."""
    n = points.shape[0]
    pts = points.copy().astype(np.float64)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1e-12, norms)
    pts = pts / norms * 1.0
    
    best_pts = pts.copy()
    best_score = _score(pts)
    
    for it in range(n_iter):
        # Compute distances
        dists = pdist(pts)
        d_min = np.min(dists)
        
        # Find the closest pair
        min_idx = np.argmin(dists)
        # Convert flat index to pair (i, j)
        i = int((np.sqrt(1 + 8 * min_idx) - 1) / 2)
        j = i + 1 + (min_idx - i*(2*n - i - 1)//2)
        if j >= n:
            j = n - 1
        
        # Gradient of min distance w.r.t. positions
        d = dists[min_idx]
        if d < 1e-12:
            grad = np.zeros_like(pts)
            grad[i] = np.random.randn(3)
            grad[j] = -np.random.randn(3)
        else:
            direction = (pts[i] - pts[j]) / d
            grad = np.zeros_like(pts)
            grad[i] = direction
            grad[j] = -direction
        
        # Project to tangent plane
        for k in range(n):
            norm_k = np.linalg.norm(pts[k])
            if norm_k < 1e-12:
                continue
            radial = (np.dot(grad[k], pts[k]) / (norm_k**2)) * pts[k]
            grad[k] = grad[k] - radial
        
        # Normalize
        gnorm = np.linalg.norm(grad)
        if gnorm > 1e-12:
            grad = grad / gnorm
        
        # Update
        lr = 0.005 * (1.0 - it / n_iter)
        pts = pts + lr * grad
        
        # Project to sphere
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1e-12, norms)
        pts = pts / norms * 1.0
        
        current_score = _score(pts)
        if current_score > best_score:
            best_score = current_score
            best_pts = pts.copy()
    
    return best_pts

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize the ratio of minimum to maximum distance.
    
    Returns
        points: np.ndarray of shape (14,3) containing the coordinates of the 14 points.
    """
    deadline = time.time() + 600  # Safety margin before 630s kill
    n = 14
    R = 1.0
    
    # Start with fibonacci sphere
    points = _fibonacci_sphere(n, R)
    best_pts = points.copy()
    best_score = _score(points)
    
    # Multi-restart optimization
    n_restarts = 10
    for restart in range(n_restarts):
        if time.time() > deadline - 60:
            break
        
        if restart == 0:
            start_pts = points.copy()
        else:
            # Perturb the best solution found so far
            start_pts = best_pts + np.random.randn(n, 3) * 0.1
            norms = np.linalg.norm(start_pts, axis=1, keepdims=True)
            norms = np.where(norms < 1e-12, 1e-12, norms)
            start_pts = start_pts / norms * R
        
        # Repulsive force optimization
        if time.time() > deadline - 120:
            break
        opt_pts = _repulsive_optimize(start_pts, n_iter=5000, learning_rate=0.01, power=12)
        opt_score = _score(opt_pts)
        if opt_score > best_score:
            best_score = opt_score
            best_pts = opt_pts.copy()
        
        # Local refinement
        if time.time() > deadline - 60:
            break
        refined_pts = _local_optimize(best_pts, n_iter=2000, power=15)
        refined_score = _score(refined_pts)
        if refined_score > best_score:
            best_score = refined_score
            best_pts = refined_pts.copy()
    
    return best_pts

# EVOLVE-BLOCK-END
