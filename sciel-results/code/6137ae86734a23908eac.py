"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
import scipy.optimize
from scipy.spatial.distance import cdist


def _covering_radius(points):
    """Compute the covering radius in radians for points on the unit sphere."""
    # Compute pairwise angular distances
    cos_d = np.clip(points @ points.T, -1.0, 1.0)
    # The covering radius is half the maximum pairwise angular distance
    # (the worst-case point is at the midpoint of the farthest pair)
    max_ang = np.arccos(cos_d)
    return 0.5 * np.max(max_ang)


def _objective_and_grad(theta_vec):
    """Objective: minimize sum of squared deviations from target angle."""
    n = theta_vec.shape[0]
    # Reshape to points
    pts = theta_vec.reshape(-1, 3)
    # Normalize
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / np.maximum(norms, 1e-12)
    
    # Compute pairwise distances
    cos_d = np.clip(pts @ pts.T, -1.0, 1.0)
    ang_d = np.arccos(cos_d)
    
    # Target: optimal covering radius ~ 11.3165625 degrees in radians
    target = np.radians(11.3165625)
    
    # Objective: sum of squared deviations from target for all pairs
    mask = ~np.eye(n, dtype=bool)
    obj = np.sum((ang_d[mask] - target) ** 2)
    
    return obj


def _cvt_iteration(points, n_iters=100, n_samples=10000):
    """CVT (Centroidal Voronoi Tessellation) optimization on sphere."""
    rng = np.random.default_rng(42)
    pts = points.copy()
    
    for _ in range(n_iters):
        # Sample random points on sphere
        samples = rng.normal(size=(n_samples, 3))
        samples /= np.linalg.norm(samples, axis=1, keepdims=True)
        
        # Find nearest site for each sample
        cos_d = samples @ pts.T
        nearest = np.argmax(cos_d, axis=1)
        
        # Compute centroids
        new_pts = np.zeros_like(pts)
        for i in range(len(pts)):
            mask = nearest == i
            if np.sum(mask) > 0:
                centroid = np.mean(samples[mask], axis=0)
                norm = np.linalg.norm(centroid)
                if norm > 1e-10:
                    new_pts[i] = centroid / norm
                else:
                    new_pts[i] = pts[i]
            else:
                new_pts[i] = pts[i]
        
        pts = new_pts
    
    return pts


def _polish(points, max_time=10):
    """Final polish using scipy.optimize to minimize covering radius."""
    start_time = time.time()
    pts = points.copy()
    
    # Use L-BFGS-B to minimize the max pairwise distance / 2
    def objective(x):
        p = x.reshape(-1, 3)
        p = p / np.linalg.norm(p, axis=1, keepdims=True)
        cos_d = np.clip(p @ p.T, -1.0, 1.0)
        ang_d = np.arccos(cos_d)
        return 0.5 * np.max(ang_d)
    
    x0 = pts.flatten()
    
    result = scipy.optimize.minimize(
        objective, x0, method='L-BFGS-B',
        options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-15}
    )
    
    if result.success:
        new_pts = result.x.reshape(-1, 3)
        new_pts = new_pts / np.linalg.norm(new_pts, axis=1, keepdims=True)
        if _covering_radius(new_pts) < _covering_radius(pts):
            pts = new_pts
    
    return pts


def construct():
    """Construct 130 points on the unit sphere minimizing covering radius."""
    deadline = 110.0  # Safety margin from 120s limit
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_pts = pts
    best_radius = _covering_radius(pts)
    
    # Multiple restarts with different random initializations
    rng = np.random.default_rng(0)
    
    while time.time() < deadline - 30:  # Leave 30s for polishing
        # Random initial points
        rand_pts = rng.normal(size=(N, 3))
        rand_pts /= np.linalg.norm(rand_pts, axis=1, keepdims=True)
        
        # CVT optimization
        cvt_pts = _cvt_iteration(rand_pts, n_iters=50, n_samples=5000)
        
        # Check if better
        cr = _covering_radius(cvt_pts)
        if cr < best_radius:
            best_radius = cr
            best_pts = cvt_pts
        elif cr < best_radius * 1.01:  # Keep if close
            # Also try from Fibonacci start
            fib_cvt = _cvt_iteration(pts, n_iters=50, n_samples=5000)
            cr2 = _covering_radius(fib_cvt)
            if cr2 < best_radius:
                best_radius = cr2
                best_pts = fib_cvt
    
    # Final polish
    if time.time() < deadline - 5:
        best_pts = _polish(best_pts, max_time=10)
    
    # Normalize final points
    best_pts = best_pts / np.linalg.norm(best_pts, axis=1, keepdims=True)
    
    return best_pts
# EVOLVE-BLOCK-END
