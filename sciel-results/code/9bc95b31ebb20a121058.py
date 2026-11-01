import numpy as np
import time

N = 130

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 40s margin for safety
    
    # 1. Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_pts = pts.copy()
    best_radius = _estimate_covering_radius(best_pts)
    
    # 2. Local optimization using repulsive forces
    # Objective: Minimize sum of d_ij^-p, which promotes uniform distribution
    # This correlates well with minimizing covering radius.
    p = 4.0  # Power of inverse distance
    step_size = 0.05
    decay = 0.999
    momentum = 0.5
    v = np.zeros_like(pts)
    
    while time.time() < deadline:
        # Calculate pairwise distances
        # pts is (N, 3)
        # dists[i, j] is angular distance
        dot = np.dot(pts, pts.T)
        np.clip(dot, -1.0, 1.0, out=dot)
        dists = np.arccos(dot)
        
        # Mask diagonal to avoid division by zero
        np.fill_diagonal(dists, 1.0)
        
        # Calculate forces
        # Force on i due to j is proportional to -grad_i (d_ij^-p)
        # grad d_ij = (pts_i - pts_j) / ||pts_i - pts_j||_tan? 
        # Simpler: Use vector repulsion in 3D space, then project.
        # F_ij = (pts_i - pts_j) * (p * dists[i,j]^{-(p+1)}) 
        # This is an approximation of the gradient on the sphere for small angles.
        # Let's use the exact gradient of the potential V = sum d_ij^-p on the sphere.
        # dV/dpts_i = sum_j -p * d_ij^{-(p+1)} * (pts_i - pts_j) / ||pts_i - pts_j||
        # Note: ||pts_i - pts_j|| = 2 * sin(d_ij / 2)
        
        # Vector difference
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :] # (N, N, 3)
        norms = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(norms, 1.0)
        
        # Gradient of d_ij w.r.t pts_i is (pts_i - pts_j) / ||pts_i - pts_j||
        # So grad V = sum_j -p * d_ij^{-(p+1)} * (pts_i - pts_j) / ||pts_i - pts_j||
        # We need d_ij in the power term.
        # weights = -p * dists**(-(p+1))
        # But we must exclude diagonal.
        
        # Let's compute the force vector
        # w[i,j] = -p * dists[i,j]**(-(p+1))
        w = -p * (dists ** (-(p+1)))
        np.fill_diagonal(w, 0.0)
        
        # Force on i is sum_j w[i,j] * (pts_i - pts_j) / norms[i,j]
        # diff is pts_i - pts_j
        unit_diff = diff / norms[:, :, np.newaxis]
        forces = np.sum(w[:, :, np.newaxis] * unit_diff, axis=1)
        
        # Project forces onto tangent plane
        # Tangent plane at pts_i is orthogonal to pts_i
        # proj = force - (force . pts_i) * pts_i
        dot_products = np.sum(forces * pts, axis=1)
        forces = forces - dot_products[:, np.newaxis] * pts
        
        # Update with momentum
        v = momentum * v + step_size * forces
        pts = pts + v
        
        # Normalize
        norms_pts = np.linalg.norm(pts, axis=1)
        pts = pts / norms_pts[:, np.newaxis]
        
        # Decay step size
        step_size *= decay
        
        # Evaluate and keep best
        current_radius = _estimate_covering_radius(pts)
        if current_radius < best_radius:
            best_radius = current_radius
            best_pts = pts.copy()
            
        # Restart mechanism if stuck?
        # If step_size is very small, we might be stuck.
        # We can add a small random perturbation occasionally.
        if step_size < 0.001:
            # Perturb
            pts = pts + 0.01 * np.random.randn(N, 3)
            norms_pts = np.linalg.norm(pts, axis=1)
            pts = pts / norms_pts[:, np.newaxis]
            step_size = 0.05
            v = np.zeros_like(pts)

    return best_pts

def _estimate_covering_radius(pts):
    """
    Estimate the covering radius by sampling the sphere.
    This is an approximation. The true covering radius is the max over all y on sphere
    of min_i dist(y, pts_i).
    We sample a dense set of points on the sphere.
    """
    # Use a Fibonacci sphere for sampling too, but denser
    M = 5000
    i = np.arange(M) + 0.5
    z = 1.0 - 2.0 * i / M
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    samples = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Calculate min distance from each sample to nearest pt
    # dot = samples @ pts.T
    dot = np.dot(samples, pts.T)
    np.clip(dot, -1.0, 1.0, out=dot)
    dists = np.arccos(dot)
    min_dists = np.min(dists, axis=1)
    max_dist = np.max(min_dists)
    
    return np.degrees(max_dist)
