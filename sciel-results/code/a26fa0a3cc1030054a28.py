"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time


def _normalize(pts):
    """Normalize vectors to unit length."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms


def _covering_radius(pts, n_samples=20000):
    """
    Estimate the covering radius by sampling points on the sphere.
    Returns the max angular distance in degrees.
    """
    # Generate sample points on sphere using Fibonacci spiral
    i = np.arange(n_samples) + 0.5
    z = 1.0 - 2.0 * i / n_samples
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    samples = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # For each sample, find nearest point
    # Compute all pairwise dot products
    dots = samples @ pts.T  # shape (n_samples, N)
    max_dots = np.max(dots, axis=1)  # shape (n_samples,)
    # Clamp for numerical stability
    max_dots = np.clip(max_dots, -1.0, 1.0)
    # Angular distance = arccos(dot)
    angles = np.arccos(max_dots)
    return np.degrees(np.max(angles))


def _repulsion_step(pts, n_iter=50, learning_rate=0.01):
    """
    Iterative repulsion method to minimize covering radius.
    Points are pushed away from neighbors based on local density.
    """
    pts = _normalize(pts.copy())
    
    for it in range(n_iter):
        # Compute pairwise distances (angular)
        dots = pts @ pts.T
        dots = np.clip(dots, -1.0, 1.0)
        angles = np.arccos(dots)  # shape (N, N)
        
        # For each point, find its k-nearest neighbors and compute local covering radius
        # Use the maximum distance to nearest neighbor as a proxy
        np.fill_diagonal(angles, np.inf)
        
        # For each point, the covering radius is related to the max of:
        # - distance to nearest neighbor (if isolated)
        # - half the distance to the two nearest neighbors in a triangle
        # Approximation: use the max of distances to 3 nearest neighbors / 2
        sorted_angles = np.sort(angles, axis=1)
        # Take up to 3 nearest (or fewer if N < 4)
        k = min(3, N - 1)
        local_r = sorted_angles[:, :k].max(axis=1) / 2.0
        
        # Repulsion: each point is pushed away from its neighbors
        # Force is proportional to local covering radius
        forces = np.zeros_like(pts)
        for i in range(N):
            # Get neighbors sorted by distance
            dists = angles[i]
            # Exclude self
            dists[i] = np.inf
            # Get top 3 nearest
            top_k = np.argsort(dists)[:k]
            
            for j in top_k:
                # Repulsion force: direction away from j, magnitude based on local_r
                # If point i has large local_r, it's in a sparse region, push it less
                # If point j has large local_r, it's isolated, push i away more
                weight = local_r[j] / (local_r[i] + 1e-10)
                # Force direction: from j to i
                direction = pts[i] - pts[j]
                norm = np.linalg.norm(direction)
                if norm > 1e-10:
                    direction = direction / norm
                    # Tangential component only
                    tangential = direction - np.dot(direction, pts[i]) * pts[i]
                    forces[i] += weight * tangential
        
        # Normalize forces to be tangential and update
        for i in range(N):
            # Project force to tangent plane
            f = forces[i]
            f_tang = f - np.dot(f, pts[i]) * pts[i]
            norm = np.linalg.norm(f_tang)
            if norm > 1e-10:
                f_tang = f_tang / norm
                # Update with adaptive step size
                step = learning_rate * local_r[i]
                pts[i] = pts[i] + step * f_tang
                pts[i] = _normalize(pts[i].reshape(1, -1))[0]
    
    return pts


def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Start from Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts = _normalize(pts)
    
    best_pts = pts.copy()
    best_cr = _covering_radius(pts)
    
    # Multiple restarts with different initializations and refinement
    n_restarts = 5
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        if restart == 0:
            # Use the Fibonacci spiral
            current = pts.copy()
        elif restart == 1:
            # Slightly perturbed Fibonacci
            rng = np.random.RandomState(42)
            current = pts + 0.01 * rng.randn(N, 3)
            current = _normalize(current)
        elif restart == 2:
            # Different golden ratio phase
            i2 = np.arange(N) + 0.5
            z2 = 1.0 - 2.0 * i2 / N
            phi2 = np.pi * (1.0 + np.sqrt(5.0)) * i2 + np.pi / N
            r2 = np.sqrt(1.0 - z2 * z2)
            current = np.column_stack([r2 * np.cos(phi2), r2 * np.sin(phi2), z2])
            current = _normalize(current)
        elif restart == 3:
            # Random initialization
            rng = np.random.RandomState(123)
            current = rng.randn(N, 3)
            current = _normalize(current)
        else:
            # Best found so far with perturbation
            rng = np.random.RandomState(999)
            current = best_pts + 0.05 * rng.randn(N, 3)
            current = _normalize(current)
        
        # Refine with repulsion
        # Use more iterations if time permits
        remaining = deadline - time.time()
        if remaining > 30:
            n_iter = 100
            lr = 0.01
        elif remaining > 15:
            n_iter = 50
            lr = 0.01
        elif remaining > 5:
            n_iter = 20
            lr = 0.005
        else:
            n_iter = 5
            lr = 0.002
        
        current = _repulsion_step(current, n_iter=n_iter, learning_rate=lr)
        
        # Evaluate
        cr = _covering_radius(current)
        if cr < best_cr:
            best_cr = cr
            best_pts = current.copy()
    
    # Final check
    if time.time() > deadline - 5:
        return best_pts
    
    return best_pts
# EVOLVE-BLOCK-END
