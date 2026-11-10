"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_score(pts):
    """Compute the packing score: min of pairwise distances/2 and boundary margins."""
    n = len(pts)
    # Pairwise distances
    diff = pts[:, None, :] - pts[None, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists) / 2.0
    
    # Boundary margins
    norms = np.sqrt(np.sum(pts ** 2, axis=1))
    min_boundary = np.min(1.0 - norms)
    
    return min(min_pair_dist, min_boundary)


def _optimize(pts, max_time=110.0, initial_lr=0.001, decay=0.995):
    """Local optimization using subgradient method on the score function."""
    start_time = time.time()
    n = len(pts)
    pts = pts.astype(np.float64).copy()
    best_score = _compute_score(pts)
    best_pts = pts.copy()
    lr = initial_lr
    
    # Precompute initial distances for efficiency
    while time.time() - start_time < max_time:
        # Compute pairwise distances
        diff = pts[:, None, :] - pts[None, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Find minimum pair distance
        min_idx = np.argmin(dists)
        i, j = np.unravel_index(min_idx, (n, n))
        min_pair_dist = dists[i, j] / 2.0
        
        # Compute boundary margins
        norms = np.sqrt(np.sum(pts ** 2, axis=1))
        min_boundary_idx = np.argmin(1.0 - norms)
        min_boundary = 1.0 - norms[min_boundary_idx]
        
        current_score = min(min_pair_dist, min_boundary)
        
        # If we improved, update best
        if current_score > best_score:
            best_score = current_score
            best_pts = pts.copy()
        
        # Compute gradient
        grad = np.zeros_like(pts)
        
        if min_pair_dist <= min_boundary:
            # Gradient from pair constraint
            d = pts[i] - pts[j]
            norm_d = np.linalg.norm(d)
            if norm_d > 1e-10:
                grad[i] += d / norm_d
                grad[j] -= d / norm_d
        else:
            # Gradient from boundary constraint
            p = pts[min_boundary_idx]
            norm_p = np.linalg.norm(p)
            if norm_p > 1e-10:
                grad[min_boundary_idx] -= p / norm_p
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-10:
            grad /= grad_norm
        
        # Update
        pts += lr * grad
        
        # Project back: ensure points stay inside unit circle
        norms = np.sqrt(np.sum(pts ** 2, axis=1))
        mask = norms > 0.999
        if np.any(mask):
            pts[mask] *= 0.999 / norms[mask]
        
        # Decay learning rate
        lr *= decay
        
        # Check for convergence
        if lr < 1e-8:
            break
    
    return best_pts, best_score


def construct():
    # Generate hexagonal lattice
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            break
        spacing *= 0.98
    
    # Select N points closest to center
    idx = np.argsort(radius, kind="stable")[:N]
    pts = points[idx].copy()
    
    # Optimize
    optimized_pts, score = _optimize(pts, max_time=110.0)
    
    return optimized_pts
# EVOLVE-BLOCK-END
