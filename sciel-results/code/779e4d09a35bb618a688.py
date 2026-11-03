"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint
from itertools import combinations

def _compute_score(points):
    """Compute the packing radius for a set of points."""
    n = len(points)
    if n < 2:
        return 0.0
    
    # Distance to boundary
    dist_boundary = 1.0 - np.linalg.norm(points, axis=1)
    
    # Pairwise distances
    # Use a more memory-efficient approach for large N
    # But for N=300, full matrix is fine
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    min_pair_dist = np.min(dist)
    
    return min(min_pair_dist / 2.0, np.min(dist_boundary))


def _hexagonal_initial(n, max_radius=1.0):
    """Generate a hexagonal lattice initial configuration."""
    # Start with a reasonable spacing
    # For n circles in unit circle, approximate radius ~ 0.05
    # Spacing ~ 2*r
    spacing = 2.0 / (np.sqrt(n) + 1.0)
    
    points = []
    # Generate hexagonal grid
    max_k = int(1.0 / spacing) + 2
    for j in range(-max_k, max_k + 1):
        for i in range(-max_k, max_k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if x*x + y*y <= (1.0 - spacing/2.0)**2:
                points.append([x, y])
    
    points = np.array(points)
    if len(points) == 0:
        return np.zeros((n, 2))
    
    # Sort by distance from center and take first n
    radii = np.linalg.norm(points, axis=1)
    idx = np.argsort(radii)[:n]
    return points[idx]


def _gradient_ascent(points, max_iters=1000, lr=0.01, deadline=None):
    """
    Gradient ascent to maximize the minimum distance.
    We maximize min(min_pair_dist/2, min_boundary_dist)
    This is equivalent to maximizing the radius r.
    
    We use a smooth approximation: minimize the negative of a smooth function
    that approximates the min.
    """
    n = len(points)
    points = points.copy()
    
    # Precompute pairwise indices
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    def objective_and_grad(pts):
        """
        Minimize -logsumexp(-distances) to approximate -min(distances).
        Actually, let's use a different approach:
        Maximize r where r = min(d_ij/2, 1-|p_i|)
        
        We can use: maximize the minimum of all constraints.
        A common trick: minimize -r where r is the minimum.
        But min is non-differentiable.
        
        Alternative: use the "barrier" method or just do a simple 
        gradient ascent on a smooth approximation.
        
        Let's use: f(x) = logsumexp(-d_ij) + logsumexp(-(1-|p_i|))
        Minimizing this encourages all distances to be large.
        """
        pts = pts.reshape(-1, 2)
        
        # Pairwise distances
        diff = pts[i_idx] - pts[j_idx]
        dists = np.sqrt(np.sum(diff**2, axis=1))
        
        # Boundary distances
        norms = np.linalg.norm(pts, axis=1)
        bdist = 1.0 - norms
        
        # Smooth min approximation: -logsumexp(-x) approximates min(x)
        # We want to maximize min, so minimize -min
        # -min(x) ≈ logsumexp(-x)
        
        # Use a temperature for smoothness
        T = 0.01
        obj = np.log(np.sum(np.exp(-dists / T))) * T + np.log(np.sum(np.exp(-bdist / T))) * T
        
        # Gradient
        # d/dpts logsumexp(-dists/T)
        # = sum_i exp(-d_i/T) / sum_j exp(-d_j/T) * d/dpts (-d_i/T)
        # d/dpts (-d_i/T) = (pts_i - pts_j) / (T * d_i)
        
        weights_d = np.exp(-dists / T) / (np.sum(np.exp(-dists / T)) * T)
        # Gradient w.r.t. pts[i] for pair (i,j):
        # w_ij * (pts[i] - pts[j]) / d_ij
        # But we need to accumulate for all pairs
        
        grad = np.zeros_like(pts)
        for k in range(len(i_idx)):
            i, j = i_idx[k], j_idx[k]
            d = dists[k]
            if d < 1e-10:
                continue
            w = weights_d[k] / d
            grad[i] += w * (pts[i] - pts[j])
            grad[j] -= w * (pts[i] - pts[j])
        
        # Boundary gradient
        weights_b = np.exp(-bdist / T) / (np.sum(np.exp(-bdist / T)) * T)
        # d/dpts_i (-(1-|p_i|)/T) = (pts_i / |p_i|) / T
        for i in range(n):
            norm = norms[i]
            if norm < 1e-10:
                continue
            w = weights_b[i] / norm
            grad[i] += w * pts[i]
        
        return obj, grad
    
    # Simple gradient ascent with adaptive learning rate
    current_obj, current_grad = objective_and_grad(points)
    best_points = points.copy()
    best_obj = current_obj
    best_radius = _compute_score(points)
    
    lr = 0.01
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Gradient step (we're minimizing, so go in negative gradient direction)
        points_new = points - lr * current_grad
        
        # Clip to stay within unit circle (with margin)
        norms = np.linalg.norm(points_new, axis=1)
        mask = norms > 0.999
        if np.any(mask):
            points_new[mask] *= 0.999 / norms[mask]
        
        new_obj, new_grad = objective_and_grad(points_new)
        
        if new_obj < current_obj:
            points = points_new
            current_obj = new_obj
            current_grad = new_grad
            # Decrease learning rate
            lr *= 0.99
        else:
            # Increase learning rate back a bit
            lr *= 1.1
            lr = min(lr, 0.05)
        
        # Check improvement
        if it % 10 == 0:
            r = _compute_score(points)
            if r > best_radius:
                best_radius = r
                best_points = points.copy()
    
    return best_points


def _local_optimize(points, deadline=None):
    """
    Local optimization using coordinate descent and gradient methods.
    """
    points = points.copy()
    n = len(points)
    
    # First, do a few rounds of coordinate descent
    for round_num in range(5):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        for i in range(n):
            if deadline is not None and time.time() > deadline:
                break
            
            # Try to move point i to maximize the minimum distance
            # involving point i
            p = points[i].copy()
            best_p = p
            best_r = -1
            
            # Sample some directions
            for angle in np.linspace(0, 2*np.pi, 12, endpoint=False):
                for step in [0.001, 0.005, 0.01, 0.02]:
                    new_p = p + step * np.array([np.cos(angle), np.sin(angle)])
                    
                    # Check boundary
                    if np.linalg.norm(new_p) > 1.0:
                        continue
                    
                    # Check distances to all other points
                    min_dist = 1.0 - np.linalg.norm(new_p)
                    for j in range(n):
                        if j == i:
                            continue
                        d = np.linalg.norm(new_p - points[j])
                        min_dist = min(min_dist, d/2.0)
                    
                    if min_dist > best_r:
                        best_r = min_dist
                        best_p = new_p
            
            if np.any(best_p != p):
                points[i] = best_p
                improved = True
        
        if not improved:
            break
    
    # Then do gradient ascent
    points = _gradient_ascent(points, max_iters=500, deadline=deadline)
    
    return points


def construct():
    """
    Construct 300 equal circles in a unit circle to maximize the common radius.
    """
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    # Initial hexagonal packing
    points = _hexagonal_initial(N)
    
    # Local optimization
    points = _local_optimize(points, deadline=deadline)
    
    # Try a few more restarts if time permits
    best_points = points.copy()
    best_score = _compute_score(points)
    
    while time.time() < deadline - 10:
        # Random perturbation
        np.random.seed(int(time.time() * 1000) % 10000)
        perturbed = points + np.random.randn(N, 2) * 0.01
        # Clip to unit circle
        norms = np.linalg.norm(perturbed, axis=1)
        mask = norms > 0.99
        if np.any(mask):
            perturbed[mask] *= 0.99 / norms[mask]
        
        optimized = _local_optimize(perturbed, deadline=deadline)
        score = _compute_score(optimized)
        
        if score > best_score:
            best_score = score
            best_points = optimized.copy()
    
    return best_points
# EVOLVE-BLOCK-END
