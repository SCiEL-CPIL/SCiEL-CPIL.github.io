"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_radius(pts):
    """Compute the max common radius for the given centers."""
    if pts.shape[0] != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # Check inside square [-0.5, 0.5]^2
    if np.any(pts < -0.5) or np.any(pts > 0.5):
        return 0.0
    # Wall constraint
    r_wall = np.min(np.minimum(0.5 - np.abs(pts), 0.5))
    # Pairwise distances
    # Use efficient computation
    from scipy.spatial.distance import cdist
    dists = cdist(pts, pts)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    r_pair = min_pair / 2.0
    return min(r_wall, r_pair)


def _repulsion_energy(pts, k=100):
    """Compute a smooth surrogate objective: maximize min distance.
    We minimize negative of a soft-min of distances and wall distances."""
    # Wall distances
    wall_dists = np.minimum(0.5 - np.abs(pts), 0.5)
    # Pairwise distances
    from scipy.spatial.distance import cdist
    dists = cdist(pts, pts)
    np.fill_diagonal(dists, np.inf)
    
    # Soft-min over all constraints
    # r = min(wall_dists, pairwise_dists/2)
    # We want to maximize r, so minimize -r
    # Use log-sum-exp for soft-min: softmin(a) = -logsumexp(-a)/n
    all_constraints = np.concatenate([wall_dists, dists.ravel() / 2.0])
    # softmin = -log(sum(exp(-a_i / t))) * t / n for small t
    # For stability, shift by min
    m = np.min(all_constraints)
    # If m is very small, the packing is bad
    # Use a temperature parameter
    t = 0.01
    # softmin ≈ m + t * log(mean(exp(-(a_i - m)/t)))
    # = m - t * logsumexp(-(a_i - m)/t) / n
    shifted = -(all_constraints - m) / t
    # Clip for numerical stability
    shifted = np.clip(shifted, -100, 100)
    log_sum = np.log(np.mean(np.exp(shifted)))
    softmin = m + t * log_sum
    return -softmin  # minimize this


def _gradient_repulsion(pts, k=100):
    """Compute gradient of the repulsion energy."""
    # This is complex; we'll use a simpler approach with direct optimization
    pass


def _optimize_repulsion(pts0, max_iter=5000, lr=0.001, deadline=None):
    """Optimize using gradient descent on a smooth surrogate."""
    pts = pts0.copy()
    best_pts = pts.copy()
    best_radius = _compute_radius(pts)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute current radius
        r = _compute_radius(pts)
        if r > best_radius:
            best_radius = r
            best_pts = pts.copy()
        
        # Compute pairwise forces
        from scipy.spatial.distance import cdist
        dists = cdist(pts, pts)
        np.fill_diagonal(dists, 1.0)
        
        # Force: for pairs closer than some threshold, push apart
        # Use inverse distance squared for strong repulsion at close range
        # But we need to be careful about stability
        
        # Alternative: use a potential energy approach
        # E = sum_{i<j} 1/d_{ij}^2 + wall penalties
        # Gradient: dE/dp_i = sum_{j!=i} -2*(p_i - p_j)/d_{ij}^4 + wall_grad
        
        # Compute gradient
        grad = np.zeros_like(pts)
        
        # Pairwise repulsion
        # Vectorized: for each pair (i,j), force on i is -2*(p_i - p_j)/d^4
        # We can compute this as: grad[i] += sum_j 2*(p_j - p_i)/d_{ij}^4
        # = -2 * sum_j (p_i - p_j)/d_{ij}^4
        
        # Using matrix operations
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 2)
        dist_sq = np.sum(diff**2, axis=2)  # (N, N)
        np.fill_diagonal(dist_sq, 1.0)
        
        # Force magnitude: 2 / dist_sq^2 = 2/d^4
        force_mag = 2.0 / (dist_sq**2)
        np.fill_diagonal(force_mag, 0.0)
        
        # grad[i] = -sum_j force_mag[i,j] * (p_i - p_j)
        # = -sum_j force_mag[i,j] * diff[i,j]
        grad -= np.sum(force_mag[:, :, np.newaxis] * diff, axis=1)
        
        # Wall constraints: penalty for being too close to walls
        # If x < -0.5 + r or x > 0.5 - r, push inward
        # We'll add a soft wall force
        wall_margin = 0.001
        # For x coordinate
        # If pts[:,0] < -0.5 + margin, push right (positive)
        # If pts[:,0] > 0.5 - margin, push left (negative)
        # Use a smooth approximation
        eps = 0.01
        # Left wall: force = 1/(pts[:,0] + 0.5 + eps) if close
        # Actually, let's use a quadratic penalty for going outside a safe zone
        safe_left = -0.5 + 0.01
        safe_right = 0.5 - 0.01
        
        # Quadratic penalty for being outside [safe_left, safe_right]
        # For x
        x = pts[:, 0]
        y = pts[:, 1]
        
        # Penalty: 0.5 * max(0, safe_left - x)^2 + 0.5 * max(0, x - safe_right)^2
        # Gradient: -max(0, safe_left - x) + max(0, x - safe_right)
        grad[:, 0] += np.maximum(0, safe_left - x) - np.maximum(0, x - safe_right)
        grad[:, 1] += np.maximum(0, safe_left - y) - np.maximum(0, y - safe_right)
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-10:
            # Adaptive learning rate
            # Use a simple scheme: reduce lr if energy increases
            new_pts = pts - lr * grad / grad_norm * 0.1
            # Clamp to square
            new_pts = np.clip(new_pts, -0.5, 0.5)
            
            # Check if this improves
            new_r = _compute_radius(new_pts)
            if new_r >= r - 1e-10:
                pts = new_pts
                # Decay learning rate slowly
                lr *= 0.999
            else:
                # Try smaller step
                lr *= 0.5
                if lr < 1e-8:
                    lr = 0.001
    
    return best_pts, best_radius


def _initial_grid():
    """Create initial grid placement."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_pts = None
    best_radius = 0.0
    
    # Start with grid
    pts = _initial_grid()
    r = _compute_radius(pts)
    best_pts = pts.copy()
    best_radius = r
    
    # Multiple random restarts
    np.random.seed(42)
    n_restarts = 20
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Create perturbed initial point
        if restart == 0:
            pts = _initial_grid()
        else:
            # Start from best found so far with perturbation, or random
            if best_radius > 0.03:
                pts = best_pts + np.random.randn(N, 2) * 0.01
                pts = np.clip(pts, -0.5, 0.5)
            else:
                pts = np.random.uniform(-0.45, 0.45, (N, 2))
        
        # Optimize
        opt_pts, opt_r = _optimize_repulsion(pts, max_iter=2000, lr=0.01, deadline=deadline)
        
        if opt_r > best_radius:
            best_radius = opt_r
            best_pts = opt_pts.copy()
    
    # Final polish: try to improve with smaller steps
    if best_pts is not None and time.time() < deadline - 5:
        # Do a final optimization pass with very small steps
        for polish_iter in range(100):
            if time.time() > deadline:
                break
            from scipy.spatial.distance import cdist
            dists = cdist(best_pts, best_pts)
            np.fill_diagonal(dists, 1.0)
            diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
            dist_sq = np.sum(diff**2, axis=2)
            np.fill_diagonal(dist_sq, 1.0)
            force_mag = 2.0 / (dist_sq**2)
            np.fill_diagonal(force_mag, 0.0)
            grad = -np.sum(force_mag[:, :, np.newaxis] * diff, axis=1)
            
            # Wall forces
            safe_margin = 0.005
            x = best_pts[:, 0]
            y = best_pts[:, 1]
            grad[:, 0] += np.maximum(0, -0.5 + safe_margin - x) - np.maximum(0, x - 0.5 + safe_margin)
            grad[:, 1] += np.maximum(0, -0.5 + safe_margin - y) - np.maximum(0, y - 0.5 + safe_margin)
            
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-12:
                step = 0.001 * 0.95**polish_iter
                new_pts = best_pts - step * grad / grad_norm
                new_pts = np.clip(new_pts, -0.5, 0.5)
                new_r = _compute_radius(new_pts)
                if new_r > best_radius:
                    best_radius = new_r
                    best_pts = new_pts
    
    return best_pts
# EVOLVE-BLOCK-END
