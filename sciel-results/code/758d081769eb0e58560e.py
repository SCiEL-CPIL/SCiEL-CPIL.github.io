"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _compute_radius(pts):
    """Compute the valid packing radius for given centers."""
    if len(pts) < 2:
        return 0.0
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Boundary distances
    boundary_dists = np.minimum(
        np.minimum(0.5 - pts[:, 0], 0.5 + pts[:, 0]),
        np.minimum(0.5 - pts[:, 1], 0.5 + pts[:, 1])
    )
    min_boundary = np.min(boundary_dists)
    
    return min(min_pair_dist / 2, min_boundary)


def _soft_min_obj(pts_flat, beta=50.0):
    """Smooth surrogate for -min(pairwise_dist/2, boundary_dist)."""
    pts = pts_flat.reshape(-1, 2)
    
    # Pairwise distances (vectorized)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=-1)
    np.fill_diagonal(sq_dists, np.inf)
    
    # Soft min of pairwise distances / 2
    # Use log-sum-exp trick for numerical stability
    vals = sq_dists / 4.0  # (dist/2)^2
    max_val = np.max(vals)
    soft_min_sq = max_val + np.log(np.mean(np.exp(-vals + max_val)))
    soft_min_pair = np.sqrt(soft_min_sq)
    
    # Boundary distances
    bx = np.minimum(0.5 - pts[:, 0], 0.5 + pts[:, 0])
    by = np.minimum(0.5 - pts[:, 1], 0.5 + pts[:, 1])
    boundary = np.minimum(bx, by)
    
    # Soft min combining pair and boundary
    # Combine: we want min(soft_min_pair, min_boundary)
    # Use a smooth min of the two values
    a = soft_min_pair
    b = np.min(boundary)
    m = np.minimum(a, b)
    if beta > 0:
        soft_combined = m - np.logaddexp(-a * beta, -b * beta) / beta
    else:
        soft_combined = m
    
    return -soft_combined  # Negate for minimization


def _soft_min_grad(pts_flat, beta=50.0):
    """Gradient of the soft min objective."""
    pts = pts_flat.reshape(-1, 2)
    n = len(pts)
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=-1)
    np.fill_diagonal(sq_dists, np.inf)
    
    vals = sq_dists / 4.0
    max_val = np.max(vals)
    exp_neg = np.exp(-vals + max_val)
    mean_exp = np.mean(exp_neg)
    soft_min_sq = max_val + np.log(mean_exp)
    
    # Gradient of soft_min_sq w.r.t. sq_dists
    # d(soft_min_sq)/d(sq_dists_ij) = -exp(-sq_dists_ij/4 + max_val) / (4 * mean_exp)
    grad_sq = -exp_neg / (4.0 * mean_exp)
    np.fill_diagonal(grad_sq, 0.0)
    
    # Gradient of soft_min_pair = sqrt(soft_min_sq)
    if soft_min_sq > 1e-12:
        grad_pair = grad_sq / (2.0 * np.sqrt(soft_min_sq))
    else:
        grad_pair = np.zeros_like(grad_sq)
    
    # Accumulate gradient for pairwise part
    grad_pts = np.zeros_like(pts)
    for i in range(n):
        for j in range(n):
            if i != j:
                grad_pts[i] += 2.0 * (pts[i] - pts[j]) * grad_pair[i, j]
    
    # Boundary gradient
    bx = np.minimum(0.5 - pts[:, 0], 0.5 + pts[:, 0])
    by = np.minimum(0.5 - pts[:, 1], 0.5 + pts[:, 1])
    boundary = np.minimum(bx, by)
    b_min = np.min(boundary)
    
    # Gradient of b_min
    grad_b = np.zeros_like(pts)
    min_idx = np.argmin(boundary)
    if boundary[min_idx] == bx[min_idx]:
        if 0.5 - pts[min_idx, 0] <= 0.5 + pts[min_idx, 0]:
            grad_b[min_idx, 0] = -1.0
        else:
            grad_b[min_idx, 0] = 1.0
    else:
        if 0.5 - pts[min_idx, 1] <= 0.5 + pts[min_idx, 1]:
            grad_b[min_idx, 1] = -1.0
        else:
            grad_b[min_idx, 1] = 1.0
    
    # Combine gradients for soft min of a and b
    a = np.sqrt(soft_min_sq)
    b = b_min
    m = np.minimum(a, b)
    
    if beta > 0:
        ea = np.exp(-a * beta)
        eb = np.exp(-b * beta)
        w_a = ea / (ea + eb)
        w_b = eb / (ea + eb)
    else:
        if a <= b:
            w_a, w_b = 1.0, 0.0
        else:
            w_a, w_b = 0.0, 1.0
    
    total_grad = w_a * grad_pts + w_b * grad_b
    
    # Reshape to flat
    return -total_grad.flatten()  # Negate because we're minimizing -soft_min


def _hex_init(n_points):
    """Generate hexagonal lattice initialization with jitter."""
    # Hexagonal packing: rows offset by half spacing
    # For n points, use roughly sqrt(n) per side
    m = int(np.ceil(np.sqrt(n_points * 2 / np.sqrt(3))))
    
    pts = []
    row = 0
    while len(pts) < n_points:
        y = -0.5 + (row + 0.5) * (np.sqrt(3) / (2 * m))
        if y > 0.5:
            break
        x_start = -0.5 + (0.5 if row % 2 == 1 else 0.0) * (1.0 / m)
        for col in range(m):
            if len(pts) >= n_points:
                break
            x = x_start + col * (1.0 / m)
            if -0.49 <= x <= 0.49 and -0.49 <= y <= 0.49:
                pts.append([x, y])
        row += 1
    
    pts = np.array(pts[:n_points])
    
    # Add small jitter to break symmetry
    rng = np.random.default_rng(42)
    jitter = rng.uniform(-0.005, 0.005, size=pts.shape)
    pts += jitter
    
    # Clip to valid range
    pts = np.clip(pts, -0.49, 0.49)
    
    return pts


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    pts = _hex_init(N)
    pts_flat = pts.flatten()
    
    best_pts = pts.copy()
    best_radius = _compute_radius(pts)
    
    # Try multiple beta values for soft min
    for beta in [10.0, 30.0, 100.0]:
        if time.time() > deadline - 10:
            break
        
        def obj(x):
            return _soft_min_obj(x, beta)
        
        def grad(x):
            return _soft_min_grad(x, beta)
        
        try:
            result = minimize(
                obj,
                pts_flat.copy(),
                jac=grad,
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
            )
            
            new_pts = result.x.reshape(-1, 2)
            new_radius = _compute_radius(new_pts)
            
            if new_radius > best_radius:
                best_radius = new_radius
                best_pts = new_pts.copy()
                pts_flat = new_pts.flatten()
                
        except Exception:
            pass
    
    # Final polish: local search on closest pairs
    if time.time() < deadline - 5:
        pts = best_pts.copy()
        for _ in range(500):
            if time.time() > deadline - 2:
                break
            
            # Find closest pair
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            sq_dists = np.sum(diff**2, axis=-1)
            np.fill_diagonal(sq_dists, np.inf)
            min_idx = np.unravel_index(np.argmin(sq_dists), sq_dists.shape)
            i, j = min_idx
            min_dist = np.sqrt(sq_dists[i, j])
            
            # Check boundary
            boundary_dists = np.minimum(
                np.minimum(0.5 - pts[:, 0], 0.5 + pts[:, 0]),
                np.minimum(0.5 - pts[:, 1], 0.5 + pts[:, 1])
            )
            min_boundary = np.min(boundary_dists)
            
            current_r = min(min_dist / 2, min_boundary)
            
            # Move i and j apart
            direction = pts[i] - pts[j]
            norm = np.linalg.norm(direction)
            if norm < 1e-10:
                direction = np.array([1.0, 0.0])
                norm = 1.0
            direction /= norm
            
            # Try small moves
            for step in [0.001, 0.005, 0.01]:
                new_pts = pts.copy()
                new_pts[i] += direction * step
                new_pts[j] -= direction * step
                
                # Check bounds
                if np.all(new_pts >= -0.5) and np.all(new_pts <= 0.5):
                    new_r = _compute_radius(new_pts)
                    if new_r > current_r:
                        pts = new_pts
                        break
        
        final_radius = _compute_radius(pts)
        if final_radius > best_radius:
            best_radius = final_radius
            best_pts = pts.copy()
    
    return best_pts
# EVOLVE-BLOCK-END
