"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _hilbert3d(i, n):
    """
    Convert integer i in [0, n^3) to 3D coordinates in [0, n)^3
    using a 3D Hilbert curve.
    """
    bits = int(np.log2(n))
    if n != (1 << bits):
        while (1 << bits) < n:
            bits += 1
    
    x, y, z = 0, 0, 0
    t = i
    
    for p in range(bits - 1, -1, -1):
        rx = (t >> 2) & 1
        ry = (t >> 1) & 1
        rz = t & 1
        
        if rz == 0:
            if ry == 0:
                x, z = z, x
            if rx == 0:
                y, z = z, y
        
        if rx == 1:
            x = (1 << p) - 1 - x
        if ry == 1:
            y = (1 << p) - 1 - y
        
        x |= rx << p
        y |= ry << p
        z |= rz << p
        
        t >>= 3
    
    return np.array([x, y, z], dtype=float)


def _hilbert3d_curve(n, count):
    """Generate 'count' points from a 3D Hilbert curve of size n^3."""
    points = []
    for i in range(count):
        p = _hilbert3d(i, n)
        points.append(p / n)
    return np.array(points)


def _compute_radius(centers):
    """Compute the maximum common radius for given centers in [-1/2, 1/2]^3."""
    centers = np.asarray(centers)
    N = len(centers)
    
    if N < 2:
        return 0.5
    
    # Pairwise distances
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diffs ** 2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_pair_dist = np.sqrt(np.min(dist_sq))
    
    # Boundary distances
    boundary_dists = np.minimum(centers + 0.5, 0.5 - centers)
    min_boundary = np.min(boundary_dists, axis=1)
    min_boundary_dist = np.min(min_boundary)
    
    return min(min_pair_dist / 2.0, min_boundary_dist)


def _objective_soft(x, power=2.0):
    """
    Smooth objective: sum of inverse powered distances.
    Higher power emphasizes smaller distances more.
    """
    pts = x.reshape(N, 3)
    
    # Pairwise distances
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diffs ** 2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    
    # Use inverse powered distance
    dist_sq_clipped = np.maximum(dist_sq, 1e-10)
    pair_obj = np.sum(dist_sq_clipped ** (-power / 2.0))
    
    # Boundary distances
    boundary_dists = np.minimum(pts + 0.5, 0.5 - pts)
    boundary_dists_clipped = np.maximum(boundary_dists, 1e-5)
    boundary_obj = np.sum(boundary_dists_clipped ** (-power))
    
    # Weight boundary more heavily to keep spheres inside
    return pair_obj + 2.0 * boundary_obj


def _gradient_soft(x, power=2.0):
    """Gradient of the soft objective."""
    pts = x.reshape(N, 3)
    grad = np.zeros_like(pts)
    
    # Pairwise gradient
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diffs ** 2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    dist_sq_clipped = np.maximum(dist_sq, 1e-10)
    
    # d/d(pts_i) sum_j (d_ij^2)^(-power/2) = -power * sum_j diffs_ij * (d_ij^2)^(-(power/2)-1)
    inv_power = (power / 2.0) + 1.0
    factor = -power * np.ones((N, N))
    np.fill_diagonal(factor, 0.0)
    
    grad_pair = np.zeros_like(pts)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            d2 = dist_sq_clipped[i, j]
            grad_pair[i] += factor[i, j] * diffs[i, j] * (d2 ** (-inv_power))
    
    grad += grad_pair
    
    # Boundary gradient
    boundary_dists = np.minimum(pts + 0.5, 0.5 - pts)
    boundary_dists_clipped = np.maximum(boundary_dists, 1e-5)
    
    grad_boundary = np.zeros_like(pts)
    for i in range(N):
        for k in range(3):
            bd = boundary_dists[i, k]
            bd_clipped = boundary_dists_clipped[i, k]
            
            # Determine which face is closer
            if pts[i, k] + 0.5 <= 0.5 - pts[i, k]:
                sign = 1.0  # closer to -0.5 face
            else:
                sign = -1.0  # closer to 0.5 face
            
            # d/d(pts_ik) (bd^(-power)) = -power * sign * bd^(-power-1)
            grad_boundary[i, k] = 2.0 * (-power) * sign * (bd_clipped ** (-power - 1))
    
    grad += grad_boundary
    
    return grad.flatten()


def _objective_repulsive(x, beta=10.0):
    """
    Alternative objective: minimize sum of exp(-beta * dist) for pairs,
    plus boundary penalty.
    This is smoother and may escape local minima better.
    """
    pts = x.reshape(N, 3)
    
    # Pairwise: sum of exp(-beta * d_ij)
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diffs ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    # Clip to avoid overflow
    dist_clipped = np.maximum(dist, 1e-8)
    pair_obj = np.sum(np.exp(-beta * dist_clipped))
    
    # Boundary: sum of exp(-beta * bd)
    boundary_dists = np.minimum(pts + 0.5, 0.5 - pts)
    boundary_dists_clipped = np.maximum(boundary_dists, 1e-8)
    boundary_obj = np.sum(np.exp(-beta * boundary_dists_clipped))
    
    return pair_obj + 2.0 * boundary_obj


def _gradient_repulsive(x, beta=10.0):
    """Gradient of the repulsive objective."""
    pts = x.reshape(N, 3)
    grad = np.zeros_like(pts)
    
    # Pairwise gradient
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diffs ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    dist_clipped = np.maximum(dist, 1e-8)
    
    # d/d(pts_i) sum_j exp(-beta * d_ij) = -beta * sum_j exp(-beta * d_ij) * (diffs_ij / d_ij)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            d = dist_clipped[i, j]
            if d > 1e-8:
                grad[i] += -beta * np.exp(-beta * d) * diffs[i, j] / d
    
    # Boundary gradient
    boundary_dists = np.minimum(pts + 0.5, 0.5 - pts)
    boundary_dists_clipped = np.maximum(boundary_dists, 1e-8)
    
    for i in range(N):
        for k in range(3):
            bd = boundary_dists[i, k]
            bd_clipped = boundary_dists_clipped[i, k]
            
            if pts[i, k] + 0.5 <= 0.5 - pts[i, k]:
                sign = 1.0
            else:
                sign = -1.0
            
            grad[i, k] += 2.0 * (-beta) * np.exp(-beta * bd_clipped) * sign
    
    return grad.flatten()


def _optimize_single(x0, bounds, method='L-BFGS-B', maxiter=300, time_limit=20):
    """
    Optimize from a single starting point.
    Uses multiple objective functions in sequence.
    """
    start_time = time.time()
    
    # Stage 1: Soft objective with moderate power
    try:
        result = minimize(
            _objective_soft,
            x0,
            jac=lambda x: _gradient_soft(x, power=2.0),
            method=method,
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 10000}
        )
        x_opt = result.x
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            return x_opt
    except Exception:
        pass
    
    # Stage 2: Higher power for more emphasis on minimum distance
    try:
        result = minimize(
            _objective_soft,
            x_opt,
            jac=lambda x: _gradient_soft(x, power=4.0),
            method=method,
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 10000}
        )
        x_opt = result.x
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            return x_opt
    except Exception:
        pass
    
    # Stage 3: Repulsive objective for final polishing
    try:
        result = minimize(
            _objective_repulsive,
            x_opt,
            jac=lambda x: _gradient_repulsive(x, beta=8.0),
            method=method,
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 10000}
        )
        x_opt = result.x
    except Exception:
        pass
    
    return x_opt


def construct():
    """
    Construct 50 sphere centers in the unit cube [-1/2, 1/2]^3.
    """
    # Step 1: Generate multiple initializations
    initializations = []
    
    # Hilbert curve with different grid sizes
    for n in [4, 5, 6, 7]:
        hilbert_pts = _hilbert3d_curve(n, N)
        hilbert_pts = hilbert_pts - 0.5
        initializations.append(hilbert_pts.copy())
    
    # Grid-based initialization with different spacings
    for m in [3, 4, 5]:
        ticks = (np.arange(m) + 0.5) / m - 0.5
        all_points = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        if len(all_points) >= N:
            np.random.seed(42 + m)
            perm = np.random.permutation(len(all_points))
            grid_init = all_points[perm[:N]].copy()
            initializations.append(grid_init)
    
    # Add a few random initializations for diversity
    np.random.seed(123)
    for _ in range(3):
        rand_init = np.random.uniform(-0.4, 0.4, size=(N, 3))
        initializations.append(rand_init)
    
    # Step 2: Optimize each initialization
    best_centers = None
    best_radius = -1.0
    bounds = [(-0.5 + 1e-9, 0.5 - 1e-9)] * (N * 3)
    
    # Track time to ensure we finish within 120s
    overall_start = time.time()
    time_per_opt = 8.0  # seconds per optimization
    
    for idx, init in enumerate(initializations):
        # Check remaining time
        elapsed = time.time() - overall_start
        if elapsed + time_per_opt > 110:
            break
        
        x0 = init.flatten()
        
        try:
            x_opt = _optimize_single(x0, bounds, maxiter=200, time_limit=time_per_opt)
            
            centers = x_opt.reshape(N, 3)
            centers = np.clip(centers, -0.5 + 1e-8, 0.5 - 1e-8)
            
            radius = _compute_radius(centers)
            
            if radius > best_radius:
                best_radius = radius
                best_centers = centers.copy()
                
        except Exception:
            continue
    
    # If no optimization succeeded, use the best initialization
    if best_centers is None:
        best_centers = initializations[0]
        best_radius = _compute_radius(best_centers)
    
    return best_centers

# EVOLVE-BLOCK-END
