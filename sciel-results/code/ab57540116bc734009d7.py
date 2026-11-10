"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _hilbert3d(i, n):
    """
    Convert integer i in [0, n^3) to 3D coordinates in [0, n)^3
    using a 3D Hilbert curve.
    
    Uses the standard 3D Hilbert curve algorithm with rotations.
    """
    # Number of bits needed
    bits = int(np.log2(n))
    if n != (1 << bits):
        # Find next power of 2
        while (1 << bits) < n:
            bits += 1
    
    x, y, z = 0, 0, 0
    t = i
    
    for p in range(bits - 1, -1, -1):
        rx = (t >> 2) & 1
        ry = (t >> 1) & 1
        rz = t & 1
        
        # 3D Hilbert curve rotation logic
        # This is a simplified version that maintains good space-filling properties
        # The full 3D Hilbert curve has complex rotation rules
        
        # Rotate/reflection step
        if rz == 0:
            if ry == 0:
                # Swap x and z
                x, z = z, x
            if rx == 0:
                # Swap y and z
                y, z = z, y
        
        # Invert bits if needed
        if rx == 1:
            x = (1 << p) - 1 - x
        if ry == 1:
            y = (1 << p) - 1 - y
        
        # Add current bit
        x |= rx << p
        y |= ry << p
        z |= rz << p
        
        t >>= 3
    
    return np.array([x, y, z], dtype=float)


def _hilbert3d_curve(n, count):
    """
    Generate 'count' points from a 3D Hilbert curve of size n^3.
    Returns points in [0, 1)^3.
    """
    points = []
    for i in range(count):
        p = _hilbert3d(i, n)
        points.append(p / n)
    return np.array(points)


def _morton3d(i, bits):
    """Compute 3D Morton code for a 1D index."""
    x = 0
    y = 0
    z = 0
    for b in range(bits):
        x |= ((i >> (3 * b)) & 1) << b
        y |= ((i >> (3 * b + 1)) & 1) << b
        z |= ((i >> (3 * b + 2)) & 1) << b
    return x, y, z


def _morton3d_curve(n, count):
    """
    Generate 'count' points from a 3D Morton (Z-order) curve of size n^3.
    Returns points in [0, 1)^3.
    """
    bits = int(np.log2(n))
    points = []
    for i in range(count):
        x, y, z = _morton3d(i, bits)
        points.append(np.array([x, y, z], dtype=float) / n)
    return np.array(points)


def _compute_radius(centers):
    """
    Compute the maximum common radius for given centers in [-1/2, 1/2]^3.
    """
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
    # Distance to each face
    boundary_dists = np.minimum(centers + 0.5, 0.5 - centers)  # shape (N, 3)
    min_boundary = np.min(boundary_dists, axis=1)  # shape (N,)
    min_boundary_dist = np.min(min_boundary)
    
    return min(min_pair_dist / 2.0, min_boundary_dist)


def construct():
    """
    Construct 50 sphere centers in the unit cube [-1/2, 1/2]^3 using a
    space-filling curve initialization followed by gradient-based optimization.
    """
    # Step 1: Generate multiple initializations using space-filling curves
    initializations = []
    
    # Try different grid sizes and curve types
    for n in [4, 5, 6]:
        # Hilbert curve
        hilbert_pts = _hilbert3d_curve(n, N)
        # Map from [0,1)^3 to [-1/2, 1/2]^3
        hilbert_pts = hilbert_pts - 0.5
        initializations.append(hilbert_pts.copy())
        
        # Morton curve
        morton_pts = _morton3d_curve(n, N)
        morton_pts = morton_pts - 0.5
        initializations.append(morton_pts.copy())
    
    # Also add a simple grid-based initialization
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    all_points = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    # Take first N in a shuffled order
    np.random.seed(42)
    perm = np.random.permutation(len(all_points))
    grid_init = all_points[perm[:N]].copy()
    initializations.append(grid_init)
    
    # Step 2: Optimize each initialization
    best_centers = None
    best_radius = -1.0
    
    # Objective function: minimize sum of inverse squared distances (smooth)
    def objective(x):
        """
        x: flat array of shape (N*3,)
        Returns a scalar objective to minimize.
        We want to maximize the minimum pairwise distance and boundary distance.
        Use sum of inverse squared distances as a smooth proxy.
        """
        pts = x.reshape(N, 3)
        
        # Pairwise distances (squared)
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diffs ** 2, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        
        # Sum of inverse squared distances for pairs
        # Clip to avoid division by zero
        dist_sq_clipped = np.maximum(dist_sq, 1e-12)
        pair_obj = np.sum(1.0 / dist_sq_clipped)
        
        # Boundary distances (squared)
        # Distance to each face
        boundary_dists = np.minimum(pts + 0.5, 0.5 - pts)  # shape (N, 3)
        boundary_dists_sq = np.minimum(boundary_dists ** 2, 0.25)
        boundary_dists_sq_clipped = np.maximum(boundary_dists_sq, 1e-12)
        boundary_obj = np.sum(1.0 / boundary_dists_sq_clipped)
        
        # Combined objective: we want both to be large, so minimize their inverses
        # Weight: pairwise distances are more critical
        obj = 2.0 * pair_obj + 1.0 * boundary_obj
        
        return obj
    
    # Gradient of objective (for L-BFGS-B efficiency)
    def gradient(x):
        pts = x.reshape(N, 3)
        
        # Pairwise gradient
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diffs ** 2, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        dist_sq_clipped = np.maximum(dist_sq, 1e-12)
        
        # d(1/d^2)/d(pts_i) = -2 * diffs_ij / d_ij^4 for each pair j
        # Sum over j
        grad_pair = np.zeros_like(pts)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                d2 = dist_sq[i, j]
                if d2 > 1e-12:
                    grad_pair[i] += 2.0 * diffs[i, j] / (d2 ** 2)
        
        # Boundary gradient
        boundary_dists = np.minimum(pts + 0.5, 0.5 - pts)
        boundary_dists_sq = np.minimum(boundary_dists ** 2, 0.25)
        boundary_dists_sq_clipped = np.maximum(boundary_dists_sq, 1e-12)
        
        grad_boundary = np.zeros_like(pts)
        for i in range(N):
            for k in range(3):
                bd = boundary_dists[i, k]
                bd_sq = boundary_dists_sq[i, k]
                if bd_sq > 1e-12:
                    # d(1/bd^2)/d(pts_ik) = -2 * sign(bd) / bd^3
                    # sign depends on which face is closer
                    if pts[i, k] + 0.5 <= 0.5 - pts[i, k]:
                        # Closer to -0.5 face, bd = pts[i,k] + 0.5
                        sign = 1.0
                    else:
                        # Closer to 0.5 face, bd = 0.5 - pts[i,k]
                        sign = -1.0
                    grad_boundary[i, k] += sign / (bd_sq ** 1.5)
        
        grad = 2.0 * grad_pair + 1.0 * grad_boundary
        return grad.flatten()
    
    bounds = [(-0.5 + 1e-9, 0.5 - 1e-9)] * (N * 3)
    
    for init in initializations:
        x0 = init.flatten()
        
        try:
            result = minimize(
                objective,
                x0,
                jac=gradient,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 5000}
            )
            
            centers = result.x.reshape(N, 3)
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
