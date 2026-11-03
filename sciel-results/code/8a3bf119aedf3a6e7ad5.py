"""Optimized circle packing: 300 equal circles in unit circle.
Uses hexagonal lattice initialization + vectorized force-directed optimization
with deadline guard.
"""
import numpy as np
import time

N = 300
DEADLINE_FRACTION = 0.85  # safety margin

def _hex_lattice_init(n):
    """Generate a hexagonal lattice of points inside the unit disk, return n points."""
    # Estimate spacing: area per point ~ pi/n, hex spacing d ~ sqrt(4*pi*n/(3*sqrt(3)*n)) ... 
    # Actually for hex packing, density = pi/(2*sqrt(3)) ≈ 0.9069
    # Total area = pi, so effective area per circle center = pi/n
    # Hex cell area = d^2 * sqrt(3)/2, so d ≈ sqrt(2*pi/(n*sqrt(3)/2)) = sqrt(4*pi/(n*sqrt(3)))
    d = np.sqrt(4.0 * np.pi / (n * np.sqrt(3.0)))
    
    while True:
        k = int(1.0 / d) + 3
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * d
                y = j * d * np.sqrt(3.0) / 2.0
                rows.append((x, y))
        pts = np.array(rows, dtype=np.float64)
        r = np.hypot(pts[:, 0], pts[:, 1])
        # Keep points well inside
        pts = pts[r <= 1.0 - d * 0.6]
        if len(pts) >= n:
            # Sort by distance from origin, take closest n
            idx = np.argsort(r[r <= 1.0 - d * 0.6], kind="stable")[:n]
            return pts[idx]
        d *= 0.97


def _compute_score(pts):
    """Compute the packing radius: min of half-min-pairwise-dist and min(1-|c_i|)."""
    n = len(pts)
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    # Distance to boundary
    radii = np.hypot(pts[:, 0], pts[:, 1])
    min_boundary = np.min(1.0 - radii)
    return min(0.5 * min_pair, min_boundary)


def construct():
    start_time = time.time()
    deadline = start_time + 120.0 * DEADLINE_FRACTION
    
    # Initialize with hex lattice
    pts = _hex_lattice_init(N)
    if len(pts) != N:
        # Fallback: random points
        pts = np.random.randn(N, 2) * 0.5
        # Project inside unit circle
        radii = np.hypot(pts[:, 0], pts[:, 1])
        mask = radii > 0.9
        pts[mask] = pts[mask] / radii[mask, np.newaxis] * 0.9
    
    best_pts = pts.copy()
    best_score = _compute_score(best_pts)
    
    # Force-directed optimization (vectorized)
    # We want to maximize min(0.5 * min_pairwise_dist, min_boundary)
    # Use a soft-min approach: minimize -soft_min of the two terms
    # Or equivalently, use a repulsive force that pushes points apart and toward boundary
    
    # Strategy: iterative local optimization
    # At each iteration:
    #   1. Compute all pairwise distances
    #   2. For each pair, if distance < 2*r_est, apply repulsive force
    #   3. For each point, if it's too close to boundary, push inward
    #   4. Update positions with adaptive step size
    
    # Estimate initial radius
    r_est = _compute_score(pts)
    if r_est <= 0:
        r_est = 0.05
    
    lr = 0.001
    n_iter = 0
    no_improve_count = 0
    patience = 2000
    
    while time.time() < deadline:
        n_iter += 1
        
        # Compute pairwise distances (vectorized)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        # Compute radii
        radii = np.hypot(pts[:, 0], pts[:, 1])
        
        # Current score
        min_pair = np.min(dist)
        min_boundary = np.min(1.0 - radii)
        current_score = min(0.5 * min_pair, min_boundary)
        
        if current_score > best_score + 1e-12:
            best_score = current_score
            best_pts = pts.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        if no_improve_count > patience:
            # Try to escape local minimum with a small perturbation
            pts = pts + np.random.randn(N, 2) * 0.005
            # Keep inside
            radii = np.hypot(pts[:, 0], pts[:, 1])
            mask = radii > 0.99
            pts[mask] = pts[mask] / radii[mask, np.newaxis] * 0.99
            no_improve_count = 0
            lr = max(lr * 0.8, 0.0001)
            continue
        
        # Compute forces
        # Repulsive force between pairs: F = (2*r_est - dist) / dist^2 * direction (if dist < 2*r_est)
        # Attractive force to maintain spacing: for pairs with dist < 2*r_est, push apart
        # Boundary force: push points inward if too close to boundary
        
        forces = np.zeros_like(pts)
        
        # Pairwise repulsion: for each pair (i,j) with dist < 2*r_est
        # Force on i from j: (2*r_est - dist_ij) / dist_ij^2 * (pts_i - pts_j)
        # This is O(N^2) but vectorized
        
        # Only consider pairs closer than 2*r_est
        threshold = 2.0 * r_est * 1.5  # slightly larger to allow some attraction
        close_mask = dist < threshold
        np.fill_diagonal(close_mask, False)
        
        # Compute forces from close pairs
        # For each i, sum over j where close_mask[i,j]
        # F_i = sum_j (2*r_est - dist_ij) / dist_ij^2 * (pts_i - pts_j) for dist < 2*r_est
        #      + sum_j (dist_ij - 2*r_est) / dist_ij^2 * (pts_j - pts_i) for 2*r_est < dist < threshold (attractive)
        
        # Simplified: just use repulsion for all close pairs
        # F_i = sum_{j: dist_ij < threshold} (2*r_est - dist_ij) / dist_ij^2 * (pts_i - pts_j)
        
        # Vectorized computation:
        # For each pair (i,j), contribution to force on i is:
        #   coeff * diff_ij where coeff = (2*r_est - dist_ij) / dist_ij^2
        # and diff_ij = pts_i - pts_j
        
        # Build coefficient matrix
        coeff = np.zeros_like(dist)
        valid = close_mask & (dist > 1e-10)
        coeff[valid] = (2.0 * r_est - dist[valid]) / (dist[valid] ** 2)
        
        # Force on i = sum_j coeff[i,j] * (pts_i - pts_j)
        # = sum_j coeff[i,j] * pts_i - sum_j coeff[i,j] * pts_j
        # = pts_i * sum_j coeff[i,j] - sum_j coeff[i,j] * pts_j
        
        row_sums = np.sum(coeff, axis=1)  # shape (N,)
        # sum_j coeff[i,j] * pts_j: matrix-vector product
        force_pair = pts * row_sums[:, np.newaxis] - coeff @ pts
        
        # Boundary force: push points inward proportional to how close they are to boundary
        # If 1 - |c_i| < r_est, push inward
        boundary_dist = 1.0 - radii
        boundary_force = np.zeros_like(pts)
        close_to_boundary = boundary_dist < r_est * 1.5
        for idx in np.where(close_to_boundary)[0]:
            direction = -pts[idx] / (radii[idx] + 1e-10)  # inward direction
            strength = (r_est * 1.5 - boundary_dist[idx]) / (r_est + 1e-10)
            boundary_force[idx] = direction * strength * 0.5
        
        forces = force_pair + boundary_force
        
        # Update positions
        pts += lr * forces
        
        # Ensure points stay inside the unit circle
        radii = np.hypot(pts[:, 0], pts[:, 1])
        mask = radii > 0.999
        if np.any(mask):
            pts[mask] = pts[mask] / radii[mask, np.newaxis] * 0.999
        
        # Adapt learning rate
        if current_score > best_score:
            lr = min(lr * 1.02, 0.01)
        else:
            lr = max(lr * 0.999, 0.00005)
        
        # Update radius estimate
        r_est = max(current_score, 0.01)
        
        # Periodic check
        if n_iter % 1000 == 0:
            pass  # Could add logging
    
    # Final polish: use the best points found
    return best_pts
