"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as so
import scipy.spatial as ss

def _hex_init(n):
    """
    Generate a hexagonal lattice, then prune to exactly n points
    by iteratively removing the point with the smallest minimum neighbor distance.
    """
    # Estimate spacing
    # Area per point ~ 1/n
    # Hex area ~ sqrt(3)/2 * d^2
    # d ~ sqrt(2 / (n * sqrt(3)))
    d = np.sqrt(2.0 / (n * np.sqrt(3.0)))
    # Adjust d slightly to ensure we have enough points
    # We want M >= n.
    # Let's try a range of d values or just start with a slightly smaller d.
    d = d * 0.95
    
    row_h = d * np.sqrt(3.0) / 2.0
    col_w = d
    
    # Generate points
    pts = []
    y = -0.5 - row_h
    j = 0
    while y <= 0.5 + row_h:
        offset = (col_w / 2.0) if (j % 2 != 0) else 0.0
        x = -0.5 - col_w + offset
        while x <= 0.5 + col_w:
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                pts.append([x, y])
            x += col_w
        y += row_h
        j += 1
        
    pts = np.array(pts)
    
    if len(pts) < n:
        # Fallback if too few points (shouldn't happen with 0.95 factor, but safe)
        # Just return a grid
        m = int(np.ceil(np.sqrt(n)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:n]
        
    # Prune
    current_pts = pts
    while len(current_pts) > n:
        # Compute pairwise distances
        diff = current_pts[:, np.newaxis, :] - current_pts[np.newaxis, :, :]
        dists_sq = np.sum(diff ** 2, axis=-1)
        np.fill_diagonal(dists_sq, np.inf)
        
        # Find the point with the smallest minimum distance to any other point
        min_dists = np.min(dists_sq, axis=1)
        idx_to_remove = np.argmin(min_dists)
        
        # Remove it
        current_pts = np.delete(current_pts, idx_to_remove, axis=0)
        
    return current_pts

def _energy_and_grad(x_flat):
    """
    Compute repulsive energy and gradient for circle packing.
    E = sum_{i<j} 1/d_ij^2 + boundary_term
    """
    n = N
    pts = x_flat.reshape((n, 2))
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=-1)
    
    # Avoid division by zero
    dists_sq_safe = np.maximum(dists_sq, 1e-12)
    np.fill_diagonal(dists_sq_safe, 1.0) # Ignore self
    
    # Energy: sum 1/d^2
    # We only sum i < j to avoid double counting
    iu = np.triu_indices(n, k=1)
    e_pairs = np.sum(1.0 / dists_sq_safe[iu])
    
    # Gradient of 1/d^2 w.r.t pts
    # d/dx (1/||x-y||^2) = -2 (x-y) / ||x-y||^4
    # Sum over j for each i
    # grad_i = -2 * sum_{j != i} (x_i - x_j) / dists_sq_ij^2
    
    # dists_sq_safe is (N, N)
    # We need to compute sum_j (x_i - x_j) / dists_sq_ij^2
    # Let W_ij = 1 / dists_sq_ij^2
    # grad_i = -2 * ( W_i . (x_i - X) ) = -2 * ( x_i * sum_j W_ij - sum_j W_ij x_j )
    
    W = 1.0 / (dists_sq_safe ** 2)
    np.fill_diagonal(W, 0.0)
    
    row_sums = np.sum(W, axis=1)
    weighted_sum = W @ pts
    
    grad_pairs = -2.0 * (pts * row_sums[:, np.newaxis] - weighted_sum)
    
    # Boundary term
    # We want to keep points away from boundary.
    # E_b = sum_i ( 1/(0.5-|x_i|^2) + 1/(0.5-|y_i|^2) )
    # This is steep near boundary.
    # To be safe, add a small penalty if they get too close.
    # Actually, L-BFGS-B handles bounds. 
    # But adding a soft boundary term helps prevent sticking to corners.
    eps = 1e-6
    bx = 0.5 - np.abs(pts[:, 0])
    by = 0.5 - np.abs(pts[:, 1])
    
    # Penalty if bx < 0.01 or by < 0.01
    # Let's use a simple quadratic penalty for violation, or just rely on bounds.
    # Reliance on bounds is usually sufficient for L-BFGS-B.
    # However, if we want to maximize the *minimum* distance, and the optimal solution
    # has points on the boundary, the repulsive force from other points will push them
    # towards the boundary, and the bound will stop them. This is correct.
    
    # So we can just return the pairwise gradient.
    # But to ensure they don't cluster *at* the boundary corners, a slight boundary repulsion
    # might help symmetry breaking.
    
    # Let's add a very small boundary repulsion to avoid degenerate corner solutions
    # if the lattice is symmetric.
    lambda_b = 0.1
    grad_bx = np.zeros_like(pts[:, 0])
    grad_by = np.zeros_like(pts[:, 1])
    
    # Only apply if very close to boundary
    mask_x = bx < 0.05
    mask_y = by < 0.05
    
    # Force pushing away from boundary
    # If x > 0, push left (-1). If x < 0, push right (+1).
    # Magnitude proportional to 1/bx^2
    grad_bx[mask_x] = -lambda_b * 2.0 * np.sign(pts[mask_x, 0]) / (bx[mask_x] ** 2)
    grad_by[mask_y] = -lambda_b * 2.0 * np.sign(pts[mask_y, 1]) / (by[mask_y] ** 2)
    
    # Wait, the energy term for boundary would be 1/bx^2.
    # d/dx (1/(0.5-x)^2) = 2/(0.5-x)^3.
    # My gradient above is for 1/bx^2 where bx = 0.5-|x|.
    # If x > 0, bx = 0.5-x. d/dx (1/(0.5-x)^2) = 2/(0.5-x)^3.
    # So grad = 2 / bx^3 * sign?
    # Let's just use the pairwise gradient to keep it simple and robust.
    # L-BFGS-B bounds are hard.
    
    total_grad = grad_pairs.ravel()
    
    return e_pairs, total_grad

def construct():
    # Step 1: Initialize
    pts = _hex_init(N)
    x0 = pts.ravel()
    
    # Step 2: Optimize
    # Use L-BFGS-B
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    # We minimize the repulsive energy.
    # A lower energy corresponds to larger minimum distances.
    
    res = so.minimize(
        _energy_and_grad, 
        x0, 
        method='L-BFGS-B', 
        jac=True, 
        bounds=bounds,
        options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    
    pts_opt = res.x.reshape((N, 2))
    
    # Optional: A few more steps of simple gradient descent to polish?
    # L-BFGS-B should be good.
    
    return pts_opt
# EVOLVE-BLOCK-END
