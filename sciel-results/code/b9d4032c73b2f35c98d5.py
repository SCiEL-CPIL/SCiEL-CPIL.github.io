"""Rewritten program: vectorized soft-constraint gradient descent for circle packing."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def construct():
    # Target radius from best known packing
    r_target = 0.036612798904
    
    # 1. Initialization: Triangular lattice
    # Calculate spacing for triangular lattice to fit N points in unit square
    # Density of triangular lattice: 2 / (sqrt(3) * s^2)
    # We want N points in area 1.
    # s = sqrt(2 / (N * sqrt(3)))
    s = np.sqrt(2.0 / (N * np.sqrt(3)))
    # Compress slightly to ensure initial overlaps (so gradient is non-zero)
    s *= 0.9
    h = s * np.sqrt(3) / 2.0
    
    # Generate a grid of points larger than the square
    # Estimate size
    n_cols = int(1.0 / s) + 4
    n_rows = int(1.0 / h) + 4
    
    xs = []
    ys = []
    for i in range(n_rows):
        for j in range(n_cols):
            x = j * s + (i % 2) * s / 2.0
            y = i * h
            xs.append(x)
            ys.append(y)
    
    pts = np.column_stack((xs, ys))
    # Center the lattice
    pts -= pts.mean(axis=0)
    
    # Select the N points closest to the center to form a roughly circular/square cluster
    dists_from_center = np.sum(pts**2, axis=1)
    indices = np.argsort(dists_from_center)
    centers = pts[indices[:N]].copy()
    
    # 2. Optimization
    # Define objective and gradient
    
    def loss_and_grad(x_flat):
        centers = x_flat.reshape(-1, 2)
        
        # Pairwise interactions
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2) + 1e-12)
        np.fill_diagonal(dist, np.inf)
        
        gap = 2.0 * r_target - dist
        mask = gap > 0
        
        # Loss: 0.5 * sum(gap^2)
        loss = 0.5 * np.sum(gap[mask] ** 2)
        
        # Gradient
        # d(0.5 * gap^2) / d(c_i) = gap * d(gap)/d(c_i)
        # gap = 2r - dist
        # d(dist)/d(c_i) = (c_i - c_j) / dist
        # d(gap)/d(c_i) = - (c_i - c_j) / dist
        # So grad_i = sum_j gap_ij * (-(c_i - c_j) / dist_ij)
        
        safe_dist = np.where(mask, dist, 1.0)
        factor = np.zeros_like(dist)
        factor[mask] = -gap[mask] / safe_dist[mask]
        
        grad_pair = np.sum(factor[:, :, np.newaxis] * diff, axis=1)
        
        # Wall interactions
        x = centers[:, 0]
        y = centers[:, 1]
        
        grad_wall = np.zeros_like(centers)
        loss_wall = 0.0
        
        # Left wall: x + 0.5 >= r
        gap_l = r_target - (x + 0.5)
        m_l = gap_l > 0
        loss_wall += 0.5 * np.sum(gap_l[m_l] ** 2)
        grad_wall[m_l, 0] += -gap_l[m_l]
        
        # Right wall: 0.5 - x >= r
        gap_r = r_target - (0.5 - x)
        m_r = gap_r > 0
        loss_wall += 0.5 * np.sum(gap_r[m_r] ** 2)
        grad_wall[m_r, 0] += gap_r[m_r]
        
        # Bottom wall: y + 0.5 >= r
        gap_b = r_target - (y + 0.5)
        m_b = gap_b > 0
        loss_wall += 0.5 * np.sum(gap_b[m_b] ** 2)
        grad_wall[m_b, 1] += -gap_b[m_b]
        
        # Top wall: 0.5 - y >= r
        gap_t = r_target - (0.5 - y)
        m_t = gap_t > 0
        loss_wall += 0.5 * np.sum(gap_t[m_t] ** 2)
        grad_wall[m_t, 1] += gap_t[m_t]
        
        total_loss = loss + loss_wall
        total_grad = (grad_pair + grad_wall).ravel()
        
        return total_loss, total_grad

    # Bounds for L-BFGS-B
    # Centers must be in [-0.5 + r, 0.5 - r]
    lo = -0.5 + r_target
    hi = 0.5 - r_target
    bounds = [(lo, hi)] * (2 * N)
    
    x0 = centers.ravel()
    
    # Run optimizer
    # L-BFGS-B is robust for this size
    res = minimize(
        loss_and_grad, 
        x0, 
        method='L-BFGS-B', 
        jac=True, 
        bounds=bounds, 
        options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
    )
    
    best_centers = res.x.reshape(-1, 2)
    
    # Optional: A few steps of simple gradient descent to polish if L-BFGS-B gets stuck?
    # Or just return the result.
    
    return best_centers
# EVOLVE-BLOCK-END
