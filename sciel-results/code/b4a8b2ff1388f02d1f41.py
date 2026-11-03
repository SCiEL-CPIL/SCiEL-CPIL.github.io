import numpy as np
import time
import math
from scipy.optimize import minimize

N = 300

def compute_score(points):
    """
    Compute the packing radius for a set of points.
    """
    # Boundary
    rad = np.linalg.norm(points, axis=1)
    min_bound = 1.0 - np.max(rad)
    
    # Pairwise
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    score = min(min_pair, min_bound)
    return score

def objective_and_grad(x):
    """
    Objective function: Sum of repulsive potentials between points and between points and boundary.
    We minimize this energy.
    """
    pts = x.reshape(-1, 2)
    n = pts.shape[0]
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, 1.0) # Avoid div by zero, will be masked or ignored
    
    # We want to push points apart. 
    # Using 1/d^2 potential. 
    # To avoid singularity, we can clip or use a softer potential.
    # Let's use V_pair = sum(1 / (d^2 + eps))
    eps = 1e-6
    V_pair = np.sum(1.0 / (dists_sq + eps))
    
    # Boundary potential
    rad_sq = np.sum(pts**2, axis=1)
    # We want points to be close to boundary but not outside.
    # If we just minimize 1/(1-r)^2, they go to boundary.
    # But if they go outside, r > 1, 1-r is negative, square is positive.
    # So they will try to minimize distance to boundary from inside?
    # Actually, if r > 1, (1-r)^2 is small, so 1/(1-r)^2 is large.
    # So it penalizes being outside.
    # It also penalizes being far from boundary (r small -> 1-r large -> 1/(1-r)^2 small).
    # So it pushes points to the boundary.
    
    # Let's use a potential that is zero at r=1 and increases as r moves away from 1?
    # Or just a penalty for being inside?
    # Standard approach: Minimize sum(1/d^2) for pairs, and for boundary, 
    # if we want them on the boundary, we can maximize r? 
    # But maximize r is not smooth if we use min.
    # Let's use a potential that pushes them to the boundary:
    # V_bound = sum( (1 - r)^2 ) ? No, that minimizes at r=1.
    # If we minimize V_bound, points go to r=1.
    # If we maximize r, points go to r=1.
    # Let's use V_bound = sum( -r ) ? No, that pushes to infinity.
    # Let's use V_bound = sum( (1-r)^2 ) for r < 1, and large penalty for r >= 1.
    # Actually, if we just use the pairwise repulsion, the points will spread out.
    # The boundary constraint is handled by the optimizer bounds or penalty.
    # Let's use bounds in minimize: -1 <= x, y <= 1.
    # And add a boundary repulsion to keep them from clumping in the center?
    # If we only use pairwise repulsion, they will form a cluster in the center?
    # No, they will expand until they hit the boundary or each other.
    # With bounds [-1, 1], they will hit the box boundary, not the circle boundary.
    # So we need a circular boundary penalty.
    
    # Let's use V_bound = sum( max(0, r - 1)^2 ) + sum( (1 - r)^2 * lambda )?
    # If we minimize sum(1/d^2), the system wants to expand.
    # If we constrain them to r <= 1, they will press against the boundary.
    # So we can just use bounds? No, bounds are box.
    # We need a soft constraint for the circle.
    
    # Let's use V_bound = sum( (r - 1)^2 ) for all points?
    # This minimizes at r=1.
    # So if we minimize V_pair + V_bound, points will try to be far apart AND at r=1.
    # This is exactly what we want.
    
    V_bound = np.sum((rad_sq - 1.0)**2) # Minimized at r=1
    
    # Gradient
    # d/dx V_pair = sum_j d/dx (1/(d_ij^2 + eps))
    # d/dx (1/(d^2+eps)) = -1/(d^2+eps)^2 * 2 * (x_i - x_j)
    
    grad = np.zeros_like(pts)
    
    # Pairwise gradient
    # dV_pair/dx_i = sum_{j != i} -2 * (x_i - x_j) / (d_ij^2 + eps)^2
    # Note: diff[i,j] = x_i - x_j
    
    # Vectorized gradient for pairs
    # dV/dx_i = -2 * sum_j (x_i - x_j) / (d_ij^2 + eps)^2
    # Let's compute the matrix of forces
    # F_ij = -2 * (x_i - x_j) / (d_ij^2 + eps)^2
    # grad_i = sum_j F_ij
    
    inv_d4 = 1.0 / (dists_sq + eps)**2
    # Mask diagonal
    mask = np.eye(n, dtype=bool)
    inv_d4[mask] = 0.0
    
    # Force on i from j is proportional to (x_i - x_j) * inv_d4[i,j]
    # Sum over j for each i
    # grad = -2 * (diff * inv_d4[:, :, np.newaxis]).sum(axis=1)
    # diff[i,j] = x_i - x_j
    # Force on i is sum_j -2 * (x_i - x_j) * inv_d4[i,j]
    # This is correct.
    
    grad_pair = -2.0 * np.sum(diff * inv_d4[:, :, np.newaxis], axis=1)
    
    # Boundary gradient
    # V_bound = sum((r^2 - 1)^2) = sum((x^2+y^2-1)^2)
    # d/dx = 2*(x^2+y^2-1) * 2x = 4x(x^2+y^2-1)
    grad_bound = 4.0 * pts * (rad_sq[:, np.newaxis] - 1.0)
    
    total_grad = grad_pair + grad_bound
    total_val = V_pair + V_bound
    
    return total_val, total_grad.flatten()

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin
    
    best_points = None
    best_score = -np.inf
    
    # Multi-start with optimization
    # We will run several restarts.
    # Each restart: Random init or Lattice init, then minimize.
    
    num_restarts = 20
    
    for r in range(num_restarts):
        if time.time() - start_time > time_limit - 5.0:
            break
            
        # Initialization
        if r % 2 == 0:
            # Hexagonal lattice
            s_init = np.sqrt(2 * np.pi / (np.sqrt(3) * N))
            s_gen = s_init * 0.9
            y_step = s_gen * np.sqrt(3) / 2
            max_j = int(1.0 / y_step) + 2
            max_i = int(1.0 / s_gen) + 2
            
            points_list = []
            for j in range(-max_j, max_j + 1):
                y = j * y_step
                offset = 0 if j % 2 == 0 else s_gen / 2
                for i in range(-max_i, max_i + 1):
                    x = i * s_gen + offset
                    if np.hypot(x, y) < 0.99:
                        points_list.append([x, y])
                        
            points = np.array(points_list)
            if len(points) > N:
                dists_to_center = np.linalg.norm(points, axis=1)
                idx = np.argsort(dists_to_center)[:N]
                points = points[idx]
            elif len(points) < N:
                while len(points) < N:
                    p = np.random.uniform(-0.9, 0.9, 2)
                    points = np.vstack([points, p])
                points = points[:N]
        else:
            # Random init
            points = np.random.uniform(-0.9, 0.9, size=(N, 2))
            
        # Add small noise to escape symmetry
        points += np.random.normal(0, 0.01, size=points.shape)
        
        # Flatten for optimizer
        x0 = points.flatten()
        
        # Bounds: -1 to 1 for each coordinate
        bounds = [(-1.0, 1.0)] * (2 * N)
        
        # Optimize
        try:
            res = minimize(objective_and_grad, x0, method='L-BFGS-B', jac=True, bounds=bounds, options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if res.success or res.x is not None:
                opt_points = res.x.reshape(-1, 2)
                
                # Check validity
                if np.all(np.isfinite(opt_points)):
                    score = compute_score(opt_points)
                    if score > best_score:
                        best_score = score
                        best_points = opt_points.copy()
                        
        except Exception as e:
            pass
            
    if best_points is None:
        # Fallback
        best_points = np.random.uniform(-0.9, 0.9, size=(N, 2))
        
    return best_points
