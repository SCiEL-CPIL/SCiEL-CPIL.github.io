# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # 1. Initialization: Hexagonal lattice
    # Estimate optimal radius r ~ 1/sqrt(N) * constant
    # For hex packing, density ~ 0.9069. r ~ sqrt(0.9069 * pi / (N * pi)) * 1? 
    # Area of circle = pi. Area of N circles = N * pi * r^2.
    # N * r^2 ~ 0.9. r ~ sqrt(0.9/N) ~ sqrt(0.9/300) ~ 0.0547.
    # Spacing d ~ 2r ~ 0.109.
    
    def hex_lattice_init():
        r_est = np.sqrt(0.9 / N)
        spacing = 2.0 * r_est
        # Generate points in a square region larger than unit circle
        k = int(2.0 / spacing) + 5
        points = []
        for i in range(-k, k + 1):
            for j in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if x**2 + y**2 <= (1.0 - spacing/2)**2:
                    points.append([x, y])
        
        points = np.array(points)
        if len(points) < N:
            # Fallback to grid if lattice failed
            n = int(np.sqrt(N)) + 5
            xs = np.linspace(-0.95, 0.95, n)
            ys = np.linspace(-0.95, 0.95, n)
            X, Y = np.meshgrid(xs, ys)
            points = np.column_stack([X.ravel(), Y.ravel()])
            mask = X.ravel()**2 + Y.ravel()**2 <= 0.95**2
            points = points[mask]
        
        # Sort by distance from center to get the innermost N points
        radii = np.linalg.norm(points, axis=1)
        idx = np.argsort(radii)[:N]
        return points[idx]

    points = hex_lattice_init()
    x0 = points.ravel()
    
    # 2. Optimization using L-BFGS-B with smooth objective
    # Objective: Minimize sum of exp(-lambda * dist) for pairs and boundary
    # This pushes points apart and away from boundary.
    
    lambda_val = 30.0  # Softness parameter. Higher = steeper.
    
    def objective_and_grad(x):
        pts = x.reshape(-1, 2)
        n = len(pts)
        
        # Calculate pairwise distances
        # diff[i, j] = pts[i] - pts[j]
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2) + 1e-12)
        np.fill_diagonal(dist, 1.0)  # Avoid self-interaction in sum
        
        # Mask upper triangle to avoid double counting
        iu = np.triu_indices(n, 1)
        d_pairs = dist[iu]
        
        # Objective term 1: Pairwise repulsion
        # J_pair = sum(exp(-lambda * d))
        exp_pairs = np.exp(-lambda_val * d_pairs)
        J_pair = np.sum(exp_pairs)
        
        # Gradient for pairs
        # dJ/dpts[i] = sum_j exp(-lambda * d_ij) * (-lambda) * (pts[i] - pts[j]) / d_ij
        # Vectorized gradient accumulation
        grad = np.zeros_like(pts)
        
        # We need to sum contributions from all pairs (i, j) where i < j
        # Contribution to grad[i] from pair (i, j): lambda * exp(-lambda * d_ij) * (pts[i] - pts[j]) / d_ij
        # Contribution to grad[j] from pair (i, j): -lambda * exp(-lambda * d_ij) * (pts[i] - pts[j]) / d_ij
        
        # To do this efficiently without loops:
        # Create a matrix of weights W[i, j] = lambda * exp(-lambda * d_ij) / d_ij for i != j
        # Then grad[i] = sum_j W[i, j] * (pts[i] - pts[j])
        # Note: W is symmetric.
        
        # Avoid division by zero
        dist_safe = dist.copy()
        dist_safe[dist_safe < 1e-8] = 1e-8
        
        W = lambda_val * np.exp(-lambda_val * dist_safe) / dist_safe
        np.fill_diagonal(W, 0.0)
        
        # grad[i] = sum_j W[i, j] * (pts[i] - pts[j])
        # = sum_j W[i, j] * pts[i] - sum_j W[i, j] * pts[j]
        # = pts[i] * sum_j W[i, j] - sum_j (W[i, j] * pts[j])
        
        row_sums = np.sum(W, axis=1)
        # Matrix multiplication: (W @ pts) gives sum_j W[i, j] * pts[j]
        W_pts = W @ pts
        
        grad_pair = pts * row_sums[:, np.newaxis] - W_pts
        
        # Objective term 2: Boundary repulsion
        # J_bound = sum(exp(-lambda * (1 - |c_i|)))
        radii = np.linalg.norm(pts, axis=1)
        radii_safe = np.maximum(radii, 1e-8)
        boundary_dist = 1.0 - radii
        exp_bound = np.exp(-lambda_val * boundary_dist)
        J_bound = np.sum(exp_bound)
        
        # Gradient for boundary
        # dJ/dpts[i] = -lambda * exp(-lambda * (1 - |c_i|)) * (pts[i] / |c_i|)
        grad_bound = -lambda_val * exp_bound[:, np.newaxis] * (pts / radii_safe[:, np.newaxis])
        
        total_grad = grad_pair + grad_bound
        total_obj = J_pair + J_bound
        
        return total_obj, total_grad.ravel()

    # Bounds: each coordinate in [-1, 1]
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    # Run optimization
    # maxiter limit to ensure we finish in time
    max_iter = 2000
    result = minimize(
        objective_and_grad, 
        x0, 
        method='L-BFGS-B', 
        jac=True, 
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
    )
    
    if time.time() > deadline:
        # If we are running out of time, return the current best (x0) or result.x if available
        # result.x is available even if not converged
        pass

    best_points = result.x.reshape(-1, 2)
    
    # 3. Post-processing: Ensure validity
    # Clip points to be strictly inside the unit circle
    radii = np.linalg.norm(best_points, axis=1)
    mask = radii > 1.0
    if np.any(mask):
        best_points[mask] = best_points[mask] / radii[mask, np.newaxis] * 0.9999
    
    # Optional: One more small refinement step if time permits?
    # The L-BFGS-B with bounds should keep them inside, but numerical errors can creep in.
    # The clipping above fixes the validity.
    
    return best_points
# EVOLVE-BLOCK-END
