"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # 1. Initialize with Hexagonal Lattice
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        points = points[mask]
        if len(points) >= N:
            points = points[np.argsort(radius, kind="stable")[:N]]
            break
        spacing *= 0.98
    
    points = points.astype(np.float64)
    
    # 2. Define Energy Function and Gradient
    # We minimize E = sum_{i<j} 1/d_ij^2 + lambda * sum_i 1/(1-|c_i|)^2
    # This maximizes the minimum distance between points and to the boundary.
    
    def energy(params):
        p = params.reshape(-1, 2)
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=-1)
        np.fill_diagonal(dist_sq, 1.0)
        
        repulsion = np.sum(1.0 / dist_sq)
        
        dist_to_center = np.sqrt(np.sum(p**2, axis=1))
        margin = 1.0 - dist_to_center
        margin_clipped = np.maximum(margin, 0.001)
        boundary = np.sum(1.0 / margin_clipped**2)
        
        return repulsion + 1.0 * boundary

    def gradient(params):
        p = params.reshape(-1, 2)
        grad = np.zeros_like(p)
        
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=-1)
        np.fill_diagonal(dist_sq, 1.0)
        
        inv_dist_4 = 1.0 / (dist_sq**2)
        np.fill_diagonal(inv_dist_4, 0.0)
        
        sum_w = np.sum(inv_dist_4, axis=1)
        sum_wp = np.sum(inv_dist_4[:, :, np.newaxis] * p[np.newaxis, :, :], axis=1)
        
        grad_rep = -2.0 * (p * sum_w[:, np.newaxis] - sum_wp)
        
        dist_to_center = np.sqrt(np.sum(p**2, axis=1))
        margin = 1.0 - dist_to_center
        margin_clipped = np.maximum(margin, 0.001)
        
        dist_safe = np.maximum(dist_to_center, 0.001)
        grad_bound = 2.0 * p / (dist_safe[:, np.newaxis] * margin_clipped**3)
        
        return grad_rep + 1.0 * grad_bound

    def calc_score(p):
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dist, np.inf)
        min_dist = np.min(dist)
        dist_to_center = np.sqrt(np.sum(p**2, axis=1))
        min_margin = np.min(1.0 - dist_to_center)
        return min(min_dist / 2.0, min_margin)

    best_points = points.copy()
    best_score = calc_score(points)
    
    # Optimization Loop
    # We run L-BFGS-B. If time runs out, we return best found.
    
    try:
        # First pass
        res = minimize(
            energy, 
            points.flatten(), 
            jac=gradient, 
            method='L-BFGS-B', 
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if res.success and time.time() < deadline:
            new_points = res.x.reshape(-1, 2)
            new_score = calc_score(new_points)
            if new_score > best_score:
                best_points = new_points
                best_score = new_score
                
        # Second pass (Refinement)
        if time.time() < deadline - 10:
            res2 = minimize(
                energy, 
                best_points.flatten(), 
                jac=gradient, 
                method='L-BFGS-B', 
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res2.success:
                new_points2 = res2.x.reshape(-1, 2)
                new_score2 = calc_score(new_points2)
                if new_score2 > best_score:
                    best_points = new_points2
                    best_score = new_score2
                    
    except Exception:
        pass

    return best_points

# EVOLVE-BLOCK-END
