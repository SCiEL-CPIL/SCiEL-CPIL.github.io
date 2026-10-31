"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
D = 2

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    def get_sq_dists(pts):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        sq_dists = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        return sq_dists[iu]

    def score_raw(pts):
        d_sq = get_sq_dists(pts)
        d_min = np.sqrt(d_sq.min())
        d_max = np.sqrt(d_sq.max())
        if d_min < 1e-12:
            return 1e10
        return (d_max / d_min) ** 2

    def normalize(pts):
        pts = pts - np.mean(pts, axis=0)
        scale = np.sqrt(np.mean(np.sum(pts**2, axis=1)))
        if scale > 1e-9:
            pts = pts / scale
        return pts

    # Strategy: Smoothed Rank-Based Optimization
    # Instead of directly optimizing the non-differentiable ratio (max/min), 
    # we optimize a surrogate function that approximates the behavior of the 
    # extreme distances. 
    # We use the sum of p-th powers of normalized distances. As p -> infinity, 
    # this function is dominated by the largest distance. However, to also 
    # control the minimum distance (which we want to maximize), we can 
    # reformulate. 
    # A better approach: Minimize the variance of the log-distances, or 
    # minimize (max - min) directly using a smooth approximation.
    # Let's use a "Soft Min-Max" approach:
    # Objective: min (soft_max(d_sq) / soft_min(d_sq))
    # But division is tricky. 
    # Alternative: Minimize the spread of distances.
    # Let d_sq be the vector of squared distances.
    # We want d_sq to be as constant as possible.
    # Minimize sum((d_sq - mean(d_sq))^2) subject to mean(d_sq)=1 (via normalization).
    # This minimizes the variance of the squared distances.
    # Minimizing variance of d_sq is equivalent to minimizing (max-min) roughly, 
    # but it pulls all distances together.
    # Let's try minimizing the variance of the actual distances (sqrt(d_sq)).
    # Or even better, use a high-order moment.
    
    # Let's use the objective: sum(d_sq**p) - p * (mean(d_sq))**p ? No.
    # Simple and robust: Minimize the variance of the squared distances.
    # This encourages all pairs to be equidistant (which is impossible for 16 pts in 2D, 
    # but it guides the search towards highly symmetric configurations like grids or circles).
    
    # Actually, a more direct target for the ratio is minimizing (max_d - min_d) / min_d.
    # Let's use a smooth approximation of max and min.
    # soft_max(x, T) = T * log(sum(exp(x/T)))
    # soft_min(x, T) = -T * log(sum(exp(-x/T)))
    # Ratio = soft_max(d_sq) / soft_min(d_sq)
    # This is differentiable.
    
    def objective_smooth(x):
        if time.time() > deadline:
            return 1e10
        pts = x.reshape(N, 2)
        d_sq = get_sq_dists(pts)
        
        # Normalize distances to prevent overflow/underflow in exp
        # Scale d_sq such that mean is 1
        mean_d_sq = np.mean(d_sq)
        if mean_d_sq < 1e-12:
            return 1e10
        d_norm = d_sq / mean_d_sq
        
        # Temperature for soft min/max
        # Smaller T -> sharper approximation to true min/max
        # We need T to be small enough to approximate well, but large enough for numerical stability
        # Range of d_norm is typically [0.1, 10]
        T = 0.05
        
        # soft_max
        max_d_norm = T * np.log(np.sum(np.exp(d_norm / T)))
        # soft_min
        min_d_norm = -T * np.log(np.sum(np.exp(-d_norm / T)))
        
        if min_d_norm < 1e-6:
            return 1e10
            
        ratio = max_d_norm / min_d_norm
        return ratio

    def gradient_smooth(x):
        # Numerical gradient to avoid complex analytic derivations of soft min/max
        # This is slower but more robust for a first implementation.
        # Given the 120s limit, we need to be careful.
        # L-BFGS-B with numerical gradient on 32 variables is feasible.
        return None # Let scipy handle numerical gradient if we pass None? 
                    # Actually, minimize with method='L-BFGS-B' requires gradient or uses finite diff.
                    # Finite diff on 32D is 33 evaluations per step. 
                    # If we do 1000 steps, that's 33000 evals.
                    # Each eval is ~100us. 33000 * 100us = 3.3s. This is fast enough.
                    # So we can just use finite differences.

    # Multi-start L-BFGS-B
    best_x = None
    best_score = 1e10
    
    # Generate diverse initial points
    # 1. Grid
    x_grid = np.arange(4) - 1.5
    xx, yy = np.meshgrid(x_grid, x_grid)
    grid_pts = np.column_stack([xx.ravel(), yy.ravel()])
    grid_pts = normalize(grid_pts)
    
    # 2. Circle
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    circle_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    circle_pts = normalize(circle_pts)
    
    # 3. Random
    random_pts = normalize(np.random.uniform(-1, 1, (N, 2)))
    
    # 4. Two circles
    angles1 = np.linspace(0, 2*np.pi, 8, endpoint=False)
    angles2 = np.linspace(0, 2*np.pi, 8, endpoint=False) + np.pi/8
    c1 = np.column_stack([np.cos(angles1), np.sin(angles1)])
    c2 = np.column_stack([0.5*np.cos(angles2), 0.5*np.sin(angles2)])
    two_circles = np.vstack([c1, c2])
    two_circles = normalize(two_circles)
    
    initial_sets = [grid_pts, circle_pts, random_pts, two_circles]
    
    for init_pts in initial_sets:
        if time.time() > deadline:
            break
            
        x0 = init_pts.ravel()
        
        try:
            res = minimize(
                objective_smooth,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
            )
            
            if res.fun < best_score:
                best_score = res.fun
                best_x = res.x
                
        except Exception as e:
            pass

    # If no valid result, fall back to best initial
    if best_x is None:
        best_x = grid_pts.ravel()
        best_score = score_raw(grid_pts)
        
    # Final polish with Nelder-Mead on the raw score to correct for soft-min/max approximation errors
    # Raw score is the true objective.
    def objective_raw(x):
        if time.time() > deadline:
            return 1e10
        pts = x.reshape(N, 2)
        return score_raw(pts)

    try:
        res_nm = minimize(
            objective_raw,
            best_x,
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
        )
        if res_nm.fun < best_score:
            best_score = res_nm.fun
            best_x = res_nm.x
    except Exception:
        pass

    final_pts = best_x.reshape(N, 2)
    
    # Ensure distinct points
    # Check for duplicates
    d_sq = get_sq_dists(final_pts)
    if d_sq.min() < 1e-12:
        # Perturb if necessary
        np.random.seed(42)
        final_pts += np.random.normal(0, 1e-6, (N, 2))
        
    return final_pts

# EVOLVE-BLOCK-END
