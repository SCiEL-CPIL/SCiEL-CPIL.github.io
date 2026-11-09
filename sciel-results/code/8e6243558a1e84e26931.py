"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _soft_min_obj(pts, beta=50.0):
    """
    Objective: -soft_min(min_pairwise_dist/2, min_container_margin)
    Using log-sum-exp for differentiability.
    """
    # Compute pairwise distances
    # To save memory/time, we can subsample or use a trick, but for 300 points (45k pairs) it's manageable.
    # pts is (N, 2)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dists, np.inf)
    
    # Container margin
    radii = np.linalg.norm(pts, axis=1)
    margins = 1.0 - radii
    
    # Combine all constraints: pairwise/2 and margins
    # We want to maximize the minimum of these.
    # So we minimize the negative of the soft-min.
    # soft_min(x) = -1/beta * log(sum(exp(-beta * x)))
    # We want max(min(x)), so we minimize -max(min(x)) = min(-soft_max(x))? No.
    # Let's use: f = -soft_min(constraints)
    # soft_min(c) approximates min(c).
    # We want to maximize min(c). So we minimize -min(c).
    # Using soft-min: -soft_min(c) = 1/beta * log(sum(exp(-beta * c)))
    
    # Combine pairwise distances (divided by 2) and margins
    # Pairwise: dists / 2
    # Margins: margins
    
    # To avoid huge arrays, we can flatten
    # But 300*300 = 90000 values. exp(-50 * val) might underflow/overflow?
    # Values are around 0.05. -50 * 0.05 = -2.5. exp(-2.5) is fine.
    # If val is 0, exp(0)=1.
    
    pair_vals = dists / 2.0
    # Mask diagonal
    pair_vals = pair_vals[~np.eye(N, dtype=bool)]
    
    all_vals = np.concatenate([pair_vals, margins])
    
    # Soft min
    # max_val = np.max(all_vals) for numerical stability
    max_val = np.max(all_vals)
    # exp(-beta * (all_vals - max_val))
    # soft_min = max_val - (1/beta) * log(sum(exp(-beta * (all_vals - max_val))))
    # We want to MINIMIZE -soft_min
    # So objective = -soft_min = -max_val + (1/beta) * log(sum(exp(-beta * (all_vals - max_val))))
    
    exp_terms = np.exp(-beta * (all_vals - max_val))
    log_sum = np.log(np.sum(exp_terms))
    
    obj = -max_val + (1.0 / beta) * log_sum
    
    return obj


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # 1. Generate initial hexagonal lattice
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    best_pts = None
    best_score = -np.inf
    
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            init_pts = points[np.argsort(radius, kind="stable")[:N]]
            break
        spacing *= 0.98
        
    # Evaluate initial
    def _score(pts):
        # Exact score
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        radii = np.linalg.norm(pts, axis=1)
        min_margin = np.min(1.0 - radii)
        return min(min_pair, min_margin)
        
    best_pts = init_pts.copy()
    best_score = _score(best_pts)
    
    # 2. Optimization with L-BFGS-B on soft-min objective
    # We need to handle the fact that L-BFGS-B works on flat vectors
    def _obj_flat(x):
        pts = x.reshape(-1, 2)
        return _soft_min_obj(pts)
        
    # Gradient is not provided, so L-BFGS-B will use numerical gradients.
    # For 600 dimensions, this is expensive but manageable if we limit iterations.
    # Alternatively, we can provide an analytical gradient? 
    # Let's try without first, or provide a simple one if possible.
    # Numerical gradient with 600 vars takes 600 function evals per step.
    # If each eval takes ~1ms (optimized numpy), that's 0.6s per step.
    # 100 steps = 60s. Doable.
    
    # Let's try a few restarts with different betas or initial perturbations if time permits.
    
    current_pts = init_pts.copy()
    
    # Add small noise to escape symmetry?
    # The hex lattice is symmetric. L-BFGS-B might struggle if the objective is flat in some directions.
    # But soft-min should provide a gradient towards the tightest constraint.
    
    try:
        res = minimize(
            _obj_flat, 
            current_pts.flatten(), 
            method='L-BFGS-B', 
            options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-5, 'maxls': 50}
        )
        
        if time.time() < deadline:
            new_pts = res.x.reshape(-1, 2)
            new_score = _score(new_pts)
            if new_score > best_score:
                best_pts = new_pts
                best_score = new_score
                
    except Exception:
        pass
        
    # If time remains, try a second round with a higher beta (sharper min) or different start
    if time.time() < deadline - 30:
        try:
            # Perturb best solution
            perturbed = best_pts + np.random.normal(0, 0.01, best_pts.shape)
            res2 = minimize(
                _obj_flat, 
                perturbed.flatten(), 
                method='L-BFGS-B', 
                options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-5, 'maxls': 50}
            )
            new_pts2 = res2.x.reshape(-1, 2)
            new_score2 = _score(new_pts2)
            if new_score2 > best_score:
                best_pts = new_pts2
                best_score = new_score2
        except Exception:
            pass
            
    return best_pts
# EVOLVE-BLOCK-END
