"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 16
D = 2

# EVOLVE-BLOCK-START

def _pairwise_distances(pts):
    """Compute all pairwise squared distances for n points in 2D."""
    # pts shape (n, 2)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=-1)
    return dist_sq

def _score_from_points(pts):
    """Compute the score: (max_dist / min_dist)^2. Lower is better."""
    dist_sq = _pairwise_distances(pts)
    # Get upper triangle indices
    iu = np.triu_indices(N, k=1)
    d_sq = dist_sq[iu]
    min_d_sq = np.min(d_sq)
    max_d_sq = np.max(d_sq)
    if min_d_sq <= 1e-15:
        return np.inf
    ratio_sq = max_d_sq / min_d_sq
    return ratio_sq

def _objective(params):
    """Objective for differential evolution: flatten 2D coords to 1D."""
    pts = params.reshape(N, D)
    return _score_from_points(pts)

def _refine_local(pts, deadline):
    """Local refinement using L-BFGS-B on a smooth surrogate."""
    start_time = time.time()
    x0 = pts.flatten().copy()
    
    def smooth_obj(x):
        pts = x.reshape(N, D)
        dist_sq = _pairwise_distances(pts)
        iu = np.triu_indices(N, k=1)
        d_sq = dist_sq[iu]
        # Soft min and max to make it differentiable
        # Use log-sum-exp for max, and -log-sum-exp(-d) for min
        beta = 50.0
        log_max = np.log(np.sum(np.exp(beta * d_sq))) / beta
        log_min = -np.log(np.sum(np.exp(-beta * d_sq))) / beta
        # Objective: exp(log_max - log_min) = max/min
        # We want to minimize (max/min)^2, so minimize 2*(log_max - log_min)
        return 2.0 * (log_max - log_min)
    
    best_x = x0.copy()
    best_val = _score_from_points(pts)
    
    try:
        res = minimize(smooth_obj, x0, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
        if res.x is not None:
            val = _score_from_points(res.x.reshape(N, D))
            if val < best_val:
                best_val = val
                best_x = res.x.copy()
    except Exception:
        pass
    
    # If we still have time, try a few more random restarts of local optimization
    remaining = deadline - time.time()
    if remaining > 5.0:
        rng = np.random.RandomState(42)
        for _ in range(10):
            if time.time() > deadline - 2.0:
                break
            # Perturb the best solution
            x_pert = best_x + rng.normal(0, 0.1, size=best_x.shape)
            try:
                res2 = minimize(smooth_obj, x_pert, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-10})
                if res2.x is not None:
                    val2 = _score_from_points(res2.x.reshape(N, D))
                    if val2 < best_val:
                        best_val = val2
                        best_x = res2.x.copy()
            except Exception:
                pass
    
    return best_x.reshape(N, D), best_val

def construct():
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    
    # Initial guess: 4x4 grid
    side = 4
    grid = np.array([(i, j) for i in range(side) for j in range(side)], dtype=float)
    # Normalize
    grid = grid - grid.mean(axis=0)
    grid = grid / np.max(np.abs(grid))
    
    best_pts = grid.copy()
    best_score = _score_from_points(grid)
    
    # Differential Evolution
    bounds = [(-1.0, 1.0)] * (N * D)
    try:
        de_res = differential_evolution(
            _objective, 
            bounds, 
            seed=42, 
            maxiter=100, 
            popsize=20, 
            tol=1e-8, 
            mutation=(0.5, 1.5),
            recombination=0.9,
            polish=False,
            update_interval=10,
            callback=lambda xk, convergence: (time.time() > deadline - 20.0)
        )
        if de_res.x is not None:
            score_de = _score_from_points(de_res.x.reshape(N, D))
            if score_de < best_score:
                best_score = score_de
                best_pts = de_res.x.reshape(N, D).copy()
    except Exception:
        pass
    
    # Local refinement
    remaining = deadline - time.time()
    if remaining > 10.0:
        refined_pts, refined_score = _refine_local(best_pts, deadline)
        if refined_score < best_score:
            best_score = refined_score
            best_pts = refined_pts.copy()
    
    # Final check: ensure points are distinct
    dist_sq = _pairwise_distances(best_pts)
    iu = np.triu_indices(N, k=1)
    d_sq = dist_sq[iu]
    if np.min(d_sq) < 1e-12:
        # Fallback to grid if points collapsed
        best_pts = grid
    
    return best_pts

# EVOLVE-BLOCK-END
