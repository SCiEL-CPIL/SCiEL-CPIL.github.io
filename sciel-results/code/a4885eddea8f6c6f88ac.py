# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize


def _evaluate(points: np.ndarray) -> float:
    """Compute the metric (d_min / d_max)^2 for a set of points."""
    if len(points) < 2:
        return 0.0
    dists = pdist(points)
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max == 0 or not np.isfinite(d_min) or not np.isfinite(d_max):
        return 0.0
    ratio = d_min / d_max
    return ratio * ratio


def _soft_min(dists, beta):
    """
    Smooth approximation of the minimum using log-sum-exp.
    As beta -> inf, this approaches the true minimum.
    """
    # Use log-sum-exp trick for numerical stability
    # log(sum(exp(-beta * d))) = -beta * min(d) + log(sum(exp(-beta * (d - min(d)))))
    min_d = np.min(dists)
    # If min_d is very large or beta is very large, we might have issues, but dists are bounded
    # exp(-beta * (d - min_d)) <= 1
    shifted = -beta * (dists - min_d)
    # Clip to avoid underflow/overflow in exp, though shifted <= 0
    shifted = np.clip(shifted, -700, 700)
    sum_exp = np.sum(np.exp(shifted))
    log_sum_exp = np.log(sum_exp) if sum_exp > 0 else 0.0
    return min_d - log_sum_exp / beta


def _soft_max(dists, beta):
    """
    Smooth approximation of the maximum using log-sum-exp.
    As beta -> inf, this approaches the true maximum.
    """
    max_d = np.max(dists)
    shifted = beta * (dists - max_d)
    shifted = np.clip(shifted, -700, 700)
    sum_exp = np.sum(np.exp(shifted))
    log_sum_exp = np.log(sum_exp) if sum_exp > 0 else 0.0
    return max_d + log_sum_exp / beta


def _smooth_metric(points, beta_min, beta_max):
    """
    Compute the smooth surrogate for (d_min / d_max)^2.
    Higher is better. We will minimize the negative of this.
    """
    dists = pdist(points)
    d_min_soft = _soft_min(dists, beta_min)
    d_max_soft = _soft_max(dists, beta_max)
    
    if d_max_soft <= 0 or not np.isfinite(d_min_soft) or not np.isfinite(d_max_soft):
        return -1e10  # Bad configuration
    
    ratio = d_min_soft / d_max_soft
    return ratio * ratio


def _objective(params, beta_min, beta_max):
    """
    Objective function for L-BFGS-B.
    params: shape (14, 3) flattened.
    Returns negative of the smooth metric.
    """
    points = params.reshape(14, 3)
    metric = _smooth_metric(points, beta_min, beta_max)
    return -metric


def _icosahedron_vertices() -> np.ndarray:
    """Return the 12 vertices of a regular icosahedron on the unit sphere."""
    phi = (1 + np.sqrt(5)) / 2
    vertices = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(vertices, axis=1, keepdims=True)
    return vertices / norms


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin for 600s limit
    
    # Initial configurations
    ico = _icosahedron_vertices()
    
    # Generate candidates
    candidates = []
    
    # 1. Icosahedron + 2 symmetric additions
    # The icosahedron has 5-fold symmetry. Adding 2 points breaks this.
    # Try adding points along the axis of 5-fold symmetry (z-axis) or other high-symmetry directions.
    
    # Candidate: Ico + (0,0,1) + (0,0,-1)
    p1 = np.array([0, 0, 1])
    p2 = np.array([0, 0, -1])
    cand1 = np.vstack([ico, p1.reshape(1, -1), p2.reshape(1, -1)])
    # Normalize to unit sphere to keep scale consistent during optimization
    norms = np.linalg.norm(cand1, axis=1, keepdims=True)
    cand1 = cand1 / norms
    candidates.append(cand1)
    
    # Candidate: Ico + 2 points at "midpoints" of opposite faces?
    # Or just random restarts from ico
    
    rng = np.random.RandomState(42)
    
    # Add some random perturbations of the icosahedron + poles
    for _ in range(10):
        noise = rng.randn(14, 3) * 0.1
        cand = cand1 + noise
        norms = np.linalg.norm(cand, axis=1, keepdims=True)
        cand = cand / norms
        candidates.append(cand)
        
    # Pure random restarts
    for _ in range(10):
        cand = rng.randn(14, 3)
        norms = np.linalg.norm(cand, axis=1, keepdims=True)
        cand = cand / norms
        candidates.append(cand)
        
    best_points = None
    best_metric = 0.0
    
    # Optimization loop
    # We use a multi-stage approach with increasing beta for the soft min/max
    # Stage 1: Low beta (smooth), find basin
    # Stage 2: Higher beta (sharper), refine
    
    betas = [10.0, 50.0, 200.0, 1000.0]
    
    for i, candidate in enumerate(candidates):
        if time.time() > deadline - 60:  # Leave time for polish
            break
            
        current_points = candidate.copy()
        
        for beta in betas:
            if time.time() > deadline - 60:
                break
                
            try:
                # Optimize with L-BFGS-B
                # Unconstrained optimization in R^42
                # The metric is scale invariant, so the optimizer might drift in scale.
                # To prevent this, we can add a small penalty for deviation from unit sphere or fix scale.
                # However, L-BFGS-B handles unconstrained well.
                # Let's add a small penalty term to keep points near unit sphere to avoid numerical issues with pdist at large scales
                def obj_with_penalty(params):
                    pts = params.reshape(14, 3)
                    metric = _smooth_metric(pts, beta, beta)
                    # Penalty for large norms
                    norms = np.linalg.norm(pts, axis=1)
                    penalty = 0.01 * np.sum((norms - 1.0)**2)
                    return -metric + penalty
                
                result = minimize(
                    obj_with_penalty, 
                    current_points.flatten(), 
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12}
                )
                
                optimized_pts = result.x.reshape(14, 3)
                
                # Evaluate true metric on the optimized points
                true_metric = _evaluate(optimized_pts)
                
                if true_metric > best_metric:
                    best_metric = true_metric
                    best_points = optimized_pts.copy()
                
                # Use the optimized points as start for the next beta stage
                current_points = optimized_pts.copy()
                
            except Exception:
                continue
        
        # If this candidate yielded a good result, we can try a few more random restarts from it if time permits
        if best_points is not None and i < 5:
             if time.time() > deadline - 30:
                 break
             # Generate 2 more random perturbations of the best found so far
             for _ in range(2):
                 noise = rng.randn(14, 3) * 0.05
                 cand = best_points + noise
                 norms = np.linalg.norm(cand, axis=1, keepdims=True)
                 cand = cand / norms
                 candidates.append(cand)
                 # Break out of inner loop if we've added enough
                 break

    # Final Polish
    # If we have time, do a very high precision polish on the best point
    if best_points is not None and time.time() < deadline - 10:
        try:
            # Use a very high beta for the final polish
            beta_final = 5000.0
            
            def obj_polish(params):
                pts = params.reshape(14, 3)
                metric = _smooth_metric(pts, beta_final, beta_final)
                norms = np.linalg.norm(pts, axis=1)
                penalty = 0.01 * np.sum((norms - 1.0)**2)
                return -metric + penalty
            
            result = minimize(
                obj_polish, 
                best_points.flatten(), 
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-15}
            )
            
            polished_pts = result.x.reshape(14, 3)
            polished_metric = _evaluate(polished_pts)
            
            if polished_metric > best_metric:
                best_points = polished_pts
                best_metric = polished_metric
                
        except Exception:
            pass

    # Fallback
    if best_points is None:
        best_points = candidates[0]
    
    # Ensure finite
    if not np.all(np.isfinite(best_points)):
        best_points = np.vstack([ico, np.array([[0, 0, 1]]), np.array([[0, 0, -1]])])
        norms = np.linalg.norm(best_points, axis=1, keepdims=True)
        best_points = best_points / norms
        
    return best_points

# EVOLVE-BLOCK-END
