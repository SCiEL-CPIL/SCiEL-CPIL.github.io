# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize
import time
import itertools

def _precompute_indices(n):
    """Precompute triangle indices for efficiency."""
    return np.array(list(itertools.combinations(range(n), 3)))

def _compute_dets(points, idx):
    """Compute 2 * signed area (determinant) for all triangles."""
    p1 = points[idx[:, 0]]
    p2 = points[idx[:, 1]]
    p3 = points[idx[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    det = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return det

def _compute_hull_area(points):
    """Compute convex hull area safely."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def compute_score(points):
    """
    Compute the Heilbronn score: A_min / A_hull.
    Returns 0 if invalid.
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    if not np.all(np.isfinite(points)):
        return 0.0
        
    A_hull = _compute_hull_area(points)
    if A_hull <= 1e-12:
        return 0.0
        
    idx = _precompute_indices(n)
    dets = _compute_dets(points, idx)
    A_min = 0.5 * np.min(np.abs(dets))
    
    return A_min / A_hull

def heilbronn_convex13() -> np.ndarray:
    n = 13
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
    best_points = None
    best_score = 0.0
    
    idx = _precompute_indices(n)
    
    # Objective function for Differential Evolution
    # We want to MAXIMIZE A_min / A_hull
    # DE minimizes, so we minimize -score
    # To avoid division by zero and handle invalid points, we return a large penalty
    def objective(x):
        points = x.reshape((n, 2))
        
        # Basic validity check
        if not np.all(np.isfinite(points)):
            return 1e6
            
        A_hull = _compute_hull_area(points)
        if A_hull <= 1e-12:
            return 1e6
            
        dets = _compute_dets(points, idx)
        A_min = 0.5 * np.min(np.abs(dets))
        
        score = A_min / A_hull
        return -score

    # Define bounds for DE: points in [-1, 1] x [-1, 1]
    # Scaling doesn't affect the ratio, but helps DE converge
    bounds = [(-1.0, 1.0)] * (n * 2)
    
    # Run Differential Evolution
    # strategy='best1bin' is a common default
    # maxiter and popsize are tuned to fit within time limit
    # 600s limit. DE can be slow. We need to be careful.
    # Let's try a moderate population and iterations.
    
    try:
        # DE is stochastic, so we set a seed for reproducibility if we want, 
        # but the task implies we just need *a* good result. 
        # However, deterministic behavior is safer for evaluation.
        # scipy's DE uses a random seed internally if not specified? 
        # Actually, DE in scipy doesn't take a seed directly in older versions, 
        # but we can control it via np.random if we use a custom engine, 
        # or just rely on the default. 
        # For safety, let's run DE with a time limit check inside if possible, 
        # or just set conservative parameters.
        
        # If DE takes too long, it might hit the 600s limit.
        # Let's estimate: 26 dimensions, popsize=100, maxiter=100 -> 10k evaluations.
        # Each eval is ~286 triangles + hull. ~1ms. 10s total. This is safe.
        # We can increase popsize/iter if time permits.
        
        # Let's run DE
        res_de = differential_evolution(
            objective, 
            bounds, 
            strategy='best1bin', 
            maxiter=200, 
            popsize=150, 
            tol=1e-8, 
            mut=(0.5, 1.5), 
            reccomb=0.7, 
            seed=42
        )
        
        de_points = res_de.x.reshape((n, 2))
        de_score = -res_de.fun
        
        if de_score > best_score:
            best_score = de_score
            best_points = de_points.copy()
            
    except Exception as e:
        # Fallback if DE fails
        pass

    # Check time
    if time.time() > deadline - 50:
        if best_points is None:
            # Fallback to regular polygon
            angles = np.linspace(0, 2*np.pi, n, endpoint=False)
            best_points = np.column_stack([np.cos(angles), np.sin(angles)])
            best_score = compute_score(best_points)
        return best_points

    # If we have a good candidate from DE, polish it with Nelder-Mead
    # NM is good for fine-tuning in the local neighborhood
    if best_points is not None:
        def nm_objective(x):
            pts = x.reshape((n, 2))
            s = compute_score(pts)
            if s == 0.0:
                return 1.0
            return -s
        
        try:
            # Run NM for a limited number of iterations to stay within time
            res_nm = minimize(
                nm_objective, 
                best_points.flatten(), 
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True}
            )
            final_pts_nm = res_nm.x.reshape((n, 2))
            final_score_nm = compute_score(final_pts_nm)
            
            if final_score_nm > best_score:
                best_score = final_score_nm
                best_points = final_pts_nm.copy()
                
        except Exception:
            pass

    # If we still have time, try a few more DE runs with different seeds or restarts?
    # Or just return the best found.
    # Given the 600s limit and DE taking ~10-20s, we could run multiple DE instances.
    
    remaining_time = deadline - time.time()
    if remaining_time > 100:
        # Run a second DE instance with a different seed to explore a different basin
        try:
            res_de2 = differential_evolution(
                objective, 
                bounds, 
                strategy='best1bin', 
                maxiter=150, 
                popsize=100, 
                tol=1e-8, 
                mut=(0.5, 1.5), 
                reccomb=0.7, 
                seed=123
            )
            de2_points = res_de2.x.reshape((n, 2))
            de2_score = -res_de2.fun
            
            if de2_score > best_score:
                best_score = de2_score
                best_points = de2_points.copy()
                
            # Polish the new best if it improved
            if best_score > 0:
                def nm_objective2(x):
                    pts = x.reshape((n, 2))
                    s = compute_score(pts)
                    if s == 0.0:
                        return 1.0
                    return -s
                    
                res_nm2 = minimize(
                    nm_objective2, 
                    best_points.flatten(), 
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True}
                )
                final_pts_nm2 = res_nm2.x.reshape((n, 2))
                final_score_nm2 = compute_score(final_pts_nm2)
                
                if final_score_nm2 > best_score:
                    best_score = final_score_nm2
                    best_points = final_pts_nm2.copy()
                    
        except Exception:
            pass

    # Fallback
    if best_points is None or best_score < 0.01:
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        best_score = compute_score(best_points)

    return best_points

# EVOLVE-BLOCK-END
