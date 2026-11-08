import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution

N = 13

def _all_triangle_areas(points):
    """Vectorized computation of all C(n,3) triangle areas."""
    n = len(points)
    if n < 3:
        return np.array([0.0])
    idx = np.arange(n)
    i_idx, j_idx, k_idx = np.meshgrid(idx, idx, idx, indexing='ij')
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_idx, j_idx, k_idx = i_idx[mask], j_idx[mask], k_idx[mask]
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return areas

def _hull_area(points):
    """Compute area of the convex hull."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 1.0

def _exact_score(points):
    """Calculate the exact score: min triangle area / hull area."""
    areas = _all_triangle_areas(points)
    min_area = np.min(areas)
    hull_area = _hull_area(points)
    if hull_area <= 0:
        return 0.0
    return min_area / hull_area

def _score_neg(points):
    """Negative score for minimization."""
    return -_exact_score(points)

def construct():
    n = N
    t0 = time.time()
    deadline = 140.0  # Safety margin from 150s limit
    
    best_points = None
    best_score = -np.inf
    
    # Strategy 1: Differential Evolution on raw coordinates
    # This is a global optimizer that doesn't get stuck in local minima like L-BFGS-B might.
    # We restrict the search space to [-1, 1] x [-1, 1] which is sufficient for normalized area problems.
    
    # Bounds for each coordinate
    bounds = [(-1.0, 1.0)] * (2 * n)
    
    # Callback to track best
    best_found = {'points': None, 'score': -np.inf}
    
    def callback(xk, convergence):
        pts = xk.reshape(-1, 2)
        score = _exact_score(pts)
        if score > best_found['score']:
            best_found['score'] = score
            best_found['points'] = pts.copy()
        if time.time() - t0 > deadline - 20:
            raise KeyboardInterrupt()

    try:
        # Differential Evolution parameters
        # popsize: 15 * dim
        # maxiter: enough to converge but not too long
        # tol: precision
        # seed: for reproducibility
        res = differential_evolution(
            _score_neg, 
            bounds, 
            maxiter=50, 
            popsize=15, 
            tol=1e-6, 
            seed=42, 
            callback=callback,
            init='latinhypercube'
        )
        if res.x is not None:
            pts = res.x.reshape(-1, 2)
            score = _exact_score(pts)
            if score > best_found['score']:
                best_found['score'] = score
                best_found['points'] = pts
    except KeyboardInterrupt:
        pass
    except Exception:
        pass
    
    if best_found['points'] is not None:
        best_points = best_found['points']
        best_score = best_found['score']
    
    # Strategy 2: Local refinement using L-BFGS-B with a smooth objective
    # Start from the best point found by DE
    if best_points is not None and time.time() - t0 < deadline - 10:
        def _soft_objective(params, beta=50.0):
            points = params.reshape(-1, 2)
            areas = _all_triangle_areas(points)
            scaled_areas = -beta * areas
            max_val = np.max(scaled_areas)
            if max_val == -np.inf:
                return 1e10
            log_sum_exp = max_val + np.log(np.sum(np.exp(scaled_areas - max_val)))
            soft_min = -log_sum_exp / beta
            hull_area = _hull_area(points)
            if hull_area <= 0:
                return 1e10
            return -soft_min / hull_area
        
        try:
            x0 = best_points.flatten()
            res2 = minimize(_soft_objective, x0, args=(50.0,), method='L-BFGS-B',
                            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
            cand = res2.x.reshape(-1, 2)
            score2 = _exact_score(cand)
            if score2 > best_score:
                best_score = score2
                best_points = cand
        except Exception:
            pass
        
        # Strategy 3: Nelder-Mead on exact objective for final polish
        if time.time() - t0 < deadline - 5:
            try:
                x0 = best_points.flatten()
                res3 = minimize(lambda x: -_exact_score(x.reshape(-1, 2)), x0, method='Nelder-Mead',
                                options={'maxiter': 500, 'fatol': 1e-10, 'xatol': 1e-10, 'adaptive': True})
                cand = res3.x.reshape(-1, 2)
                score3 = _exact_score(cand)
                if score3 > best_score:
                    best_score = score3
                    best_points = cand
            except Exception:
                pass

    # Fallback: Regular 13-gon if optimization failed
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
        best_score = _exact_score(best_points)

    return best_points
