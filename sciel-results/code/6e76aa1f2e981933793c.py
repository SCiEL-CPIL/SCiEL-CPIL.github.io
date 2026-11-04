import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
import time
from itertools import combinations

N = 13

# Precompute triangle indices for vectorization
TRI_INDICES = list(combinations(range(N), 3))
NUM_TRI = len(TRI_INDICES)

def _compute_areas(pts):
    """
    Compute areas of all triangles.
    pts: (N, 2) array
    returns: (NUM_TRI,) array of areas
    """
    # pts[i] is (x_i, y_i)
    # Area = 0.5 * | (x_j - x_i)*(y_k - y_i) - (x_k - x_i)*(y_j - y_i) |
    # Vectorized approach:
    # p_i, p_j, p_k
    # We can use broadcasting or just loop since 286 is small.
    # Loop is likely faster in Python overhead than creating huge temporary arrays for such small N.
    areas = np.empty(NUM_TRI)
    for idx, (i, j, k) in enumerate(TRI_INDICES):
        xi, yi = pts[i]
        xj, yj = pts[j]
        xk, yk = pts[k]
        areas[idx] = 0.5 * abs((xj - xi) * (yk - yi) - (xk - xi) * (yj - yi))
    return areas

def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score(pts):
    """Calculate the actual score for the final answer."""
    areas = _compute_areas(pts)
    min_area = np.min(areas)
    hull_area = _hull_area(pts)
    if hull_area < 1e-12:
        return 0.0
    return min_area / hull_area

def _objective(x, beta):
    """
    Objective function for optimization.
    x: (2*N,) flattened points
    beta: temperature for soft-min
    Returns: -soft_min_area / hull_area (to be minimized)
    """
    pts = x.reshape(N, 2)
    
    # Compute hull area
    hull_area = _hull_area(pts)
    if hull_area < 1e-10:
        return 1e6
        
    # Compute triangle areas
    areas = _compute_areas(pts)
    
    # Soft min approximation
    # min(areas) approx -1/beta * log(sum(exp(-beta * areas)))
    # We want to maximize min(areas)/hull_area
    # So minimize -min(areas)/hull_area
    # Using soft min:
    # soft_min = -1/beta * log(sum(exp(-beta * areas)))
    # obj = -soft_min / hull_area
    
    # Numerical stability for log-sum-exp
    # log_sum_exp(-beta * areas)
    # = log(sum(exp(-beta * areas)))
    # Let M = max(-beta * areas) = -beta * min(areas)
    # sum(exp(-beta * areas)) = exp(M) * sum(exp(-beta * areas - M))
    # log(sum(...)) = M + log(sum(exp(...)))
    
    scaled_areas = -beta * areas
    max_scaled = np.max(scaled_areas)
    if max_scaled > 700: # overflow protection, though unlikely with beta ~ 10-100 and areas ~ 1
         # If areas are very small, -beta*areas is large positive? No, areas are positive.
         # -beta * areas is negative. Max is close to 0.
         pass
         
    lse = max_scaled + np.log(np.sum(np.exp(scaled_areas - max_scaled)))
    soft_min = -lse / beta
    
    return -soft_min / hull_area

def _regular_polygon(n, radius=1.0):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _halton_sequence(n, base):
    """Generate Halton sequence points in [0,1]^2"""
    def halton(i, base):
        f = 1.0
        r = 0.0
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r
    
    points = []
    for i in range(1, n + 1):
        points.append([halton(i, 2), halton(i, 3)])
    return np.array(points)

def _random_points(n, seed=None):
    rng = np.random.default_rng(seed)
    # Points in unit disk
    angles = rng.uniform(0, 2 * np.pi, n)
    radii = np.sqrt(rng.uniform(0, 1, n)) # Uniform area distribution
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

def _normalize_hull_unit(pts):
    """Scale points so convex hull area is 1."""
    hull_area = _hull_area(pts)
    if hull_area < 1e-12:
        return pts
    scale = 1.0 / np.sqrt(hull_area)
    return pts * scale

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    best_pts = None
    best_score = -np.inf
    
    # Initial candidates
    candidates = []
    candidates.append(_regular_polygon(N))
    candidates.append(_halton_sequence(N, 2) * 2 - 1) # Map to [-1,1]
    candidates.append(_random_points(N, seed=42))
    candidates.append(_random_points(N, seed=123))
    
    # Evaluate initial candidates
    for cand in candidates:
        if time.time() > deadline:
            break
        try:
            s = _score(cand)
            if s > best_score:
                best_score = s
                best_pts = cand.copy()
        except Exception:
            pass
    
    # Phase 1: Differential Evolution (Global Search)
    # Use a moderate beta for smoothness
    beta_de = 10.0
    
    # Bounds for DE: points should be roughly in [-2, 2] to avoid extreme scaling
    bounds = [(-2.0, 2.0)] * (2 * N)
    
    # Use the best initial candidate as the starting point or bounds center?
    # DE doesn't take a start point, it uses bounds.
    # We can run DE on a subset of the best candidates or just let it explore.
    
    if time.time() < deadline - 30: # Leave 30s for local refinement
        try:
            # Differential Evolution
            # maxiter is number of generations
            # popsize is factor * N
            de_result = differential_evolution(
                _objective,
                bounds,
                args=(beta_de,),
                maxiter=50,
                popsize=15,
                tol=1e-8,
                seed=42,
                polish=False # We will do local polish separately
            )
            
            if de_result.success:
                pts_de = de_result.x.reshape(N, 2)
                s = _score(pts_de)
                if s > best_score:
                    best_score = s
                    best_pts = pts_de.copy()
        except Exception:
            pass
    
    # Phase 2: Local Refinement with L-BFGS-B and Annealing Beta
    # Start from best_pts
    if best_pts is not None:
        current_pts = best_pts.copy()
        
        # Annealing schedule
        betas = [10, 50, 100, 200, 500]
        
        for beta in betas:
            if time.time() > deadline - 5:
                break
                
            try:
                # Flatten current points
                x0 = current_pts.flatten()
                
                res = minimize(
                    _objective,
                    x0,
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                if res.success or res.fun < 1e5:
                    new_pts = res.x.reshape(N, 2)
                    s = _score(new_pts)
                    if s > best_score:
                        best_score = s
                        best_pts = new_pts.copy()
                        current_pts = new_pts.copy()
                    else:
                        # If no improvement, keep current_pts but maybe try again with same beta?
                        # Or just move to next beta.
                        pass
            except Exception:
                pass
        
        # Final polish with very high beta
        if time.time() < deadline - 2:
            try:
                x0 = best_pts.flatten()
                res = minimize(
                    _objective,
                    x0,
                    args=(1000,),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if res.fun < 1e5:
                    new_pts = res.x.reshape(N, 2)
                    s = _score(new_pts)
                    if s > best_score:
                        best_score = s
                        best_pts = new_pts.copy()
            except Exception:
                pass

    # Fallback
    if best_pts is None:
        best_pts = _regular_polygon(N)
        
    return best_pts
