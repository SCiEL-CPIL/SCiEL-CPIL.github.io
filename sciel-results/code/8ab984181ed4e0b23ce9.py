import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 14

def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _all_triangle_areas(points):
    """
    Compute all triangle areas for N points.
    Returns array of shape (N, N, N) where entry [i,j,k] is 0.5*|cross| for i<j<k,
    but we only compute for i<j<k and store in a flattened list or just return min.
    For efficiency, let's return the min and the areas.
    Actually, for the soft objective, we need the areas.
    Let's return an array of all 364 areas.
    """
    n = len(points)
    x = points[:, 0]
    y = points[:, 1]
    
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = 0.5 * abs((x[j] - x[i]) * (y[k] - y[i]) - (x[k] - x[i]) * (y[j] - y[i]))
                areas.append(area)
    return np.array(areas)

def _min_triangle_area_vec(points):
    """Vectorized minimum triangle area."""
    return np.min(_all_triangle_areas(points))

def _score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    """
    hull_area = _hull_area(points)
    if hull_area <= 0:
        return 0.0
    min_area = _min_triangle_area_vec(points)
    return min_area / hull_area

def _soft_obj(points, beta):
    """
    Soft objective: - (sum of exp(-area/beta) / N_triangles) / hull_area
    We want to maximize the min area, so we minimize the soft min.
    Soft min approximates min as beta -> 0.
    """
    areas = _all_triangle_areas(points)
    # Soft min: -beta * log(sum(exp(-area/beta)))
    # To avoid overflow, use the log-sum-exp trick
    # log(sum(exp(-a/beta))) = -min(a)/beta + log(sum(exp(-(a-min(a))/beta)))
    min_a = np.min(areas)
    # If min_a is very small, exp(-(a-min_a)/beta) can be large if beta is small
    # But a - min_a >= 0, so exp(-(a-min_a)/beta) <= 1.
    # So log(sum(exp(-(a-min_a)/beta))) is safe.
    soft_min = -beta * np.log(np.sum(np.exp(-(areas - min_a) / beta))) - min_a
    # Wait, the formula is:
    # soft_min = -beta * log( sum( exp(-area_i / beta) ) )
    # = -beta * ( -min_a/beta + log( sum( exp(-(area_i - min_a)/beta) ) ) )
    # = min_a + beta * log( sum( exp(-(area_i - min_a)/beta) ) )
    # Let's re-verify.
    # log(sum(exp(-a_i/beta))) = log(exp(-min_a/beta) * sum(exp(-(a_i-min_a)/beta)))
    # = -min_a/beta + log(sum(exp(-(a_i-min_a)/beta)))
    # So soft_min = -beta * ( -min_a/beta + log(...) ) = min_a + beta * log(...)
    
    # Let's just use the direct formula with clipping if needed, but N=14 is small.
    # Let's use the stable version.
    soft_min_val = min_a + beta * np.log(np.sum(np.exp(-(areas - min_a) / beta)))
    
    hull_area = _hull_area(points)
    if hull_area <= 0:
        return 1e6 # Large penalty
    
    return soft_min_val / hull_area

def _neg_soft_obj(p_flat, beta):
    points = p_flat.reshape(N, 2)
    points = np.clip(points, 0, 1)
    return _soft_obj(points, beta)

def _neg_score(p_flat):
    points = p_flat.reshape(N, 2)
    points = np.clip(points, 0, 1)
    return -_score(points)

def _initial_regular_polygon():
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])

def _initial_grid():
    # 4x4 grid, remove 2 corners
    grid = np.array([[i, j] for i in range(4) for j in range(4)], dtype=float)
    grid = grid / 3.0
    # Remove (0,0) and (3,3)
    mask = ~((grid[:,0]==0 & grid[:,1]==0) | (grid[:,0]==3/3.0 & grid[:,1]==3/3.0))
    grid = grid[mask][:14]
    return grid

def _initial_random(seed):
    rng = np.random.default_rng(seed)
    return rng.random((N, 2))

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_score = 0.0
    best_points = None
    
    seeds = [
        _initial_regular_polygon(),
        _initial_grid(),
        _initial_random(42),
        _initial_random(123),
        _initial_random(7)
    ]
    
    # Normalize seeds to [0,1]
    for i in range(len(seeds)):
        s = seeds[i]
        min_val = s.min(axis=0)
        max_val = s.max(axis=0)
        diff = max_val - min_val
        diff[diff == 0] = 1
        seeds[i] = (s - min_val) / diff
    
    # Multi-start with Beta Annealing
    # We'll run a few starts.
    # For each start:
    # 1. Optimize with L-BFGS-B using a soft objective with decreasing beta.
    # 2. Polish with Nelder-Mead on the hard objective.
    
    # Since time is limited, we'll do 3-5 starts.
    # Each start:
    # - Beta annealing: ~20-30 L-BFGS-B calls.
    # - Polish: ~500-1000 Nelder-Mead iterations.
    
    # Let's allocate time:
    # 5 starts * (annealing + polish)
    
    # We'll try 4 starts to be safe.
    num_starts = 4
    
    for idx, seed in enumerate(seeds[:num_starts]):
        if time.time() > deadline:
            break
            
        x0 = seed.flatten()
        
        # Phase 1: Beta Annealing with L-BFGS-B
        # Start with a large beta, decrease it.
        # Beta values: 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001
        # We'll use a smaller set to save time.
        betas = [0.05, 0.02, 0.01, 0.005, 0.002]
        
        x_current = x0.copy()
        
        for beta in betas:
            if time.time() > deadline:
                break
                
            try:
                res = minimize(
                    _neg_soft_obj,
                    x_current,
                    args=(beta,),
                    method='L-BFGS-B',
                    bounds=[(0, 1)] * (2 * N),
                    options={'maxiter': 200, 'ftol': 1e-8}
                )
                x_current = res.x
            except Exception:
                pass
        
        # Phase 2: Polish with Nelder-Mead on hard objective
        if time.time() > deadline:
            break
            
        try:
            res = minimize(
                _neg_score,
                x_current,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            x_best = res.x
        except Exception:
            x_best = x_current
            
        x_best = np.clip(x_best, 0, 1).reshape(N, 2)
        score = _score(x_best)
        
        if score > best_score:
            best_score = score
            best_points = x_best.copy()
            
        # Check time
        if time.time() > deadline:
            break
            
    if best_points is None:
        best_points = seeds[0]
        
    return best_points

# EVOLVE-BLOCK-END
