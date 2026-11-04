"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import itertools
import time

N = 13

# EVOLVE-BLOCK-START
def _triangle_areas(points):
    """
    Compute areas of all triangles formed by triples of points.
    Returns an array of areas (positive).
    """
    n = len(points)
    # Indices for all combinations of 3
    idx = np.array(list(itertools.combinations(range(n), 3)))
    # idx shape: (286, 3)
    p0 = points[idx[:, 0]]
    p1 = points[idx[:, 1]]
    p2 = points[idx[:, 2]]
    
    # Vector = p1 - p0, p2 - p0
    v1 = p1 - p0
    v2 = p2 - p0
    # Cross product in 2D: v1_x * v2_y - v1_y * v2_x
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return areas

def _score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    Returns the score (higher is better).
    """
    points = np.asarray(points)
    
    # 1. Calculate Convex Hull Area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-12:
            return 0.0
    except Exception:
        return 0.0

    # 2. Calculate Minimum Triangle Area
    areas = _triangle_areas(points)
    if len(areas) == 0:
        return 0.0
    min_area = np.min(areas)
    
    return min_area / hull_area

def _negative_score(points):
    """
    Objective function for minimization (we want to maximize score).
    """
    return -_score(points)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_score = -np.inf
    best_points = None
    
    # Strategy:
    # 1. Start with a regular 13-gon (or similar symmetric starting points)
    # 2. Use L-BFGS-B to optimize
    
    # Generate diverse initial candidates
    rng = np.random.default_rng(42)
    
    # Candidate 1: Regular 13-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    p0 = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Candidate 2: Random
    p1 = rng.random((N, 2))
    
    # Candidate 3: Random scaled
    p2 = rng.uniform(-1, 1, (N, 2))
    
    # Candidate 4: Grid-like
    # 13 points... maybe a 4x4 grid with one removed or specific arrangement
    # Let's just use random with different seeds for more diversity
    
    candidates = [p0, p1, p2]
    
    # Add a few more random candidates
    for i in range(5):
        candidates.append(rng.random((N, 2)))
    
    # Evaluate initial candidates
    for c in candidates:
        s = _score(c)
        if s > best_score:
            best_score = s
            best_points = c.copy()
            
        if time.time() > deadline:
            break
            
    # Optimize each candidate (or the best few)
    # We will optimize the top 3 initial candidates
    
    # Sort candidates by initial score
    scored_candidates = []
    for c in candidates:
        s = _score(c)
        scored_candidates.append((s, c))
    scored_candidates.sort(key=lambda x: -x[0])
    
    # Try to optimize the top 3
    for i in range(min(3, len(scored_candidates))):
        if time.time() > deadline - 10.0:
            break
            
        s_init, c_init = scored_candidates[i]
        
        # L-BFGS-B
        try:
            res = minimize(
                _negative_score,
                c_init,
                method='L-BFGS-B',
                options={'maxiter': 200, 'maxls': 50}
            )
            
            if res.success or res.fun < -best_score:
                s_new = -res.fun
                if s_new > best_score:
                    best_score = s_new
                    best_points = res.x.copy()
        except Exception:
            pass

    # If we still have time, try a few more random restarts with shorter optimization
    if best_points is None:
        return rng.random((N, 2))
        
    # Refine the best point found so far with a tighter optimization
    if time.time() < deadline - 5.0:
        try:
            res = minimize(
                _negative_score,
                best_points,
                method='L-BFGS-B',
                options={'maxiter': 500, 'maxls': 100}
            )
            s_new = -res.fun
            if s_new > best_score:
                best_score = s_new
                best_points = res.x.copy()
        except Exception:
            pass

    return best_points

# EVOLVE-BLOCK-END
