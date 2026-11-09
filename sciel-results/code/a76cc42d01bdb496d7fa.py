"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull
import time

# Pre-compute indices for all triples
# C(14, 3) = 364 triples
_IDX_I = []
_IDX_J = []
_IDX_K = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _IDX_I.append(i)
            _IDX_J.append(j)
            _IDX_K.append(k)
_IDX_I = np.array(_IDX_I)
_IDX_J = np.array(_IDX_J)
_IDX_K = np.array(_IDX_K)

def _score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    Returns 0.0 if invalid.
    """
    try:
        if np.any(np.isnan(points)) or np.any(np.isinf(points)):
            return 0.0
        
        # Compute convex hull area
        try:
            hull = ConvexHull(points)
            hull_area = hull.volume
        except Exception:
            return 0.0
            
        if hull_area < 1e-10:
            return 0.0
        
        # Compute triangle areas
        pi = points[_IDX_I]
        pj = points[_IDX_J]
        pk = points[_IDX_K]
        
        v1 = pj - pi
        v2 = pk - pi
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        
        min_area = np.min(areas)
        
        return min_area / hull_area
    except Exception:
        return 0.0

def _objective(params):
    """
    Objective function for minimization: -score
    """
    points = params.reshape(N, 2)
    s = _score(points)
    if s == 0.0:
        return 1e6 # Penalty for invalid
    return -s

def _generate_initial_guesses():
    """
    Generate diverse initial configurations.
    """
    guesses = []
    
    # 1. Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    guesses.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 2. Perturbed Regular 14-gon
    rng = np.random.default_rng(42)
    for _ in range(5):
        a = angles + rng.normal(0, 0.2, N)
        r = 1.0 + rng.normal(0, 0.1, N)
        guesses.append(np.column_stack([np.cos(a) * r, np.sin(a) * r]))
        
    # 3. Concentric Rings: 1 center, 13 outer
    # 13 is prime, so regular 13-gon outer ring
    a_outer = np.linspace(0, 2 * np.pi, 13, endpoint=False)
    r_outer = 1.0
    center = np.array([[0.0, 0.0]])
    outer = np.column_stack([np.cos(a_outer) * r_outer, np.sin(a_outer) * r_outer])
    guesses.append(np.vstack([center, outer]))
    
    # 4. Concentric Rings: 2 inner, 12 outer
    # Inner: 2 points (diameter)
    inner = np.array([[0.5, 0.0], [-0.5, 0.0]])
    a_outer = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    r_outer = 1.0
    outer = np.column_stack([np.cos(a_outer) * r_outer, np.sin(a_outer) * r_outer])
    guesses.append(np.vstack([inner, outer]))
    
    # 5. Concentric Rings: 3 inner (triangle), 11 outer
    a_inner = np.linspace(0, 2 * np.pi, 3, endpoint=False)
    r_inner = 0.5
    inner = np.column_stack([np.cos(a_inner) * r_inner, np.sin(a_inner) * r_inner])
    a_outer = np.linspace(0, 2 * np.pi, 11, endpoint=False)
    r_outer = 1.0
    outer = np.column_stack([np.cos(a_outer) * r_outer, np.sin(a_outer) * r_outer])
    guesses.append(np.vstack([inner, outer]))
    
    # 6. Concentric Rings: 4 inner (square), 10 outer
    a_inner = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi/4
    r_inner = 0.5
    inner = np.column_stack([np.cos(a_inner) * r_inner, np.sin(a_inner) * r_inner])
    a_outer = np.linspace(0, 2 * np.pi, 10, endpoint=False)
    r_outer = 1.0
    outer = np.column_stack([np.cos(a_outer) * r_outer, np.sin(a_outer) * r_outer])
    guesses.append(np.vstack([inner, outer]))
    
    # 7. Grid-like (5x3 minus corners? 14 points)
    # Try a rectangular grid
    # 14 points: 4x4 grid has 16. Remove 2 corners?
    # Or 5x3 = 15, remove 1.
    xs = np.linspace(-1, 1, 5)
    ys = np.linspace(-1, 1, 3)
    xx, yy = np.meshgrid(xs, ys)
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    # Remove one point to make 14
    pts = pts[:14]
    guesses.append(pts)
    
    # 8. Random convex hull
    for _ in range(5):
        a = np.sort(rng.uniform(0, 2 * np.pi, N))
        r = 1.0 + rng.uniform(-0.2, 0.2, N)
        guesses.append(np.column_stack([np.cos(a) * r, np.sin(a) * r]))
        
    return guesses

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_points = None
    best_score = -np.inf
    
    guesses = _generate_initial_guesses()
    
    # Phase 1: Global Optimization (Differential Evolution) on a few best guesses
    # DE is good for escaping local minima
    # Bounds: -1.5 to 1.5
    bounds = [(-1.5, 1.5)] * (N * 2)
    
    # We will run DE on the top 3 initial guesses (by score)
    scored_guesses = []
    for g in guesses:
        s = _score(g)
        scored_guesses.append((s, g))
    scored_guesses.sort(key=lambda x: x[0], reverse=True)
    
    # Run DE on the best guess
    if time.time() < deadline - 60:
        try:
            res = differential_evolution(
                _objective, 
                bounds, 
                maxiter=300, 
                popsize=20, 
                seed=42,
                tol=1e-6,
                polish=False # We will polish manually
            )
            if res.fun < -best_score:
                best_score = -res.fun
                best_points = res.x.reshape(N, 2)
        except Exception:
            pass
            
    # Phase 2: Local Refinement (Powell) on best candidates
    # We take the best DE result and the top initial guesses
    candidates = []
    if best_points is not None:
        candidates.append(best_points)
    
    for s, g in scored_guesses[:3]:
        if g not in candidates:
            candidates.append(g)
            
    for guess in candidates:
        if time.time() > deadline - 10:
            break
            
        # Powell refinement
        try:
            res = minimize(
                _objective, 
                guess.flatten(), 
                method='Powell',
                options={'maxiter': 2000, 'ftol': 1e-10, 'xtol': 1e-10}
            )
            if res.fun < -best_score:
                best_score = -res.fun
                best_points = res.x.reshape(N, 2)
        except Exception:
            pass
            
    # Phase 3: Coordinate Ascent / Fine-tuning on the best point
    if best_points is not None and time.time() < deadline - 5:
        # Simple coordinate ascent with small steps
        pts = best_points.copy()
        current_score = best_score
        step_sizes = [0.05, 0.01, 0.001]
        
        for ss in step_sizes:
            improved = True
            while improved and time.time() < deadline - 2:
                improved = False
                for i in range(N):
                    for dim in range(2):
                        for delta in [ss, -ss]:
                            trial = pts.copy()
                            trial[i, dim] += delta
                            s = _score(trial)
                            if s > current_score + 1e-12:
                                pts = trial
                                current_score = s
                                improved = True
                                if current_score > best_score:
                                    best_score = current_score
                                    best_points = pts.copy()
                
    if best_points is None:
        # Fallback to regular polygon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_points
# EVOLVE-BLOCK-END
