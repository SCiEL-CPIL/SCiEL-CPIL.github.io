import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time

N = 14

# Pre-compute indices for all triples
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

def _angles_to_points(angles):
    """Convert sorted angles on unit circle to points."""
    angles = np.sort(angles) % (2 * np.pi)
    return np.column_stack([np.cos(angles), np.sin(angles)])

def _circular_objective(angles):
    """
    Objective for circular parameterization: minimize -score
    """
    try:
        if np.any(np.isnan(angles)) or np.any(np.isinf(angles)):
            return 1e6
        points = _angles_to_points(angles)
        s = _score(points)
        if s <= 1e-15:
            return 1e6
        return -s
    except Exception:
        return 1e6

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s safety margin
    
    best_points = None
    best_score = -np.inf
    best_angles = None
    
    # Initial guesses
    guesses = []
    
    # 1. Regular polygon
    angles_reg = np.linspace(0, 2 * np.pi, N, endpoint=False)
    guesses.append(angles_reg)
    
    # 2. Perturbed regular polygons
    rng = np.random.default_rng(42)
    for _ in range(20):
        noise = rng.normal(0, 0.5, N)
        guesses.append(angles_reg + noise)
        
    # 3. Random angles
    for _ in range(20):
        guesses.append(np.sort(rng.uniform(0, 2 * np.pi, N)))
        
    # 4. Clustered angles (to explore non-uniform distributions)
    for spread in [0.5, 1.0, 2.0]:
        for _ in range(5):
            base = rng.uniform(0, 2 * np.pi, N)
            # Add large offsets to some points to break symmetry
            offsets = rng.uniform(-spread, spread, N)
            guesses.append(np.sort(base + offsets))

    # Evaluate initial guesses
    scored_guesses = []
    for g in guesses:
        points = _angles_to_points(g)
        s = _score(points)
        scored_guesses.append((s, g))
        if s > best_score:
            best_score = s
            best_points = points.copy()
            best_angles = g.copy()
            
    # Sort by score
    scored_guesses.sort(key=lambda x: x[0], reverse=True)
    
    # Refine top candidates
    # Use Nelder-Mead for local refinement
    for idx in range(min(10, len(scored_guesses))):
        if time.time() > deadline - 10:
            break
            
        s_init, angles = scored_guesses[idx]
        
        try:
            res = minimize(
                _circular_objective,
                angles,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            s = -res.fun
            if s > best_score:
                best_score = s
                best_points = _angles_to_points(res.x)
                best_angles = res.x
        except Exception:
            pass

    # Final polish on the best
    if best_angles is not None and time.time() < deadline - 2:
        try:
            res = minimize(
                _circular_objective,
                best_angles,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            s = -res.fun
            if s > best_score:
                best_score = s
                best_points = _angles_to_points(res.x)
                best_angles = res.x
        except Exception:
            pass

    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = _angles_to_points(angles)
        
    return best_points
