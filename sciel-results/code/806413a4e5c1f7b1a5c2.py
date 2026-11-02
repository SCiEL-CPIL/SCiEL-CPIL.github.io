import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize

N = 13
# Pre-compute all triangle index triples
TRI_INDICES = []
for i in range(N):
    for j in range(i+1, N):
        for k in range(j+1, N):
            TRI_INDICES.append((i, j, k))
TRI_INDICES = np.array(TRI_INDICES)

def _all_triangle_areas(pts):
    """
    Compute areas of all triangles formed by triples of points.
    pts: (N, 2) array
    Returns: array of shape (286,) with areas
    """
    # pts[i], pts[j], pts[k]
    p1 = pts[TRI_INDICES[:, 0]]
    p2 = pts[TRI_INDICES[:, 1]]
    p3 = pts[TRI_INDICES[:, 2]]
    
    # Area = 0.5 * | (p2-p1) x (p3-p1) |
    # Cross product in 2D: (x2-x1)*(y3-y1) - (y2-y1)*(x3-x1)
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    return areas

def _hull_area(pts):
    """
    Compute convex hull area.
    """
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score(pts):
    """
    Compute score: min triangle area / convex hull area.
    """
    pts = np.asarray(pts)
    if pts.shape != (N, 2):
        return 0.0
    
    areas = _all_triangle_areas(pts)
    min_area = np.min(areas)
    
    if min_area <= 0:
        return 0.0
        
    hull_area = _hull_area(pts)
    if hull_area <= 1e-12:
        return 0.0
        
    return min_area / hull_area

def _neg_score(pts):
    """
    Objective for minimization.
    """
    return -_score(pts)

def _soft_min_score(pts, beta=50.0):
    """
    Smooth surrogate for the minimum area to aid gradient-based methods or DE.
    Uses log-sum-exp style soft minimum.
    """
    areas = _all_triangle_areas(pts)
    # Avoid log(0)
    areas_safe = np.maximum(areas, 1e-12)
    # Soft min: -1/beta * log(sum(exp(-beta * areas)))
    # This is smooth and approximates min(areas) as beta -> inf
    log_sum = np.log(np.sum(np.exp(-beta * areas_safe)))
    soft_min_area = -1.0 / beta * log_sum
    
    hull_area = _hull_area(pts)
    if hull_area <= 1e-12:
        return 1.0 # Penalty
    
    return -soft_min_area / hull_area

def _initialization():
    """
    Generate an initial configuration.
    We try a mix of regular polygon, center+ring, and random.
    """
    r = np.random.random()
    if r < 0.3:
        # Regular 13-gon
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        return np.column_stack((np.cos(angles), np.sin(angles)))
    elif r < 0.6:
        # 1 center + 12 ring
        pts = np.zeros((N, 2))
        angles = np.linspace(0, 2*np.pi, 12, endpoint=False)
        for i in range(12):
            pts[i+1] = [np.cos(angles[i]), np.sin(angles[i])]
        # Center is 0,0
        return pts
    else:
        # Random in unit circle
        angles = np.random.uniform(0, 2*np.pi, N)
        radii = np.sqrt(np.random.uniform(0.1, 1.0, N))
        return np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin for 150s kill
    
    best_score = 0.0
    best_points = None
    
    # Strategy 1: Differential Evolution on full 26D space
    # Bounds: [-1.5, 1.5] for x and y
    bounds = [(-1.5, 1.5)] * (2 * N)
    
    # We will use a custom objective that checks time
    global _current_deadline
    _current_deadline = deadline
    
    def _de_objective(x_flat):
        if time.time() > _current_deadline:
            return 1e10
        pts = x_flat.reshape((N, 2))
        # Normalize points to avoid scaling issues? 
        # The score is scale invariant. DE might wander to large/small scales.
        # Let's normalize the hull area to 1 inside the objective? No, that breaks differentiability slightly but is okay.
        # Actually, score is scale invariant. Let's just compute it.
        # To help DE, maybe center the points?
        pts_centered = pts - np.mean(pts, axis=0)
        # Scale to unit area hull?
        # If we don't scale, DE has to find the right scale too.
        # Let's enforce unit area hull in the objective to reduce search space effectively.
        hull_area = _hull_area(pts_centered)
        if hull_area < 1e-6:
            return 1e10
        scale_factor = 1.0 / np.sqrt(hull_area)
        pts_scaled = pts_centered * scale_factor
        return _neg_score(pts_scaled)
        
    try:
        # Use DE with moderate population
        # maxiter is limited by time
        result = differential_evolution(
            _de_objective, 
            bounds, 
            seed=42, 
            maxiter=100, 
            tol=1e-5, 
            popsize=15, # 15 * 26 = 390 evaluations per generation
            polish=True, # Use L-BFGS-B polish
            update_interval=1,
            callback=lambda xk, convergence: (time.time() > _current_deadline) # Stop if time exceeded
        )
        
        if result.success or result.fun < 1e9:
            pts_de = result.x.reshape((N, 2))
            # Normalize
            pts_de = pts_de - np.mean(pts_de, axis=0)
            hull_area = _hull_area(pts_de)
            if hull_area > 1e-6:
                pts_de = pts_de * (1.0 / np.sqrt(hull_area))
                score_de = _score(pts_de)
                if score_de > best_score:
                    best_score = score_de
                    best_points = pts_de
                    
    except Exception:
        pass

    # Strategy 2: Local refinement from DE result or other starts
    # We can try to refine the best point found so far using Nelder-Mead or Powell
    # on the normalized space.
    
    def _local_objective(x_flat):
        if time.time() > _current_deadline:
            return 1e10
        pts = x_flat.reshape((N, 2))
        pts_centered = pts - np.mean(pts, axis=0)
        hull_area = _hull_area(pts_centered)
        if hull_area < 1e-6:
            return 1e10
        pts_scaled = pts_centered * (1.0 / np.sqrt(hull_area))
        return _neg_score(pts_scaled)

    starts = []
    if best_points is not None:
        starts.append(best_points.flatten())
    
    # Add a few random starts for local search
    rng = np.random.default_rng(123)
    for _ in range(3):
        if time.time() > deadline - 10:
            break
        pts_rand = _initialization()
        pts_rand = pts_rand - np.mean(pts_rand, axis=0)
        hull_area = _hull_area(pts_rand)
        if hull_area > 1e-6:
             pts_rand = pts_rand * (1.0 / np.sqrt(hull_area))
        starts.append(pts_rand.flatten())

    for start in starts:
        if time.time() > deadline - 5:
            break
        try:
            res_local = minimize(
                _local_objective, 
                start, 
                method='Powell', 
                options={'maxiter': 200, 'ftol': 1e-8, 'xtol': 1e-8}
            )
            if time.time() > deadline:
                break
            pts_local = res_local.x.reshape((N, 2))
            pts_local = pts_local - np.mean(pts_local, axis=0)
            hull_area = _hull_area(pts_local)
            if hull_area > 1e-6:
                pts_local = pts_local * (1.0 / np.sqrt(hull_area))
                score_local = _score(pts_local)
                if score_local > best_score:
                    best_score = score_local
                    best_points = pts_local
        except Exception:
            continue

    # Fallback if nothing worked
    if best_points is None:
        # Regular 13-gon
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
        
    return best_points

# EVOLVE-BLOCK-END
