import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 14
START_TIME = time.time()
DEADLINE = START_TIME + 140.0  # Safety margin for 150s limit

def _get_triple_indices(n):
    indices = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                indices.append((i, j, k))
    return np.array(indices, dtype=int)

TRIPLE_INDICES = _get_triple_indices(N)
I_IDX = TRIPLE_INDICES[:, 0]
J_IDX = TRIPLE_INDICES[:, 1]
K_IDX = TRIPLE_INDICES[:, 2]

def _score(points):
    """Compute min triangle area / convex hull area."""
    if len(points) < 3:
        return 0.0
    
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0
        
    if hull_area < 1e-12:
        return 0.0
        
    p_i = points[I_IDX]
    p_j = points[J_IDX]
    p_k = points[K_IDX]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    return min_area / hull_area

def _soft_score(points, k=100.0):
    """
    Compute a smooth approximation of the min area / hull area.
    Uses log-sum-exp to approximate the minimum.
    -min(a_i) approx - (1/k) log(sum(exp(-k * a_i)))
    """
    if len(points) < 3:
        return 0.0
    
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0
        
    if hull_area < 1e-12:
        return 0.0
        
    p_i = points[I_IDX]
    p_j = points[J_IDX]
    p_k = points[K_IDX]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Normalize areas by hull area to make it scale-invariant in the soft term?
    # The ratio is what we want.
    # min(areas) / hull_area
    # Let's optimize - (min(areas) / hull_area)
    # Soft min of areas: -1/k * log(sum(exp(-k * areas)))
    # This is tricky because hull_area is in the denominator.
    # Let's just use the soft min of the areas, and divide by hull area at the end?
    # No, the objective should be smooth.
    
    # Approximation: - (1/k) * log(sum(exp(-k * areas))) / hull_area
    # This is not exactly smooth because hull_area is in denominator, but it's better than hard min.
    
    max_area = np.max(areas)
    exp_terms = np.exp(-k * (areas - max_area))
    log_sum_exp = max_area + (1.0/k) * np.log(np.sum(exp_terms))
    soft_min_area = log_sum_exp
    
    return soft_min_area / hull_area

def _neg_soft_score(flat_points, k=100.0):
    points = flat_points.reshape((N, 2))
    return -_soft_score(points, k)

def _neg_hard_score(flat_points):
    points = flat_points.reshape((N, 2))
    return -_score(points)

def _make_regular_polygon(n, radius=1.0, offset=0.0):
    angles = np.linspace(offset, 2*np.pi + offset, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _polish(points, time_budget=10.0):
    """
    Polish points using Powell method on soft score, then hard score.
    """
    start_time = time.time()
    
    # 1. Polish with soft score to smooth out the landscape
    try:
        res_soft = minimize(
            _neg_soft_score,
            points.flatten(),
            args=(50.0,), # k parameter
            method='Powell',
            options={'maxiter': 100, 'maxfev': 5000}
        )
        points_soft = res_soft.x.reshape((N, 2))
        
        # Check if it improved
        s_soft = _score(points_soft)
        s_orig = _score(points)
        if s_soft > s_orig:
            points = points_soft
    except Exception:
        pass
    
    if time.time() > start_time + time_budget * 0.5:
        return points
        
    # 2. Polish with hard score for final accuracy
    try:
        res_hard = minimize(
            _neg_hard_score,
            points.flatten(),
            method='Powell',
            options={'maxiter': 200, 'maxfev': 10000}
        )
        points_hard = res_hard.x.reshape((N, 2))
        
        s_hard = _score(points_hard)
        s_curr = _score(points)
        if s_hard > s_curr:
            points = points_hard
    except Exception:
        pass
        
    return points

def construct():
    global START_TIME, DEADLINE
    START_TIME = time.time()
    DEADLINE = START_TIME + 140.0
    
    np.random.seed(42)
    
    best_score = -np.inf
    best_points = None
    
    # 1. Regular Polygon Baseline
    reg_poly = _make_regular_polygon(N)
    s_reg = _score(reg_poly)
    best_score = s_reg
    best_points = reg_poly.copy()
    
    # 2. Polish the regular polygon
    polished = _polish(reg_poly, time_budget=30.0)
    s_polished = _score(polished)
    if s_polished > best_score:
        best_score = s_polished
        best_points = polished
        
    # 3. Try small perturbations of the regular polygon
    # The optimal solution might be slightly distorted from the regular polygon.
    # We will try several random perturbations and polish them.
    
    num_perturbations = 10
    for i in range(num_perturbations):
        if time.time() > DEADLINE - 20.0:
            break
            
        # Generate a perturbed regular polygon
        # Add small noise to the radii
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        radii = 1.0 + np.random.normal(0, 0.05, N)
        # Ensure convexity by keeping radii close to 1
        radii = np.clip(radii, 0.8, 1.2)
        
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        
        # Polish this perturbation
        polished_pt = _polish(pts, time_budget=10.0)
        s_pt = _score(polished_pt)
        
        if s_pt > best_score:
            best_score = s_pt
            best_points = polished_pt
            
    # 4. Try a two-ring configuration as a fallback/alternative
    # 7 points on outer ring, 7 on inner ring
    # This is a different structural family.
    if time.time() < DEADLINE - 15.0:
        for r in [0.5, 0.7, 0.8]:
            if time.time() > DEADLINE - 10.0:
                break
            # Generate two rings
            angles_out = np.linspace(0, 2*np.pi, 7, endpoint=False)
            angles_in = np.linspace(np.pi/7, 2*np.pi + np.pi/7, 7, endpoint=False) # Rotated
            
            pts_out = np.column_stack([1.0 * np.cos(angles_out), 1.0 * np.sin(angles_out)])
            pts_in = np.column_stack([r * np.cos(angles_in), r * np.sin(angles_in)])
            
            pts = np.vstack([pts_out, pts_in])
            
            polished_ring = _polish(pts, time_budget=5.0)
            s_ring = _score(polished_ring)
            
            if s_ring > best_score:
                best_score = s_ring
                best_points = polished_ring

    # 5. Final high-precision polish on the absolute best
    if best_points is not None and time.time() < DEADLINE - 5.0:
        try:
            res_final = minimize(
                _neg_hard_score,
                best_points.flatten(),
                method='Powell',
                options={'maxiter': 500, 'maxfev': 20000}
            )
            final_points = res_final.x.reshape((N, 2))
            final_score = _score(final_points)
            
            if final_score > best_score:
                best_score = final_score
                best_points = final_points
        except Exception:
            pass

    return best_points
