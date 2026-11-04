import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import itertools
import time

N = 13

# EVOLVE-BLOCK-START

# Precompute all triangle index triplets for vectorized area computation
_TRIPLETS = np.array(list(itertools.combinations(range(N), 3)))
_N_TRIPLES = _TRIPLETS.shape[0]
_I, _J, _K = _TRIPLETS.T

def _triangle_areas(points):
    """Compute areas of all C(N,3) triangles."""
    p1 = points[_I]
    p2 = points[_J]
    p3 = points[_K]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def _convex_hull_area(points):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return None

def _score(points):
    """Exact score: min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area is None or hull_area < 1e-9:
        return -1e6
    areas = _triangle_areas(points)
    min_area = np.min(areas)
    if min_area < 1e-9:
        return -1e6
    return min_area / hull_area

def _soft_min_score(points, beta=50.0):
    """
    Soft-min surrogate: -log(sum(exp(-beta * areas))) / beta / hull_area
    This is smooth and differentiable, suitable for L-BFGS-B.
    """
    hull_area = _convex_hull_area(points)
    if hull_area is None or hull_area < 1e-9:
        return 1e6  # Large value to be minimized
    
    areas = _triangle_areas(points)
    
    # Soft min: -1/beta * log(sum(exp(-beta * areas)))
    # Use log-sum-exp trick for numerical stability
    min_a = np.min(areas)
    if min_a < 1e-9:
        return 1e6
    
    exp_vals = np.exp(-beta * (areas - min_a))
    soft_min = min_a - np.log(np.sum(exp_vals)) / beta
    
    return soft_min / hull_area

def _neg_soft_min_score(x, beta=50.0):
    """Objective for L-BFGS-B: minimize -soft_min_score."""
    points = x.reshape(N, 2)
    val = _soft_min_score(points, beta)
    if val > 1e5:
        return 1e6
    return -val

def _halton_sequence(n, base):
    """Generate n points in the Halton sequence with given base."""
    result = np.zeros(n)
    f = 1.0 / base
    for i in range(1, n + 1):
        x = i
        result[i-1] = 0.0
        while x > 0:
            f *= base
            x, r = divmod(x, base)
            result[i-1] += r * f
    return result

def _halton_points(n_points, seed=1):
    """Generate n_points in 2D using Halton sequences for low-discrepancy sampling."""
    # Use two different bases for x and y
    x = _halton_sequence(n_points, 2)
    y = _halton_sequence(n_points, 3)
    # Scale to [-1, 1]
    points = np.column_stack([x * 2 - 1, y * 2 - 1])
    return points

def _center_ring(n_outer, n_inner, r_inner=0.5):
    """Create points on concentric circles."""
    angles_outer = np.linspace(0, 2*np.pi, n_outer, endpoint=False)
    outer = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    
    if n_inner > 0:
        angles_inner = np.linspace(0, 2*np.pi, n_inner, endpoint=False)
        inner = r_inner * np.column_stack([np.cos(angles_inner), np.sin(angles_inner)])
        return np.vstack([outer, inner])
    return outer

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin
    
    best_points = None
    best_score = -np.inf
    
    def try_optimize(init_points, beta=50.0, maxiter=300):
        nonlocal best_points, best_score
        if time.time() > deadline:
            return
        
        # Normalize initial points
        try:
            hull = ConvexHull(init_points)
            area = hull.volume
            if area < 1e-6:
                return
            scale = np.sqrt(area)
            init_points = init_points / scale
        except Exception:
            init_points = init_points + np.random.normal(0, 0.01, init_points.shape)
            try:
                hull = ConvexHull(init_points)
                area = hull.volume
                if area < 1e-6:
                    return
                scale = np.sqrt(area)
                init_points = init_points / scale
            except Exception:
                return
        
        try:
            # Phase 1: L-BFGS-B on soft-min surrogate (smooth, fast convergence)
            res = minimize(_neg_soft_min_score, init_points.flatten(), 
                          args=(beta,),
                          method='L-BFGS-B',
                          options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10})
            
            new_points = res.x.reshape(N, 2)
            
            # Phase 2: Polish with Nelder-Mead on exact objective
            if time.time() < deadline - 5:
                def exact_obj(x):
                    return -_score(x.reshape(N, 2))
                
                res2 = minimize(exact_obj, new_points.flatten(),
                               method='Nelder-Mead',
                               options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12})
                new_points = res2.x.reshape(N, 2)
            
            # Verify and update best
            direct_score = _score(new_points)
            if direct_score > best_score:
                best_score = direct_score
                best_points = new_points.copy()
                
        except Exception:
            pass

    # --- Initializations ---
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    p1 = np.column_stack([np.cos(angles), np.sin(angles)])
    try_optimize(p1)
    
    # 2. Regular 12-gon + center
    angles12 = np.linspace(0, 2*np.pi, 12, endpoint=False)
    p2 = np.column_stack([np.cos(angles12), np.sin(angles12)])
    p2 = np.vstack([p2, [[0, 0]]])
    try_optimize(p2)
    
    # 3. Two concentric hexagons + center
    p3 = _center_ring(6, 6, 0.5)
    p3 = np.vstack([p3, [[0, 0]]])
    try_optimize(p3)
    
    # 4. Halton sequence points
    p4 = _halton_points(N)
    try_optimize(p4)
    
    # 5. More Halton with different scaling
    p5 = _halton_points(N) * 1.2
    try_optimize(p5)
    
    # 6. Grid-based: 4x4 grid minus 3 corners
    x = np.linspace(-1, 1, 4)
    y = np.linspace(-1, 1, 4)
    grid = np.array([[xi, yi] for xi in x for yi in y])
    keep_indices = [i for i in range(16) if i not in [12, 14, 15]]
    p6 = grid[keep_indices]
    try_optimize(p6)
    
    # 7. Random initializations
    rng = np.random.RandomState(42)
    for i in range(8):
        if time.time() > deadline - 10:
            break
        p7 = rng.rand(N, 2) * 2 - 1
        try_optimize(p7, beta=30.0, maxiter=200)
    
    # If no valid points found, return a default regular polygon
    if best_points is None:
        return p1
        
    return best_points

# EVOLVE-BLOCK-END
