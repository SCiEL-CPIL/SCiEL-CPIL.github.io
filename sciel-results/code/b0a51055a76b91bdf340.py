import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution
import warnings
warnings.filterwarnings('ignore')

N = 13
# Pre-compute triangle indices for vectorized area calculation
_TRIPLE_IDX = np.array([
    (i, j, k)
    for i in range(N)
    for j in range(i + 1, N)
    for k in range(j + 1, N)
], dtype=np.int32)

def _triangle_areas_vectorized(points):
    """Compute areas of all triangles formed by pairs of points."""
    pi = points[_TRIPLE_IDX[:, 0]]
    pj = points[_TRIPLE_IDX[:, 1]]
    pk = points[_TRIPLE_IDX[:, 2]]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def _hull_area(points):
    """Compute area of the convex hull of the points."""
    try:
        if len(points) < 3:
            return 0.0
        # Check if points are collinear or identical
        if np.allclose(points, points[0]):
            return 0.0
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Compute the score: min triangle area / hull area."""
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    min_area = np.min(areas)
    if min_area < 1e-15:
        return 0.0
    return min_area / hull_area

def _objective_full(x_flat):
    """Objective function for optimization: minimize negative score."""
    points = x_flat.reshape(N, 2)
    score = _score(points)
    if score <= 0:
        return 1e10
    return -score

def _smooth_objective(x_flat, p=30):
    """
    Smooth surrogate objective using power mean.
    This helps gradient-based methods navigate the non-differentiable min operation.
    """
    points = x_flat.reshape(N, 2)
    centroid = np.mean(points, axis=0)
    points = points - centroid
    
    if np.allclose(points, 0):
        return 1e10
    
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 1e10
    
    areas = _triangle_areas_vectorized(points)
    areas = np.maximum(areas, 1e-15)
    
    # Soft min approximation
    inv_areas = 1.0 / areas
    soft_min = (np.mean(inv_areas ** p)) ** (-1.0 / p)
    
    score = soft_min / hull_area
    if score <= 0:
        return 1e10
    return -score

def _make_ring_config(n_outer, n_inner, r_inner, angle_offset):
    """Create a two-ring configuration."""
    angles_outer = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
    angles_inner = np.linspace(angle_offset, 2 * np.pi + angle_offset, n_inner, endpoint=False)
    
    outer_pts = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    inner_pts = np.column_stack([r_inner * np.cos(angles_inner), r_inner * np.sin(angles_inner)])
    
    return np.vstack([outer_pts, inner_pts])

def construct():
    start_time = time.time()
    # Safety margin: evaluator kills at 150s. We aim to finish by 140s.
    deadline = start_time + 140.0
    
    best_points = None
    best_score = 0.0
    
    def _update_best(points):
        nonlocal best_points, best_score
        if points is None or len(points) != N:
            return
        s = _score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()
            # print(f"New best: {s:.6f}", flush=True) # Disabled for speed

    # --- Phase 1: Global Search with Differential Evolution ---
    # DE is robust to the non-smoothness of the min-triangle-area objective
    # and can find good global configurations.
    if time.time() < deadline - 60:
        try:
            bounds = [(-2.0, 2.0)] * (2 * N)
            # DE parameters tuned for this problem size
            de_params = {
                'maxiter': 100,
                'popsize': 20,
                'mutation': (0.4, 0.9),
                'recombination': 0.7,
                'tol': 1e-8,
                'seed': 42,
                'polish': False,
                'init': 'sobol',
                'callback': None # We will check time manually if needed, but maxiter is low enough
            }
            
            # Run DE
            result = differential_evolution(
                _objective_full,
                bounds,
                **de_params
            )
            
            if result.success or result.fun < 1e9:
                de_points = result.x.reshape(N, 2)
                _update_best(de_points)
                # print(f"DE result: {best_score:.6f}", flush=True)
                
        except Exception as e:
            # print(f"DE failed: {e}", flush=True)
            pass

    # --- Phase 2: Local Refinement from Multiple Starts ---
    # We take the best point from DE (if any) and add diverse structured starts.
    # Then we run a multi-phase local optimization on each.
    
    start_points = []
    
    # 1. Add DE result if we have it
    if best_points is not None:
        start_points.append(best_points.flatten())
    
    # 2. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    regular_13 = np.column_stack([np.cos(angles), np.sin(angles)])
    start_points.append(regular_13.flatten())
    
    # 3. Two-ring configurations
    # Theoretical optimal for small N often involves concentric rings.
    for n_out in [5, 6, 7, 8]:
        n_in = N - n_out
        if n_in < 1:
            continue
        for r_in in [0.2, 0.4, 0.6, 0.8]:
            for offset in [0, np.pi / n_out, np.pi / (2 * n_out)]:
                pts = _make_ring_config(n_out, n_in, r_in, offset)
                start_points.append(pts.flatten())
    
    # 4. Random starts (just in case)
    for _ in range(5):
        start_points.append(np.random.uniform(-1, 1, (N, 2)).flatten())

    # Optimization Pipeline for each start
    for x0_flat in start_points:
        if time.time() > deadline - 10:
            break
            
        current_x = x0_flat.copy()
        current_score = _score(current_x.reshape(N, 2))
        
        if current_score > best_score:
            best_score = current_score
            best_points = current_x.reshape(N, 2).copy()
        
        # Phase 2a: Smooth Optimization (Powell)
        # Powell is derivative-free and good for this dimensionality.
        # Using the smooth objective helps it navigate better.
        if time.time() < deadline - 5:
            try:
                res = minimize(
                    lambda x: _smooth_objective(x, p=30),
                    current_x,
                    method='Powell',
                    options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8}
                )
                new_score = _score(res.x.reshape(N, 2))
                if new_score > best_score:
                    best_score = new_score
                    best_points = res.x.reshape(N, 2).copy()
                    current_x = res.x.copy()
            except Exception:
                pass
        
        # Phase 2b: Hard Objective Polish (Nelder-Mead)
        # Now that we are close, use the exact objective for final precision.
        if time.time() < deadline - 2:
            try:
                res_nm = minimize(
                    _objective_full,
                    current_x,
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
                )
                new_score = _score(res_nm.x.reshape(N, 2))
                if new_score > best_score:
                    best_score = new_score
                    best_points = res_nm.x.reshape(N, 2).copy()
            except Exception:
                pass

    # Fallback
    if best_points is None:
        return np.random.rand(N, 2)
        
    return best_points
