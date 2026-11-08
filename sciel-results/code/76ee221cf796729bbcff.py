"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time

N = 14

# EVOLVE-BLOCK-START

def _all_triangle_areas(points):
    """Compute all triangle areas for N points. Returns array of C(N,3) areas."""
    n = points.shape[0]
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                cross = abs(v1[0]*v2[1] - v1[1]*v2[0])
                areas.append(0.5 * cross)
    return np.array(areas)

def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by three points."""
    n = points.shape[0]
    min_area = np.inf
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                cross = abs(v1[0]*v2[1] - v1[1]*v2[0])
                area = 0.5 * cross
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(points):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 1e-10

def _score(points):
    """Score is min_triangle_area / convex_hull_area"""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _soft_min_triangle_area(points, beta=50.0):
    """
    Soft-min surrogate for the minimum triangle area.
    Uses log-sum-exp trick: soft_min(x) = -1/beta * log(sum(exp(-beta*x)))
    This is smooth and approximates min as beta -> infinity.
    """
    n = points.shape[0]
    # Compute all triangle areas
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                cross = abs(v1[0]*v2[1] - v1[1]*v2[0])
                areas.append(0.5 * cross)
    areas = np.array(areas)
    
    # Soft min using log-sum-exp
    min_val = np.min(areas)
    # Shift for numerical stability
    shifted = -beta * (areas - min_val)
    # Clip to avoid overflow
    shifted = np.clip(shifted, -700, 700)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min = min_val - log_sum / beta
    return soft_min

def _soft_score(points, beta=50.0):
    """Soft score using soft-min triangle area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    soft_min_area = _soft_min_triangle_area(points, beta)
    return soft_min_area / hull_area

def _negative_score(flat_points):
    """Objective for minimizer - exact score."""
    points = flat_points.reshape(N, 2)
    return -_score(points)

def _negative_soft_score(flat_points, beta=50.0):
    """Objective for minimizer - soft score."""
    points = flat_points.reshape(N, 2)
    return -_soft_score(points, beta)

def _generate_initializations():
    """Generate diverse initial point configurations."""
    candidates = []
    
    # 1. Regular 14-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    poly = np.column_stack([np.cos(angles), np.sin(angles)])
    candidates.append(poly.copy())
    
    # 2. Grid-like: 4x4 grid, take 14 points
    grid_x, grid_y = np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, 1, 4))
    grid_pts = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    # Remove 2 points from corners to get 14
    grid_14 = np.delete(grid_pts, [0, 15], axis=0)  # Remove two opposite corners
    candidates.append(grid_14.copy())
    
    # 3. Two concentric circles: 7 inner, 7 outer
    inner_angles = np.linspace(0, 2*np.pi, 7, endpoint=False)
    outer_angles = np.linspace(np.pi/7, 2*np.pi, 7, endpoint=False)  # Offset
    inner_r = 0.5
    outer_r = 1.0
    inner_pts = np.column_stack([inner_r*np.cos(inner_angles), inner_r*np.sin(inner_angles)])
    outer_pts = np.column_stack([outer_r*np.cos(outer_angles), outer_r*np.sin(outer_angles)])
    two_circles = np.vstack([inner_pts, outer_pts])
    candidates.append(two_circles.copy())
    
    # 4. Perturbed regular polygon
    rng = np.random.RandomState(42)
    for _ in range(5):
        noise = rng.normal(0, 0.15, (N, 2))
        candidates.append((poly + noise).copy())
    
    # 5. Random points
    for seed in [42, 123, 456]:
        rng = np.random.RandomState(seed)
        candidates.append(rng.rand(N, 2))
    
    return candidates

def _optimize_from_start(start_pts, deadline, method='Nelder-Mead', maxiter=3000, 
                          xatol=1e-10, fatol=1e-12, use_soft=False, beta=50.0):
    """Optimize from a given starting point."""
    x0 = start_pts.flatten()
    
    if use_soft:
        obj = lambda x: _negative_soft_score(x, beta)
    else:
        obj = _negative_score
    
    try:
        res = minimize(
            obj,
            x0,
            method=method,
            options={'maxiter': maxiter, 'xatol': xatol, 'fatol': fatol}
        )
        if time.time() > deadline:
            return None, None
        points = res.x.reshape(N, 2)
        score = _score(points)
        return points, score
    except Exception:
        return None, None

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    best_score = -np.inf
    best_points = None
    
    def update_best(points, score):
        nonlocal best_score, best_points
        if score > best_score:
            best_score = score
            best_points = points
    
    # Phase 1: Coarse optimization with soft surrogate from multiple starts
    inits = _generate_initializations()
    
    for i, init_pts in enumerate(inits):
        if time.time() > deadline:
            break
        
        # Use soft score for smoother landscape
        pts, score = _optimize_from_start(
            init_pts, deadline, 
            method='Nelder-Mead',
            maxiter=2000,
            xatol=1e-8,
            fatol=1e-10,
            use_soft=True,
            beta=30.0
        )
        if pts is not None:
            # Evaluate with exact score
            exact_score = _score(pts)
            update_best(pts, exact_score)
    
    if time.time() > deadline:
        if best_points is not None:
            return best_points
        return inits[0]
    
    # Phase 2: Fine-tuning with exact objective from best candidates
    # Keep top candidates from Phase 1
    candidates_phase2 = []
    if best_points is not None:
        candidates_phase2.append(best_points.copy())
    
    # Also try direct optimization from a few good starts
    for init_pts in inits[:3]:
        if time.time() > deadline:
            break
        pts, score = _optimize_from_start(
            init_pts, deadline,
            method='Nelder-Mead',
            maxiter=3000,
            xatol=1e-10,
            fatol=1e-12,
            use_soft=False
        )
        if pts is not None:
            candidates_phase2.append(pts)
    
    # Phase 3: High-precision polish on best candidates
    for cand in candidates_phase2[:5]:
        if time.time() > deadline:
            break
        
        # Tighter tolerances
        pts, score = _optimize_from_start(
            cand, deadline,
            method='Nelder-Mead',
            maxiter=5000,
            xatol=1e-12,
            fatol=1e-14,
            use_soft=False
        )
        if pts is not None:
            update_best(pts, score)
        
        # Also try with slightly perturbed version
        rng = np.random.RandomState(99)
        perturbed = cand + rng.normal(0, 0.01, (N, 2))
        pts2, score2 = _optimize_from_start(
            perturbed, deadline,
            method='Nelder-Mead',
            maxiter=3000,
            xatol=1e-11,
            fatol=1e-13,
            use_soft=False
        )
        if pts2 is not None:
            update_best(pts2, score2)
    
    # Final check
    if time.time() > deadline:
        if best_points is not None:
            return best_points
        return inits[0]
    
    # One more round: try L-BFGS-B with bounds if time permits
    if time.time() < deadline - 20:
        # Normalize points to be in a reasonable range
        center = best_points.mean(axis=0) if best_points is not None else np.zeros(2)
        scale = np.max(np.abs(best_points - center)) if best_points is not None else 1.0
        if scale < 1e-10:
            scale = 1.0
        
        # Try optimization with BFGS
        try:
            x0 = best_points.flatten() if best_points is not None else inits[0].flatten()
            res = minimize(
                _negative_score,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if time.time() < deadline:
                pts = res.x.reshape(N, 2)
                score = _score(pts)
                update_best(pts, score)
        except Exception:
            pass
    
    if best_points is None:
        return inits[0]
    
    return best_points

# EVOLVE-BLOCK-END
