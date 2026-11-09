# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time


def _triangle_areas_vectorized(points):
    """Compute all 286 triangle areas vectorized."""
    n = len(points)
    idx = np.array([(i, j, k) for i in range(n) for j in range(i+1, n) for k in range(j+1, n)])
    p1 = points[idx[:, 0]]
    p2 = points[idx[:, 1]]
    p3 = points[idx[:, 2]]
    areas = 0.5 * np.abs(
        p1[:, 0] * (p2[:, 1] - p3[:, 1]) +
        p2[:, 0] * (p3[:, 1] - p1[:, 1]) +
        p3[:, 0] * (p1[:, 1] - p2[:, 1])
    )
    return areas


def _min_triangle_area(points):
    """Compute the minimum triangle area among all C(13,3)=286 triangles."""
    return np.min(_triangle_areas_vectorized(points))


def _hull_area(points):
    """Compute the convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0


def _score(points):
    """Compute the Heilbronn score: A_min / A_hull."""
    a_min = _min_triangle_area(points)
    a_hull = _hull_area(points)
    if a_hull <= 0 or not np.isfinite(a_min) or not np.isfinite(a_hull):
        return -1.0
    return a_min / a_hull


def _soft_min_objective(params, temperature=0.01):
    """
    Smooth surrogate for the min triangle area using log-sum-exp.
    Minimizing this encourages all triangle areas to be large.
    """
    points = params.reshape(13, 2)
    if not np.all(np.isfinite(points)):
        return 1e10
    
    areas = _triangle_areas_vectorized(points)
    
    # Soft min via log-sum-exp: -log(sum(exp(-areas/T))) * T
    # This is a smooth approximation of min(areas)
    max_neg = -np.max(areas)
    log_sum = max_neg + np.log(np.sum(np.exp(-areas / temperature - max_neg)))
    soft_min = -log_sum * temperature
    
    # Also penalize if hull area is too small or invalid
    a_hull = _hull_area(points)
    if a_hull <= 0:
        return 1e10
    
    # We want to maximize soft_min / a_hull, so minimize -(soft_min / a_hull)
    return -(soft_min / a_hull)


def _objective(params):
    """Objective to minimize (negative score) for SLSQP."""
    points = params.reshape(13, 2)
    if not np.all(np.isfinite(points)):
        return 1e10
    score = _score(points)
    if score < 0:
        return 1e10
    return -score


def _make_seed_hexagonal():
    """Create a hexagonal/triangular lattice seed for 13 points."""
    pts = []
    # Row 0: 1 point
    pts.append([0.0, 0.0])
    # Row 1: 2 points
    pts.append([-0.5, np.sqrt(3)/2])
    pts.append([0.5, np.sqrt(3)/2])
    # Row 2: 3 points
    pts.append([-1.0, np.sqrt(3)])
    pts.append([0.0, np.sqrt(3)])
    pts.append([1.0, np.sqrt(3)])
    # Row 3: 4 points
    pts.append([-1.5, 1.5*np.sqrt(3)])
    pts.append([-0.5, 1.5*np.sqrt(3)])
    pts.append([0.5, 1.5*np.sqrt(3)])
    pts.append([1.5, 1.5*np.sqrt(3)])
    # Row 4: 3 points (to make 13 total: 1+2+3+4+3=13)
    pts.append([-1.0, 2*np.sqrt(3)])
    pts.append([0.0, 2*np.sqrt(3)])
    pts.append([1.0, 2*np.sqrt(3)])
    
    points = np.array(pts)
    points = points - points.mean(axis=0)
    scale = np.max(np.abs(points))
    if scale > 0:
        points = points / scale
    return points


def _make_seed_grid():
    """Create a grid-based seed for 13 points."""
    pts = []
    for i in range(4):
        for j in range(4):
            pts.append([i, j])
    points = np.array(pts[:13], dtype=float)
    points = points - points.mean(axis=0)
    scale = np.max(np.abs(points))
    if scale > 0:
        points = points / scale
    return points


def _make_seed_circular():
    """Create a circular seed with center point."""
    points = np.zeros((13, 2))
    points[0] = [0.0, 0.0]
    for i in range(12):
        angle = i * 2 * np.pi / 12
        points[i + 1] = [np.cos(angle), np.sin(angle)]
    return points


def _make_seed_mixed():
    """Create a mixed seed: center + inner ring + outer ring."""
    points = np.zeros((13, 2))
    points[0] = [0.0, 0.0]
    for i in range(4):
        angle = i * np.pi / 2 + np.pi / 4
        points[i + 1] = [0.5 * np.cos(angle), 0.5 * np.sin(angle)]
    for i in range(8):
        angle = i * 2 * np.pi / 8
        points[i + 5] = [np.cos(angle), np.sin(angle)]
    return points


def _make_seed_pentagonal():
    """Create a pentagonal-based seed."""
    points = np.zeros((13, 2))
    # Center
    points[0] = [0.0, 0.0]
    # Inner ring: 5 points at distance 0.5
    for i in range(5):
        angle = i * 2 * np.pi / 5
        points[i + 1] = [0.5 * np.cos(angle), 0.5 * np.sin(angle)]
    # Outer ring: 7 points at distance 1.0
    for i in range(7):
        angle = i * 2 * np.pi / 7
        points[i + 6] = [np.cos(angle), np.sin(angle)]
    return points


def _make_seed_triangular_ring():
    """Create a triangular ring seed."""
    points = np.zeros((13, 2))
    points[0] = [0.0, 0.0]
    # 6 points on inner hexagon
    for i in range(6):
        angle = i * np.pi / 3
        points[i + 1] = [0.6 * np.cos(angle), 0.6 * np.sin(angle)]
    # 6 points on outer hexagon, offset by 30 degrees
    for i in range(6):
        angle = i * np.pi / 3 + np.pi / 6
        points[i + 7] = [np.cos(angle), np.sin(angle)]
    return points


def _normalize_points(points):
    """Center and scale points to unit max norm."""
    points = points - points.mean(axis=0)
    scale = np.max(np.abs(points))
    if scale > 0:
        points = points / scale
    return points


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the minimum triangle area
    relative to the convex hull area (Heilbronn problem).
    
    Uses structured seeds with multi-stage optimization:
    1. SLSQP with exact objective from multiple seeds
    2. Soft-min smoothing for better gradient behavior
    3. Final polish with tighter tolerance
    
    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 600s limit
    
    # Generate candidate seeds
    seeds = [
        _make_seed_hexagonal(),
        _make_seed_grid(),
        _make_seed_circular(),
        _make_seed_mixed(),
        _make_seed_pentagonal(),
        _make_seed_triangular_ring(),
    ]
    
    # Add randomized perturbations
    rng = np.random.default_rng(seed=42)
    for seed_pts in seeds[:3]:
        for _ in range(4):
            perturbed = seed_pts + rng.normal(0, 0.08, (13, 2))
            seeds.append(perturbed)
    
    best_points = None
    best_score = -1.0
    
    # Stage 1: SLSQP with exact objective
    for seed in seeds:
        if time.time() > deadline - 60:
            break
        
        seed_norm = _normalize_points(seed)
        initial_params = seed_norm.flatten()
        
        try:
            result = minimize(
                _objective,
                initial_params,
                method='SLSQP',
                options={'maxiter': 150, 'ftol': 1e-10},
            )
            
            if result.fun < -best_score:
                opt_points = result.x.reshape(13, 2)
                opt_points = _normalize_points(opt_points)
                score = _score(opt_points)
                if score > best_score:
                    best_score = score
                    best_points = opt_points.copy()
                    
        except Exception:
            continue
    
    # Stage 2: Soft-min optimization for smoother gradients
    if best_points is not None:
        if time.time() < deadline - 30:
            # Use best points as starting point for soft-min optimization
            for temp in [0.05, 0.02, 0.01]:
                if time.time() > deadline - 30:
                    break
                
                try:
                    result = minimize(
                        lambda p: _soft_min_objective(p, temperature=temp),
                        best_points.flatten(),
                        method='SLSQP',
                        options={'maxiter': 100, 'ftol': 1e-10},
                    )
                    
                    if result.fun < _soft_min_objective(best_points.flatten(), temperature=temp):
                        opt_points = result.x.reshape(13, 2)
                        opt_points = _normalize_points(opt_points)
                        score = _score(opt_points)
                        if score > best_score:
                            best_score = score
                            best_points = opt_points.copy()
                            
                except Exception:
                    continue
    
    # Stage 3: Final high-precision polish
    if best_points is not None and time.time() < deadline - 10:
        try:
            result = minimize(
                _objective,
                best_points.flatten(),
                method='SLSQP',
                options={'maxiter': 200, 'ftol': 1e-12},
            )
            
            if result.fun < -best_score:
                opt_points = result.x.reshape(13, 2)
                opt_points = _normalize_points(opt_points)
                score = _score(opt_points)
                if score > best_score:
                    best_score = score
                    best_points = opt_points.copy()
                    
        except Exception:
            pass
    
    # If optimization didn't find anything good, use the best seed
    if best_points is None:
        best_points = _normalize_points(seeds[0])
    
    # Final verification
    if not np.all(np.isfinite(best_points)) or best_points.shape != (13, 2):
        best_points = _make_seed_hexagonal()
    
    return best_points


# EVOLVE-BLOCK-END
