"""Heilbronn convex regions: maximize min triangle area / convex hull area for 13 points."""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 13
DEADLINE = 150.0  # seconds
SAFETY_MARGIN = 10.0  # seconds before deadline to stop

def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 points."""
    n = len(points)
    min_area = float('inf')
    # Precompute all pair differences
    # For efficiency, use vectorized approach
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                # Area = 0.5 * |cross product|
                ax, ay = points[i]
                bx, by = points[j]
                cx, cy = points[k]
                area = 0.5 * abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))
                if area < min_area:
                    min_area = area
    return min_area

def _hull_area(points):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Compute the score: min triangle area / convex hull area."""
    points = np.asarray(points)
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _negative_score(points):
    """Negative score for minimization."""
    return -_score(points)

def _smooth_min_score(points, tau=0.01):
    """
    Smooth approximation of the minimum triangle area using log-sum-exp.
    This makes the objective differentiable for gradient-based optimization.
    """
    n = len(points)
    areas = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                ax, ay = points[i]
                bx, by = points[j]
                cx, cy = points[k]
                area = 0.5 * abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))
                areas.append(area)
    areas = np.array(areas)
    # Smooth minimum: -tau * log(sum(exp(-areas/tau)))
    # As tau -> 0, this approaches min(areas)
    min_area_smooth = -tau * np.log(np.sum(np.exp(-areas / tau)))
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    return min_area_smooth / hull_area

def _negative_smooth_score(points, tau=0.01):
    return -_smooth_min_score(points, tau)

def _grid_init(rng):
    """Initialize points on a structured grid within the unit square."""
    # Use a 4x4 grid (16 positions) and pick 13, or a 3x5 grid (15) pick 13
    # Let's use a 4x4 grid and randomly select 13 of the 16 positions
    grid_x = np.linspace(0.1, 0.9, 4)
    grid_y = np.linspace(0.1, 0.9, 4)
    gx, gy = np.meshgrid(grid_x, grid_y)
    grid_points = np.column_stack([gx.ravel(), gy.ravel()])
    # Shuffle and pick 13
    idx = rng.sample(range(len(grid_points)), N)
    points = grid_points[idx].copy()
    # Add small random perturbation
    points += rng.gauss(0, 0.02, size=points.shape)
    return points

def _random_init(rng):
    """Random initialization within the unit square."""
    return np.array([rng.random(2) for _ in range(N)])

def _optimize_from_start(start_points, tau_start=0.1, tau_end=0.005, max_iter=200, tol=1e-8):
    """
    Optimize from a starting configuration using gradient-based methods.
    Uses a smooth approximation of the min triangle area, with decreasing tau.
    """
    current = start_points.copy()
    best_points = current.copy()
    best_score = _score(current)
    
    # Multi-phase optimization with decreasing tau
    num_phases = 5
    taus = np.geomspace(tau_start, tau_end, num_phases)
    
    for tau in taus:
        # Optimize with current tau
        res = minimize(
            _negative_smooth_score,
            current,
            args=(tau,),
            method='L-BFGS-B',
            options={'maxiter': max_iter // num_phases, 'ftol': tol, 'gtol': tol}
        )
        if res.success or res.fun < _negative_smooth_score(current, tau):
            current = res.x
            current_score = _score(current)
            if current_score > best_score:
                best_score = current_score
                best_points = current.copy()
    
    return best_points, best_score

def construct():
    """
    Construct 13 points in the unit square to maximize min triangle area / convex hull area.
    """
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN
    
    rng = np.random.RandomState(42)
    
    best_points = None
    best_score = 0.0
    
    # Phase 1: Grid-based initialization with optimization
    for attempt in range(3):
        if time.time() > deadline:
            break
        start = _grid_init(rng)
        # Add some randomness to the grid
        start += rng.uniform(-0.05, 0.05, size=start.shape)
        points, score = _optimize_from_start(start)
        if score > best_score:
            best_score = score
            best_points = points.copy()
    
    # Phase 2: Random initialization with optimization
    for attempt in range(5):
        if time.time() > deadline:
            break
        start = _random_init(rng)
        points, score = _optimize_from_start(start)
        if score > best_score:
            best_score = score
            best_points = points.copy()
    
    # Phase 3: Local perturbations around the best solution
    if best_points is not None and time.time() < deadline:
        for _ in range(20):
            if time.time() > deadline:
                break
            perturbed = best_points + rng.normal(0, 0.01, size=best_points.shape)
            points, score = _optimize_from_start(perturbed, tau_start=0.05, tau_end=0.003)
            if score > best_score:
                best_score = score
                best_points = points.copy()
    
    # Final polish with very small tau
    if best_points is not None and time.time() < deadline:
        points, score = _optimize_from_start(
            best_points, 
            tau_start=0.01, 
            tau_end=0.001, 
            max_iter=300
        )
        if score > best_score:
            best_score = score
            best_points = points.copy()
    
    if best_points is None:
        # Fallback: return random points
        best_points = np.array([rng.random(2) for _ in range(N)])
    
    return best_points
