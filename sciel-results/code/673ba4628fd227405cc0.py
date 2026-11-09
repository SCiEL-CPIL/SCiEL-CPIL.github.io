# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
import scipy.optimize


def _compute_score(points: np.ndarray) -> float:
    """Compute A_min / A_hull for a set of points."""
    n = points.shape[0]
    if n < 3:
        return 0.0
    
    # Check for NaN or inf
    if not np.all(np.isfinite(points)):
        return 0.0
    
    # Compute convex hull
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 0.0
    
    if A_hull <= 0:
        return 0.0
    
    # Compute all triangle areas
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                area = abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0
                if area < min_area:
                    min_area = area
                if min_area == 0:
                    return 0.0
    
    if min_area == 0:
        return 0.0
    
    return min_area / A_hull


def _local_search(points: np.ndarray, max_iters: int = 500, step_size: float = 0.1,
                  learning_rate: float = 0.95, min_step: float = 1e-7,
                  deadline: float = None) -> np.ndarray:
    """Perform local search to improve the point configuration."""
    best_points = points.copy()
    best_score = _compute_score(best_points)
    current_step = step_size
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        for i in range(len(best_points)):
            # Try 8 directions + maybe diagonal
            for dx in [-current_step, 0, current_step]:
                for dy in [-current_step, 0, current_step]:
                    if dx == 0 and dy == 0:
                        continue
                    candidate = best_points.copy()
                    candidate[i, 0] += dx
                    candidate[i, 1] += dy
                    
                    # Keep points in [0, 1]^2
                    if np.any(candidate < 0) or np.any(candidate > 1):
                        continue
                    
                    score = _compute_score(candidate)
                    if score > best_score:
                        best_points = candidate
                        best_score = score
                        improved = True
        
        if not improved:
            current_step *= learning_rate
            if current_step < min_step:
                break
        else:
            # Slightly increase step if we found improvement
            current_step = max(current_step, min_step)
    
    return best_points


def _powell_refine(points: np.ndarray, deadline: float, max_iters: int = 300) -> np.ndarray:
    """Use Powell's method to refine points."""
    n = 13
    
    def objective(points_flat):
        pts = points_flat.reshape(n, 2)
        # Penalize points outside [0,1]^2
        if np.any(pts < -0.1) or np.any(pts > 1.1):
            return 1e6
        score = _compute_score(pts)
        if score == 0.0:
            return 1e6
        return -score
    
    try:
        res = scipy.optimize.minimize(
            objective,
            points.flatten(),
            method='Powell',
            options={'maxiter': max_iters, 'ftol': 1e-10, 'xatol': 1e-10,
                     'disp': False}
        )
        candidate = res.x.reshape(n, 2)
        # Clip to [0,1]
        candidate = np.clip(candidate, 0, 1)
        return candidate
    except Exception:
        return points


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 13.

    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    deadline = time.time() + 590  # Safety margin before 630s kill
    
    best_points = None
    best_score = 0.0
    
    # Strategy 1: Start with regular polygon (inspiration 1)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    init_poly = np.column_stack([0.5 + 0.45 * np.cos(angles), 
                                  0.5 + 0.45 * np.sin(angles)])
    
    # Refine with local search
    if time.time() < deadline - 100:
        refined = _local_search(init_poly, max_iters=300, step_size=0.05,
                                learning_rate=0.9, min_step=1e-6,
                                deadline=deadline - 100)
        score = _compute_score(refined)
        if score > best_score:
            best_score = score
            best_points = refined.copy()
    
    # Refine with Powell
    if time.time() < deadline - 50 and best_points is not None:
        powell_result = _powell_refine(best_points, deadline, max_iters=200)
        score = _compute_score(powell_result)
        if score > best_score:
            best_score = score
            best_points = powell_result.copy()
    
    # Strategy 2: Multiple random restarts with local search
    num_restarts = 20
    for seed in range(num_restarts):
        if time.time() > deadline - 30:
            break
        
        rng = np.random.default_rng(seed=seed + 100)
        points = rng.random((n, 2))
        
        # Local search
        points = _local_search(points, max_iters=200, step_size=0.08,
                               learning_rate=0.9, min_step=1e-6,
                               deadline=deadline - 30)
        
        score = _compute_score(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()
    
    # Strategy 3: Try Powell from a few different starting points
    if time.time() < deadline - 20 and best_points is not None:
        # Perturb best points and run Powell
        rng = np.random.default_rng(seed=999)
        perturbed = best_points + rng.normal(0, 0.02, (n, 2))
        perturbed = np.clip(perturbed, 0, 1)
        
        powell_result2 = _powell_refine(perturbed, deadline, max_iters=150)
        score = _compute_score(powell_result2)
        if score > best_score:
            best_score = score
            best_points = powell_result2.copy()
    
    # Fallback: simple grid if nothing worked
    if best_points is None:
        xs = np.linspace(0.1, 0.9, 4)
        ys = np.linspace(0.1, 0.9, 4)
        grid_points = []
        for x in xs:
            for y in ys:
                grid_points.append([x, y])
        best_points = np.array(grid_points[:13])
    
    return best_points


# EVOLVE-BLOCK-END
