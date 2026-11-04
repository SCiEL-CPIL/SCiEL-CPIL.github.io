"""Heilbronn convex regions n=14: maximize min triangle area / convex hull area."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14
DEADLINE = 145.0  # safety margin before 150s kill


def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = 0.5 * abs(
                    (points[j][0] - points[i][0]) * (points[k][1] - points[i][1]) -
                    (points[j][1] - points[i][1]) * (points[k][0] - points[i][0])
                )
                if area < min_area:
                    min_area = area
    return min_area


def _convex_hull_area(points):
    """Compute the area of the convex hull of a set of 2D points."""
    if len(points) < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 1.0


def _score(points):
    """Score = min triangle area / convex hull area."""
    min_ta = _min_triangle_area(points)
    hull_area = _convex_hull_area(points)
    if hull_area <= 1e-12:
        return 0.0
    return min_ta / hull_area


def _soft_min_score(points, temperature=0.01):
    """Soft minimum of triangle areas divided by hull area, for differentiable optimization."""
    n = len(points)
    areas = np.zeros((n * (n - 1) * (n - 2)) // 6)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = 0.5 * abs(
                    (points[j][0] - points[i][0]) * (points[k][1] - points[i][1]) -
                    (points[j][1] - points[i][1]) * (points[k][0] - points[i][0])
                )
                areas[idx] = area
                idx += 1
    # Soft min: log-sum-exp trick
    log_areas = np.log(areas + 1e-15)
    log_min = np.log(np.sum(np.exp(log_areas / temperature))) * temperature
    soft_min_area = np.exp(log_min)
    
    hull_area = _convex_hull_area(points)
    if hull_area <= 1e-12:
        return 0.0
    return soft_min_area / hull_area


def _objective_neg(x):
    """Negative soft-min score for minimization."""
    points = x.reshape(-1, 2)
    return -_soft_min_score(points, temperature=0.02)


def _gradient_neg(x):
    """Numerical gradient of negative soft-min score."""
    points = x.reshape(-1, 2)
    eps = 1e-5
    grad = np.zeros_like(x)
    f0 = _objective_neg(x)
    for i in range(len(x)):
        xp = x.copy()
        xp[i] += eps
        grad[i] = (_objective_neg(xp) - f0) / eps
    return grad


def _local_update_polish(points, max_iter=2000, step_init=0.05):
    """Coordinate descent with local updates for fine-tuning."""
    n = len(points)
    points = points.copy()
    current_score = _score(points)
    step = step_init
    
    # Precompute all triple indices
    triples = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                triples.append((i, j, k))
    
    for it in range(max_iter):
        if time.time() > DEADLINE - 5:
            break
        improved = False
        for i in range(n):
            # Compute min area involving point i
            min_area_i = float('inf')
            for (a, b, c) in triples:
                if i in (a, b, c):
                    area = 0.5 * abs(
                        (points[b][0] - points[a][0]) * (points[c][1] - points[a][1]) -
                        (points[b][1] - points[a][1]) * (points[c][0] - points[a][0])
                    )
                    if area < min_area_i:
                        min_area_i = area
            
            # Try small perturbations in each direction
            best_move = None
            best_area = min_area_i
            for dx in [-step, step]:
                for dy in [-step, step]:
                    new_pt = points[i] + np.array([dx, dy])
                    # Check bounds (keep in [0,1]^2 roughly)
                    if new_pt[0] < -0.5 or new_pt[0] > 1.5 or new_pt[1] < -0.5 or new_pt[1] > 1.5:
                        continue
                    points[i] = new_pt
                    new_min = float('inf')
                    for (a, b, c) in triples:
                        if i in (a, b, c):
                            area = 0.5 * abs(
                                (points[b][0] - points[a][0]) * (points[c][1] - points[a][1]) -
                                (points[b][1] - points[a][1]) * (points[c][0] - points[a][0])
                            )
                            if area < new_min:
                                new_min = area
                    if new_min > best_area + 1e-10:
                        best_area = new_min
                        best_move = new_pt.copy()
            
            if best_move is not None:
                points[i] = best_move
                improved = True
            else:
                # Try smaller step
                step = max(step * 0.9, 1e-6)
        
        if not improved:
            step = max(step * 0.7, 1e-7)
            if step < 1e-8:
                break
    
    new_score = _score(points)
    if new_score > current_score:
        return points
    return points  # return anyway, caller checks


def _initial_grid():
    """Create a reasonable initial configuration."""
    # Start with points on a circle and perturb
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    points = np.column_stack([0.5 + 0.4 * np.cos(angles), 0.5 + 0.4 * np.sin(angles)])
    return points


def _random_start(rng):
    """Generate a random starting configuration."""
    # Mix of circle-based and random
    if rng.random() < 0.5:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + rng.uniform(-0.3, 0.3)
        r = 0.35 + rng.uniform(-0.1, 0.15)
        points = np.column_stack([0.5 + r * np.cos(angles), 0.5 + r * np.sin(angles)])
    else:
        points = rng.uniform(0.1, 0.9, size=(N, 2))
    return points


def construct():
    start_time = time.time()
    rng = np.random.RandomState(N)
    
    best_points = _initial_grid()
    best_score = _score(best_points)
    
    # Multi-start L-BFGS-B optimization
    n_starts = 8
    for s in range(n_starts):
        if time.time() > DEADLINE - 20:
            break
        
        if s == 0:
            x0 = _initial_grid().flatten()
        else:
            x0 = _random_start(rng).flatten()
        
        # L-BFGS-B with soft-min objective
        try:
            res = minimize(
                _objective_neg,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if res.success or res.fun < 0:
                candidate = res.x.reshape(-1, 2)
                cand_score = _score(candidate)
                if cand_score > best_score:
                    best_score = cand_score
                    best_points = candidate.copy()
        except Exception:
            pass
        
        # Local polish
        if time.time() > DEADLINE - 15:
            break
        polished = _local_update_polish(best_points, max_iter=500, step_init=0.03)
        pol_score = _score(polished)
        if pol_score > best_score:
            best_score = pol_score
            best_points = polished.copy()
    
    # Final polish with more iterations
    if time.time() < DEADLINE - 10:
        final = _local_update_polish(best_points, max_iter=3000, step_init=0.01)
        final_score = _score(final)
        if final_score > best_score:
            best_points = final
            best_score = final_score
    
    # Normalize to unit area (scale so convex hull area = 1)
    hull_area = _convex_hull_area(best_points)
    if hull_area > 1e-12:
        scale = 1.0 / np.sqrt(hull_area)
        best_points = best_points * scale
    
    return best_points
