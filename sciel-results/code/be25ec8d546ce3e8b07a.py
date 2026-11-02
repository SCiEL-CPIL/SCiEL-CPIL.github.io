"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull


def _min_triangle_area(points):
    """Compute the minimum area over all triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
                if area < min_area:
                    min_area = area
    return min_area


def _convex_hull_area(points):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0


def _score(points):
    """Compute the score: min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area <= 1e-12:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area


def _objective(params):
    """Objective function to minimize (negative of score)."""
    points = params.reshape(N, 2)
    s = _score(points)
    return -s


def _objective_soft(params, alpha=100.0):
    """Soft-min objective for smoother gradients."""
    points = params.reshape(N, 2)
    n = len(points)
    # Compute all triangle areas
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
                areas.append(area)
    areas = np.array(areas)
    # Soft minimum
    soft_min = np.log(np.sum(np.exp(-alpha * areas))) / (-alpha)
    hull_area = _convex_hull_area(points)
    if hull_area <= 1e-12:
        return 1e6
    return soft_min / hull_area


def _make_initial_points(seed):
    """Generate diverse initial points."""
    rng = np.random.RandomState(seed)
    method = seed % 5
    
    if method == 0:
        # Regular polygon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        points = np.column_stack([np.cos(angles), np.sin(angles)])
    elif method == 1:
        # Concentric rings
        points = np.zeros((N, 2))
        for i in range(N):
            r = 1.0 if i < N // 2 else 0.5
            theta = 2 * np.pi * i / N
            points[i] = [r * np.cos(theta), r * np.sin(theta)]
    elif method == 2:
        # Hexagonal-like lattice
        points = []
        for i in range(N):
            x = (i % 4) - 1.5
            y = (i // 4) - 1.5
            points.append([x, y])
        points = np.array(points)
        points = points - points.mean(axis=0)
    elif method == 3:
        # Random on circle
        angles = np.sort(rng.uniform(0, 2 * np.pi, N))
        points = np.column_stack([np.cos(angles), np.sin(angles)])
    else:
        # Random
        points = rng.uniform(-1, 1, (N, 2))
    
    # Add small random perturbation
    points += rng.uniform(-0.01, 0.01, (N, 2))
    return points


def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin
    
    best_points = None
    best_score = 0.0
    
    # Try multiple initializations and optimization methods
    for seed in range(20):
        if time.time() > deadline - 10:
            break
        
        points = _make_initial_points(seed)
        
        # Evaluate initial score
        s = _score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()
        
        # Try Nelder-Mead
        try:
            result = minimize(
                _objective,
                points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if time.time() > deadline - 5:
                break
            candidate_points = result.x.reshape(N, 2)
            candidate_score = _score(candidate_points)
            if candidate_score > best_score:
                best_score = candidate_score
                best_points = candidate_points.copy()
        except Exception:
            pass
        
        # Try L-BFGS-B with soft objective
        try:
            result2 = minimize(
                _objective_soft,
                points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10}
            )
            if time.time() > deadline - 5:
                break
            candidate_points2 = result2.x.reshape(N, 2)
            candidate_score2 = _score(candidate_points2)
            if candidate_score2 > best_score:
                best_score = candidate_score2
                best_points = candidate_points2.copy()
        except Exception:
            pass
    
    # If we have time, try differential evolution
    if time.time() < deadline - 30 and best_points is not None:
        try:
            bounds = [(-1, 1)] * (2 * N)
            de_result = differential_evolution(
                _objective,
                bounds,
                maxiter=50,
                popsize=15,
                seed=42,
                tol=1e-8,
                init='latinhypercube'
            )
            if time.time() < deadline - 5:
                candidate_points3 = de_result.x.reshape(N, 2)
                candidate_score3 = _score(candidate_points3)
                if candidate_score3 > best_score:
                    best_score = candidate_score3
                    best_points = candidate_points3.copy()
        except Exception:
            pass
    
    if best_points is None:
        # Fallback: regular polygon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_points
# EVOLVE-BLOCK-END

