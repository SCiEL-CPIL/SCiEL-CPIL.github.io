"""Heilbronn triangles: place 11 points in unit equilateral triangle to maximize min triangle area."""
import numpy as np
import time
import itertools
from scipy.optimize import differential_evolution, minimize

N = 11
H = np.sqrt(3) / 2.0
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, H])

# Precompute all C(11,3) = 165 triangle index triplets
_idx_i, _idx_j, _idx_k = [], [], []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _idx_i.append(i)
            _idx_j.append(j)
            _idx_k.append(k)
_idx_i = np.array(_idx_i)
_idx_j = np.array(_idx_j)
_idx_k = np.array(_idx_k)


def _min_area(points):
    """Compute minimum triangle area among all C(N,3) triples. points: (N,2)."""
    p1 = points[_idx_i]
    p2 = points[_idx_j]
    p3 = points[_idx_k]
    areas = 0.5 * np.abs(np.cross(p2 - p1, p3 - p1))
    return np.min(areas)


def _is_inside(points):
    """Check if all points are inside or on the boundary of the equilateral triangle."""
    # Triangle vertices: A(0,0), B(1,0), C(0.5, H)
    # Use barycentric-style half-plane tests
    x = points[:, 0]
    y = points[:, 1]
    # Edge AB: y >= 0
    # Edge BC: line from (1,0) to (0.5, H). Normal pointing inward.
    #   Direction: (-0.5, H). Normal: (H, 0.5). Point inside if (p - B) . (H, 0.5) >= 0? 
    #   Let's check: A is inside. (A - B) = (-1, 0). (-1,0).(H,0.5) = -H < 0. So normal should be (-H, -0.5)? 
    #   Actually, let's use the standard half-plane method.
    #   For edge from P1 to P2, the interior is on one side.
    #   Edge AB: y >= 0
    #   Edge BC: line through (1,0) and (0.5, sqrt(3)/2). Equation: y - 0 = (sqrt(3)/2 - 0)/(0.5 - 1) * (x - 1) => y = -sqrt(3)(x-1). So sqrt(3)x + y - sqrt(3) <= 0.
    #   Edge CA: line through (0.5, sqrt(3)/2) and (0,0). Equation: y = sqrt(3)x. So -sqrt(3)x + y <= 0.
    cond1 = y >= -1e-9
    cond2 = np.sqrt(3) * x + y - np.sqrt(3) <= 1e-9
    cond3 = -np.sqrt(3) * x + y <= 1e-9
    return np.all(cond1) and np.all(cond2) and np.all(cond3)


def _params_to_points(params):
    """Convert 2N parameters to N points, clipping to stay inside the triangle."""
    points = params.reshape(N, 2)
    # Clip to bounding box
    points[:, 0] = np.clip(points[:, 0], 0.0, 1.0)
    points[:, 1] = np.clip(points[:, 1], 0.0, H)
    # Project points that are outside the triangle back to the boundary
    # For simplicity and robustness, we'll do a simple rejection/clipping in the objective,
    # but for DE we need valid points. Let's use a mapping that guarantees validity.
    # Actually, the simplest robust way for DE is to use the bounds and then in the objective,
    # if a point is outside, penalize heavily. But DE can get stuck.
    # Better: use a transform. But the triangle is not a box.
    # Let's stick with bounds and a penalty in the objective.
    return points


def _objective(params):
    """Objective function to minimize: -min_area. Penalize if points are outside."""
    points = _params_to_points(params)
    if not _is_inside(points):
        # Penalty for being outside
        # Compute how far outside
        x = points[:, 0]
        y = points[:, 1]
        dist1 = np.maximum(0, -y)
        dist2 = np.maximum(0, np.sqrt(3) * x + y - np.sqrt(3))
        dist3 = np.maximum(0, -np.sqrt(3) * x + y)
        total_dist = np.sum(dist1) + np.sum(dist2) + np.sum(dist3)
        return 1.0 + total_dist  # Large penalty
    return -_min_area(points)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    # Bounds for each point: x in [0, 1], y in [0, H]
    bounds = [(0.0, 1.0), (0.0, H)] * N

    best_params = None
    best_value = -np.inf

    # Phase 1: Differential Evolution
    # Use multiple restarts if time permits
    n_restarts = 5
    for r in range(n_restarts):
        if time.time() > deadline - 30.0:  # Leave 30s for refinement
            break
        try:
            result = differential_evolution(
                _objective,
                bounds,
                seed=42 + r,
                maxiter=200,
                popsize=15,
                tol=1e-6,
                polish=False,
                init='latinhypercube',
                mutation=(0.5, 1.5),
                recombination=0.7,
                update_interval=10,
                callback=None,
            )
            if not result.success or result.fun >= 0:  # If penalty was applied
                continue
            val = -result.fun
            if val > best_value:
                best_value = val
                best_params = result.x.copy()
        except Exception:
            continue

    # Phase 2: Nelder-Mead refinement from best DE result
    if best_params is not None:
        if time.time() > deadline - 10.0:
            # No time for refinement, return best DE result
            points = _params_to_points(best_params)
            return points

        try:
            result2 = minimize(
                _objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True},
            )
            if result2.fun < 0:  # Valid
                val2 = -result2.fun
                if val2 > best_value:
                    best_value = val2
                    best_params = result2.x.copy()
        except Exception:
            pass

    # Phase 3: Final high-precision polish
    if best_params is not None and time.time() < deadline - 2.0:
        try:
            result3 = minimize(
                _objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True},
            )
            if result3.fun < 0:
                val3 = -result3.fun
                if val3 > best_value:
                    best_value = val3
                    best_params = result3.x.copy()
        except Exception:
            pass

    if best_params is None:
        # Fallback: random points
        rng = np.random.RandomState(42)
        points = rng.rand(N, 2) * [1.0, H]
        # Ensure they are inside
        for _ in range(1000):
            if _is_inside(points):
                break
            points = rng.rand(N, 2) * [1.0, H]
        return points

    points = _params_to_points(best_params)
    return points
