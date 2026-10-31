"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import random

DEADLINE = 140.0  # safety margin before 150s kill

def _check_validity(circles):
    """Check if circles are valid: within rectangle of perimeter <= 4, disjoint interiors."""
    n = len(circles)
    if n != N:
        return False, 0.0
    
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    
    # All radii must be positive
    if np.any(rs <= 0):
        return False, 0.0
    
    # Determine bounding rectangle
    min_x = np.min(xs - rs)
    max_x = np.max(xs + rs)
    min_y = np.min(ys - rs)
    max_y = np.max(ys + rs)
    
    w = max_x - min_x
    h = max_y - min_y
    
    if w <= 0 or h <= 0:
        return False, 0.0
    
    # Perimeter constraint: 2*(w + h) <= 4
    perim = 2.0 * (w + h)
    if perim > 4.0 + 1e-9:
        return False, 0.0
    
    # Check disjoint interiors: distance between centers >= sum of radii
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < rs[i] + rs[j] - 1e-9:
                return False, 0.0
    
    # Compute objective: sum of radii
    obj = np.sum(rs)
    return True, obj


def _penalty_objective(params):
    """Objective with penalties for constraint violations. Minimize negative sum of radii."""
    n = N
    xs = params[0:n]
    ys = params[n:2*n]
    rs = params[2*n:3*n]
    
    # Penalize negative radii
    if np.any(rs <= 0):
        return 1e6
    
    # Bounding box
    min_x = np.min(xs - rs)
    max_x = np.max(xs + rs)
    min_y = np.min(ys - rs)
    max_y = np.max(ys + rs)
    
    w = max_x - min_x
    h = max_y - min_y
    
    if w <= 0 or h <= 0:
        return 1e6
    
    perim = 2.0 * (w + h)
    perim_penalty = max(0, perim - 4.0) * 100.0
    
    # Disjointness penalty
    overlap_penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = rs[i] + rs[j]
            if dist < min_dist:
                overlap_penalty += (min_dist - dist) ** 2
    
    # Negative sum of radii (we minimize)
    obj = -np.sum(rs) + perim_penalty + overlap_penalty * 10.0
    return obj


def _grid_initialization():
    """Create initial grid-like configuration."""
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) < N:
                cells.append([0.5, 0.5, 0.1])
    
    # Spread them out
    for k in range(N):
        i = k % side
        j = k // side
        cells[k][0] = (i + 0.5) / side
        cells[k][1] = (j + 0.5) / side
        cells[k][2] = 0.5 / side - 0.01
    
    return np.array(cells)


def _random_initialization():
    """Create random initial configuration."""
    circles = np.zeros((N, 3))
    # Place in a square of side 1 (perimeter 4)
    for i in range(N):
        circles[i, 0] = random.uniform(0.1, 0.9)
        circles[i, 1] = random.uniform(0.1, 0.9)
        circles[i, 2] = random.uniform(0.02, 0.15)
    return circles


def _hill_climb(circles, max_iter=200, step_size=0.01):
    """Simple hill climbing to improve a configuration."""
    best_circles = circles.copy()
    best_obj, _ = _check_validity(best_circles)
    if not best_obj:
        best_obj = 0.0
    
    for _ in range(max_iter):
        improved = False
        for i in range(N):
            for coord in range(3):
                for direction in [-1, 1]:
                    new_circles = best_circles.copy()
                    delta = direction * step_size
                    if coord == 2:  # radius
                        delta = direction * step_size * 0.5
                    new_circles[i, coord] += delta
                    if coord == 2 and new_circles[i, coord] <= 0:
                        continue
                    valid, obj = _check_validity(new_circles)
                    if valid and obj > best_obj + 1e-10:
                        best_circles = new_circles
                        best_obj = obj
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    
    return best_circles, best_obj


def _slsqp_optimize(circles, time_limit=10.0):
    """Use SLSQP to optimize a configuration."""
    start_time = time.time()
    params = circles.flatten().copy()
    
    result = minimize(_penalty_objective, params, method='L-BFGS-B',
                      options={'maxiter': 50, 'ftol': 1e-8})
    
    if result.success or result.fun < _penalty_objective(params):
        new_params = result.x
        new_circles = new_params.reshape(N, 3)
        # Ensure radii are positive
        new_circles[:, 2] = np.maximum(new_circles[:, 2], 0.001)
        return new_circles
    
    return circles


def construct():
    """Construct 21 circles in a perimeter-4 rectangle maximizing sum of radii."""
    start_time = time.time()
    
    best_circles = None
    best_score = 0.0
    
    # Try multiple strategies
    strategies = [
        ("grid", _grid_initialization),
        ("random1", lambda: _random_initialization()),
        ("random2", lambda: _random_initialization()),
        ("random3", lambda: _random_initialization()),
    ]
    
    for name, init_func in strategies:
        if time.time() - start_time > DEADLINE:
            break
        
        circles = init_func()
        
        # Hill climb first
        circles, score = _hill_climb(circles, max_iter=100, step_size=0.01)
        if score > best_score:
            best_score = score
            best_circles = circles.copy()
        
        # Then try SLSQP if time permits
        if time.time() - start_time < DEADLINE - 15:
            try:
                optimized = _slsqp_optimize(circles, time_limit=10.0)
                valid, obj = _check_validity(optimized)
                if valid and obj > best_score:
                    best_score = obj
                    best_circles = optimized.copy()
            except Exception:
                pass
    
    # Final hill climb on best
    if best_circles is not None and time.time() - start_time < DEADLINE - 5:
        best_circles, best_score = _hill_climb(best_circles, max_iter=150, step_size=0.005)
    
    if best_circles is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    cells.append([(i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 0.01])
        best_circles = np.array(cells)
    
    # Scale to fit perimeter constraint exactly if possible
    # Normalize: compute bounding box and scale
    xs = best_circles[:, 0]
    ys = best_circles[:, 1]
    rs = best_circles[:, 2]
    
    min_x = np.min(xs - rs)
    max_x = np.max(xs + rs)
    min_y = np.min(ys - rs)
    max_y = np.max(ys + rs)
    
    w = max_x - min_x
    h = max_y - min_y
    perim = 2.0 * (w + h)
    
    if perim > 4.0:
        scale = 4.0 / perim
        # Scale positions and radii
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        best_circles[:, 0] = center_x + (best_circles[:, 0] - center_x) * scale
        best_circles[:, 1] = center_y + (best_circles[:, 1] - center_y) * scale
        best_circles[:, 2] *= scale
    
    # Final validity check
    valid, final_score = _check_validity(best_circles)
    if not valid:
        # Return a known-valid grid configuration
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    cells.append([(i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 0.01])
        best_circles = np.array(cells)
    
    return best_circles
# EVOLVE-BLOCK-END
