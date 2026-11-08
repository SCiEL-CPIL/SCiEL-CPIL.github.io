"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution

DEADLINE = 135.0  # Safety margin before 150s kill


def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0


def _min_triangle_area_vec(points):
    """Vectorized minimum triangle area for N points."""
    n = len(points)
    min_area = float('inf')
    x = points[:, 0]
    y = points[:, 1]
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = (x[j] - x[i]) * dy_ki - dx_ki * (y[j] - y[i])
            local_min = 0.5 * np.min(np.abs(cross))
            if local_min < min_area:
                min_area = local_min
                if min_area < 1e-14:
                    return min_area
    return min_area


def _score(pts):
    """Compute min triangle area / hull area."""
    hull_a = _hull_area(pts)
    if hull_a <= 0:
        return 0.0
    min_area = _min_triangle_area_vec(pts)
    return min_area / hull_a


def _de_objective(x_flat):
    """Objective for differential evolution: negative score."""
    pts = x_flat.reshape(N, 2)
    # Ensure points are within bounds
    if np.any(pts < -2.0) or np.any(pts > 2.0):
        return 1e10
    s = _score(pts)
    return -s


def _initial_regular_polygon(n):
    """Points on a regular n-gon inscribed in unit circle, scaled to unit area."""
    angles = 2 * np.pi * np.arange(n) / n
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    return _normalize_unit_area(pts)


def _normalize_unit_area(pts):
    """Scale points so convex hull area is 1."""
    hull_a = _hull_area(pts)
    if hull_a > 0:
        return pts / np.sqrt(hull_a)
    return pts


def construct():
    start_time = time.time()
    
    # Define bounds for differential evolution
    # We'll search in a square [-2, 2] x [-2, 2]
    bounds = [(-2.0, 2.0)] * (2 * N)
    
    best_pts = None
    best_score = 0.0
    
    # Initial best from regular polygon
    init_pts = _initial_regular_polygon(N)
    best_pts = init_pts.copy()
    best_score = _score(best_pts)
    
    # Run differential evolution with discrete-like search
    # We use a coarse grid by setting popsize and using integer-like steps
    # DE works on continuous space, but we can use a small mutation step
    # to simulate discrete search
    
    try:
        # Use differential evolution with a relatively small population
        # and limited generations to fit in time budget
        result = differential_evolution(
            _de_objective,
            bounds,
            maxiter=50,
            popsize=15,
            mutation=(0.3, 0.8),
            recombination=0.7,
            tol=1e-8,
            seed=42,
            polish=False,  # Disable polish to save time
            update_interval=1,
            callback=_de_callback
        )
        
        if result.x is not None:
            cand = result.x.reshape(N, 2)
            s = _score(cand)
            if s > best_score:
                best_score = s
                best_pts = cand
    except Exception:
        pass
    
    # If time permits, do a local polish on the best point found
    if best_pts is not None and time.time() - start_time < DEADLINE - 20:
        try:
            # Simple local search: try small perturbations
            current_best = best_pts.copy()
            current_score = best_score
            step_size = 0.05
            rng = np.random.RandomState(123)
            
            for _ in range(1000):
                if time.time() - start_time > DEADLINE - 10:
                    break
                # Random perturbation
                perturbed = current_best + rng.normal(0, step_size, size=current_best.shape)
                s = _score(perturbed)
                if s > current_score:
                    current_score = s
                    current_best = perturbed
                    step_size *= 0.9  # Reduce step size when improving
                else:
                    step_size *= 0.99  # Slowly reduce step size
                    if step_size < 1e-4:
                        step_size = 1e-4
            
            if current_score > best_score:
                best_score = current_score
                best_pts = current_best
        except Exception:
            pass
    
    # Final normalization and return
    if best_pts is None:
        best_pts = _initial_regular_polygon(N)
    
    return best_pts


def _de_callback(xk, convergence=True):
    """Callback for differential evolution to check time limit."""
    global _de_stop_flag
    if time.time() - _de_start_time > DEADLINE - 15:
        _de_stop_flag = True
        return True
    return _de_stop_flag


# Global variables for DE callback
_de_start_time = 0
_de_stop_flag = False

# Override construct to properly initialize time tracking
def construct():
    global _de_start_time, _de_stop_flag
    _de_start_time = time.time()
    _de_stop_flag = False
    
    start_time = time.time()
    
    # Define bounds for differential evolution
    bounds = [(-2.0, 2.0)] * (2 * N)
    
    best_pts = None
    best_score = 0.0
    
    # Initial best from regular polygon
    init_pts = _initial_regular_polygon(N)
    best_pts = init_pts.copy()
    best_score = _score(best_pts)
    
    try:
        result = differential_evolution(
            _de_objective,
            bounds,
            maxiter=50,
            popsize=15,
            mutation=(0.3, 0.8),
            recombination=0.7,
            tol=1e-8,
            seed=42,
            polish=False,
            update_interval=1,
            callback=_de_callback
        )
        
        if result.x is not None:
            cand = result.x.reshape(N, 2)
            s = _score(cand)
            if s > best_score:
                best_score = s
                best_pts = cand
    except Exception:
        pass
    
    if best_pts is not None and time.time() - start_time < DEADLINE - 20:
        try:
            current_best = best_pts.copy()
            current_score = best_score
            step_size = 0.05
            rng = np.random.RandomState(123)
            
            for _ in range(1000):
                if time.time() - start_time > DEADLINE - 10:
                    break
                perturbed = current_best + rng.normal(0, step_size, size=current_best.shape)
                s = _score(perturbed)
                if s > current_score:
                    current_score = s
                    current_best = perturbed
                    step_size *= 0.9
                else:
                    step_size *= 0.99
                    if step_size < 1e-4:
                        step_size = 1e-4
            
            if current_score > best_score:
                best_score = current_score
                best_pts = current_best
        except Exception:
            pass
    
    if best_pts is None:
        best_pts = _initial_regular_polygon(N)
    
    return best_pts
# EVOLVE-BLOCK-END
