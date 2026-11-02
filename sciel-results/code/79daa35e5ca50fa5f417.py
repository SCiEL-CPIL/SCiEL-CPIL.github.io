# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from itertools import combinations

N = 11
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2.0])
AREA_EQ = np.sqrt(3) / 4.0

TRIPLES = list(combinations(range(N), 3))
N_TRIPLES = len(TRIPLES)
T_I = np.array([t[0] for t in TRIPLES])
T_J = np.array([t[1] for t in TRIPLES])
T_K = np.array([t[2] for t in TRIPLES])

def _barycentric_to_cartesian(params):
    """
    params: shape (N, 2) array of (a, b) for each point.
    Converts to Cartesian coordinates using softmax-like mapping to simplex.
    w0 = 1, w1 = exp(a), w2 = exp(b)
    u = w1 / (w0+w1+w2), v = w2 / (w0+w1+w2)
    P = u * V1 + v * V2
    """
    a = params[:, 0]
    b = params[:, 1]
    w1 = np.exp(a)
    w2 = np.exp(b)
    denom = 1.0 + w1 + w2
    u = w1 / denom
    v = w2 / denom
    # P = u * V1 + v * V2
    # V1 = (1, 0), V2 = (0.5, sqrt(3)/2)
    x = u * 1.0 + v * 0.5
    y = u * 0.0 + v * (np.sqrt(3) / 2.0)
    return np.column_stack((x, y))

def _min_area_cart(pts):
    """
    pts: shape (N, 2)
    Returns the minimum area of any triangle formed by 3 points.
    """
    # Vectorized area calculation
    # Area = 0.5 * | (x_j - x_i)(y_k - y_i) - (y_j - y_i)(x_k - x_i) |
    # We can compute this for all triples at once using broadcasting or precomputed indices.
    pi = pts[T_I]
    pj = pts[T_J]
    pk = pts[T_K]
    
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _objective(params):
    """
    Objective function to minimize: -min_area
    """
    pts = _barycentric_to_cartesian(params)
    min_area = _min_area_cart(pts)
    return -min_area

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Bounds for a and b
    # exp(10) is ~22000, exp(-10) is ~0.000045.
    # This gives a good dynamic range for barycentric coordinates.
    bounds = [(-10.0, 10.0)] * (2 * N)
    
    best_score = -1.0
    best_params = None
    
    # Run Differential Evolution
    # maxiter controls the number of generations
    # popsize is multiplier for population size
    try:
        result = differential_evolution(
            _objective,
            bounds=bounds,
            maxiter=100,
            popsize=15,
            tol=1e-5,
            seed=42,
            polish=False # We will polish manually to handle non-smoothness
        )
        
        if result.success:
            best_params = result.x
            best_score = -result.fun
        else:
            # Even if not successful, result.x is the best found
            best_params = result.x
            best_score = -result.fun
            
    except Exception as e:
        # Fallback if DE fails
        pass
    
    # Check time
    if time.time() > deadline - 5.0:
        if best_params is not None:
            return _barycentric_to_cartesian(best_params)
        # Fallback to random if no result
        return _random_fallback()

    # Polish the result using Nelder-Mead on the barycentric parameters
    # Nelder-Mead does not require gradients and handles the non-smooth min well.
    if best_params is not None:
        try:
            res_polish = minimize(
                _objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if -res_polish.fun > best_score:
                best_params = res_polish.x
                best_score = -res_polish.fun
        except Exception:
            pass
    
    # Final check
    if time.time() > deadline:
        pass # Return what we have

    if best_params is None:
        return _random_fallback()
        
    return _barycentric_to_cartesian(best_params)

def _random_fallback():
    """Generate random points in triangle as a last resort."""
    rng = np.random.RandomState(42)
    pts = np.zeros((N, 2))
    for i in range(N):
        u = rng.random()
        v = rng.random()
        while u + v > 1.0:
            u = rng.random()
            v = rng.random()
        pts[i] = u * V1 + v * V2
    return pts
# EVOLVE-BLOCK-END
