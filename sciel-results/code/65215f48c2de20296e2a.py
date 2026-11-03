"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = np.sqrt(3) / 4.0  # area of the unit-base equilateral triangle

# Precompute indices for all triples to speed up area calculation
_IDX_TRIPLES = None
def _get_indices():
    global _IDX_TRIPLES
    if _IDX_TRIPLES is None:
        n = N
        _IDX_TRIPLES = np.array([(i, j, k) for i in range(n) for j in range(i+1, n) for k in range(j+1, n)])
    return _IDX_TRIPLES

def _min_triangle_area_vectorized(points):
    """Compute the smallest area among all triangles formed by triples of points, vectorized."""
    idx = _get_indices()
    pi = points[idx[:, 0]]
    pj = points[idx[:, 1]]
    pk = points[idx[:, 2]]
    # Area = 0.5 * |cross(pj-pi, pk-pi)|
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _bary_to_cart(bary):
    """Convert barycentric coords (u, v) with u>=0, v>=0, u+v<=1 to cartesian."""
    # Point = u*V1 + v*V2 (since V0 is origin)
    x = bary[:, 0] * V1[0] + bary[:, 1] * V2[0]
    y = bary[:, 0] * V1[1] + bary[:, 1] * V2[1]
    return np.column_stack([x, y])

def _cart_to_bary(points):
    """Convert cartesian points to barycentric (u, v) where point = u*V1 + v*V2."""
    u = points[:, 0] - 0.5 * points[:, 1]
    v = 2.0 * points[:, 1] / np.sqrt(3)
    return np.column_stack([u, v])

def _clamp_bary(bary):
    """Clamp barycentric coordinates to the valid region."""
    bary = np.clip(bary, 0.0, None)
    s = bary[:, 0] + bary[:, 1]
    mask = s > 1.0
    if np.any(mask):
        bary[mask] = bary[mask] / s[mask, None]
    return bary

def _objective_bary(x):
    """Objective: negative min triangle area, with points in barycentric coords."""
    bary = x.reshape(-1, 2)
    bary = _clamp_bary(bary)
    points = _bary_to_cart(bary)
    return -_min_triangle_area_vectorized(points)

def _make_initial_bary(seed):
    """Generate initial barycentric coordinates using a heuristic structure."""
    rng = np.random.RandomState(seed)
    # Start with vertices and midpoints for a good base structure
    bary_list = [
        [0, 0],      # V0
        [1, 0],      # V1
        [0, 1],      # V2
        [0.5, 0],    # midpoint V0-V1
        [0.5, 0.5],  # midpoint V1-V2
        [0, 0.5],    # midpoint V2-V0
        [1/3, 1/3],  # Centroid
    ]
    # Add remaining points randomly but biased towards interior
    while len(bary_list) < N:
        u = rng.uniform(0.1, 0.9)
        v = rng.uniform(0.1, 0.9)
        if u + v < 0.8:
            bary_list.append([u, v])
        else:
            # Retry if too close to edge
            continue
    return np.array(bary_list[:N])

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    best_points = None
    best_score = -np.inf
    best_bary = None

    def time_left():
        return deadline - time.time()

    # Multi-restart Nelder-Mead optimization
    rng = np.random.RandomState(42)
    
    # Initial guess
    bary_init = _make_initial_bary(0)
    points_init = _bary_to_cart(bary_init)
    best_score = _min_triangle_area_vectorized(points_init)
    best_points = points_init.copy()
    best_bary = bary_init.copy()

    # Run many restarts
    # Nelder-Mead is fast and effective for this small dimension
    n_restarts = 200  # We will break early based on time
    for i in range(n_restarts):
        if time_left() < 1.0:
            break
            
        # Generate a new starting point
        if i == 0:
            x0 = best_bary.flatten()
        else:
            # Perturb the best solution found so far
            # Scale perturbation by current best score to allow finer tuning later
            scale = 0.1 / (1.0 + best_score) 
            bary_pert = best_bary + rng.normal(0, scale, size=best_bary.shape)
            bary_pert = _clamp_bary(bary_pert)
            x0 = bary_pert.flatten()
            
        try:
            res = minimize(
                _objective_bary, x0, method='Nelder-Mead',
                options={'maxiter': 1000, 'fatol': 1e-10, 'xatol': 1e-10}
            )
            bary_opt = res.x.reshape(-1, 2)
            bary_opt = _clamp_bary(bary_opt)
            points = _bary_to_cart(bary_opt)
            score = _min_triangle_area_vectorized(points)
            
            if score > best_score:
                best_score = score
                best_points = points.copy()
                best_bary = bary_opt.copy()
                
        except Exception:
            pass

    # If we have time left, do a final polish with a tighter tolerance
    if time_left() > 5.0 and best_bary is not None:
        try:
            x0 = best_bary.flatten()
            res = minimize(
                _objective_bary, x0, method='Nelder-Mead',
                options={'maxiter': 5000, 'fatol': 1e-12, 'xatol': 1e-12}
            )
            bary_opt = res.x.reshape(-1, 2)
            bary_opt = _clamp_bary(bary_opt)
            points = _bary_to_cart(bary_opt)
            score = _min_triangle_area_vectorized(points)
            
            if score > best_score:
                best_score = score
                best_points = points.copy()
                best_bary = bary_opt.copy()
        except Exception:
            pass

    if best_points is None:
        bary0 = _make_initial_bary(0)
        best_points = _bary_to_cart(bary0)

    return np.array(best_points)
# EVOLVE-BLOCK-END
