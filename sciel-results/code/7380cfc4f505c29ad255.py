"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = np.sqrt(3) / 4.0
H = np.sqrt(3) / 2.0

def _bary_to_cart(bary):
    """Vectorized barycentric to Cartesian."""
    return bary[:, 0:1] * V1 + bary[:, 1:2] * V2

def _clamp_bary(bary):
    """Clamp barycentric coords to valid region."""
    b = np.clip(bary, 0, 1)
    s = b[:, 0] + b[:, 1]
    mask = s > 1
    if np.any(mask):
        b[mask] /= s[mask, np.newaxis]
    return b

def _min_triangle_area_vectorized(points):
    """Compute min triangle area among all triples, vectorized."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                ax, ay = points[i]
                bx, by = points[j]
                cx, cy = points[k]
                area = 0.5 * abs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay))
                if area < min_area:
                    min_area = area
    return min_area

def _objective_de(x_flat):
    """Objective for Differential Evolution. Returns negative min area."""
    bary = x_flat.reshape(-1, 2)
    # Check validity: u>=0, v>=0, u+v<=1
    if np.any(bary < -1e-9) or np.any(bary > 1 + 1e-9):
        return 1e6
    if np.any(bary[:, 0] + bary[:, 1] > 1 + 1e-9):
        return 1e6
    
    points = _bary_to_cart(bary)
    min_area = _min_triangle_area_vectorized(points)
    return -min_area

def _nelder_mead_refine(x0, maxiter=2000):
    """Refine a solution using Nelder-Mead."""
    def obj(x):
        bary = x.reshape(-1, 2)
        # Soft penalty for boundary violation to help NM stay in bounds
        penalty = 0.0
        if np.any(bary < 0):
            penalty += 100 * np.sum(bary[bary < 0] ** 2)
        if np.any(bary > 1):
            penalty += 100 * np.sum((bary[bary > 1] - 1) ** 2)
        s = bary[:, 0] + bary[:, 1]
        if np.any(s > 1):
            penalty += 100 * np.sum((s[s > 1] - 1) ** 2)
        
        bary_clamped = _clamp_bary(bary)
        points = _bary_to_cart(bary_clamped)
        min_area = _min_triangle_area_vectorized(points)
        return -min_area + penalty

    try:
        result = minimize(
            obj, x0, method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
        )
        return result.x
    except Exception:
        return x0

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_points = None
    best_score = -float('inf')
    
    def update_best(points):
        nonlocal best_points, best_score
        score = _min_triangle_area_vectorized(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()

    # Bounds for barycentric coordinates: u in [0, 1], v in [0, 1]
    bounds = [(0, 1), (0, 1)] * 11
    
    try:
        # Run Differential Evolution
        de_result = differential_evolution(
            _objective_de, bounds,
            maxiter=100, 
            popsize=20, 
            seed=42,
            polish=False,
            tol=1e-8,
            init='latinhypercube'
        )
        
        if time.time() < deadline - 10.0:
            x_best_de = de_result.x
            # Refine with Nelder-Mead
            x_refined = _nelder_mead_refine(x_best_de, maxiter=3000)
            
            bary_opt = _clamp_bary(x_refined.reshape(-1, 2))
            points = _bary_to_cart(bary_opt)
            update_best(points)
            
            # If time allows, try another DE run with different seed
            if time.time() < deadline - 15.0:
                 de_result2 = differential_evolution(
                    _objective_de, bounds,
                    maxiter=100, 
                    popsize=20, 
                    seed=123,
                    polish=False,
                    tol=1e-8,
                    init='latinhypercube'
                 )
                 x_best_de2 = de_result2.x
                 x_refined2 = _nelder_mead_refine(x_best_de2, maxiter=3000)
                 bary_opt2 = _clamp_bary(x_refined2.reshape(-1, 2))
                 points2 = _bary_to_cart(bary_opt2)
                 update_best(points2)
                 
    except Exception as e:
        print(f"DE failed: {e}")
        
    # Fallback if DE didn't run or failed
    if best_points is None or best_score < 1e-5:
        # Try a simple grid + NM approach as backup
        for seed in [42, 123]:
            rng = np.random.RandomState(seed)
            bary_init = rng.uniform(0, 1, size=(11, 2))
            bary_init = _clamp_bary(bary_init)
            x0 = bary_init.flatten()
            x_nm = _nelder_mead_refine(x0, maxiter=1000)
            bary_opt = _clamp_bary(x_nm.reshape(-1, 2))
            points = _bary_to_cart(bary_opt)
            update_best(points)

    if best_points is None:
        # Absolute fallback: vertices + centroid + random
        points = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, np.sqrt(3)/2],
            [0.5, np.sqrt(3)/6], # Centroid
        ])
        # Add 7 random points
        rng = np.random.RandomState(0)
        for _ in range(7):
            u = rng.uniform(0, 1)
            v = rng.uniform(0, 1-u)
            p = _bary_to_cart(np.array([[u, v]]))[0]
            points = np.vstack([points, p])
        best_points = points

    return best_points
# EVOLVE-BLOCK-END
