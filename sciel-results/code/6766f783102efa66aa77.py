"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def _min_area_bary(points_bary):
    """
    Calculate the minimum area ratio of triangles formed by N points 
    given in barycentric coordinates (u, v) where w = 1 - u - v.
    The area of a triangle formed by points i, j, k is proportional to 
    the absolute value of the determinant of their barycentric coordinates.
    """
    n = len(points_bary)
    u = points_bary[:, 0]
    v = points_bary[:, 1]
    w = 1.0 - u - v
    
    min_area = float('inf')
    
    # Iterate over all unique triplets
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                det = (u[i] * (v[j] * w[k] - w[j] * v[k])
                     - v[i] * (u[j] * w[k] - w[j] * u[k])
                     + w[i] * (u[j] * v[k] - v[j] * u[k]))
                area = abs(det)
                if area < min_area:
                    min_area = area
    return min_area

def _objective(params):
    """
    Objective function for minimization.
    Minimizes negative min area + penalty for invalid points.
    """
    uv = params.reshape(N, 2)
    u = uv[:, 0]
    v = uv[:, 1]
    w = 1.0 - u - v
    
    # Penalty for points outside the simplex (u,v,w >= 0)
    penalty = 0.0
    if np.any(w < -1e-9):
        violations = -w[w < -1e-9]
        penalty = 100.0 * np.sum(violations ** 2)
    
    if penalty > 1e-6:
        return penalty
    
    min_a = _min_area_bary(uv)
    return -min_a + penalty

def _generate_grid_init():
    """Generate an initial set of points based on a grid in barycentric coords."""
    points = []
    step = 0.2
    for i in range(int(1/step) + 1):
        u = i * step
        for j in range(int((1-u)/step) + 1):
            v = j * step
            w = 1 - u - v
            if w >= -1e-9:
                points.append([u, v])
    
    selected_indices = [0, 5, 15, 2, 10, 7, 12, 1, 13, 4, 8]
    if len(points) < 16:
         selected_indices = list(range(min(11, len(points))))
         
    init = np.array([points[i] for i in selected_indices])
    return init

def construct():
    start_time = time.time()
    deadline = start_time + 135.0 # Safety margin
    
    best_score = -float('inf')
    best_points_bary = None
    
    def _update_best(uv):
        nonlocal best_score, best_points_bary
        u = uv[:, 0]
        v = uv[:, 1]
        w = 1.0 - u - v
        if np.any(u < -1e-5) or np.any(v < -1e-5) or np.any(w < -1e-5):
            return
        score = _min_area_bary(uv)
        if score > best_score:
            best_score = score
            best_points_bary = uv.copy()

    # 1. Grid-based init
    try:
        grid_init = _generate_grid_init()
        _update_best(grid_init)
    except Exception:
        pass
        
    # 2. Random inits
    rng = np.random.RandomState(42)
    for _ in range(5):
        bary = rng.dirichlet([1, 1, 1], size=N)
        uv = bary[:, :2]
        _update_best(uv)
        
    # 3. Perturbed grid init
    try:
        grid_perturbed = _generate_grid_init() + rng.uniform(-0.05, 0.05, size=(N, 2))
        grid_perturbed[:, 0] = np.clip(grid_perturbed[:, 0], 0, 1)
        grid_perturbed[:, 1] = np.clip(grid_perturbed[:, 1], 0, 1)
        sum_uv = grid_perturbed[:, 0] + grid_perturbed[:, 1]
        mask = sum_uv > 1
        if np.any(mask):
            factor = 1.0 / sum_uv[mask]
            grid_perturbed[mask, 0] *= factor
            grid_perturbed[mask, 1] *= factor
        _update_best(grid_perturbed)
    except Exception:
        pass

    # 4. Differential Evolution (Global Search)
    if time.time() < deadline - 30:
        try:
            bounds = [(0, 1)] * (2 * N)
            # Use a subset of points for DE to save time, or all if fast enough
            # 22 dimensions is okay for DE with small popsize
            popsize = 10
            maxiter = 50
            
            def de_objective(x):
                uv = x.reshape(N, 2)
                u = uv[:, 0]
                v = uv[:, 1]
                w = 1.0 - u - v
                # Check validity
                if np.any(u < 0) or np.any(v < 0) or np.any(w < 0):
                    # Return large penalty
                    return 1.0 + np.sum(np.maximum(0, -u)**2) + np.sum(np.maximum(0, -v)**2) + np.sum(np.maximum(0, -w)**2)
                
                return -_min_area_bary(uv)

            res_de = differential_evolution(
                de_objective,
                bounds,
                maxiter=maxiter,
                popsize=popsize,
                seed=42,
                tol=1e-6,
                polish=False
            )
            
            if res_de.success or res_de.fun < 0: # If it found a valid solution
                uv_de = res_de.x.reshape(N, 2)
                _update_best(uv_de)
        except Exception:
            pass

    # 5. Local Polish (L-BFGS-B) on best found
    if best_points_bary is not None and time.time() < deadline - 10:
        try:
            x0 = best_points_bary.flatten()
            bounds = [(0, 1)] * (2 * N)
            
            # Run multiple local optimizations from slightly perturbed starts
            for _ in range(3):
                if time.time() > deadline - 5:
                    break
                noise = rng.uniform(-0.01, 0.01, size=(N, 2))
                x_pert = x0 + noise.flatten()
                # Clamp to bounds
                x_pert = np.clip(x_pert, 0, 1)
                
                res = minimize(
                    _objective,
                    x_pert,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                if res.success:
                    uv_opt = res.x.reshape(N, 2)
                    _update_best(uv_opt)
        except Exception:
            pass

    # If no valid points found, use the grid init
    if best_points_bary is None:
        best_points_bary = _generate_grid_init()
        
    # Convert barycentric to Cartesian
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.5, np.sqrt(3)/2])
    
    pts = []
    for i in range(N):
        u, v = best_points_bary[i]
        w = 1 - u - v
        u = max(0, u)
        v = max(0, v)
        w = max(0, w)
        s = u + v + w
        if s > 0:
            u /= s
            v /= s
            w /= s
            
        p = u * A + v * B + w * C
        pts.append(p)
        
    return np.array(pts)

# EVOLVE-BLOCK-END
