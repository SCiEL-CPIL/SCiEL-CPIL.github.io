"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize


def _min_triangle_area(points):
    """Calculate the minimum area of any triangle formed by the points using vectorized ops."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            dx12 = x2 - x1
            dy12 = y2 - y1
            # Vectorize over k
            rest = points[j+1:]
            cross = dx12 * (rest[:, 1] - y1) - dy12 * (rest[:, 0] - x1)
            areas = 0.5 * np.abs(cross)
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
                if min_area < 1e-12:
                    break
        if min_area < 1e-12:
            break
    return min_area


def _objective_flat(flat_params):
    """Objective function: minimize negative min area."""
    n = len(flat_params) // 2
    u = flat_params[0::2]
    v = flat_params[1::2]
    
    # Check bounds - v in [0,1], u in [0,1], and u+v <= 1 (barycentric)
    if np.any(v < -0.01) or np.any(v > 1.01) or np.any(u < -0.01) or np.any(u > 1.01):
        return -1e-10
    if np.any(u + v > 1.01):
        return -1e-10
    
    # Convert to Cartesian
    x = u + 0.5 * v
    y = (np.sqrt(3) / 2) * v
    points = np.column_stack([x, y])
    
    min_area = _min_triangle_area(points)
    return -min_area


def _refine_with_l_bfgs(flat_params, deadline):
    """Refine solution using L-BFGS-B for local optimization."""
    def obj(flat):
        return _objective_flat(flat)
    
    def grad(flat):
        # Numerical gradient
        eps = 1e-7
        g = np.zeros_like(flat)
        for i in range(len(flat)):
            fp = flat.copy()
            fm = flat.copy()
            fp[i] += eps
            fm[i] -= eps
            g[i] = (obj(fp) - obj(fm)) / (2 * eps)
        return g
    
    bounds = [(0, 1)] * len(flat_params)
    
    result = minimize(
        obj,
        flat_params,
        method='L-BFGS-B',
        jac=grad,
        bounds=bounds,
        options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
    )
    
    if result.fun < _objective_flat(flat_params):
        return result.x
    return flat_params


def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    # Initial good starting point: distribute points in barycentric coords
    # Use a structured grid-like initial distribution
    barycentric_init = [
        [0.05, 0.05, 0.90],   # Near vertex A
        [0.05, 0.90, 0.05],   # Near vertex B  
        [0.90, 0.05, 0.05],   # Near vertex C
        [0.10, 0.40, 0.50],   # Edge AB
        [0.40, 0.10, 0.50],   # Edge AC
        [0.50, 0.40, 0.10],   # Edge BC
        [0.333, 0.333, 0.334], # Center
        [0.20, 0.20, 0.60],
        [0.60, 0.20, 0.20],
        [0.20, 0.60, 0.20],
        [0.40, 0.30, 0.30],
    ]
    
    # Convert barycentric (a,b,c) to (u,v) where x = u + 0.5*v, y = sqrt(3)/2 * v
    # In this triangle: vertex A=(0,0) has barycentric (1,0,0)
    # vertex B=(1,0) has barycentric (0,1,0) -> u=1, v=0
    # vertex C=(0.5, sqrt(3)/2) has barycentric (0,0,1) -> u=0, v=1
    # So: x = a*0 + b*1 + c*0.5 = b + 0.5*c
    #     y = a*0 + b*0 + c*sqrt(3)/2 = c*sqrt(3)/2
    # Therefore: v = c, u = b
    initial_flat = []
    for bary in barycentric_init:
        a, b, c = bary
        u = b
        v = c
        initial_flat.extend([u, v])
    
    initial_flat = np.array(initial_flat)
    
    # Phase 1: Differential evolution for global search
    bounds = [(0, 1)] * (2 * N)
    
    best_flat = initial_flat.copy()
    best_score = -_objective_flat(initial_flat)
    
    # Run differential evolution
    result = differential_evolution(
        _objective_flat,
        bounds,
        x0=initial_flat,
        maxiter=100,
        popsize=15,
        tol=1e-6,
        mutation=(0.3, 1.0),
        recombination=0.8,
        polish=True,
        seed=42,
        callback=lambda x, convergence: None
    )
    
    if result.fun < _objective_flat(best_flat):
        best_flat = result.x
        best_score = -result.fun
    
    # Check time
    if time.time() > deadline - 30:
        # Not enough time for refinement, return best so far
        points = _flat_to_points(best_flat)
        return points
    
    # Phase 2: Local refinement with L-BFGS-B
    refined_flat = _refine_with_l_bfgs(best_flat, deadline)
    if -_objective_flat(refined_flat) > best_score:
        best_flat = refined_flat
        best_score = -_objective_flat(refined_flat)
    
    # Phase 3: Try a few more DE runs with different seeds if time permits
    if time.time() < deadline - 40:
        for seed in [1, 2]:
            if time.time() > deadline - 30:
                break
            try:
                result2 = differential_evolution(
                    _objective_flat,
                    bounds,
                    x0=initial_flat,
                    maxiter=50,
                    popsize=10,
                    tol=1e-6,
                    mutation=(0.4, 1.2),
                    recombination=0.7,
                    polish=True,
                    seed=seed,
                    callback=lambda x, convergence: None
                )
                if result2.fun < _objective_flat(best_flat):
                    best_flat = result2.x
                    best_score = -result2.fun
                    
                    # Quick refinement
                    refined = _refine_with_l_bfgs(best_flat, deadline)
                    if -_objective_flat(refined) > best_score:
                        best_flat = refined
                        best_score = -_objective_flat(refined)
            except:
                pass
    
    points = _flat_to_points(best_flat)
    return points


def _flat_to_points(flat_params):
    """Convert flat parameters to point array."""
    n = len(flat_params) // 2
    u = flat_params[0::2]
    v = flat_params[1::2]
    x = u + 0.5 * v
    y = (np.sqrt(3) / 2) * v
    return np.column_stack([x, y])
# EVOLVE-BLOCK-END
