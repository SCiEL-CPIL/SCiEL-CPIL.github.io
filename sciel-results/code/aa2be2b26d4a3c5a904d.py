"""Heilbronn triangles: maximize min triangle area among 11 points in unit equilateral triangle."""
import numpy as np
import time
from scipy.optimize import minimize

N = 11
# Equilateral triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])

def _barycentric_to_cart(bary):
    """Convert barycentric coords (a,b,c) with a+b+c=1 to cartesian."""
    a, b, c = bary[:, 0], bary[:, 1], bary[:, 2]
    x = a * V0[0] + b * V1[0] + c * V2[0]
    y = a * V0[1] + b * V1[1] + c * V2[1]
    return np.column_stack([x, y])

def _triangle_areas(pts):
    """Compute areas of all triangles formed by triples of points.
    Returns array of shape (n, n, n) with area[i,j,k] for i<j<k stored appropriately.
    Actually returns a flat array of all C(n,3) areas."""
    n = len(pts)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Area = 0.5 * |cross(pj-pi, pk-pi)|
                v1 = pts[j] - pts[i]
                v2 = pts[k] - pts[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                areas.append(area)
    return np.array(areas)

def _smooth_min_area(pts, temperature=0.01):
    """Smooth approximation of the minimum triangle area using log-sum-exp."""
    areas = _triangle_areas(pts)
    if len(areas) == 0:
        return 0.0
    # Use log-sum-exp to approximate the min
    min_a = np.min(areas)
    # Smoothed min: -temperature * log(sum(exp(-area/temperature)))
    # This is stable when temperature is small
    shifted = -areas / temperature
    shifted -= np.max(shifted)
    log_sum_exp = np.log(np.sum(np.exp(shifted))) + np.max(shifted)
    smooth_min = -temperature * log_sum_exp
    return smooth_min

def _objective(bary_flat, temperature=0.01):
    """Objective to minimize: negative of smoothed min area (so we maximize min area).
    bary_flat is a flat array of length N*3, but we only use N*2 since a+b+c=1."""
    bary = bary_flat.reshape(N, 3)
    # Normalize barycentric coordinates
    bary_norm = bary / np.sum(bary, axis=1, keepdims=True)
    pts = _barycentric_to_cart(bary_norm)
    area = _smooth_min_area(pts, temperature)
    return -area

def _objective_grad(bary_flat, temperature=0.01):
    """Gradient of the objective using finite differences or analytical."""
    # Use numerical gradient since analytical is complex
    eps = 1e-7
    grad = np.zeros_like(bary_flat)
    f0 = _objective(bary_flat, temperature)
    for i in range(len(bary_flat)):
        bary_flat[i] += eps
        f1 = _objective(bary_flat, temperature)
        bary_flat[i] -= eps
        grad[i] = (f1 - f0) / eps
    return grad

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    best_points = None
    best_area = -np.inf
    
    rng = np.random.RandomState(42)
    
    # Multiple restarts with different initializations
    n_restarts = 50
    
    for restart in range(n_restarts):
        if time.time() - start_time > deadline:
            break
        
        # Initialize barycentric coordinates randomly inside triangle
        # Use Dirichlet-like distribution for interior points
        raw = rng.random((N, 3))
        bary_init = raw / np.sum(raw, axis=1, keepdims=True)
        # Ensure minimum distance from boundary
        bary_init = np.clip(bary_init, 0.02, 1.0)
        bary_init = bary_init / np.sum(bary_init, axis=1, keepdims=True)
        
        x0 = bary_init.flatten()
        
        # Use L-BFGS-B with numerical gradient
        temperature = 0.005 + 0.01 * (restart / n_restarts)  # Anneal temperature
        
        try:
            result = minimize(
                _objective,
                x0,
                args=(temperature,),
                method='L-BFGS-B',
                jac=_objective_grad,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
        except Exception:
            continue
        
        if time.time() - start_time > deadline:
            break
        
        if result.success or result.fun < best_area * -1:  # result.fun is -area
            bary_opt = result.x.reshape(N, 3)
            bary_opt = np.clip(bary_opt, 0.01, 1.0)
            bary_opt = bary_opt / np.sum(bary_opt, axis=1, keepdims=True)
            pts = _barycentric_to_cart(bary_opt)
            
            # Compute actual min area (not smoothed)
            actual_area = np.min(_triangle_areas(pts))
            
            if actual_area > best_area:
                best_area = actual_area
                best_points = pts
    
    # If no optimization succeeded, fall back to random
    if best_points is None:
        # Simple random placement
        raw = rng.random((N, 3))
        bary = raw / np.sum(raw, axis=1, keepdims=True)
        bary = np.clip(bary, 0.02, 1.0)
        bary = bary / np.sum(bary, axis=1, keepdims=True)
        best_points = _barycentric_to_cart(bary)
    
    return best_points
