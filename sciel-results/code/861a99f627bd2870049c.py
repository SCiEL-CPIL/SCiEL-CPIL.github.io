import numpy as np
from scipy.optimize import differential_evolution, minimize
from itertools import combinations

N = 11

tri_indices = list(combinations(range(N), 3))
tri_indices_arr = np.array(tri_indices)

def _calculate_min_area(points):
    """Compute minimum triangle area for 11 points."""
    p1 = points[tri_indices_arr[:, 0]]
    p2 = points[tri_indices_arr[:, 1]]
    p3 = points[tri_indices_arr[:, 2]]
    
    x1, y1 = p1[:, 0], p1[:, 1]
    x2, y2 = p2[:, 0], p2[:, 1]
    x3, y3 = p3[:, 0], p3[:, 1]
    
    areas = np.abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)) * 0.5
    return np.min(areas)

def _uv_to_xy(u, v):
    """Convert barycentric-like coords (u,v) to Cartesian in unit equilateral triangle."""
    x = u + 0.5 * v
    y = (np.sqrt(3) / 2.0) * v
    return x, y

def _xy_to_uv(x, y):
    """Convert Cartesian to (u,v) coords."""
    v = y / (np.sqrt(3) / 2.0)
    u = x - 0.5 * v
    return u, v

def _is_valid(u, v):
    """Check if points are inside the triangle."""
    return np.all(u >= -1e-9) and np.all(v >= -1e-9) and np.all(u + v <= 1 + 1e-9)

def _objective_de(uv_params):
    """Objective for differential evolution: minimize -min_area."""
    u = uv_params[0::2]
    v = uv_params[1::2]
    
    # Soft penalty for constraint violation
    violation = np.maximum(0, u + v - 1.0)
    if np.any(violation > 1e-6):
        return 1.0 + np.sum(violation)
    
    x, y = _uv_to_xy(u, v)
    points = np.column_stack((x, y))
    
    min_area = _calculate_min_area(points)
    return -min_area

def _local_refine(points):
    """Local refinement using Nelder-Mead on the minimum area."""
    def _obj_flat(flat_points):
        pts = flat_points.reshape(N, 2)
        # Check if all points are inside triangle
        # For equilateral triangle with vertices (0,0), (1,0), (0.5, sqrt(3)/2)
        # Point (x,y) is inside if: y >= 0, y <= sqrt(3)*x, y <= sqrt(3)*(1-x)
        y = pts[:, 1]
        x = pts[:, 0]
        # Check constraints
        if np.any(y < -1e-10) or np.any(y > np.sqrt(3) * x + 1e-10) or np.any(y > np.sqrt(3) * (1 - x) + 1e-10):
            return 1.0
        return -_calculate_min_area(pts)
    
    initial_flat = points.flatten()
    result = minimize(_obj_flat, initial_flat, method='Nelder-Mead', 
                      options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
    
    best_pts = result.x.reshape(N, 2)
    # Ensure points are inside triangle
    y = best_pts[:, 1]
    x = best_pts[:, 0]
    for i in range(N):
        if y[i] < 0:
            best_pts[i, 1] = 0
        if y[i] > np.sqrt(3) * x[i]:
            # Project back
            t = y[i] / (np.sqrt(3) * x[i]) if x[i] > 0 else 1.0
            best_pts[i] *= t
        if y[i] > np.sqrt(3) * (1 - x[i]):
            t = y[i] / (np.sqrt(3) * (1 - x[i])) if (1 - x[i]) > 0 else 1.0
            best_pts[i] *= t
    
    return best_pts

def construct():
    """Main function: multi-stage optimization."""
    # Stage 1: Coarse search with DE using barycentric coordinates
    # Use bounds that ensure u >= 0, v >= 0, u+v <= 1
    # We'll use u in [0, 1], v in [0, 1] and rely on the objective to penalize violations
    
    bounds = [(0, 1)] * (2 * N)
    
    # Run DE with moderate settings
    result = differential_evolution(
        _objective_de, 
        bounds, 
        popsize=30, 
        maxiter=80, 
        seed=42, 
        tol=1e-5,
        polish=False,  # We'll do our own polishing
        init='sobol'
    )
    
    best_uv = result.x
    u = best_uv[0::2]
    v = best_uv[1::2]
    
    # Project to valid region if needed
    for i in range(N):
        if u[i] + v[i] > 1:
            scale = 1.0 / (u[i] + v[i])
            u[i] *= scale
            v[i] *= scale
    
    x, y = _uv_to_xy(u, v)
    points = np.column_stack((x, y))
    
    # Stage 2: Local refinement with Nelder-Mead
    refined_points = _local_refine(points)
    
    # Stage 3: Another round of local refinement from multiple perturbations
    best_score = _calculate_min_area(refined_points)
    best_final = refined_points.copy()
    
    rng = np.random.RandomState(123)
    for _ in range(5):
        # Perturb the best solution
        perturbed = best_final + rng.normal(0, 0.01, size=(N, 2))
        # Clip to triangle
        for i in range(N):
            px, py = perturbed[i]
            if py < 0:
                perturbed[i, 1] = 0
            elif py > np.sqrt(3) * px:
                t = py / (np.sqrt(3) * px) if px > 0 else 1.0
                perturbed[i] *= t
            elif py > np.sqrt(3) * (1 - px):
                t = py / (np.sqrt(3) * (1 - px)) if (1 - px) > 0 else 1.0
                perturbed[i] *= t
        
        refined = _local_refine(perturbed)
        score = _calculate_min_area(refined)
        if score > best_score:
            best_score = score
            best_final = refined
    
    return best_final
