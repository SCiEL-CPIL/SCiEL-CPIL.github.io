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

def _project_to_triangle(points):
    """Project points to be inside the equilateral triangle."""
    pts = points.copy()
    for i in range(N):
        px, py = pts[i]
        # Triangle vertices: (0,0), (1,0), (0.5, sqrt(3)/2)
        # Constraints: y >= 0, y <= sqrt(3)*x, y <= sqrt(3)*(1-x)
        if py < 0:
            pts[i, 1] = 0
        if py > np.sqrt(3) * px:
            # Project onto line from (0,0) to (0.5, sqrt(3)/2)
            if px > 1e-10:
                t = py / (np.sqrt(3) * px)
                pts[i] *= min(t, 1.0)
            else:
                pts[i, 1] = 0
        if py > np.sqrt(3) * (1 - px):
            # Project onto line from (1,0) to (0.5, sqrt(3)/2)
            if (1 - px) > 1e-10:
                t = py / (np.sqrt(3) * (1 - px))
                px_new = 1 - (1 - px) * t
                py_new = py * t
                pts[i] = [px_new, py_new]
            else:
                pts[i, 1] = 0
    return pts

def _local_refine(points):
    """Local refinement using Nelder-Mead on the minimum area."""
    def _obj_flat(flat_points):
        pts = flat_points.reshape(N, 2)
        # Check if all points are inside triangle
        y = pts[:, 1]
        x = pts[:, 0]
        # Check constraints with tolerance
        if np.any(y < -1e-8) or np.any(y > np.sqrt(3) * x + 1e-8) or np.any(y > np.sqrt(3) * (1 - x) + 1e-8):
            return 1.0
        return -_calculate_min_area(pts)
    
    initial_flat = points.flatten()
    result = minimize(_obj_flat, initial_flat, method='Nelder-Mead', 
                      options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14, 'initial_simplex': 1e-3})
    
    best_pts = result.x.reshape(N, 2)
    best_pts = _project_to_triangle(best_pts)
    
    return best_pts

def _coordinate_descent_refine(points, max_iter=200):
    """Refine using coordinate descent on the minimum triangle vertices."""
    pts = points.copy()
    best_area = _calculate_min_area(pts)
    
    for iteration in range(max_iter):
        # Find the minimum area triangle
        p1 = pts[tri_indices_arr[:, 0]]
        p2 = pts[tri_indices_arr[:, 1]]
        p3 = pts[tri_indices_arr[:, 2]]
        
        x1, y1 = p1[:, 0], p1[:, 1]
        x2, y2 = p2[:, 0], p2[:, 1]
        x3, y3 = p3[:, 0], p3[:, 1]
        
        areas = np.abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)) * 0.5
        min_idx = np.argmin(areas)
        
        # Get the three points of the minimum triangle
        i, j, k = tri_indices_arr[min_idx]
        
        # Try moving each of the three points to increase the area
        for idx in [i, j, k]:
            # Try small perturbations in x and y
            for dx in [-0.001, -0.0001, 0.0001, 0.001]:
                for dy in [-0.001, -0.0001, 0.0001, 0.001]:
                    trial = pts.copy()
                    trial[idx, 0] += dx
                    trial[idx, 1] += dy
                    trial = _project_to_triangle(trial)
                    trial_area = _calculate_min_area(trial)
                    if trial_area > best_area:
                        pts = trial
                        best_area = trial_area
        
        # Also try larger steps
        for idx in [i, j, k]:
            for dx in [-0.01, -0.005, 0.005, 0.01]:
                for dy in [-0.01, -0.005, 0.005, 0.01]:
                    trial = pts.copy()
                    trial[idx, 0] += dx
                    trial[idx, 1] += dy
                    trial = _project_to_triangle(trial)
                    trial_area = _calculate_min_area(trial)
                    if trial_area > best_area:
                        pts = trial
                        best_area = trial_area
    
    return pts

def _simulated_annealing(points, max_iter=500, T_init=0.01, T_final=0.001, cooling=0.99):
    """Simulated annealing to escape local optima."""
    pts = points.copy()
    best_pts = pts.copy()
    best_score = _calculate_min_area(pts)
    current_score = best_score
    T = T_init
    
    rng = np.random.RandomState(42)
    
    for i in range(max_iter):
        T = T_init * (T_final / T_init) ** (i / max_iter)
        
        # Generate a random perturbation
        perturbed = pts + rng.normal(0, 0.005, size=(N, 2))
        perturbed = _project_to_triangle(perturbed)
        
        new_score = _calculate_min_area(perturbed)
        delta = new_score - current_score
        
        # Accept if better or with probability exp(delta/T)
        if delta > 0 or np.random.random() < np.exp(delta / max(T, 1e-10)):
            pts = perturbed
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_pts = pts.copy()
    
    return best_pts

def construct():
    """Main function: multi-stage optimization."""
    # Stage 1: Coarse search with DE using barycentric coordinates
    bounds = [(0, 1)] * (2 * N)
    
    # Run DE with moderate settings
    result = differential_evolution(
        _objective_de, 
        bounds, 
        popsize=35, 
        maxiter=100, 
        seed=42, 
        tol=1e-6,
        polish=False,
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
    
    # Stage 3: Coordinate descent refinement
    refined_points = _coordinate_descent_refine(refined_points, max_iter=100)
    
    # Stage 4: Simulated annealing
    refined_points = _simulated_annealing(refined_points, max_iter=300)
    
    # Stage 5: Final local refinement
    refined_points = _local_refine(refined_points)
    refined_points = _coordinate_descent_refine(refined_points, max_iter=50)
    
    # Stage 6: Multiple perturbations and refinements
    best_score = _calculate_min_area(refined_points)
    best_final = refined_points.copy()
    
    rng = np.random.RandomState(123)
    for _ in range(8):
        # Perturb the best solution with varying magnitudes
        magnitude = 0.005 + 0.01 * rng.random()
        perturbed = best_final + rng.normal(0, magnitude, size=(N, 2))
        perturbed = _project_to_triangle(perturbed)
        
        refined = _local_refine(perturbed)
        refined = _coordinate_descent_refine(refined, max_iter=50)
        
        score = _calculate_min_area(refined)
        if score > best_score:
            best_score = score
            best_final = refined
    
    # Final polish
    best_final = _local_refine(best_final)
    best_final = _coordinate_descent_refine(best_final, max_iter=30)
    
    return best_final
