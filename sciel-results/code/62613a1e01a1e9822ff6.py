import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
from itertools import combinations

N = 11
TIME_LIMIT = 115.0

# Triangle vertices
P0 = np.array([0.0, 0.0])
P1 = np.array([1.0, 0.0])
P2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2

# Pre-compute triangle indices for vectorization
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices) # shape (165, 3)

def perimeter_to_cartesian(t):
    """
    Convert arc length parameter t in [0, 3) to Cartesian coordinates on the perimeter.
    """
    t = np.mod(t, 3.0)
    x = np.zeros_like(t)
    y = np.zeros_like(t)
    
    # Segment 0: P0 to P1 (0 <= t < 1)
    mask0 = (t >= 0) & (t < 1)
    x[mask0] = t[mask0]
    y[mask0] = 0.0
    
    # Segment 1: P1 to P2 (1 <= t < 2)
    mask1 = (t >= 1) & (t < 2)
    t1 = t[mask1] - 1.0
    x[mask1] = 1.0 - 0.5 * t1
    y[mask1] = H * t1
    
    # Segment 2: P2 to P0 (2 <= t < 3)
    mask2 = (t >= 2) & (t < 3)
    t2 = t[mask2] - 2.0
    x[mask2] = 0.5 * (1.0 - t2)
    y[mask2] = H * (1.0 - t2)
    
    return np.column_stack([x, y])

def calc_min_area_from_perimeter(t_params):
    """
    Calculate minimum area for points defined by perimeter parameters.
    """
    points = perimeter_to_cartesian(t_params)
    
    pi = points[tri_idx[:, 0]]
    pj = points[tri_idx[:, 1]]
    pk = points[tri_idx[:, 2]]
    
    v1 = pj - pi
    v2 = pk - pi
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def objective_perimeter(t_params):
    """
    Objective function for perimeter optimization.
    """
    area = calc_min_area_from_perimeter(t_params)
    return -area

def project_to_triangle(points):
    """
    Project points to the interior or boundary of the equilateral triangle.
    """
    x = points[:, 0]
    y = points[:, 1]
    
    w = y / H
    v = x - 0.5 * w
    u = 1.0 - v - w
    
    outside = (u < 0) | (v < 0) | (w < 0)
    
    if np.any(outside):
        pts_out = points[outside]
        u_out = u[outside]
        v_out = v[outside]
        w_out = w[outside]
        
        u_clip = np.clip(u_out, 0, None)
        v_clip = np.clip(v_out, 0, None)
        w_clip = np.clip(w_out, 0, None)
        
        sums = u_clip + v_clip + w_clip
        # Avoid division by zero
        sums = np.where(sums == 0, 1e-10, sums)
        
        u_out = u_clip / sums
        v_out = v_clip / sums
        w_out = w_clip / sums
        
        x_out = v_out + 0.5 * w_out
        y_out = w_out * H
        
        points[outside] = np.column_stack([x_out, y_out])
        
    return points

def calc_min_area(points):
    """Calculate the minimum area of triangles formed by N points."""
    pi = points[tri_idx[:, 0]]
    pj = points[tri_idx[:, 1]]
    pk = points[tri_idx[:, 2]]
    
    v1 = pj - pi
    v2 = pk - pi
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def refine_perimeter(t_params, time_limit=5.0):
    """
    Refine perimeter parameters using local optimization.
    """
    start_time = time.time()
    
    # Use Nelder-Mead on the perimeter parameters
    # Bounds are not directly supported in NM, but we can wrap the objective
    # to penalize out-of-bounds or just rely on the periodic nature/mod inside.
    # Since t is in [0, 3), we can just let NM run and mod the results.
    
    try:
        res = minimize(
            objective_perimeter,
            t_params,
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        if res.fun < -calc_min_area_from_perimeter(t_params):
            t_params = res.x
    except Exception:
        pass
        
    return t_params

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_points = None
    best_score = -np.inf
    
    def update_best(points):
        nonlocal best_points, best_score
        points = project_to_triangle(points)
        score = calc_min_area(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()
            return True
        return False

    # Strategy: Optimize points on the perimeter.
    # Initialize with evenly spaced points on the perimeter.
    # Then use Differential Evolution to find better spacing.
    
    # Initial guess: evenly spaced
    t_init = np.linspace(0, 3, N, endpoint=False)
    
    # Run DE on perimeter parameters
    bounds = [(0, 3)] * N
    
    try:
        de_result = differential_evolution(
            objective_perimeter,
            bounds,
            seed=42,
            maxiter=100,
            popsize=20,
            tol=1e-6,
            polish=False,
            init='latinhypercube'
        )
        
        t_de = de_result.x
        pts_de = perimeter_to_cartesian(t_de)
        update_best(pts_de)
        
        # Refine with NM
        if time.time() < deadline - 20:
            t_refined = refine_perimeter(t_de, time_limit=15.0)
            pts_refined = perimeter_to_cartesian(t_refined)
            update_best(pts_refined)
            
    except Exception:
        pass

    # Also try some multi-starts on the perimeter
    rng = np.random.default_rng(123)
    for i in range(10):
        if time.time() > deadline - 10:
            break
        t_rand = rng.uniform(0, 3, N)
        # Sort to maintain some order? Not strictly necessary for area, but helps NM
        t_rand = np.sort(t_rand)
        
        try:
            t_refined = refine_perimeter(t_rand, time_limit=3.0)
            pts_refined = perimeter_to_cartesian(t_refined)
            update_best(pts_refined)
        except Exception:
            pass

    # If we have a best solution, try to allow points to move into the interior
    # by converting to Cartesian and running a Cartesian local optimization.
    if best_points is not None and time.time() < deadline - 5:
        try:
            x0 = best_points.flatten()
            bounds_cart = [(0, 1), (0, H)] * N
            
            def objective_cart(x):
                pts = x.reshape(N, 2)
                pts = project_to_triangle(pts)
                return -calc_min_area(pts)
            
            res_cart = minimize(
                objective_cart,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            
            if res_cart.fun < -calc_min_area(best_points):
                pts_final = project_to_triangle(res_cart.x.reshape(N, 2))
                update_best(pts_final)
        except Exception:
            pass

    return best_points

# EVOLVE-BLOCK-END
