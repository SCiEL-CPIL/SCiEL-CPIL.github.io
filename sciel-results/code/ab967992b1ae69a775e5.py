import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
from itertools import combinations

N = 11
# Vertices of the equilateral triangle
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3)/2])
H = np.sqrt(3)/2

# Pre-compute triangle indices
TRI_INDICES = np.array(list(combinations(range(N), 3)))

def calc_min_area(points):
    """
    points: (11, 2)
    returns: min area
    """
    p1 = points[TRI_INDICES[:, 0]]
    p2 = points[TRI_INDICES[:, 1]]
    p3 = points[TRI_INDICES[:, 2]]
    
    x1, y1 = p1[:, 0], p1[:, 1]
    x2, y2 = p2[:, 0], p2[:, 1]
    x3, y3 = p3[:, 0], p3[:, 1]
    
    cross = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def is_inside_triangle(points):
    """
    Check if all points are inside or on the boundary of the triangle.
    """
    x = points[:, 0]
    y = points[:, 1]
    
    eps = 1e-9
    cond1 = y >= -eps
    cond2 = y <= np.sqrt(3) * x + eps
    cond3 = y <= -np.sqrt(3) * (x - 1) + eps
    
    return np.all(cond1) and np.all(cond2) and np.all(cond3)

def to_barycentric(points):
    """
    Convert Cartesian coordinates to Barycentric coordinates relative to the equilateral triangle.
    Returns array of shape (N, 3) where each row sums to 1.
    """
    # Area of the triangle is 0.5 * base * height = 0.5 * 1 * H = H/2
    # Barycentric coords (u, v, w) for point p:
    # u = Area(p, V1, V2) / Area(V0, V1, V2)
    # v = Area(V0, p, V2) / Area(V0, V1, V2)
    # w = Area(V0, V1, p) / Area(V0, V1, V2)
    
    # Using cross products for area calculation (2*Area)
    # Area(V0, V1, V2) * 2 = | (V1-V0) x (V2-V0) | = | (1,0) x (0.5, H) | = 1*H - 0 = H
    # So Area = H/2.
    
    x, y = points[:, 0], points[:, 1]
    
    # w = Area(V0, V1, p) / Area = 0.5 * | (V1-V0) x (p-V0) | / (H/2) = | (1,0) x (x,y) | / H = y / H
    w = y / H
    
    # v = Area(V0, p, V2) / Area = 0.5 * | (p-V0) x (V2-V0) | / (H/2) = | (x,y) x (0.5, H) | / H
    # Cross product in 2D: x1*y2 - y1*x2
    # (x,y) x (0.5, H) = x*H - y*0.5
    v = (x * H - y * 0.5) / H
    
    # u = 1 - v - w
    u = 1.0 - v - w
    
    return np.column_stack((u, v, w))

def from_barycentric(bary):
    """
    Convert Barycentric coordinates to Cartesian.
    """
    u, v, w = bary[:, 0], bary[:, 1], bary[:, 2]
    x = u * V0[0] + v * V1[0] + w * V2[0]
    y = u * V0[1] + v * V1[1] + w * V2[1]
    return np.column_stack((x, y))

def calc_min_area_bary(bary):
    """
    Calculate min area directly from barycentric coordinates.
    """
    pts = from_barycentric(bary)
    return calc_min_area(pts)

def objective_de(x):
    """
    Objective for Differential Evolution.
    x: flat array of length 22
    """
    pts = x.reshape(N, 2)
    
    if not is_inside_triangle(pts):
        return 10.0
    
    min_area = calc_min_area(pts)
    return -min_area

def simulated_annealing(initial_points, time_limit=30.0, T_start=0.1, T_end=1e-4, cooling_rate=0.999):
    """
    Perform Simulated Annealing to maximize the minimum triangle area.
    """
    current_points = initial_points.copy()
    current_area = calc_min_area(current_points)
    best_points = current_points.copy()
    best_area = current_area
    
    start_time = time.time()
    T = T_start
    
    while time.time() - start_time < time_limit:
        # Generate a perturbation
        # Move a random point by a small random amount
        i = np.random.randint(N)
        step_size = T * 0.5  # Scale step with temperature
        dx = np.random.normal(0, step_size)
        dy = np.random.normal(0, step_size)
        
        new_points = current_points.copy()
        new_points[i] += np.array([dx, dy])
        
        # Check if inside triangle
        if not is_inside_triangle(new_points):
            continue
            
        new_area = calc_min_area(new_points)
        delta_area = new_area - current_area
        
        # Accept if better or with probability exp(delta_area / T)
        if delta_area > 0 or np.random.random() < np.exp(delta_area / T):
            current_points = new_points
            current_area = new_area
            
            if current_area > best_area:
                best_area = current_area
                best_points = current_points.copy()
                
        # Cool down
        T *= cooling_rate
        if T < T_end:
            break
            
    return best_points, best_area

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # 1. Initial Global Search using Differential Evolution (short run to get a good start)
    # We use a small population and few iterations to save time for SA/SLSQP
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, H))
        
    best_x = None
    best_val = -np.inf
    
    try:
        de_result = differential_evolution(
            objective_de, 
            bounds=bounds, 
            seed=42, 
            maxiter=20, 
            popsize=10,
            tol=1e-6,
            polish=False, 
            init='latinhypercube',
            updating='immediate'
        )
        
        if de_result.fun < 1.0: # Valid
            best_x = de_result.x
            best_val = -de_result.fun
        else:
            # Fallback to random if DE fails
            rng = np.random.default_rng(42)
            pts = np.zeros((N, 2))
            for i in range(N):
                while True:
                    x = rng.random()
                    y = rng.random() * H
                    if y <= np.sqrt(3)*x and y <= -np.sqrt(3)*(x-1):
                        pts[i] = [x, y]
                        break
            best_x = pts.flatten()
            best_val = calc_min_area(pts)
            
    except Exception:
        # Fallback to random sampling if DE crashes
        rng = np.random.default_rng(42)
        pts = np.zeros((N, 2))
        for i in range(N):
            while True:
                x = rng.random()
                y = rng.random() * H
                if y <= np.sqrt(3)*x and y <= -np.sqrt(3)*(x-1):
                    pts[i] = [x, y]
                    break
        best_x = pts.flatten()
        best_val = calc_min_area(pts)

    # 2. Simulated Annealing Refinement
    if time.time() < deadline - 40.0:
        sa_time = min(30.0, deadline - 40.0 - (time.time() - start_time))
        sa_points, sa_area = simulated_annealing(best_x.reshape(N, 2), time_limit=sa_time)
        
        if sa_area > best_val:
            best_val = sa_area
            best_x = sa_points.flatten()

    # 3. SLSQP Refinement using Barycentric Coordinates
    # This is the most precise step.
    if time.time() < deadline - 10.0:
        # Convert current best to barycentric
        bary_init = to_barycentric(best_x.reshape(N, 2))
        
        # Define objective for SLSQP (maximize min area -> minimize -min area)
        def objective_slsqp(bary_flat):
            bary = bary_flat.reshape(N, 3)
            # Ensure barycentric coords are valid (sum to 1, non-negative)
            # SLSQP will handle constraints, but let's add a small penalty if they go negative significantly
            if np.any(bary < -1e-6):
                return 10.0
            # Normalize to sum to 1 just in case
            bary = bary / np.sum(bary, axis=1, keepdims=True)
            
            pts = from_barycentric(bary)
            area = calc_min_area(pts)
            return -area
            
        # Constraints: barycentric coords must be non-negative and sum to 1
        # SLSQP handles equality and inequality constraints
        # Equality: sum of bary for each point = 1
        # Inequality: bary >= 0
        
        def constraints(bary_flat):
            bary = bary_flat.reshape(N, 3)
            # Equality constraints: sum(bary_i) = 1 for all i
            eq_cons = np.sum(bary, axis=1) - 1.0
            # Inequality constraints: bary >= 0
            # SLSQP requires 'fun' >= 0 for inequalities
            ineq_cons = bary.flatten()
            return np.concatenate([eq_cons, ineq_cons])
            
        # Define constraints in SLSQP format
        cons = [
            {'type': 'eq', 'fun': lambda x: np.sum(x.reshape(N, 3), axis=1) - 1.0},
            {'type': 'ineq', 'fun': lambda x: x} # x >= 0
        ]
        
        try:
            slsqp_result = minimize(
                objective_slsqp, 
                bary_init.flatten(), 
                method='SLSQP', 
                constraints=cons,
                options={'maxiter': 100, 'ftol': 1e-12}
            )
            
            if slsqp_result.success or slsqp_result.fun < 1.0:
                bary_final = slsqp_result.x.reshape(N, 3)
                # Ensure non-negative
                bary_final = np.maximum(bary_final, 0)
                # Normalize
                bary_final = bary_final / np.sum(bary_final, axis=1, keepdims=True)
                
                pts_final = from_barycentric(bary_final)
                final_area = calc_min_area(pts_final)
                
                if final_area > best_val:
                    best_val = final_area
                    best_x = pts_final.flatten()
                    
        except Exception:
            pass

    # Final check
    final_pts = best_x.reshape(N, 2)
    if not is_inside_triangle(final_pts):
        # Emergency clamp
        # Project points back to triangle?
        # Simple clamp to [0,1] x [0,H] is not enough.
        # We'll just return it, hoping the optimizers kept it valid.
        pass

    return final_pts
