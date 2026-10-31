import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

N = 11
H = np.sqrt(3) / 2

# Pre-compute indices for all triangles formed by N points
tri_indices = list(combinations(range(N), 3))
tri_indices = np.array(tri_indices)
I = tri_indices[:, 0]
J = tri_indices[:, 1]
K = tri_indices[:, 2]

# Vertices of the equilateral triangle
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, H])

def bary_to_cart(bary_points):
    """
    Convert barycentric coordinates (u, v, w) with u+v+w=1 to Cartesian coordinates.
    bary_points: array of shape (N, 3)
    """
    # P = u*A + v*B + w*C
    # A is (0,0), so term u*A is 0.
    # P = v*B + w*C
    # B = (1, 0), C = (0.5, H)
    # P_x = v * 1 + w * 0.5
    # P_y = v * 0 + w * H
    v = bary_points[:, 1]
    w = bary_points[:, 2]
    x = v + 0.5 * w
    y = H * w
    return np.column_stack((x, y))

def clamp_bary(bary_points):
    """
    Ensure barycentric coordinates are valid: non-negative and sum to 1.
    Simple clipping and normalization.
    """
    bary_points = np.maximum(bary_points, 0)
    sums = np.sum(bary_points, axis=1, keepdims=True)
    # Avoid division by zero
    sums[sums == 0] = 1
    return bary_points / sums

def calc_areas_bary(bary_points):
    """
    Calculate areas of all triangles formed by N points given in barycentric coordinates.
    Uses the property that area ratio is preserved under affine transformation.
    Area of triangle P_i P_j P_k = |det([u_i v_i w_i; u_j v_j w_j; u_k v_k w_k])| * Area(ABC)
    Area(ABC) = sqrt(3)/4.
    """
    Pi = bary_points[I]
    Pj = bary_points[J]
    Pk = bary_points[K]
    
    # Determinant of 3x3 matrix with rows Pi, Pj, Pk
    # det = u_i(v_j w_k - w_j v_k) - v_i(u_j w_k - w_j u_k) + w_i(u_j v_k - v_j u_k)
    u_i, v_i, w_i = Pi[:, 0], Pi[:, 1], Pi[:, 2]
    u_j, v_j, w_j = Pj[:, 0], Pj[:, 1], Pj[:, 2]
    u_k, v_k, w_k = Pk[:, 0], Pk[:, 1], Pk[:, 2]
    
    det = u_i * (v_j * w_k - w_j * v_k) - v_i * (u_j * w_k - w_j * u_k) + w_i * (u_j * v_k - v_j * u_k)
    
    # Area = 0.5 * |det| * Area(ABC) ? 
    # Actually, the area of the reference triangle is Area(ABC).
    # The determinant of the barycentric coordinates matrix gives the ratio of the area of the sub-triangle to the reference triangle.
    # So Area_sub = |det| * Area(ABC).
    area_abc = np.sqrt(3) / 4.0
    areas = np.abs(det) * area_abc
    
    return areas

def hard_min_obj(bary_points):
    """
    Objective for Nelder-Mead: Minimize -min_area.
    """
    bary_points = clamp_bary(bary_points)
    areas = calc_areas_bary(bary_points)
    return -np.min(areas)

def soft_min_obj(bary_points, beta=50.0):
    """
    Smooth approximation of -min_area.
    Uses log-sum-exp trick for stability.
    """
    bary_points = clamp_bary(bary_points)
    areas = calc_areas_bary(bary_points)
    
    # We want to maximize min(areas).
    # Minimize -soft_min(areas).
    # soft_min(areas) = -1/beta * log(sum(exp(-beta * areas)))
    # This is close to min(areas) when beta is large.
    
    vals = -beta * areas
    max_val = np.max(vals)
    log_sum_exp = max_val + np.log(np.sum(np.exp(vals - max_val)))
    soft_min_val = -log_sum_exp / beta
    
    return -soft_min_val

def generate_hexagonal_init(rng):
    """
    Generate an initial configuration based on a hexagonal lattice structure,
    sampled in barycentric coordinates.
    """
    # Create a grid of points in a larger space and filter/project
    # Or just generate random points and let optimization work.
    # A structured start is better.
    # Let's try placing points in rows corresponding to barycentric levels.
    
    pts = []
    # Simple heuristic: distribute points roughly evenly
    # For N=11, maybe 3-4-4 or similar distribution in barycentric space?
    # Just random uniform in simplex is a robust baseline.
    for _ in range(N):
        u = rng.random()
        v = rng.random()
        if u + v > 1:
            u = 1 - u
            v = 1 - v
        w = 1 - u - v
        pts.append([u, v, w])
        
    return np.array(pts)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s buffer
    
    rng = np.random.default_rng(42)
    
    best_score = -np.inf
    best_points = None
    
    def update_best(bary_pts):
        nonlocal best_score, best_points
        bary_pts = clamp_bary(bary_pts)
        areas = calc_areas_bary(bary_pts)
        score = np.min(areas)
        if score > best_score:
            best_score = score
            best_points = bary_pts.copy()
            
    # Initial candidates
    num_restarts = 20
    # We will perform multiple random restarts with Nelder-Mead
    
    # 1. Hexagonal/Lattice-like init (approximated by random for now, or specific logic)
    # Let's stick to robust random inits with good optimization.
    
    for i in range(num_restarts):
        if time.time() > deadline - 20:
            break
            
        # Generate initial barycentric points
        # Strategy: Random uniform in simplex
        init_bary = generate_hexagonal_init(rng)
        
        # Optimization
        # Use Nelder-Mead on the soft objective to find a good basin, then hard min for final score?
        # Nelder-Mead on hard min is non-smooth. Soft min is smooth.
        # Let's optimize soft min with high beta.
        
        def objective(x):
            return soft_min_obj(x, beta=50.0)
        
        try:
            res = minimize(
                objective, 
                init_bary, 
                method='Nelder-Mead', 
                options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
            )
            
            # Evaluate the result
            res_bary = clamp_bary(res.x)
            update_best(res_bary)
            
            # If we have time, do a second pass with higher beta (sharper min) starting from this result
            if time.time() < deadline - 10:
                def objective_sharp(x):
                    return soft_min_obj(x, beta=200.0)
                
                res2 = minimize(
                    objective_sharp, 
                    res_bary, 
                    method='Nelder-Mead', 
                    options={'maxiter': 1000, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True}
                )
                res2_bary = clamp_bary(res2.x)
                update_best(res2_bary)
                
        except Exception as e:
            pass

    # Final polish on the best found
    if best_points is not None and time.time() < deadline - 5:
        def objective_final(x):
            return soft_min_obj(x, beta=500.0)
        
        try:
            res_final = minimize(
                objective_final, 
                best_points, 
                method='Nelder-Mead', 
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            final_bary = clamp_bary(res_final.x)
            update_best(final_bary)
        except Exception:
            pass

    if best_points is None:
        # Fallback: random
        best_points = generate_hexagonal_init(rng)
        
    # Convert to Cartesian for output
    cart_points = bary_to_cart(best_points)
    
    return cart_points
