import numpy as np
from scipy.optimize import differential_evolution, minimize, basinhopping
from itertools import combinations
import time

N = 11
TIME_LIMIT = 145.0  # Safety margin for 150s limit

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2

def bary_to_cart(bary):
    """Convert barycentric coords (u, v) to Cartesian (x, y)."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    return np.column_stack([x, y])

def cart_to_bary(cart):
    """Convert Cartesian (x, y) to barycentric (u, v)."""
    x = cart[:, 0]
    y = cart[:, 1]
    v = 2.0 * y / H
    u = 1.0 - v - x
    return np.column_stack([u, v])

def calc_min_area_cartesian(points):
    """
    Calculate the minimum area of all triangles formed by N points.
    points: (N, 2) array of Cartesian coordinates.
    """
    p_i = points[tri_idx[:, 0]]
    p_j = points[tri_idx[:, 1]]
    p_k = points[tri_idx[:, 2]]
    
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_k[:, 0] - p_i[:, 0]) * (p_j[:, 1] - p_i[:, 1])
    
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def clamp_bary(bary, eps=1e-9):
    """Clamp barycentric coordinates to the strictly valid triangle region."""
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()
    
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    sums = u + v
    invalid = sums > (1 - eps)
    if np.any(invalid):
        scale = (1 - eps) / np.where(sums > 0, sums, 1.0)
        u[invalid] *= scale[invalid]
        v[invalid] *= scale[invalid]
        
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    return np.column_stack([u, v])

def hard_min_obj_bary(bary_flat):
    """Objective using hard min in barycentric coords."""
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    cart = bary_to_cart(bary)
    return -calc_min_area_cartesian(cart)

def soft_min_obj_bary(bary_flat, beta=50.0):
    """
    Objective using soft min (log-sum-exp approximation) for smoother gradients.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    cart = bary_to_cart(bary)
    
    p_i = cart[tri_idx[:, 0]]
    p_j = cart[tri_idx[:, 1]]
    p_k = cart[tri_idx[:, 2]]
    
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_k[:, 0] - p_i[:, 0]) * (p_j[:, 1] - p_i[:, 1])
    
    areas = 0.5 * np.abs(cross)
    
    if np.any(areas < 1e-15):
        return 1e10

    scaled = -beta * areas
    max_scaled = np.max(scaled)
    
    if max_scaled > 700:
        return -np.min(areas)

    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min = -log_sum / beta
    
    return -soft_min

def generate_lattice_init():
    """
    Generate a specific high-symmetry lattice initialization.
    This serves as the 'template' for the affine deformation search.
    """
    # A 5x5 grid subset that fits 11 points well
    # Rows in barycentric (u, v):
    # Row 0 (w=1): (0,0), (0.25,0), (0.5,0), (0.75,0), (1,0) -> 5 pts
    # Row 1 (w=0.75): (0.25, 0), (0.5, 0), (0.75, 0) ... wait, v = 1-u-w
    # Let's just use a standard triangular lattice subset.
    
    pts = []
    # 3 points on bottom edge
    pts.append([0.0, 0.0])
    pts.append([0.5, 0.0])
    pts.append([1.0, 0.0])
    
    # 3 points in next row up
    pts.append([0.25, 0.25]) # u=0.25, v=0.25, w=0.5 -> x=0.25+0.25=0.5? No.
    # Barycentric to Cart: x = v + 0.5w, y = w*H
    # If u=0, v=0.25, w=0.75 -> x=0.25+0.375=0.625, y=0.75H
    # Let's just generate a regular grid in barycentric coords and pick 11.
    
    grid_pts = []
    step = 0.25
    for i in range(5):
        for j in range(5 - i):
            u = i * step
            v = j * step
            w = 1.0 - u - v
            if w >= -1e-9:
                grid_pts.append([u, v])
                
    # We have 15 points in a 4-step grid. We need 11.
    # Let's pick the 11 that form a "diamond" or "hexagon" shape in the middle.
    # Or just take the first 11.
    # Better: Center them.
    # The centroid of the 15 points is (1/3, 1/3).
    # Let's select points closest to the center to maximize min area?
    # Actually, for min area, we want points spread out.
    # Let's use a specific known good configuration for 11 points if possible, 
    # or just use the 15-point grid and drop 4 corner ones?
    
    # Let's use a 5-point cross + 6 surrounding?
    # Simpler: Use the 15-point grid, but we need exactly 11.
    # Let's just return the first 11 of the 15-point grid, which is a triangular block.
    # This is a valid start.
    return np.array(grid_pts[:11])

def generate_symmetric_init():
    """
    Generate an initialization with D3 symmetry.
    1 center, 3 on axes, 6 in orbits. Total 10. Add 1 more?
    Let's try: 1 center, 1 on axis (3 pts), 1 interior (6 pts). Total 10.
    Add a 2nd interior point? Total 16.
    Add a 2nd on axis? Total 13.
    
    Let's try: 1 center, 2 on axis (6 pts total from these 2 sources? No.
    Point on axis: (1/3, 1/6) -> rotates to 3 points.
    Point on axis: (1/3, 1/12) -> rotates to 3 points.
    Point interior: (0.2, 0.2) -> rotates to 6 points.
    Center: 1 point.
    Total: 1 + 3 + 3 + 6 = 13. Too many.
    
    Let's try:
    1 Center
    1 Point on Axis (3 pts)
    1 Point Interior (6 pts)
    Total 10.
    We need 11.
    Maybe 1 Center, 1 Point on Axis (3 pts), 1 Point Interior (6 pts), and 1 Point on Boundary?
    
    For the sake of the affine search, let's just use the lattice init.
    """
    return generate_lattice_init()

def apply_affine(base_pts, params):
    """
    Apply an affine transformation to base_pts.
    params: [a, b, c, d, e, f]
    x' = a*x + b*y + e
    y' = c*x + d*y + f
    """
    a, b, c, d, e, f = params
    x = base_pts[:, 0]
    y = base_pts[:, 1]
    new_x = a * x + b * y + e
    new_y = c * x + d * y + f
    return np.column_stack([new_x, new_y])

def affine_obj(params, base_pts_bary):
    """
    Objective for affine parameters.
    We transform the base points (in Cartesian) and check if they are inside the triangle.
    If inside, return -min_area.
    If outside, return a penalty.
    """
    # Base points are in Barycentric. Convert to Cartesian for affine transform?
    # Affine transform is easier in Cartesian.
    base_cart = bary_to_cart(base_pts_bary)
    transformed = apply_affine(base_cart, params)
    
    # Check if points are inside the triangle
    bary = cart_to_bary(transformed)
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    # Penalty if any coordinate is negative (outside)
    min_coord = np.min(np.vstack([u, v, w]).flatten())
    if min_coord < -1e-6:
        # Return a large penalty
        return 1e10 - min_coord 
        
    # If slightly outside, clamp and penalize
    if min_coord < 0:
        penalty = -min_coord * 1000
    else:
        penalty = 0
        
    # Calculate min area
    min_area = calc_min_area_cartesian(transformed)
    
    return penalty - min_area

def refine_local(bary_init, time_limit=5.0):
    """
    Refine a configuration using local optimization.
    """
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    
    bary_current = bary_init.copy()
    
    # Stage 1: Nelder-Mead on hard min (robust to non-smoothness)
    try:
        result1 = minimize(
            hard_min_obj_bary,
            bary_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        bary_current = clamp_bary(result1.x.reshape(N, 2))
    except Exception:
        pass
        
    if time.time() - start_time > time_limit * 0.5:
        return bary_current
    
    # Stage 2: SLSQP polish with hard min for precise convergence
    try:
        result2 = minimize(
            hard_min_obj_bary,
            bary_current.flatten(),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-14}
        )
        bary_current = clamp_bary(result2.x.reshape(N, 2))
    except Exception:
        pass
        
    return bary_current

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_bary = None
    best_val = -np.inf
    
    def update_best(bary):
        nonlocal best_bary, best_val
        bary = clamp_bary(bary)
        cart = bary_to_cart(bary)
        val = calc_min_area_cartesian(cart)
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    # 1. Standard Lattice Init + Local Refine
    lat_init = generate_lattice_init()
    refined_lat = refine_local(lat_init, time_limit=5.0)
    update_best(refined_lat)

    # 2. Structural Change: Affine Deformation of the Lattice
    # We search for the best affine transform (scale, rotation, shear, translation)
    # applied to the base lattice. This is a 6D search, much faster than 22D.
    if time.time() < deadline - 60:
        base_pts_bary = generate_lattice_init()
        
        # Bounds for affine parameters
        # a, b, c, d: scaling/shear. Keep them around 1.
        # e, f: translation. Keep them within triangle bounds roughly.
        affine_bounds = [
            (-0.5, 2.0),  # a
            (-0.5, 2.0),  # b
            (-0.5, 2.0),  # c
            (-0.5, 2.0),  # d
            (-0.5, 1.5),  # e
            (-0.5, 1.5)   # f
        ]
        
        try:
            # Use Differential Evolution for the 6D affine search
            result_de = differential_evolution(
                affine_obj,
                affine_bounds,
                args=(base_pts_bary,),
                seed=42,
                maxiter=50,
                popsize=15,
                tol=1e-6,
                polish=False
            )
            
            best_affine_params = result_de.x
            # Apply the best affine transform to get a good starting point
            base_cart = bary_to_cart(base_pts_bary)
            transformed_cart = apply_affine(base_cart, best_affine_params)
            
            # Convert back to barycentric
            affine_bary = cart_to_bary(transformed_cart)
            affine_bary = clamp_bary(affine_bary)
            
            # Refine this affine-derived point with full local search
            refined_affine = refine_local(affine_bary, time_limit=10.0)
            update_best(refined_affine)
            
        except Exception:
            pass

    # 3. Multi-start: Random and Perturbed
    rng = np.random.default_rng(123)
    restart_count = 0
    while time.time() < deadline - 15:
        restart_count += 1
        if best_bary is not None and restart_count % 2 == 0:
            scale = 0.05
            noise = rng.normal(0, scale, best_bary.shape)
            init = clamp_bary(best_bary + noise)
        else:
            u = rng.random(N)
            v = rng.random(N) * (1 - u)
            init = np.column_stack([u, v])
        
        remaining = deadline - 15 - time.time()
        if remaining < 3:
            break
            
        t_limit = min(5.0, remaining)
        refined = refine_local(init, time_limit=t_limit)
        update_best(refined)
        
    # Final polish
    if best_bary is not None and time.time() < deadline - 5:
        try:
            bounds = [(0, 1)] * (N * 2)
            result_slsqp = minimize(
                hard_min_obj_bary,
                best_bary.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-15}
            )
            update_best(clamp_bary(result_slsqp.x.reshape(N, 2)))
        except Exception:
            pass

    cart = bary_to_cart(best_bary)
    return cart
