import numpy as np
from scipy.optimize import minimize, linprog
from itertools import combinations
import time

N = 11
TIME_LIMIT = 140.0

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2

# Pre-compute all triangle indices
tri_indices = np.array(list(combinations(range(N), 3)))
NUM_TRIS = len(tri_indices)

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
    """Calculate the minimum area of all triangles formed by N points."""
    p_i = points[tri_indices[:, 0]]
    p_j = points[tri_indices[:, 1]]
    p_k = points[tri_indices[:, 2]]
    
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_k[:, 0] - p_i[:, 0]) * (p_j[:, 1] - p_i[:, 1])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def calc_areas_cartesian(points):
    """Calculate all areas of triangles formed by N points."""
    p_i = points[tri_indices[:, 0]]
    p_j = points[tri_indices[:, 1]]
    p_k = points[tri_indices[:, 2]]
    
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_k[:, 0] - p_i[:, 0]) * (p_j[:, 1] - p_i[:, 1])
    return 0.5 * np.abs(cross)

def solve_max_min_area_for_topology(active_indices, time_limit=5.0):
    """
    Given a set of 'active' triangle indices (believed to be the minimum),
    find the configuration that maximizes the minimum area among these active triangles,
    subject to all points being inside the triangle.
    
    We use a bisection search on the minimum area A.
    For a fixed A, we check if there exists a configuration such that:
    1. All points are in the triangle.
    2. Area(i, j, k) >= A for all (i, j, k) in active_indices.
    
    Since area is non-linear, we cannot use linprog directly on the full problem.
    However, we can use a local refinement approach:
    1. Start with a good initial guess (e.g., hexagonal).
    2. Identify the active triangles (smallest 3-5).
    3. Use SLSQP to maximize the min area of these specific triangles, 
       while penalizing if any OTHER triangle becomes smaller than the target.
       
    Actually, a more robust "structural" approach for a specific topology:
    We treat the problem as: Maximize t
    Subject to: Area(i,j,k) >= t for all (i,j,k) in ActiveSet
                Points in Triangle
    
    This is a Non-Linear Program (NLP). We use SLSQP with constraints.
    """
    start_time = time.time()
    
    # Initial guess: Hexagonal lattice
    # Generate a dense hexagonal grid and pick N points
    pts = []
    scale = 0.25
    max_k = int(1.0 / scale) + 1
    for k in range(max_k + 1):
        w_val = k * scale
        if w_val > 1.0: break
        num_pts_in_row = int((1 - w_val) / scale) + 1
        for j in range(num_pts_in_row):
            u_val = j * scale
            v_val = (1 - w_val) - u_val
            if u_val >= -1e-9 and v_val >= -1e-9 and u_val + v_val <= 1 + 1e-9:
                pts.append([u_val, v_val])
    if len(pts) < N:
        scale = 0.2
        pts = []
        max_k = int(1.0 / scale) + 1
        for k in range(max_k + 1):
            w_val = k * scale
            if w_val > 1.0: break
            num_pts_in_row = int((1 - w_val) / scale) + 1
            for j in range(num_pts_in_row):
                u_val = j * scale
                v_val = (1 - w_val) - u_val
                if u_val >= -1e-9 and v_val >= -1e-9 and u_val + v_val <= 1 + 1e-9:
                    pts.append([u_val, v_val])
    
    if len(pts) < N:
        return None, -1e10
        
    # Randomly select N points from the grid as init
    rng = np.random.default_rng(42)
    indices = rng.choice(len(pts), size=N, replace=False)
    bary_init = np.array([pts[i] for i in indices])
    
    # Define the optimization problem
    # Variables: x, y for each point (2N vars)
    # Objective: Maximize t (min area)
    # We optimize -t
    
    def objective(vars):
        cart = vars.reshape(N, 2)
        # Check bounds
        if np.any(cart[:, 0] < -1e-9) or np.any(cart[:, 0] > 1 + 1e-9) or np.any(cart[:, 1] < -1e-9) or np.any(cart[:, 1] > H + 1e-9):
            return 1e10
        # Check triangle interior
        # y <= sqrt(3) * x for x < 0.5
        # y <= sqrt(3) * (1-x) for x >= 0.5
        # Simpler: use barycentric check
        bary = cart_to_bary(cart)
        if np.any(bary < -1e-9) or np.any(bary > 1 + 1e-9):
            return 1e10
            
        areas = calc_areas_cartesian(cart)
        active_areas = areas[active_indices]
        min_active = np.min(active_areas)
        
        # Penalty if any inactive triangle is smaller than min_active
        # This ensures the "topology" (set of minimum triangles) is respected
        # If an inactive triangle becomes smaller, it becomes "active", 
        # which changes the topology. We want to stay in this topology basin.
        # A soft penalty helps transition, but for "solving for topology", 
        # we might want a hard constraint or strong penalty.
        
        all_min = np.min(areas)
        # We want to maximize the global min. 
        # If we constrain to active set, we might miss a better solution where topology changes.
        # So, let's just maximize the global min, but initialize near this topology.
        # This is effectively just local optimization.
        
        # To make this "structural", we should try to enforce that the active set 
        # remains the set of minimums.
        # Let's return -min_active + penalty * max(0, min_active - min_inactive)
        inactive_areas = np.delete(areas, active_indices)
        if len(inactive_areas) > 0:
            min_inactive = np.min(inactive_areas)
            # If any inactive is smaller than min_active, penalize
            # This encourages the active set to be the true minimum set.
            penalty = 0.0
            if min_inactive < min_active:
                penalty = (min_active - min_inactive) ** 2
            return -min_active + penalty
        else:
            return -min_active

    def constraints(vars):
        cart = vars.reshape(N, 2)
        # Points inside triangle
        # Inequality constraints: g(vars) >= 0
        # 1. y >= 0
        # 2. y <= sqrt(3) x  => sqrt(3) x - y >= 0
        # 3. y <= sqrt(3) (1-x) => sqrt(3) (1-x) - y >= 0
        
        cons = []
        for i in range(N):
            x, y = cart[i]
            # y >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[2*i+1]})
            # sqrt(3) x - y >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: np.sqrt(3)*v[2*i] - v[2*i+1]})
            # sqrt(3) (1-x) - y >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: np.sqrt(3)*(1-v[2*i]) - v[2*i+1]})
            
        return cons

    x0 = bary_to_cart(bary_init).flatten()
    
    try:
        res = minimize(objective, x0, method='SLSQP', constraints=constraints, 
                       bounds=[(-0.1, 1.1), (-0.1, H+0.1)] * N,
                       options={'maxiter': 500, 'ftol': 1e-9})
        
        if res.success or (hasattr(res, 'fun') and res.fun < 1e9):
            cart_best = res.x.reshape(N, 2)
            # Clamp to valid region
            bary_best = cart_to_bary(cart_best)
            bary_best = np.clip(bary_best, 1e-9, 1-1e-9)
            # Re-normalize
            sums = bary_best[:, 0] + bary_best[:, 1]
            invalid = sums > (1 - 1e-9)
            if np.any(invalid):
                scale = (1 - 1e-9) / sums[invalid]
                bary_best[invalid, 0] *= scale
                bary_best[invalid, 1] *= scale
            
            cart_final = bary_to_cart(bary_best)
            val = calc_min_area_cartesian(cart_final)
            return cart_final, val
        else:
            return None, -1e10
            
    except Exception as e:
        return None, -1e10

def identify_active_triangles(cart, k=5):
    """Identify the k smallest triangles."""
    areas = calc_areas_cartesian(cart)
    idx = np.argsort(areas)
    return idx[:k]

def genetic_search(deadline):
    """
    Structural search using a Genetic Algorithm over topologies.
    1. Generate population of random configurations.
    2. For each, identify active triangles (topology).
    3. Solve for the best configuration for that topology.
    4. Select best topologies and mutate (swap points slightly, or random restart within topology).
    """
    start_time = time.time()
    population = []
    
    # Initial population: Hexagonal variants + Random
    def generate_hex_init(scale, offset=0.0):
        pts = []
        max_k = int(1.0 / scale) + 1
        for k in range(max_k + 1):
            w_val = k * scale + offset
            if w_val > 1.0: break
            num_pts_in_row = int((1 - w_val) / scale) + 1
            for j in range(num_pts_in_row):
                u_val = j * scale + offset
                v_val = (1 - w_val) - u_val
                if u_val >= -1e-9 and v_val >= -1e-9 and u_val + v_val <= 1 + 1e-9:
                    pts.append([u_val, v_val])
        if len(pts) >= N:
            rng = np.random.default_rng(42)
            indices = rng.choice(len(pts), size=N, replace=False)
            return np.array([pts[i] for i in indices])
        return None

    # Add some hexagonal inits
    for scale in [0.2, 0.25, 0.3]:
        for offset in [0.0, 0.05]:
            bary = generate_hex_init(scale, offset)
            if bary is not None:
                cart = bary_to_cart(bary)
                val = calc_min_area_cartesian(cart)
                active = identify_active_triangles(cart)
                population.append((cart, val, active))
                
    # Add random inits
    for i in range(10):
        rng = np.random.default_rng(i)
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        bary = np.column_stack([u, v])
        cart = bary_to_cart(bary)
        val = calc_min_area_cartesian(cart)
        active = identify_active_triangles(cart)
        population.append((cart, val, active))
        
    best_val = -1e10
    best_cart = None
    
    # Sort population by value
    population.sort(key=lambda x: x[1], reverse=True)
    
    # Iterative improvement
    while time.time() < deadline - 10:
        # Take top 5 topologies
        top_topologies = population[:5]
        
        new_population = []
        
        for cart, val, active in top_topologies:
            if time.time() > deadline - 10:
                break
                
            # Refine this specific topology
            refined_cart, refined_val = solve_max_min_area_for_topology(active, time_limit=5.0)
            
            if refined_cart is not None and refined_val > val:
                new_active = identify_active_triangles(refined_cart)
                new_population.append((refined_cart, refined_val, new_active))
                if refined_val > best_val:
                    best_val = refined_val
                    best_cart = refined_cart.copy()
            else:
                # Keep original if refinement failed
                new_population.append((cart, val, active))
                
        # Mutation: Create new topologies by perturbing points of best solutions
        if best_cart is not None:
            for _ in range(3):
                if time.time() > deadline - 10:
                    break
                # Perturb best cart
                noise = np.random.normal(0, 0.05, (N, 2))
                perturbed_cart = best_cart + noise
                # Clamp to triangle
                bary_p = cart_to_bary(perturbed_cart)
                bary_p = np.clip(bary_p, 1e-9, 1-1e-9)
                sums = bary_p[:, 0] + bary_p[:, 1]
                invalid = sums > (1 - 1e-9)
                if np.any(invalid):
                    scale = (1 - 1e-9) / sums[invalid]
                    bary_p[invalid, 0] *= scale
                    bary_p[invalid, 1] *= scale
                perturbed_cart = bary_to_cart(bary_p)
                
                val_p = calc_min_area_cartesian(perturbed_cart)
                active_p = identify_active_triangles(perturbed_cart)
                new_population.append((perturbed_cart, val_p, active_p))
                
                # Also try random restarts
                rng = np.random.default_rng(int(time.time()*1000) % 10000)
                u = rng.random(N)
                v = rng.random(N) * (1 - u)
                bary_r = np.column_stack([u, v])
                cart_r = bary_to_cart(bary_r)
                val_r = calc_min_area_cartesian(cart_r)
                active_r = identify_active_triangles(cart_r)
                new_population.append((cart_r, val_r, active_r))
                
        # Update population
        # Combine new_population with existing population, sort, take top 10
        combined = new_population + population[5:]
        combined.sort(key=lambda x: x[1], reverse=True)
        population = combined[:10]
        
        # Check for stagnation? If top 5 are same, break?
        # For now, just run until time limit.
        
    return best_cart, best_val

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_cart, best_val = genetic_search(deadline)
    
    if best_cart is None:
        # Fallback
        rng = np.random.default_rng(42)
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        bary = np.column_stack([u, v])
        best_cart = bary_to_cart(bary)
        
    return best_cart

if __name__ == "__main__":
    pts = construct()
    print(pts)
