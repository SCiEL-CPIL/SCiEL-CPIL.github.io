import numpy as np
from scipy.optimize import linprog, minimize
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
AREA_TOTAL = np.sqrt(3) / 4.0

def bary_to_cart(bary):
    """Convert barycentric coords (u, v) to Cartesian (x, y)."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    return np.column_stack([x, y])

def cart_to_bary(x, y):
    """Convert Cartesian to barycentric (u, v) where w = 1-u-v."""
    w = y / H
    v = x - 0.5 * w
    u = 1.0 - w - v
    return np.column_stack([u, v])

def calc_areas_bary(bary):
    """Calculate areas of all triangles formed by N points in barycentric coords."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    ui = u[tri_idx[:, 0]]
    vi = v[tri_idx[:, 0]]
    wi = w[tri_idx[:, 0]]
    
    uj = u[tri_idx[:, 1]]
    vj = v[tri_idx[:, 1]]
    wj = w[tri_idx[:, 1]]
    
    uk = u[tri_idx[:, 2]]
    vk = v[tri_idx[:, 2]]
    wk = w[tri_idx[:, 2]]
    
    det = ui * (vj * wk - vk * wj) - vi * (uj * wk - uk * wj) + wi * (uj * vk - uk * vj)
    
    return np.abs(det)

def clamp_bary(bary, eps=1e-9):
    """Clamp barycentric coordinates to the valid triangle region."""
    u = bary[:, 0].copy()
    v = bary[:, 1]. copy()
    
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

def get_min_area(bary):
    """Return the minimum triangle area for a configuration."""
    areas = calc_areas_bary(bary)
    return np.min(areas)

def solve_lp_for_active_set(fixed_points, active_tris, t_guess):
    """
    Given a set of fixed points and a set of active triangles (assumed to have area t),
    solve for the positions of the remaining points to maximize t (or verify feasibility).
    
    Here we fix the area of active triangles to be exactly t, and require all other
    triangles to have area >= t.
    This is a linear programming problem in the coordinates of the free points.
    Variables: x_j, y_j for free points j.
    """
    free_indices = [i for i in range(N) if i not in fixed_points]
    if not free_indices:
        return None, t_guess

    n_free = len(free_indices)
    n_vars = 2 * n_free
    
    # Objective: maximize t. 
    # However, t is a parameter here. To find the max t, we usually iterate.
    # Alternatively, we can set up the LP to maximize the minimum area directly? 
    # No, LP requires linear objective.
    # Strategy: We assume the active triangles define the bottleneck.
    # We want to find if there exists a configuration where these triangles have area >= t
    # and all others have area >= t.
    # Actually, a better LP formulation for "Given fixed points, maximize min area of triangles involving free points"
    # is hard because of the min.
    
    # Let's use a different LP formulation:
    # Maximize t
    # Subject to:
    # Area(i, j, k) >= t for all i,j,k
    # This is not linear because Area is a bilinear/quadratic function of coordinates?
    # Wait, Area in barycentric coordinates is linear in the coordinates of the points?
    # det(P_i, P_j, P_k) is linear in each P.
    # If we fix 2 points, Area is linear in the 3rd.
    # If we fix 1 point, Area is bilinear in the other 2.
    # So standard LP only works if we fix N-2 points? No, that's too restrictive.
    
    # Let's stick to a hybrid: 
    # 1. Identify a "core" of fixed points (e.g., corners and some interior).
    # 2. Assume a specific set of triangles are the minimum.
    # 3. Solve for free points such that these triangles have area t, and others >= t.
    # This requires t to be a variable.
    # Variables: x_free, y_free, t
    # Constraints:
    # For active tris (involving free points): Area = t (or >= t, but equality is better for LP bounds)
    # For other tris: Area >= t
    # This is a Linear Program if the areas are linear in the variables.
    # Area(i, j, k) is linear in x_i, y_i, etc. ONLY if the other two points are fixed.
    # If a triangle involves 2 free points, its area is quadratic.
    
    # Therefore, strict LP only works if every triangle involves at most 1 free point.
    # This implies we must fix N-2 points.
    # Let's try fixing N-2 points (9 points) and optimizing 2 points.
    
    # But which 9 points to fix?
    # We can iterate over subsets of points to fix.
    # For N=11, fixing 9 means choosing 2 free points. C(11,2) = 55 pairs.
    # For each pair, we assume the 9 fixed points are in a "good" configuration (from previous best).
    # Then we solve for the 2 free points to maximize the min area.
    
    # Let's implement a function that, given 9 fixed points, optimizes the remaining 2.
    
    return None, t_guess

def optimize_two_free_points(fixed_indices, fixed_coords):
    """
    Given 9 fixed points, optimize the remaining 2 points to maximize the minimum triangle area.
    The minimum area will be determined by triangles involving the free points.
    Since only 2 points are free, any triangle involving both is bilinear.
    However, we can use a grid search or local optimization for these 2 points (4 variables).
    This is much faster than optimizing 22 variables.
    """
    free_indices = [i for i in range(N) if i not in fixed_indices]
    if len(free_indices) != 2:
        return None, -np.inf
        
    i1, i2 = free_indices
    
    # Initial guess for free points: centroid or previous best
    # We'll use a local Nelder-Mead on the 4 variables (x1,y1,x2,y2)
    
    def obj_2pt(params):
        bary = fixed_coords.copy()
        bary[i1] = params[0:2]
        bary[i2] = params[2:4]
        bary = clamp_bary(bary)
        # Check if points are inside
        if np.any(bary[:, 0] < -1e-6) or np.any(bary[:, 1] < -1e-6) or np.any(bary[:, 0]+bary[:, 1] > 1+1e-6):
            return 1e10
        areas = calc_areas_bary(bary)
        return -np.min(areas)
    
    # Initial guess: average of fixed points or random
    # Use centroid of fixed points as a start
    avg_fixed = np.mean(fixed_coords, axis=0)
    x0 = np.array([avg_fixed[0], avg_fixed[1], avg_fixed[0], avg_fixed[1]])
    
    try:
        res = minimize(obj_2pt, x0, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
        if res.success:
            bary = fixed_coords.copy()
            bary[i1] = res.x[0:2]
            bary[i2] = res.x[2:4]
            bary = clamp_bary(bary)
            val = get_min_area(bary)
            return bary, val
    except Exception:
        pass
    return None, -np.inf

def refine_local(bary_init, time_limit=5.0):
    """
    Standard local refinement using Nelder-Mead on full 22 variables.
    Used as a polish step.
    """
    start_time = time.time()
    
    def hard_min_obj(bary_flat):
        bary = bary_flat.reshape(N, 2)
        bary = clamp_bary(bary)
        areas = calc_areas_bary(bary)
        return -np.min(areas)
        
    try:
        result = minimize(
            hard_min_obj,
            bary_init.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        new_bary = clamp_bary(result.x.reshape(N, 2))
        return new_bary
    except Exception:
        return bary_init

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT

    best_bary = None
    best_val = -np.inf

    def update_best(bary):
        nonlocal best_bary, best_val
        bary = clamp_bary(bary)
        val = get_min_area(bary)
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    # 1. Generate initial configurations
    # Use a hexagonal/symmetric pattern as a strong start
    # For N=11, a good start is 3 corners, 3 edge midpoints, 3 edge quarters, 2 interior
    # Let's create a few diverse inits
    
    inits = []
    
    # Init 1: Symmetric
    # Corners: (0,0), (1,0), (0,1)
    # Edge midpoints: (0.5, 0), (0, 0.5), (0.25, 0.75)
    # Edge quarters: (0.25, 0), (0, 0.25), (0.5, 0.5) -- wait, (0.5, 0.5) is on BC?
    # BC is u=0.25, v=0.75 line? No.
    # Barycentric:
    # A: (1,0), B: (0,1), C: (0,0) ? No, let's stick to (u,v) where A=(1,0), B=(0,1), C=(0,0) is confusing.
    # Let's use:
    # P0: (1, 0)
    # P1: (0, 1)
    # P2: (0, 0)
    # P3: (0.5, 0)
    # P4: (0, 0.5)
    # P5: (0.25, 0.25) # Midpoint of BC? No, (0,1) to (0,0) is u=0. Midpoint is (0, 0.5).
    # Let's use standard barycentric grid.
    
    base_pts = [
        [1.0, 0.0], [0.0, 1.0], [0.0, 0.0], # Corners
        [0.5, 0.0], [0.0, 0.5], [0.25, 0.75], # Edge midpoints? 
        # Midpoint of AB (1,0)-(0,1) is (0.5, 0.5). 
        # Midpoint of AC (1,0)-(0,0) is (0.5, 0).
        # Midpoint of BC (0,1)-(0,0) is (0, 0.5).
        [0.5, 0.5], # Mid AB
        [0.25, 0.0], # Quarter AC
        [0.0, 0.25], # Quarter BC
        [0.1, 0.1], # Interior 1
        [0.4, 0.4], # Interior 2
    ]
    inits.append(np.array(base_pts))
    
    # Init 2: Perturbed random
    rng = np.random.default_rng(42)
    for _ in range(5):
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        inits.append(np.column_stack([u, v]))
        
    # Evaluate inits
    for init in inits:
        if time.time() > deadline - 20:
            break
        refined = refine_local(init, time_limit=min(5.0, deadline - time.time() - 10))
        update_best(refined)

    # 2. Combinatorial Search: Fix 9 points, optimize 2
    # This is the core "different strategy".
    # We iterate over pairs of points to leave free.
    # For each pair, we take the current best configuration, fix the other 9 points,
    # and optimize the 2 free points.
    
    if best_bary is not None:
        # Use the best configuration found so far as the base for fixed points
        base_config = best_bary.copy()
        
        # We might want to try different "anchors" if time permits, but let's start with the best.
        # Also, we can try perturbing the fixed points slightly to escape local minima.
        
        pairs_to_test = list(combinations(range(N), 2))
        
        # Shuffle pairs to diversify
        rng.shuffle(pairs_to_test)
        
        for i1, i2 in pairs_to_test:
            if time.time() > deadline - 15:
                break
            
            fixed_indices = [i for i in range(N) if i != i1 and i != i2]
            fixed_coords = base_config[fixed_indices].copy()
            
            # Optimization for these 2 free points
            # We use a slightly more robust local search for the 2 points
            # to ensure we don't get stuck.
            
            def obj_2pt(params):
                bary = base_config.copy()
                bary[i1] = params[0:2]
                bary[i2] = params[2:4]
                bary = clamp_bary(bary)
                if np.any(bary[:, 0] < -1e-6) or np.any(bary[:, 1] < -1e-6) or np.any(bary[:, 0]+bary[:, 1] > 1+1e-6):
                    return 1e10
                areas = calc_areas_bary(bary)
                return -np.min(areas)
            
            # Multi-start for the 2 points
            best_local_val = -np.inf
            best_local_bary = None
            
            # Start from current best positions
            starts = [
                np.array([base_config[i1][0], base_config[i1][1], base_config[i2][0], base_config[i2][1]]),
                np.array([0.5, 0.5, 0.5, 0.5]), # Center
                np.array([0.1, 0.1, 0.1, 0.1]), # Corner-ish
                np.array([0.9, 0.1, 0.1, 0.9]), # Other corner-ish
            ]
            
            # Add random starts
            for _ in range(3):
                starts.append(rng.random(4) * np.array([1, 0.5, 1, 0.5])) # Roughly valid bary
            
            for s in starts:
                if time.time() > deadline - 12:
                    break
                try:
                    res = minimize(obj_2pt, s, method='Nelder-Mead', options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10})
                    if res.fun < best_local_val:
                        best_local_val = res.fun
                        bary_candidate = base_config.copy()
                        bary_candidate[i1] = res.x[0:2]
                        bary_candidate[i2] = res.x[2:4]
                        bary_candidate = clamp_bary(bary_candidate)
                        # Verify
                        val = get_min_area(bary_candidate)
                        if -val < best_local_val:
                            best_local_val = -val
                            best_local_bary = bary_candidate
                except Exception:
                    continue
            
            if best_local_bary is not None:
                # Polish the result with full local search
                if time.time() < deadline - 10:
                    refined = refine_local(best_local_bary, time_limit=min(2.0, deadline - time.time() - 5))
                    update_best(refined)
                else:
                    update_best(best_local_bary)
    
    # 3. Final Global Polish
    if best_bary is not None and time.time() < deadline - 5:
        try:
            remaining = deadline - 5 - time.time()
            if remaining > 1:
                # One last Nelder-Mead on the full 22 variables
                def hard_min_obj(bary_flat):
                    bary = bary_flat.reshape(N, 2)
                    bary = clamp_bary(bary)
                    areas = calc_areas_bary(bary)
                    return -np.min(areas)
                    
                result_final = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-11, 'fatol': 1e-14, 'adaptive': True}
                )
                update_best(clamp_bary(result_final.x.reshape(N, 2)))
        except Exception:
            pass

    if best_bary is None:
        # Fallback
        best_bary = inits[0]

    cart = bary_to_cart(best_bary)
    
    return cart
