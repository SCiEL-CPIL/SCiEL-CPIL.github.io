import numpy as np
from itertools import combinations
import time
from scipy.optimize import differential_evolution

N = 11
TIME_LIMIT = 145.0

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)

V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2

def bary_to_cart(bary):
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    return np.column_stack([x, y])

def cart_to_bary(cart):
    x = cart[:, 0]
    y = cart[:, 1]
    w = y / H
    v = x - 0.5 * w
    u = 1.0 - v - w
    return np.column_stack([u, v])

def calc_areas_bary(bary):
    """
    Calculate areas of all triangles formed by points in barycentric coordinates.
    Returns an array of areas.
    """
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
    
    # Determinant of the 3x3 matrix with rows [ui, vi, wi], etc.
    # Area = 0.5 * |det|, but since we are comparing relative areas or using a consistent scale,
    # we can work with |det| directly or scale it. 
    # The actual area of the triangle in Cartesian coords is 0.5 * |det| * Area_of_Container.
    # Wait, the barycentric determinant gives the ratio of the area to the container triangle's area?
    # Let's check: 
    # If points are vertices of the container, e.g., (1,0,0), (0,1,0), (0,0,1).
    # Det of [[1,0,0],[0,1,0],[0,0,1]] is 1.
    # The area of the container is A_container.
    # So Area_triangle = 0.5 * |det| * A_container? 
    # Actually, the area of a triangle defined by barycentric coordinates P1, P2, P3 is:
    # Area = |det( [u1 v1 w1; u2 v2 w2; u3 v3 w3] )| * Area_Container
    # Let's verify with a mid-point triangle.
    # P1(1/2, 1/2, 0), P2(1/2, 0, 1/2), P3(0, 1/2, 1/2)
    # Det = 1/2*0*1/2 - 1/2*1/2*1/2 + 0... = -1/8. Abs = 1/8.
    # The medial triangle has 1/4 the area of the parent.
    # So Area = |det| * Area_Container? No, 1/8 * A_container is 1/8.
    # Let's re-calculate determinant of [[0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]
    # 0.5*(0*0.5 - 0.5*0.5) - 0.5*(0.5*0.5 - 0*0) + 0
    # 0.5*(-0.25) - 0.5*(0.25) = -0.125 - 0.125 = -0.25.
    # Abs = 0.25. Correct.
    # So Area = |det| * Area_Container.
    
    det = ui * (vj * wk - vk * wj) - vi * (uj * wk - uk * wj) + wi * (uj * vk - uk * vj)
    
    return np.abs(det)

def clamp_bary(bary, eps=1e-9):
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

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # The area of the equilateral triangle with side 1 is sqrt(3)/4.
    # The problem asks for "maximize the smallest area".
    # The evaluator measures the "raw value". 
    # Looking at the seed: "divides by its area".
    # So the score is likely (min_area / area_triangle).
    # min_area = min_det * area_triangle.
    # Score = min_det.
    # So we just need to maximize the minimum determinant.
    
    # Initial guess: Hexagonal packing is usually good for packing problems.
    # For 11 points, a hexagonal arrangement on a grid is a strong initial guess.
    # Let's generate a few initial candidates.
    
    def generate_hex_init(scale=1.0, offset=0.0):
        # 11 points in a triangle.
        # Rows of a hex grid: 1, 2, 3, 4, ... 
        # 1+2+3+4 = 10. Need 1 more. 
        # Or 2, 3, 4, 2? 
        # Let's just use a scaled grid of points and pick the best 11 or fit 11 into a grid.
        # A simple grid:
        # Row 0: 1 point
        # Row 1: 2 points
        # Row 2: 3 points
        # Row 3: 4 points
        # Row 4: 1 point? No, that's 11.
        # Let's place them in barycentric coords.
        # w = 0 (bottom), w=1/4, w=1/2, w=3/4, w=1 (top).
        # But w=1 is a vertex. 
        # Let's try to distribute them roughly evenly.
        
        points = []
        # 1 at bottom center? No, corners are usually good for maximizing min area?
        # Actually, for "maximize min area", pushing points to corners creates large triangles,
        # but the triangles involving interior points might be small.
        # The optimal solution for small N often puts points on the boundary.
        
        # Let's use a known good configuration for 11 points if possible, 
        # or generate a grid.
        # Grid approach:
        # Map integer grid (i, j) to barycentric.
        # i, j >= 0, i+j <= M.
        # For 11 points, M=3 gives 1+2+3+4=10 points. M=4 gives 15.
        # We can pick 11 from the M=4 grid (remove 4).
        
        candidates = []
        for i in range(5):
            for j in range(5 - i):
                candidates.append((i, j))
        
        # We have 15 candidates. We need to select 11.
        # The corners (0,0), (4,0), (0,4) are likely important.
        # Let's just take the first 11? Or randomize?
        # Let's take the ones that form a dense cluster in the middle? No, spread out.
        
        # Let's just use the 10 points from M=3 and add one in the center of the top row?
        # Or better, use a fixed set of 11 points from the M=4 grid that are spread out.
        # Indices in M=4 grid:
        # (0,0), (1,0), (2,0), (3,0), (4,0)
        # (0,1), (1,1), (2,1), (3,1)
        # (0,2), (1,2), (2,2)
        # (0,3), (1,3)
        # (0,4)
        
        # Let's select:
        # All 4 corners of the sub-triangles?
        # Let's just pick a random subset of 11 from the 15, scaled to fit.
        # But for a deterministic start, let's pick specific ones.
        # (0,0), (4,0), (0,4) [Corners]
        # (2,0), (0,2), (1,1) [Mid edges/center]
        # (1,0), (3,0), (0,1), (0,3), (2,1)
        
        selected = [(0,0), (4,0), (0,4), (2,0), (0,2), (1,1), (1,0), (3,0), (0,1), (0,3), (2,1)]
        
        M = 4.0
        bary = np.zeros((11, 2))
        for k, (i, j) in enumerate(selected):
            # u = i/M, v = j/M, w = 1 - (i+j)/M
            # This maps to the triangle with vertices (1,0,0), (0,1,0), (0,0,1)
            # Wait, (0,0) in grid -> u=0, v=0 -> w=1 -> Vertex 2 (Top).
            # (4,0) in grid -> u=1, v=0 -> w=0 -> Vertex 1 (Right).
            # (0,4) in grid -> u=0, v=1 -> w=0 -> Vertex 0 (Left).
            # This covers the whole triangle.
            bary[k, 0] = i / M
            bary[k, 1] = j / M
            
        return clamp_bary(bary)

    def generate_random_init(seed):
        rng = np.random.default_rng(seed)
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        return clamp_bary(np.column_stack([u, v]))

    # Feasibility Objective for DE
    # We want to find x such that min(areas) >= target.
    # Penalty = max(0, target - min(areas))
    # We minimize this penalty. If 0, it's feasible.
    
    def feasibility_obj(x, target):
        bary = x.reshape(N, 2)
        # Check bounds
        if np.any(bary < -1e-6) or np.any(bary > 1 + 1e-6):
            return 1e10
        if np.any(1 - bary[:, 0] - bary[:, 1] < -1e-6):
            return 1e10
            
        areas = calc_areas_bary(bary)
        min_area = np.min(areas)
        
        if min_area >= target:
            return 0.0
        else:
            return (target - min_area)

    # Start with a low target and increase it.
    # Max possible min area is roughly 1/11 * Total Area? No, packing density.
    # For 11 points, the best known value for "smallest triangle" is likely around 0.03-0.05?
    # Let's start with 0.01 and go up.
    
    best_bary = generate_hex_init()
    current_min = np.min(calc_areas_bary(best_bary))
    
    # Also try random init
    rand_bary = generate_random_init(42)
    rand_min = np.min(calc_areas_bary(rand_bary))
    if rand_min > current_min:
        best_bary = rand_bary
        current_min = rand_min
        
    # Bisection / Incremental search
    # We will try to find a configuration for increasing targets.
    # Target range: [0.0, 0.1]
    
    # We use DE to solve feasibility for a given target.
    # We start with the current best min area as the initial feasible target.
    
    target = current_min
    best_target_found = current_min
    best_bary_found = best_bary.copy()
    
    # We will try to push the target up.
    # Step size: 0.005 initially, then smaller.
    
    steps = [0.005, 0.005, 0.002, 0.002, 0.001, 0.001, 0.0005, 0.0005, 0.0002]
    
    de_bounds = [(0.0, 1.0)] * (2 * N)
    
    for step in steps:
        if time.time() > deadline - 10:
            break
            
        target += step
        
        # If target is too high, we might fail. 
        # We can check if target is > 0.1 (unlikely to be feasible)
        if target > 0.1:
            break
            
        try:
            de_result = differential_evolution(
                lambda x: feasibility_obj(x, target),
                bounds=de_bounds,
                seed=int(target * 10000), # Different seed for different targets
                maxiter=20,
                popsize=20,
                tol=1e-8,
                polish=False,
                init='sobol',
                x0=best_bary_found.flatten() # Warm start from previous best
            )
            
            # Check if feasible
            if de_result.fun < 1e-8:
                # Feasible!
                best_bary_found = clamp_bary(de_result.x.reshape(N, 2))
                best_target_found = target
                current_min = np.min(calc_areas_bary(best_bary_found))
                # Update current_min to be the actual min, which might be > target
                if current_min > best_target_found:
                     best_target_found = current_min
            else:
                # Not feasible. 
                # We could try a smaller step, but for simplicity, we break or continue?
                # If we continue with larger steps, we'll definitely fail.
                # We should probably break and refine the last feasible point.
                break
                
        except Exception:
            break

    # Final Polish: 
    # Now that we have a good configuration (best_bary_found) with min area ~ best_target_found,
    # we can use a local optimizer to maximize the min area directly.
    # This is the "Max-Min" problem again, but starting from a very good point.
    
    if time.time() < deadline - 5:
        def max_min_obj(x):
            bary = clamp_bary(x.reshape(N, 2))
            areas = calc_areas_bary(bary)
            return -np.min(areas)
            
        from scipy.optimize import minimize
        
        # Use Nelder-Mead or SLSQP for local refinement
        # SLSQP with constraints: Area(i,j,k) >= t
        # But we don't have t in x. We can just maximize -min(areas).
        # Nelder-Mead is derivative-free and robust for this.
        
        res = minimize(
            max_min_obj,
            best_bary_found.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12}
        )
        
        if res.fun < -best_target_found:
             best_bary_found = clamp_bary(res.x.reshape(N, 2))
             best_target_found = -res.fun

    cart = bary_to_cart(best_bary_found)
    return cart

# EVOLVE-BLOCK-END
