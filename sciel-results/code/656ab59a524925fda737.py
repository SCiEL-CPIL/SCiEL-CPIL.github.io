import numpy as np
from itertools import combinations
import time
from scipy.optimize import minimize, differential_evolution

N = 11
TIME_LIMIT = 145.0

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

def get_areas_cartesian(cart):
    """Compute all triangle areas in Cartesian coordinates."""
    p_i = cart[tri_idx[:, 0]]
    p_j = cart[tri_idx[:, 1]]
    p_k = cart[tri_idx[:, 2]]
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_k[:, 0] - p_i[:, 0]) * (p_j[:, 1] - p_i[:, 1])
    return 0.5 * np.abs(cross)

def calc_min_area_cartesian(points):
    """Calculate the minimum area of all triangles formed by N points."""
    return np.min(get_areas_cartesian(points))

def find_active_triangles(cart, k=5):
    """
    Find the k smallest triangles. These are the 'active constraints'
    that define the local optimum structure.
    """
    areas = get_areas_cartesian(cart)
    # Find indices of the k smallest areas
    # Use argsort to get the smallest ones
    smallest_indices = np.argsort(areas)[:k]
    return smallest_indices, areas[smallest_indices]

def active_set_newton(bary_init, max_iter=50, tol=1e-10, time_limit=10.0):
    """
    Newton-Raphson refinement on the active set.
    
    We assume the optimum lies in the manifold where the areas of the 
    active triangles are equal to a common value t, and t is maximized.
    
    Instead of solving the full KKT system directly (which is complex due to 
    inequality constraints), we use a penalty approach:
    Minimize F(x) = -t + lambda * sum( (A_i(x) - t)^2 ) for i in active set
    
    But a more direct structural approach:
    1. Identify active triangles A.
    2. Solve for x such that A_i(x) = A_j(x) for all i,j in A.
    3. Among these, find the one that maximizes A_i.
    
    This is equivalent to finding the critical points of the min-area function 
    restricted to the active set manifold.
    
    We implement this by minimizing the variance of the active areas while 
    pushing the mean area up.
    """
    start_time = time.time()
    bary = bary_init.copy()
    
    # Get initial active set
    cart = bary_to_cart(bary)
    active_idx, active_areas = find_active_triangles(cart, k=5)
    
    # We will optimize the positions to equalize these areas and increase them.
    # Objective: Minimize (variance of active areas) - alpha * (mean active area)
    # This drives the active areas to be equal and large.
    
    alpha = 10.0
    
    def objective(bary_flat):
        if time.time() - start_time > time_limit:
            return 1e10
        b = bary_flat.reshape(N, 2)
        b = clamp_bary(b)
        c = bary_to_cart(b)
        areas = get_areas_cartesian(c)
        
        # Extract active areas
        a_active = areas[active_idx]
        
        if np.any(a_active < 1e-12):
            return 1e10
            
        mean_a = np.mean(a_active)
        var_a = np.var(a_active)
        
        # We want to minimize variance and maximize mean
        # Scaling: variance is small, mean is ~0.01-0.05
        # Use log for mean to emphasize relative improvement? 
        # No, linear is better for Newton-like behavior if scaled correctly.
        # Let's use a weighted sum.
        return var_a - alpha * mean_a

    # Use L-BFGS-B for this sub-problem as it's smooth enough
    bounds = [(0, 1)] * (N * 2)
    
    try:
        res = minimize(objective, bary.flatten(), method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
        if res.success:
            bary_new = clamp_bary(res.x.reshape(N, 2))
            cart_new = bary_to_cart(bary_new)
            min_new = calc_min_area_cartesian(cart_new)
            min_old = calc_min_area_cartesian(bary_to_cart(bary))
            
            if min_new > min_old:
                return bary_new
    except Exception:
        pass
        
    return bary

def refine_local_active_set(bary_init, time_limit=15.0):
    """
    Iteratively refine using the active set method.
    1. Find active triangles.
    2. Optimize to equalize and increase them.
    3. Repeat with new active set.
    """
    start_time = time.time()
    bary = bary_init.copy()
    best_val = calc_min_area_cartesian(bary_to_cart(bary))
    
    for _ in range(10):
        if time.time() - start_time > time_limit:
            break
            
        # Find active set
        cart = bary_to_cart(bary)
        active_idx, _ = find_active_triangles(cart, k=5)
        
        # Run Newton-like refinement on this active set
        bary_new = active_set_newton(bary, max_iter=50, time_limit=min(5.0, time_limit - (time.time() - start_time)))
        
        # Check improvement
        val_new = calc_min_area_cartesian(bary_to_cart(bary_new))
        if val_new > best_val + 1e-12:
            best_val = val_new
            bary = bary_new
        else:
            # If no improvement, break or try smaller active set?
            # Try with k=3
            bary_new2 = active_set_newton(bary, max_iter=50, time_limit=min(5.0, time_limit - (time.time() - start_time)))
            # active_set_newton uses global active_idx, so we need to pass it or restructure
            # For simplicity, if k=5 fails, we might just stop or do a generic polish.
            break
            
    # Final polish with standard Nelder-Mead on the hard min to clean up
    if time.time() - start_time < time_limit - 1:
        try:
            res = minimize(lambda x: -calc_min_area_cartesian(bary_to_cart(clamp_bary(x.reshape(N,2)))), 
                           bary.flatten(), method='Nelder-Mead', 
                           options={'maxiter': 200, 'xatol': 1e-12, 'fatol': 1e-14})
            bary_polished = clamp_bary(res.x.reshape(N, 2))
            val_polished = calc_min_area_cartesian(bary_to_cart(bary_polished))
            if val_polished > best_val:
                bary = bary_polished
                best_val = val_polished
        except Exception:
            pass
            
    return bary

def generate_structured_init():
    """
    Generate a structured initial guess based on known good configurations 
    or symmetric grids.
    """
    # 11 points in an equilateral triangle.
    # A 4x4 grid has 10 points. Adding a center point gives 11.
    # Or a hexagonal packing.
    
    # Let's try a few deterministic patterns
    
    # Pattern 1: Grid
    # 4 rows: 4, 3, 2, 1 points? No, that's 10.
    # 5 rows: 5, 4, 3, 2, 1? That's 15.
    
    # Let's use a hexagonal lattice subset.
    # Side length 3 hex grid has 19 points. We need 11.
    # Center + 6 surrounding + 4 more?
    
    # Simpler: Random but seeded, then refined.
    # But we want a "structured" start to help the active set method.
    
    # Try a "cross" pattern or "wheel" pattern.
    
    # Wheel: Center + 10 on perimeter?
    # Perimeter of unit triangle: 3 points at vertices + 8 distributed?
    
    # Let's generate a few candidates and pick the best one after a quick refine.
    candidates = []
    
    # Candidate 1: Centroid + 10 random
    rng = np.random.default_rng(42)
    for _ in range(20):
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        bary = np.column_stack([u, v])
        # Force one point to centroid
        bary[0] = [1/3, 1/3]
        bary = clamp_bary(bary)
        candidates.append(bary)
        
    # Candidate 2: Symmetric grid
    # 4x4 grid in barycentric?
    # u, v in {0, 1/3, 2/3, 1} such that u+v<=1
    # Points:
    # (0,0), (1/3,0), (2/3,0), (1,0)
    # (0,1/3), (1/3,1/3), (2/3,1/3)
    # (0,2/3), (1/3,2/3)
    # (0,1)
    # Total: 4+3+2+1 = 10.
    # Add center (1/3, 1/3) is already there.
    # We need 11. Add midpoint of a side? (1/2, 0)
    grid_pts = [
        [0.0, 0.0], [1/3, 0.0], [2/3, 0.0], [1.0, 0.0],
        [0.0, 1/3], [1/3, 1/3], [2/3, 1/3],
        [0.0, 2/3], [1/3, 2/3],
        [0.0, 1.0]
    ]
    # This is 10 points. Add (0.5, 0.0)
    grid_pts.append([0.5, 0.0])
    bary_grid = np.array(grid_pts)
    bary_grid = clamp_bary(bary_grid)
    candidates.append(bary_grid)
    
    # Candidate 3: Perturbed Grid
    bary_pert = bary_grid + 0.01 * rng.random((N, 2))
    bary_pert = clamp_bary(bary_pert)
    candidates.append(bary_pert)
    
    # Evaluate candidates quickly
    best_bary = None
    best_val = -np.inf
    for b in candidates:
        val = calc_min_area_cartesian(bary_to_cart(b))
        if val > best_val:
            best_val = val
            best_bary = b
            
    return best_bary

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT

    best_bary = None
    best_val = -np.inf

    def update_best(bary):
        nonlocal best_bary, best_val
        bary = clamp_bary(bary)
        val = calc_min_area_cartesian(bary_to_cart(bary))
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    # 1. Generate structured initial guesses
    init_bary = generate_structured_init()
    update_best(init_bary)
    
    # 2. Global Search using Differential Evolution (but shorter, to save time for AS)
    # We use a soft-min objective for DE to guide it towards good regions
    def soft_min_obj(bary_flat, beta=50.0):
        bary = bary_flat.reshape(N, 2)
        bary = clamp_bary(bary)
        areas = get_areas_cartesian(bary_to_cart(bary))
        if np.any(areas < 1e-15):
            return 1e10
        scaled = -beta * areas
        max_scaled = np.max(scaled)
        if max_scaled > 700:
            return -np.min(areas)
        log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
        soft_min = -log_sum / beta
        return -soft_min

    bounds = [(0, 1)] * (N * 2)
    
    # Run DE for a limited time to get a diverse set of starting points
    de_start = time.time()
    de_deadline = de_start + 30.0  # 30 seconds for DE
    
    def de_callback(xk, convergence):
        return time.time() > de_deadline
    
    try:
        de_result = differential_evolution(
            soft_min_obj,
            bounds,
            args=(50.0,),
            maxiter=50,
            popsize=15,
            tol=1e-6,
            seed=42,
            polish=False,
            init='sobol',
            callback=de_callback
        )
        
        if de_result.x is not None:
            de_bary = clamp_bary(de_result.x.reshape(N, 2))
            update_best(de_bary)
            
            # Also take some intermediate results from DE history if available?
            # DE doesn't easily expose history in standard scipy version without custom callback.
            # We'll just use the final result.
            
    except Exception as e:
        pass

    # 3. Active Set Refinement on the best candidate so far
    if best_bary is not None:
        remaining_time = deadline - time.time() - 5.0 # 5s safety
        if remaining_time > 10:
            refined_bary = refine_local_active_set(best_bary, time_limit=min(40.0, remaining_time))
            update_best(refined_bary)

    # 4. Try Active Set on the initial structured guess if it was different and potentially better
    # (In case DE moved us to a worse local minimum)
    if time.time() < deadline - 15:
        refined_init = refine_local_active_set(init_bary, time_limit=min(20.0, deadline - 15 - time.time()))
        update_best(refined_init)

    # 5. Random restarts with Active Set if time remains
    if time.time() < deadline - 20:
        for seed in [1, 2, 3, 4, 5]:
            if time.time() > deadline - 10:
                break
            rng = np.random.default_rng(seed)
            u = rng.random(N)
            v = rng.random(N) * (1 - u)
            rand_bary = np.column_stack([u, v])
            rand_bary = clamp_bary(rand_bary)
            
            # Quick DE polish or just AS?
            # AS is fast, let's try it directly on random starts
            refined_rand = refine_local_active_set(rand_bary, time_limit=min(10.0, deadline - 10 - time.time()))
            update_best(refined_rand)

    # 6. Final Polish
    if best_bary is not None and time.time() < deadline - 5:
        # One more round of AS on the global best
        final_bary = refine_local_active_set(best_bary, time_limit=min(10.0, deadline - 5 - time.time()))
        update_best(final_bary)

    if best_bary is None:
        # Fallback
        rng = np.random.default_rng(42)
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        best_bary = np.column_stack([u, v])

    cart = bary_to_cart(best_bary)
    return cart

if __name__ == "__main__":
    pts = construct()
    print(pts)
