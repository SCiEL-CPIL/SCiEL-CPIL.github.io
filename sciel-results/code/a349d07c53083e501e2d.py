import numpy as np
from scipy.optimize import differential_evolution, minimize
from itertools import combinations
import time

N = 11
TIME_LIMIT = 145.0  # Safety margin for 150s limit (evaluator kills at 150s, we want to return well before)

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2
AREA_EQUI = 0.5 * H  # Area of the unit equilateral triangle

def bary_to_cart(bary):
    """Convert barycentric coords (u, v) to Cartesian (x, y)."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    return np.column_stack([x, y])

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

def hard_min_obj(x):
    """
    Objective function for minimization.
    We want to maximize the minimum area, so we minimize the negative of the minimum area.
    """
    bary = x.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def generate_hexagonal_init():
    """
    Generate a hexagonal grid initialization.
    This provides a structured, spread-out starting point.
    """
    # A 3x3 grid in barycentric coordinates (10 points) plus one more
    # Let's try a 4x4 grid subset or a specific hexagonal pattern.
    # For N=11, a common good configuration is close to a hexagonal packing.
    
    # Let's place points on a triangular lattice.
    # Side length of the big triangle is 1.
    # We can divide the triangle into smaller triangles.
    
    # Simple heuristic: Place points on a grid and then optimize.
    # Let's use a 3x3 grid (10 points) and add the centroid? No, centroid is (1/3, 1/3).
    # Grid points (i/3, j/3) for i,j >= 0, i+j <= 3.
    # (0,0), (1/3,0), (2/3,0), (1,0)
    # (0,1/3), (1/3,1/3), (2/3,1/3)
    # (0,2/3), (1/3,2/3)
    # (0,1)
    # Total 10 points.
    
    pts = []
    for i in range(4):
        for j in range(4 - i):
            pts.append([i/3.0, j/3.0])
    
    # We have 10 points. We need 11.
    # Add a point near the center but not overlapping.
    # Or just perturb one.
    # Let's add a point at (1/2, 1/4)? No, let's just add a random point or shift the grid slightly.
    # Actually, for N=11, the optimal solution is known to be close to a specific configuration.
    # Let's just use the 10 grid points and one random point, or duplicate one and let DE separate them.
    # Better: Use a 4x4 grid? 1+2+3+4 = 10 points on lines.
    # Let's just use the 10 points and add a point at the centroid of the central triangle.
    # The central triangle is formed by (1/3,0), (0,1/3), (1/3,1/3)? No.
    # Let's just append a random point to the 10 grid points.
    rng = np.random.default_rng(42)
    u = rng.random()
    v = rng.random() * (1 - u)
    pts.append([u, v])
    
    bary = np.array(pts)
    return clamp_bary(bary)

def refine_local(bary, time_limit=5.0):
    """
    Perform a local refinement using Nelder-Mead to polish the solution.
    """
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    
    try:
        res = minimize(
            hard_min_obj,
            bary.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
        )
        candidate_bary = clamp_bary(res.x.reshape(N, 2))
        return candidate_bary
    except Exception:
        return bary

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # 1. Global Search using Differential Evolution
    # Bounds for barycentric coordinates: u in [0,1], v in [0,1]
    # We must ensure u+v <= 1. DE does not handle this constraint natively.
    # We can penalize violations in the objective function or use a transformation.
    # A common trick: use parameters p_i, q_i and map to barycentric via:
    # u = p / (p + q + r), v = q / (p + q + r) where p,q,r > 0.
    # Or simply penalize the violation heavily in the objective.
    
    def de_objective(x):
        bary = x.reshape(N, 2)
        # Penalize if points are outside the triangle
        u = bary[:, 0]
        v = bary[:, 1]
        sums = u + v
        penalty = np.sum(np.maximum(0, sums - 1.0 + 1e-6)**2) * 1e6
        # Also penalize if u or v are outside [0,1] (though bounds handle this)
        bary = clamp_bary(bary)
        areas = calc_areas_bary(bary)
        min_area = np.min(areas)
        return penalty - min_area
    
    # Bounds for DE: u in [0, 1], v in [0, 1]
    bounds = [(0.0, 1.0)] * (N * 2)
    
    best_val = -np.inf
    best_bary = generate_hexagonal_init()
    best_val = -hard_min_obj(best_bary.flatten())
    
    # Run DE for a significant portion of the time
    # DE can be slow. We need to limit the number of generations or use a time check.
    # scipy's differential_evolution does not have a built-in time limit.
    # We can run it in a background thread or just run it with a limited number of generations.
    # Let's run it for a fixed number of iterations that fits within ~60-80s.
    
    # Strategy:
    # 1. Run DE for ~50s.
    # 2. Take the best result from DE.
    # 3. Run local polish for the remaining time.
    
    de_start = time.time()
    de_deadline = de_start + 80.0  # 80 seconds for DE
    
    try:
        # Use a smaller population and fewer generations to fit in the time
        # popsize=10, maxiter=100 might be enough for a rough global search
        res_de = differential_evolution(
            de_objective,
            bounds,
            seed=42,
            maxiter=100,  # Limit iterations
            popsize=10,   # Smaller population
            tol=1e-8,
            init='sobol'
        )
        
        if res_de.success or res_de.fun < best_val:
            candidate_bary = clamp_bary(res_de.x.reshape(N, 2))
            candidate_val = -hard_min_obj(candidate_bary.flatten())
            if candidate_val > best_val:
                best_val = candidate_val
                best_bary = candidate_bary
    except Exception as e:
        # If DE fails or takes too long, we still have best_bary from init
        pass
    
    # 2. Local Polish
    if time.time() < deadline - 5.0:
        # Use the best found so far to polish
        polished_bary = refine_local(best_bary, time_limit=10.0)
        polished_val = -hard_min_obj(polished_bary.flatten())
        
        if polished_val > best_val:
            best_val = polished_val
            best_bary = polished_bary
            
    # 3. If time remains, try a few more random restarts + polish
    while time.time() < deadline - 10.0:
        # Generate a random start
        rng = np.random.default_rng()
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        rand_bary = clamp_bary(np.column_stack([u, v]))
        
        # Polish it
        polished_rand = refine_local(rand_bary, time_limit=5.0)
        rand_val = -hard_min_obj(polished_rand.flatten())
        
        if rand_val > best_val:
            best_val = rand_val
            best_bary = polished_rand
        else:
            # If not improving, break to save time for final safety
            # But maybe try a few more?
            # Let's just break if we've tried a couple of times without improvement
            # For simplicity, we'll just let the loop run until deadline - 10
            pass
    
    # Final check and return
    cart = bary_to_cart(best_bary)
    return cart

# EVOLVE-BLOCK-END
