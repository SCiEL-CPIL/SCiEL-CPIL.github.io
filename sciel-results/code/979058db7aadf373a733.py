import numpy as np
from scipy.optimize import minimize, differential_evolution
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

def get_min_area(bary):
    """Get the minimum area for a barycentric configuration."""
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return np.min(areas)

def smooth_min_obj(bary_flat, beta=100.0):
    """
    Smooth surrogate for the hard min via log-sum-exp.
    Minimizing this pushes ALL areas up, with emphasis on the smallest.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    if np.any(areas < 1e-15):
        return 1e10
        
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    
    if max_scaled > 700:
        return -np.min(areas)
        
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min = -log_sum / beta
    return -soft_min

def generate_hexagonal_init(k=3, seed=0):
    """
    Generate a hexagonal-like lattice of points.
    """
    rng = np.random.default_rng(seed)
    pts = []
    # Generate a grid of points in barycentric coordinates
    # Hexagonal lattice in barycentric:
    # Points are at (i/k, j/k) where i+j <= k, but shifted to form hexagons?
    # Standard lattice:
    for i in range(k + 1):
        for j in range(k - i + 1):
            # Add a small jitter to break symmetry and help optimization
            u = i / k + rng.normal(0, 0.01)
            v = j / k + rng.normal(0, 0.01)
            pts.append([u, v])
    
    pts = np.array(pts)
    pts = clamp_bary(pts)
    
    # If we have more than N points, we need to select N.
    # If we have less, we need to add more.
    # For k=3, we have 10 points. Add 1 in the center.
    if len(pts) < N:
        # Add points in the "holes"
        # Simple heuristic: add random points that are far from existing ones
        while len(pts) < N:
            u = rng.random()
            v = rng.random() * (1 - u)
            pt = np.array([u, v])
            # Check distance
            if len(pts) > 0:
                dists = np.linalg.norm(pts - pt, axis=1)
                if np.min(dists) > 0.1:
                    pts = np.vstack([pts, [pt]])
            else:
                pts = np.vstack([pts, [pt]])
    elif len(pts) > N:
        # Select N points that maximize the min area? 
        # For now, just take the first N or random N.
        # Better: take the ones closest to the "ideal" hex positions.
        # Since we generated a lattice, they are already good.
        # Just take a random subset? No, that might break the structure.
        # Let's just use the first N.
        pts = pts[:N]
        
    return pts

def refine_local(bary_init, time_limit=20.0):
    """
    Refine a configuration using L-BFGS-B with annealing beta.
    """
    start_time = time.time()
    bary_current = bary_init.copy()
    
    # Annealing schedule for beta
    betas = [10, 50, 100, 200, 500]
    
    for beta in betas:
        if time.time() - start_time > time_limit:
            break
            
        def obj(bary_flat):
            return smooth_min_obj(bary_flat, beta=beta)
        
        try:
            result = minimize(
                obj,
                bary_current.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-12}
            )
            bary_new = clamp_bary(result.x.reshape(N, 2))
            # Only accept if it improves the hard min area
            if get_min_area(bary_new) > get_min_area(bary_current):
                bary_current = bary_new
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
        val = get_min_area(bary)
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    # 1. Generate initial candidates
    candidates = []
    # Hexagonal initiations with different seeds and k values
    candidates.append(generate_hexagonal_init(k=3, seed=1))
    candidates.append(generate_hexagonal_init(k=4, seed=2))
    candidates.append(generate_hexagonal_init(k=3, seed=3))
    
    # Random initializations
    rng = np.random.default_rng(42)
    for _ in range(3):
        pts = np.zeros((N, 2))
        for i in range(N):
            while True:
                u = rng.random()
                v = rng.random() * (1 - u)
                pt = np.array([u, v])
                if i == 0:
                    break
                dists = np.linalg.norm(pts[:i] - pt, axis=1)
                if np.min(dists) > 0.15:
                    break
            pts[i] = pt
        candidates.append(pts)

    # 2. Refine initial candidates
    # Allocate 20% of time to initial refinement
    t_init = (TIME_LIMIT * 0.2) / len(candidates)
    for cand in candidates:
        if time.time() > deadline - 10:
            break
        refined = refine_local(cand, time_limit=t_init)
        update_best(refined)

    # 3. Differential Evolution for global search
    # Allocate 60% of time to DE
    t_de = TIME_LIMIT * 0.6
    bounds = [(0, 1), (0, 1)] * N
    # DE works in Cartesian? No, we defined smooth_min_obj on barycentric.
    # DE will optimize 22 variables.
    
    # We need to ensure the points are inside the triangle.
    # The clamp_bary in the objective handles this, but DE might waste time outside.
    # We can restrict bounds for v to (0, 1-u) but DE doesn't support coupled bounds.
    # So we rely on clamping.
    
    try:
        # Use a smaller population for speed
        # popsize=10 means 10 * 22 = 220 points per generation
        # maxiter=50
        # This might be too slow. Let's reduce.
        # popsize=5, maxiter=20
        de_result = differential_evolution(
            smooth_min_obj, 
            bounds, 
            args=(50,), # beta=50 for DE
            maxiter=20, 
            popsize=5, 
            tol=1e-4,
            seed=42,
            polish=False # We will polish manually
        )
        if de_result.success or de_result.fun < 1e9:
            bary_de = clamp_bary(de_result.x.reshape(N, 2))
            update_best(bary_de)
    except Exception as e:
        print(f"DE failed: {e}")

    # 4. Final Polish
    # Allocate remaining time to polish the best candidate
    if best_bary is not None and time.time() < deadline - 5:
        remaining = deadline - 5 - time.time()
        try:
            refined_best = refine_local(best_bary, time_limit=remaining)
            update_best(refined_best)
        except Exception:
            pass

    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart

# EVOLVE-BLOCK-END
