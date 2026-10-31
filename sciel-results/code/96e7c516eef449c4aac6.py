import numpy as np
from scipy.optimize import minimize, differential_evolution
from itertools import combinations
import time

N = 11
TIME_LIMIT = 138.0  # Safety margin for 150s limit

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

def hard_min_obj(bary_flat):
    """Objective using hard min."""
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def soft_min_obj(bary_flat, beta=50.0):
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

def generate_hexagonal_init():
    """
    Generate a hexagonal lattice-like initial configuration.
    """
    best_bary = None
    best_val = -np.inf
    
    for scale in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]:
        pts = []
        max_k = int(1.0 / scale) + 1
        for k in range(max_k + 1):
            w_val = k * scale
            if w_val > 1.0:
                break
            num_pts_in_row = int((1 - w_val) / scale) + 1
            for j in range(num_pts_in_row):
                u_val = j * scale
                v_val = (1 - w_val) - u_val
                if u_val >= -1e-9 and v_val >= -1e-9 and u_val + v_val <= 1 + 1e-9:
                    pts.append([u_val, v_val])
        
        if len(pts) >= N:
            # Try different subsets
            rng = np.random.default_rng(42)
            for _ in range(5):
                if len(pts) == N:
                    bary = np.array(pts)
                else:
                    idx = rng.choice(len(pts), N, replace=False)
                    bary = np.array(pts)[idx]
                val = -hard_min_obj(bary.flatten())
                if val > best_val:
                    best_val = val
                    best_bary = bary.copy()
                
    if best_bary is None:
        rng = np.random.default_rng(42)
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        best_bary = np.column_stack([u, v])
        
    return best_bary

def generate_symmetric_init():
    """
    Generate a configuration with 3-fold rotational symmetry.
    The equilateral triangle has 3-fold symmetry around its centroid.
    We place 1 point at centroid + groups of 3 orbitally symmetric points.
    11 = 1 (center) + 10 = 1 + 3*3 + 1? No, 11 = 1 + 10, 10 is not divisible by 3.
    Try: 2 center + 9 = 2 + 3*3, or 1 + 3*3 + 1 (two special)
    Better: use 3-fold symmetric orbits. 11 = 1 + 3*3 + 1 doesn't work cleanly.
    Let's try: place points in orbits of size 3, plus some on the symmetry axes.
    11 points: 1 at centroid, 3 at radius r1 (120 deg apart), 3 at radius r2, 3 at radius r3, 1 at another special location?
    Actually 11 = 1 + 3 + 3 + 3 + 1. The two "1"s can be at centroid and one on an axis.
    Or just do 1 + 3*3 = 10, plus 1 more. Let's try multiple patterns.
    """
    centroid = np.array([1.0/3, 1.0/3])
    
    def rotate_120(bary_pt):
        """Rotate a barycentric point by 120 degrees around centroid."""
        # In barycentric coords, 120-degree rotation permutes the coordinates cyclically
        # (u, v, w) -> (v, w, u) where w = 1-u-v
        u, v = bary_pt
        w = 1 - u - v
        return np.array([w, u, v][:2])  # (new_u, new_v) = (w, u)
    
    def rotate_240(bary_pt):
        u, v = bary_pt
        w = 1 - u - v
        return np.array([v, w])  # (new_u, new_v) = (v, w)
    
    best_bary = None
    best_val = -np.inf
    
    # Pattern 1: centroid + 3 orbits of 3
    for r1 in [0.1, 0.15, 0.2, 0.25, 0.3]:
        for r2 in [0.1, 0.15, 0.2, 0.25, 0.3]:
            for r3 in [0.1, 0.15, 0.2, 0.25, 0.3]:
                if r1 >= r2 or r2 >= r3:
                    continue
                pts = [centroid.copy()]
                for r in [r1, r2, r3]:
                    # Place a point at distance r from centroid along one direction, then rotate
                    # In barycentric, moving from centroid: u = 1/3 + r*cos(0), v = 1/3 - r*sin(0)*something
                    # Simpler: pick a reference point and rotate
                    ref = np.array([1/3 + r, 1/3 - r/2])
                    ref = clamp_bary(ref.reshape(1,2))[0]
                    pts.append(ref.copy())
                    pts.append(rotate_120(ref))
                    pts.append(rotate_240(ref))
                
                if len(pts) == 10:
                    # Add one more point - try on the axis of symmetry
                    for extra_r in [0.05, 0.1, 0.15, 0.2]:
                        extra = np.array([1/3, 1/3 + extra_r])
                        extra = clamp_bary(extra.reshape(1,2))[0]
                        all_pts = pts + [extra]
                        bary = np.array(all_pts)
                        val = -hard_min_obj(bary.flatten())
                        if val > best_val:
                            best_val = val
                            best_bary = bary.copy()
    
    # Pattern 2: 2 points on axis + 3 orbits of 3
    for r1 in [0.1, 0.15, 0.2, 0.25, 0.3]:
        for r2 in [0.1, 0.15, 0.2, 0.25, 0.3]:
            for r3 in [0.1, 0.15, 0.2, 0.25, 0.3]:
                if r1 >= r2 or r2 >= r3:
                    continue
                pts = []
                # Two points on the vertical axis of symmetry (u = 1/3)
                pts.append(np.array([1/3, 1/3]))  # centroid
                pts.append(np.array([1/3, 1/3 + 0.15]))
                pts = [clamp_bary(np.array([p]))[0] for p in pts]
                
                for r in [r1, r2, r3]:
                    ref = np.array([1/3 + r, 1/3 - r/2])
                    ref = clamp_bary(ref.reshape(1,2))[0]
                    pts.append(ref.copy())
                    pts.append(rotate_120(ref))
                    pts.append(rotate_240(ref))
                
                if len(pts) == 11:
                    bary = np.array(pts)
                    val = -hard_min_obj(bary.flatten())
                    if val > best_val:
                        best_val = val
                        best_bary = bary.copy()
    
    if best_bary is None:
        return generate_hexagonal_init()
    
    return best_bary

def generate_convex_polygon_init():
    """
    Generate points on a convex polygon (inscribed in the triangle) plus interior points.
    This helps ensure good coverage of the boundary.
    """
    best_bary = None
    best_val = -np.inf
    
    # Place points along the edges and interior
    # 11 points: 4 on each edge (shared vertices) + interior
    # Actually: 3 vertices + 2 on each edge (6) + 2 interior = 11
    
    # Pattern: vertices + edge midpoints + interior
    vertices = np.array([[0,0], [1,0], [0.5, 1-0.5]])  # barycentric
    # Edge midpoints
    edge_pts = [
        [0.5, 0],      # midpoint of edge 0-1
        [0.75, 0.25],  # midpoint of edge 1-2
        [0.25, 0.75],  # midpoint of edge 2-0
    ]
    
    for n_int in [2, 3, 4]:
        if 3 + 3 + n_int != 11:
            continue
        for cx in [0.3, 0.4, 0.5]:
            for cy in [0.3, 0.4, 0.5]:
                if cx + cy > 1:
                    continue
                interior = [[cx, cy]]
                # Add more interior points in a small cluster
                for i in range(1, n_int):
                    angle = 2 * np.pi * i / n_int
                    dx = 0.08 * np.cos(angle)
                    dy = 0.08 * np.sin(angle)
                    interior.append([cx + dx, cy + dy])
                
                all_pts = np.array(vertices.tolist() + edge_pts + interior)
                all_pts = clamp_bary(all_pts)
                val = -hard_min_obj(all_pts.flatten())
                if val > best_val:
                    best_val = val
                    best_bary = all_pts.copy()
    
    if best_bary is None:
        return generate_hexagonal_init()
    
    return best_bary

def generate_random_init(seed=None):
    """Generate random valid barycentric coordinates."""
    rng = np.random.default_rng(seed)
    u = rng.random(N)
    v = rng.random(N) * (1 - u)
    return np.column_stack([u, v])

def generate_perturbed_init(base_bary, sigma=0.05, seed=None):
    """Generate a perturbed version of a base configuration."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, base_bary.shape)
    return clamp_bary(base_bary + noise)

def refine_local(bary_init, time_limit=5.0):
    """
    Refine a configuration using multiple optimization stages with graduated soft-min.
    """
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    
    bary_current = bary_init.copy()
    
    # Stage 1: Coarse soft-min (low beta, more iterations)
    try:
        result1 = minimize(
            soft_min_obj,
            bary_current.flatten(),
            args=(20.0,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if result1.success or result1.fun < -hard_min_obj(bary_current.flatten()):
            bary_current = clamp_bary(result1.x.reshape(N, 2))
    except Exception:
        pass
    
    if time.time() - start_time > time_limit * 0.25:
        return bary_current

    # Stage 2: Medium soft-min (increasing beta)
    try:
        result2 = minimize(
            soft_min_obj,
            bary_current.flatten(),
            args=(50.0,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-13, 'gtol': 1e-9}
        )
        if result2.success or result2.fun < -hard_min_obj(bary_current.flatten()):
            bary_current = clamp_bary(result2.x.reshape(N, 2))
    except Exception:
        pass
        
    if time.time() - start_time > time_limit * 0.45:
        return bary_current

    # Stage 3: Sharp soft-min (high beta, close to hard min)
    try:
        result3 = minimize(
            soft_min_obj,
            bary_current.flatten(),
            args=(200.0,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        if result3.success or result3.fun < -hard_min_obj(bary_current.flatten()):
            bary_current = clamp_bary(result3.x.reshape(N, 2))
    except Exception:
        pass
        
    if time.time() - start_time > time_limit * 0.65:
        return bary_current

    # Stage 4: Hard min refinement with Nelder-Mead
    try:
        result4 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 1500, 'xatol': 1e-13, 'fatol': 1e-15, 'adaptive': True}
        )
        bary_current = clamp_bary(result4.x.reshape(N, 2))
    except Exception:
        pass
        
    if time.time() - start_time > time_limit * 0.85:
        return bary_current

    # Stage 5: Final high-precision Nelder-Mead polish
    try:
        result5 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-15, 'fatol': 1e-17, 'adaptive': True}
        )
        bary_current = clamp_bary(result5.x.reshape(N, 2))
    except Exception:
        pass
        
    return bary_current

def final_polish(bary_init, time_limit=8.0):
    """
    Aggressive final polishing using coordinate-wise optimization and multiple NM restarts.
    """
    start_time = time.time()
    bary_current = bary_init.copy()
    best_val = -hard_min_obj(bary_current.flatten())
    best_bary = bary_current.copy()
    
    def update(bary):
        nonlocal best_val, best_bary
        bary = clamp_bary(bary)
        val = -hard_min_obj(bary.flatten())
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
        return bary
    
    # Multiple Nelder-Mead passes with decreasing tolerance
    for i, (maxiter, xatol, fatol) in enumerate([
        (2000, 1e-12, 1e-14),
        (1500, 1e-14, 1e-16),
        (1000, 1e-15, 1e-17),
    ]):
        if time.time() - start_time > time_limit * (0.3 + 0.3 * i):
            break
        try:
            result = minimize(
                hard_min_obj,
                best_bary.flatten(),
                method='Nelder-Mead',
                options={'maxiter': maxiter, 'xatol': xatol, 'fatol': fatol, 'adaptive': True}
            )
            bary_current = update(result.x.reshape(N, 2))
        except Exception:
            pass
    
    # SLSQP with bounds for constraint handling
    if time.time() - start_time < time_limit * 0.85:
        try:
            bounds = [(0, 1)] * (N * 2)
            result = minimize(
                hard_min_obj,
                best_bary.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14}
            )
            bary_current = update(result.x.reshape(N, 2))
        except Exception:
            pass
    
    # One more NM if time permits
    if time.time() - start_time < time_limit * 0.95:
        try:
            result = minimize(
                hard_min_obj,
                best_bary.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-15, 'fatol': 1e-17, 'adaptive': True}
            )
            bary_current = update(result.x.reshape(N, 2))
        except Exception:
            pass
    
    return best_bary

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT

    best_bary = None
    best_val = -np.inf

    def update_best(bary):
        nonlocal best_bary, best_val
        bary = clamp_bary(bary)
        val = -hard_min_obj(bary.flatten())
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    # Phase 1: Quick global search with Differential Evolution (reduced time)
    try:
        def de_obj(x):
            bary = x.reshape(N, 2)
            bary = clamp_bary(bary)
            areas = calc_areas_bary(bary)
            return -np.min(areas)
        
        bounds = [(0, 1)] * (N * 2)
        
        # Run differential evolution with fewer iterations (it's slow in 22D)
        result_de = differential_evolution(
            de_obj,
            bounds,
            seed=42,
            maxiter=15,
            popsize=12,
            tol=1e-4,
            polish=False
        )
        
        de_bary = clamp_bary(result_de.x.reshape(N, 2))
        update_best(de_bary)
        
        # Quick refine of DE result
        remaining = deadline - time.time() - 20
        if remaining > 4:
            refined_de = refine_local(de_bary, time_limit=min(5.0, remaining))
            update_best(refined_de)
    except Exception:
        pass

    # Phase 2: Local refinement from multiple diverse starting points
    candidates = []
    
    # 1. Hexagonal lattice based candidate
    hex_bary = generate_hexagonal_init()
    candidates.append(hex_bary)
    
    # 2. Symmetric candidate
    sym_bary = generate_symmetric_init()
    candidates.append(sym_bary)
    
    # 3. Convex polygon candidate
    poly_bary = generate_convex_polygon_init()
    candidates.append(poly_bary)
    
    # 4. Random candidates
    for i in range(2):
        candidates.append(generate_random_init(seed=42 + i))
    
    # 5. Perturbed versions of the best found so far
    if best_bary is not None:
        for i in range(2):
            candidates.append(generate_perturbed_init(best_bary, sigma=0.03, seed=999 + i))
    
    # Process candidates with time allocation
    for i, cand in enumerate(candidates):
        if time.time() > deadline - 20:
            break
            
        remaining = deadline - 20 - time.time()
        if remaining < 3:
            break
            
        # Allocate time: structured inits get more time
        if i in [0, 1, 2]:  # Hex, symmetric, polygon
            t_limit = min(5.0, remaining)
        else:  # Random and perturbed
            t_limit = min(3.0, remaining)
            
        try:
            refined = refine_local(cand, time_limit=t_limit)
            update_best(refined)
        except Exception:
            continue

    # Phase 3: Final polish on best solution
    if best_bary is not None and time.time() < deadline - 3:
        remaining = deadline - 3 - time.time()
        try:
            polished = final_polish(best_bary, time_limit=min(10.0, remaining))
            update_best(polished)
        except Exception:
            pass

    if best_bary is None:
        # Fallback
        best_bary = generate_hexagonal_init()

    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart
