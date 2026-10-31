import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
from itertools import combinations

N = 11
TIME_LIMIT = 115.0

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

def clamp_bary(bary, eps=1e-10):
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

def generate_hex_init():
    """
    Generate a high-quality hexagonal lattice initialization.
    This is a strong prior for maximizing minimum distances/areas.
    """
    inits = []
    
    for scale in [0.15, 0.18, 0.22, 0.25]:
        center = np.array([1.0/3, 1.0/3])
        
        cart_pts = []
        cart_center = bary_to_cart(center.reshape(1,2))[0]
        cart_pts.append(cart_center)
        
        for k in range(6):
            theta = np.pi/3 * k
            dx = scale * np.cos(theta)
            dy = scale * np.sin(theta)
            cart_pts.append(cart_center + np.array([dx, dy]))
            
        for k in range(6):
            if len(cart_pts) >= N:
                break
            theta = np.pi/3 * k + np.pi/6
            dx = 2 * scale * np.cos(theta)
            dy = 2 * scale * np.sin(theta)
            cart_pts.append(cart_center + np.array([dx, dy]))
            
        if len(cart_pts) >= N:
            cart_pts = np.array(cart_pts[:N])
            
            y_vals = cart_pts[:, 1]
            x_vals = cart_pts[:, 0]
            S = 1 - y_vals / H
            u = 0.5 + 0.5 * S - x_vals
            v = S - u
            
            bary_pts = np.column_stack([u, v])
            bary_pts = clamp_bary(bary_pts)
            
            if np.all(bary_pts[:, 0] > 0) and np.all(bary_pts[:, 1] > 0) and np.all(bary_pts[:, 0] + bary_pts[:, 1] < 1):
                inits.append(bary_pts)
                
    return inits

def generate_symmetric_init():
    """
    Generate symmetric configurations based on known optimal patterns.
    For 11 points in an equilateral triangle, try:
    - 3 points near vertices + 3 on edges + 5 interior
    - Hexagonal patterns with different symmetries
    """
    inits = []
    
    # Pattern 1: 3 corners + 3 midpoints + 5 interior hexagonal
    # Corners (slightly inset)
    eps = 0.05
    corners_bary = np.array([
        [eps, eps],
        [1-2*eps, eps],
        [eps, 1-2*eps]
    ])
    
    # Edge midpoints (slightly inset)
    m = 0.5 - eps
    edges_bary = np.array([
        [m, eps],
        [eps, m],
        [1-eps-2*m, m]
    ])
    
    # Interior: 5 points in a small hexagon around center
    center = np.array([1/3, 1/3])
    cart_center = bary_to_cart(center.reshape(1,2))[0]
    interior_cart = [cart_center]
    r = 0.12
    for k in range(4):
        theta = 2*np.pi*k/4 + np.pi/4
        interior_cart.append(cart_center + r * np.array([np.cos(theta), np.sin(theta)]))
    
    all_cart = []
    for b in corners_bary:
        all_cart.append(bary_to_cart(b.reshape(1,2))[0])
    for b in edges_bary:
        all_cart.append(bary_to_cart(b.reshape(1,2))[0])
    for c in interior_cart:
        all_cart.append(c)
    
    if len(all_cart) >= N:
        all_cart = np.array(all_cart[:N])
        y_vals = all_cart[:, 1]
        x_vals = all_cart[:, 0]
        S = 1 - y_vals / H
        u = 0.5 + 0.5 * S - x_vals
        v = S - u
        bary_pts = np.column_stack([u, v])
        bary_pts = clamp_bary(bary_pts)
        if np.all(bary_pts[:, 0] > 0) and np.all(bary_pts[:, 1] > 0) and np.all(bary_pts[:, 0] + bary_pts[:, 1] < 1):
            inits.append(bary_pts)
    
    # Pattern 2: Pure hexagonal with 3-fold symmetry
    # Center + 6 ring + 4 more at corners of a smaller hexagon
    for r1, r2 in [(0.15, 0.25), (0.18, 0.30), (0.12, 0.20)]:
        cart_center = bary_to_cart(np.array([[1/3, 1/3]]))[0]
        pts = [cart_center]
        for k in range(6):
            theta = 2*np.pi*k/6
            pts.append(cart_center + r1 * np.array([np.cos(theta), np.sin(theta)]))
        for k in range(4):
            if len(pts) >= N:
                break
            theta = 2*np.pi*k/4 + np.pi/4
            pts.append(cart_center + r2 * np.array([np.cos(theta), np.sin(theta)]))
        
        if len(pts) >= N:
            pts = np.array(pts[:N])
            y_vals = pts[:, 1]
            x_vals = pts[:, 0]
            S = 1 - y_vals / H
            u = 0.5 + 0.5 * S - x_vals
            v = S - u
            bary_pts = np.column_stack([u, v])
            bary_pts = clamp_bary(bary_pts)
            if np.all(bary_pts[:, 0] > 0) and np.all(bary_pts[:, 1] > 0) and np.all(bary_pts[:, 0] + bary_pts[:, 1] < 1):
                inits.append(bary_pts)
    
    # Pattern 3: Grid-like arrangement
    # 3 rows: 3-5-3 or 4-3-4
    for row_pattern in [(3,5,3), (4,3,4)]:
        pts = []
        total = sum(row_pattern)
        if total != N:
            continue
        
        # Row heights
        h_frac = [0.15, 0.5, 0.85]
        for row_idx, (count, hf) in enumerate(zip(row_pattern, h_frac)):
            y = hf * H
            # x range at this height
            x_min = 0.5 * hf
            x_max = 1 - x_min
            if count == 1:
                xs = [(x_min + x_max)/2]
            else:
                xs = np.linspace(x_min + 0.05*(x_max-x_min), x_max - 0.05*(x_max-x_min), count)
            for x in xs:
                pts.append(np.array([x, y]))
        
        if len(pts) >= N:
            pts = np.array(pts[:N])
            y_vals = pts[:, 1]
            x_vals = pts[:, 0]
            S = 1 - y_vals / H
            u = 0.5 + 0.5 * S - x_vals
            v = S - u
            bary_pts = np.column_stack([u, v])
            bary_pts = clamp_bary(bary_pts)
            if np.all(bary_pts[:, 0] > 0) and np.all(bary_pts[:, 1] > 0) and np.all(bary_pts[:, 0] + bary_pts[:, 1] < 1):
                inits.append(bary_pts)
    
    return inits

def coordinate_descent_polish(bary, max_iter=200, tol=1e-14):
    """
    Coordinate descent: for each point, try to move it to increase the min area.
    Focus on the smallest triangles.
    """
    bary = bary.copy()
    best_val = -hard_min_obj(bary.flatten())
    
    for iteration in range(max_iter):
        improved = False
        
        # Find the smallest triangles
        areas = calc_areas_bary(bary)
        min_area = np.min(areas)
        
        # Get indices of the 10 smallest triangles
        small_tri_idx = np.argsort(areas)[:10]
        
        # For each point involved in small triangles, try to move it
        for tri_i in small_tri_idx:
            i, j, k = tri_idx[tri_i]
            
            for p in [i, j, k]:
                # Try moving point p in small steps
                for dx in [-0.02, -0.01, -0.005, 0.005, 0.01, 0.02]:
                    for dy in [-0.02, -0.01, -0.005, 0.005, 0.01, 0.02]:
                        new_bary = bary.copy()
                        new_bary[p, 0] += dx
                        new_bary[p, 1] += dy
                        new_bary = clamp_bary(new_bary)
                        
                        new_val = -hard_min_obj(new_bary.flatten())
                        if new_val > best_val + tol:
                            bary = new_bary
                            best_val = new_val
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            
            if improved:
                break
        
        if not improved:
            break
    
    return bary

def refine_local(bary_init, time_limit=5.0):
    """
    Refine a configuration using multiple optimization stages.
    """
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    
    bary_current = bary_init.copy()
    
    # Stage 1: Smooth soft-min minimization with increasing beta
    for beta in [10.0, 50.0, 200.0, 1000.0]:
        if time.time() - start_time > time_limit * 0.4:
            break
        try:
            result1 = minimize(
                soft_min_obj,
                bary_current.flatten(),
                args=(beta,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            curr_hard = -hard_min_obj(bary_current.flatten())
            new_hard = -hard_min_obj(result1.x.reshape(N, 2).flatten())
            if new_hard > curr_hard:
                bary_current = clamp_bary(result1.x.reshape(N, 2))
        except Exception:
            pass
        
    # Stage 2: Hard min refinement with SLSQP
    if time.time() - start_time < time_limit * 0.7:
        try:
            result2 = minimize(
                hard_min_obj,
                bary_current.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-16}
            )
            if result2.fun < hard_min_obj(bary_current.flatten()):
                bary_current = clamp_bary(result2.x.reshape(N, 2))
        except Exception:
            pass
            
    # Stage 3: Nelder-Mead for final polish
    if time.time() - start_time < time_limit * 0.9:
        try:
            result3 = minimize(
                hard_min_obj,
                bary_current.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-14, 'fatol': 1e-18, 'adaptive': True}
            )
            if result3.fun < hard_min_obj(bary_current.flatten()):
                bary_current = clamp_bary(result3.x.reshape(N, 2))
        except Exception:
            pass
        
    # Stage 4: Coordinate descent polish
    if time.time() - start_time < time_limit:
        try:
            bary_current = coordinate_descent_polish(bary_current, max_iter=50)
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
        val = -hard_min_obj(bary.flatten())
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    # Generate diverse initial points
    rng = np.random.default_rng(123)
    inits = []
    
    # 1. High-quality Hexagonal starts
    hex_inits = generate_hex_init()
    inits.extend(hex_inits)
    
    # 2. Symmetric pattern starts
    sym_inits = generate_symmetric_init()
    inits.extend(sym_inits)
    
    # 3. Random starts
    for i in range(8):
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        inits.append(np.column_stack([u, v]))
        
    # 4. Perturbed Hex starts
    for hex_init in hex_inits[:3]:
        for _ in range(3):
            perturbed = hex_init + rng.normal(0, 0.05, (N, 2))
            perturbed = clamp_bary(perturbed)
            inits.append(perturbed)

    # Deduplicate inits
    unique_inits = []
    for init in inits:
        init_clamped = clamp_bary(init)
        is_unique = True
        for existing in unique_inits:
            if np.allclose(init_clamped, existing, atol=0.02):
                is_unique = False
                break
        if is_unique:
            unique_inits.append(init_clamped)
    
    inits = unique_inits
    
    # Refine each initial point with time budget
    # Sort by initial quality
    init_scores = []
    for init in inits:
        score = -hard_min_obj(init.flatten())
        init_scores.append(score)
    sorted_indices = np.argsort(init_scores)[::-1]
    inits = [inits[i] for i in sorted_indices]
    
    for i, init in enumerate(inits):
        if time.time() > deadline - 15:
            break
            
        remaining_inits = len(inits) - i
        base_time = 5.0 if i < 5 else 2.5
        time_per_init = min(base_time, (deadline - time.time() - 15) / max(1, remaining_inits))
        if time_per_init < 0.5:
            time_per_init = 0.5
            
        refined = refine_local(init, time_limit=time_per_init)
        update_best(refined)
        
    # If we have time, do a global search with differential evolution
    if best_bary is not None and time.time() < deadline - 30:
        try:
            def de_objective(x):
                bary = x.reshape(N, 2)
                bary = clamp_bary(bary)
                return -np.min(calc_areas_bary(bary))
            
            bounds_de = [(0, 1)] * (N * 2)
            result_de = differential_evolution(
                de_objective,
                bounds_de,
                seed=42,
                maxiter=60,
                tol=1e-10,
                popsize=15,
                mutation=(0.5, 1.0),
                recombination=0.7,
                polish=False,
                update_interval=1
            )
            if result_de.success or result_de.fun < hard_min_obj(best_bary.flatten()):
                bary_de = clamp_bary(result_de.x.reshape(N, 2))
                if update_best(bary_de):
                    if time.time() < deadline - 10:
                        refined_de = refine_local(bary_de, time_limit=min(10.0, deadline - time.time() - 5))
                        update_best(refined_de)
        except Exception:
            pass
        
    # Final high-precision polish on the best found
    if best_bary is not None and time.time() < deadline - 8:
        try:
            bounds = [(0, 1)] * (N * 2)
            result_slsqp = minimize(
                hard_min_obj,
                best_bary.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-16}
            )
            if result_slsqp.fun < hard_min_obj(best_bary.flatten()):
                update_best(clamp_bary(result_slsqp.x.reshape(N, 2)))
        except Exception:
            pass
        
        if best_bary is not None and time.time() < deadline - 3:
            try:
                result_nm = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-15, 'fatol': 1e-19, 'adaptive': True}
                )
                if result_nm.fun < hard_min_obj(best_bary.flatten()):
                    update_best(clamp_bary(result_nm.x.reshape(N, 2)))
            except Exception:
                pass
    
    # Final coordinate descent polish
    if best_bary is not None and time.time() < deadline - 2:
        try:
            best_bary = coordinate_descent_polish(best_bary, max_iter=100)
            update_best(best_bary)
        except Exception:
            pass

    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart

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

def clamp_bary(bary, eps=1e-10):
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

def generate_hex_init():
    """
    Generate a high-quality hexagonal lattice initialization.
    This is a strong prior for maximizing minimum distances/areas.
    """
    # The triangle has height H = sqrt(3)/2.
    # We want to place points in a hexagonal pattern.
    # Center is (1/3, 1/3) in barycentric.
    
    inits = []
    
    # Define a few scales for the hexagonal grid
    # The "radius" of the hexagonal cluster
    for scale in [0.15, 0.18, 0.22, 0.25]:
        center = np.array([1.0/3, 1.0/3])
        pts = [center.copy()]
        
        # 6 neighbors at distance scale
        # In barycentric coords, the "distance" is not uniform, 
        # so we use Cartesian offsets and convert back.
        # Actually, it's easier to generate in Cartesian and convert to Barycentric.
        
        cart_pts = []
        # Center in Cartesian
        cart_center = bary_to_cart(center.reshape(1,2))[0]
        cart_pts.append(cart_center)
        
        # 6 points around center
        for k in range(6):
            theta = np.pi/3 * k
            dx = scale * np.cos(theta)
            dy = scale * np.sin(theta)
            cart_pts.append(cart_center + np.array([dx, dy]))
            
        # 6 more points at distance 2*scale? Or 3 at distance 2*scale?
        # We need 11 points total. 1 + 6 = 7. Need 4 more.
        # Let's add 4 points at the "corners" of the hexagon, i.e., distance 2*scale * cos(30)?
        # Or just expand the ring.
        # Let's try adding 4 points from the next ring.
        for k in range(6):
            if len(cart_pts) >= N:
                break
            theta = np.pi/3 * k + np.pi/6
            dx = 2 * scale * np.cos(theta)
            dy = 2 * scale * np.sin(theta)
            cart_pts.append(cart_center + np.array([dx, dy]))
            
        if len(cart_pts) >= N:
            cart_pts = np.array(cart_pts[:N])
            
            # Convert back to barycentric
            # x = v + 0.5*(1-u-v) = 0.5 - 0.5u + 0.5v
            # y = (1-u-v)*H
            # y/H = 1 - u - v => u + v = 1 - y/H
            # x = 0.5 - 0.5u + 0.5v
            # Let S = u+v = 1 - y/H
            # x = 0.5 - 0.5u + 0.5(S-u) = 0.5 - 0.5u + 0.5S - 0.5u = 0.5 + 0.5S - u
            # u = 0.5 + 0.5S - x
            # v = S - u
            
            y_vals = cart_pts[:, 1]
            x_vals = cart_pts[:, 0]
            S = 1 - y_vals / H
            u = 0.5 + 0.5 * S - x_vals
            v = S - u
            
            bary_pts = np.column_stack([u, v])
            bary_pts = clamp_bary(bary_pts)
            
            # Check if points are inside
            if np.all(bary_pts[:, 0] > 0) and np.all(bary_pts[:, 1] > 0) and np.all(bary_pts[:, 0] + bary_pts[:, 1] < 1):
                inits.append(bary_pts)
                
    # Also try a "3-4-3" or "1-6-4" specific arrangement
    # Center, 6 ring, 4 corners
    return inits

def refine_local(bary_init, time_limit=5.0):
    """
    Refine a configuration using multiple optimization stages.
    """
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    
    bary_current = bary_init.copy()
    
    # Stage 1: Smooth soft-min minimization with increasing beta
    # This helps navigate the landscape without getting stuck in the non-differentiable min
    for beta in [10.0, 50.0, 200.0, 1000.0]:
        if time.time() - start_time > time_limit * 0.5:
            break
        try:
            result1 = minimize(
                soft_min_obj,
                bary_current.flatten(),
                args=(beta,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            # Accept if it improves the hard min objective (since soft min is a surrogate)
            curr_hard = -hard_min_obj(bary_current.flatten())
            new_hard = -hard_min_obj(result1.x.reshape(N, 2).flatten())
            if new_hard > curr_hard:
                bary_current = clamp_bary(result1.x.reshape(N, 2))
        except Exception:
            pass
        
    # Stage 2: Hard min refinement with SLSQP
    if time.time() - start_time < time_limit * 0.8:
        try:
            result2 = minimize(
                hard_min_obj,
                bary_current.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-16}
            )
            if result2.fun < hard_min_obj(bary_current.flatten()):
                bary_current = clamp_bary(result2.x.reshape(N, 2))
        except Exception:
            pass
            
    # Stage 3: Nelder-Mead for final polish (handles non-differentiability well)
    if time.time() - start_time < time_limit:
        try:
            result3 = minimize(
                hard_min_obj,
                bary_current.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-18, 'adaptive': True}
            )
            if result3.fun < hard_min_obj(bary_current.flatten()):
                bary_current = clamp_bary(result3.x.reshape(N, 2))
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
        val = -hard_min_obj(bary.flatten())
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    # Generate diverse initial points
    rng = np.random.default_rng(123)
    inits = []
    
    # 1. High-quality Hexagonal starts
    hex_inits = generate_hex_init()
    inits.extend(hex_inits)
    
    # 2. Random starts (fewer, as hex is better)
    for i in range(10):
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        inits.append(np.column_stack([u, v]))
        
    # 3. Perturbed Hex starts
    for hex_init in hex_inits[:3]:
        for _ in range(5):
            perturbed = hex_init + rng.normal(0, 0.05, (N, 2))
            perturbed = clamp_bary(perturbed)
            inits.append(perturbed)

    # Deduplicate inits (approximately)
    unique_inits = []
    for init in inits:
        init_clamped = clamp_bary(init)
        is_unique = True
        for existing in unique_inits:
            if np.allclose(init_clamped, existing, atol=0.02):
                is_unique = False
                break
        if is_unique:
            unique_inits.append(init_clamped)
    
    inits = unique_inits
    
    # Refine each initial point with time budget
    # Allocate more time to the first few (hex) starts
    for i, init in enumerate(inits):
        if time.time() > deadline - 15:
            break
            
        # Allocate time based on remaining time and number of inits left
        remaining_inits = len(inits) - i
        # Give more time to early iterations
        base_time = 4.0 if i < 5 else 2.0
        time_per_init = min(base_time, (deadline - time.time() - 15) / max(1, remaining_inits))
        if time_per_init < 0.5:
            time_per_init = 0.5
            
        refined = refine_local(init, time_limit=time_per_init)
        update_best(refined)
        
    # If we have time, do a global search with differential evolution
    if best_bary is not None and time.time() < deadline - 30:
        try:
            def de_objective(x):
                bary = x.reshape(N, 2)
                bary = clamp_bary(bary)
                return -np.min(calc_areas_bary(bary))
            
            bounds_de = [(0, 1)] * (N * 2)
            result_de = differential_evolution(
                de_objective,
                bounds_de,
                seed=42,
                maxiter=80,
                tol=1e-10,
                popsize=20,
                mutation=(0.5, 1.0),
                recombination=0.7,
                polish=False,
                update_interval=1,
                callback=None,
                init='sobol'
            )
            if result_de.success or result_de.fun < hard_min_obj(best_bary.flatten()):
                bary_de = clamp_bary(result_de.x.reshape(N, 2))
                if update_best(bary_de):
                    # Refine the DE result
                    if time.time() < deadline - 10:
                        refined_de = refine_local(bary_de, time_limit=min(10.0, deadline - time.time() - 5))
                        update_best(refined_de)
        except Exception:
            pass
        
    # Final high-precision polish on the best found
    if best_bary is not None and time.time() < deadline - 8:
        try:
            bounds = [(0, 1)] * (N * 2)
            result_slsqp = minimize(
                hard_min_obj,
                best_bary.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 3000, 'ftol': 1e-16}
            )
            if result_slsqp.fun < hard_min_obj(best_bary.flatten()):
                update_best(clamp_bary(result_slsqp.x.reshape(N, 2)))
        except Exception:
            pass
        
        if best_bary is not None and time.time() < deadline - 3:
            try:
                result_nm = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 10000, 'xatol': 1e-15, 'fatol': 1e-19, 'adaptive': True}
                )
                if result_nm.fun < hard_min_obj(best_bary.flatten()):
                    update_best(clamp_bary(result_nm.x.reshape(N, 2)))
            except Exception:
                pass

    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart

# EVOLVE-BLOCK-END
