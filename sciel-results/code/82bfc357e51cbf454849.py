import numpy as np
from scipy.optimize import differential_evolution, minimize
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

def get_active_triangles(bary, threshold=1.05):
    """
    Get triangles whose area is close to the minimum (within threshold factor).
    Returns list of (i, j, k, area) tuples.
    """
    areas = calc_areas_bary(bary)
    min_area = np.min(areas)
    active_mask = areas <= min_area * threshold
    active_indices = np.where(active_mask)[0]
    
    active_tris = []
    for idx in active_indices:
        i, j, k = tri_idx[idx]
        active_tris.append((i, j, k, areas[idx]))
    
    return active_tris

def epigraph_obj(x, beta=20.0):
    """
    Epigraph formulation: minimize t subject to all areas >= t.
    x = [bary_coords (N*2), t]
    Uses soft constraints for smoothness.
    """
    bary_flat = x[:N*2]
    t = x[N*2]
    
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    # Soft penalty for violations: sum of (t - area)^2 for areas < t
    violations = np.maximum(t - areas, 0)
    penalty = np.sum(violations ** 2)
    
    # Also add a small regularization to keep t reasonable
    reg = 1e-6 * t
    
    return t + penalty + reg

def epigraph_constrained_obj(x, beta=20.0):
    """
    Alternative epigraph: maximize t with log-sum-exp soft constraint.
    x = [bary_coords (N*2), t]
    """
    bary_flat = x[:N*2]
    t = x[N*2]
    
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    # Soft constraint: -log(sum(exp(beta*(t - area)))) / beta
    # This is large when any area < t
    scaled = beta * (t - areas)
    max_scaled = np.max(scaled)
    
    if max_scaled > 700:
        # Hard constraint violation
        return t + 1e6 * np.sum(np.maximum(t - areas, 0))
    
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_penalty = log_sum / beta
    
    return t + soft_penalty

def coordinated_ascent_step(bary, step_size=0.02, n_steps=5):
    """
    Perform coordinated ascent: for each active (near-min) triangle,
    try to move its three points to increase the triangle area.
    """
    bary = bary.copy()
    best_val = -hard_min_obj(bary.flatten())
    
    for step in range(n_steps):
        active_tris = get_active_triangles(bary, threshold=1.1)
        if not active_tris:
            break
        
        for (i, j, k, area) in active_tris[:10]:  # Limit to top 10 most critical
            # Compute gradient-like direction to increase area of triangle (i,j,k)
            pi = bary[i].copy()
            pj = bary[j].copy()
            pk = bary[k].copy()
            
            # Current area vector (2D cross product magnitude)
            v1 = pj - pi
            v2 = pk - pi
            cross_val = v1[0] * v2[1] - v1[1] * v2[0]
            area_sign = np.sign(cross_val) if cross_val != 0 else 1
            
            # Perturb each point in a direction that increases the area
            for pt_idx in [i, j, k]:
                for dx in [-step_size, 0, step_size]:
                    for dy in [-step_size, 0, step_size]:
                        if dx == 0 and dy == 0:
                            continue
                        
                        trial = bary.copy()
                        trial[pt_idx] += [dx, dy]
                        trial = clamp_bary(trial)
                        
                        trial_val = -hard_min_obj(trial.flatten())
                        if trial_val > best_val:
                            bary = trial.copy()
                            best_val = trial_val
    
    return bary

def refine_local(bary_init, time_limit=5.0):
    """
    Refine a configuration using multiple optimization stages with epigraph formulation.
    Combines graduated soft-min, epigraph, and coordinated ascent.
    """
    start_time = time.time()
    bounds_bary = [(0, 1)] * (N * 2)
    
    bary_current = bary_init.copy()
    best_val = -hard_min_obj(bary_current.flatten())
    
    def try_update(bary_candidate):
        nonlocal bary_current, best_val
        bary_candidate = clamp_bary(bary_candidate)
        val = -hard_min_obj(bary_candidate.flatten())
        if val > best_val:
            bary_current = bary_candidate
            best_val = val
            return True
        return False
    
    # Stage 1: Coarse soft-min (low beta)
    if time.time() < start_time + time_limit * 0.2:
        try:
            result1 = minimize(
                soft_min_obj,
                bary_current.flatten(),
                args=(20.0,),
                method='L-BFGS-B',
                bounds=bounds_bary,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            try_update(result1.x.reshape(N, 2))
        except Exception:
            pass
    
    # Stage 2: Medium soft-min
    if time.time() < start_time + time_limit * 0.4:
        try:
            result2 = minimize(
                soft_min_obj,
                bary_current.flatten(),
                args=(50.0,),
                method='L-BFGS-B',
                bounds=bounds_bary,
                options={'maxiter': 300, 'ftol': 1e-13, 'gtol': 1e-9}
            )
            try_update(result2.x.reshape(N, 2))
        except Exception:
            pass
    
    # Stage 3: Sharp soft-min (high beta)
    if time.time() < start_time + time_limit * 0.55:
        try:
            result3 = minimize(
                soft_min_obj,
                bary_current.flatten(),
                args=(200.0,),
                method='L-BFGS-B',
                bounds=bounds_bary,
                options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            try_update(result3.x.reshape(N, 2))
        except Exception:
            pass
    
    # Stage 4: Epigraph refinement with SLSQP
    if time.time() < start_time + time_limit * 0.7:
        try:
            init_areas = calc_areas_bary(bary_current)
            init_t = np.min(init_areas) * 0.95
            
            x_init = np.concatenate([bary_current.flatten(), [init_t]])
            bounds_full = bounds_bary + [(0, 1)]
            
            result4 = minimize(
                epigraph_obj,
                x_init,
                method='SLSQP',
                bounds=bounds_full,
                options={'maxiter': 500, 'ftol': 1e-12}
            )
            
            if result4.success:
                try_update(result4.x[:N*2].reshape(N, 2))
        except Exception:
            pass
    
    # Stage 5: Coordinated ascent on active triangles
    if time.time() < start_time + time_limit * 0.8:
        try:
            bary_candidate = coordinated_ascent_step(bary_current, step_size=0.012, n_steps=5)
            try_update(bary_candidate)
        except Exception:
            pass
    
    # Stage 6: Hard min refinement with Nelder-Mead
    if time.time() < start_time + time_limit * 0.9:
        try:
            result5 = minimize(
                hard_min_obj,
                bary_current.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-13, 'fatol': 1e-17, 'adaptive': True}
            )
            try_update(result5.x.reshape(N, 2))
        except Exception:
            pass
    
    # Stage 7: SLSQP polish
    if time.time() < start_time + time_limit * 0.98:
        try:
            result6 = minimize(
                hard_min_obj,
                bary_current.flatten(),
                method='SLSQP',
                bounds=bounds_bary,
                options={'maxiter': 800, 'ftol': 1e-15}
            )
            try_update(result6.x.reshape(N, 2))
        except Exception:
            pass
        
    return bary_current

def generate_symmetric_init():
    """
    Generate a symmetric initial configuration based on triangular lattice symmetry.
    Uses the 3-fold rotational symmetry of the equilateral triangle.
    """
    best_bary = None
    best_val = -np.inf
    
    # Center point
    center = np.array([1.0/3, 1.0/3])
    
    # Try different ring configurations
    for n_rings in range(1, 4):
        for ring_spacing in [0.1, 0.15, 0.2, 0.25, 0.3]:
            pts = [center.copy()]
            
            for ring in range(1, n_rings + 1):
                r = ring * ring_spacing
                # 6 points per ring (hexagonal symmetry)
                for angle_idx in range(6):
                    angle = angle_idx * np.pi / 3
                    # In barycentric: x = v + 0.5*w, y = w*H
                    # Convert from polar to barycentric
                    x_cart = 0.5 + r * np.cos(angle + np.pi/6)  # offset to center
                    y_cart = H / 2 + r * np.sin(angle + np.pi/6)
                    
                    # Convert back to barycentric
                    w_bary = y_cart / H
                    v_bary = x_cart - 0.5 * (1 - w_bary)
                    u_bary = 1 - v_bary - w_bary
                    
                    if u_bary >= -1e-9 and v_bary >= -1e-9 and w_bary >= -1e-9 and u_bary + v_bary <= 1 + 1e-9:
                        pts.append(np.array([max(0, u_bary), max(0, v_bary)]))
            
            # Select best subset of N points
            if len(pts) >= N:
                rng = np.random.default_rng(42)
                for trial in range(10):
                    if len(pts) == N:
                        bary = np.array(pts)
                    else:
                        idx = rng.choice(len(pts), N, replace=False)
                        bary = np.array(pts)[idx]
                    bary = clamp_bary(bary)
                    val = -hard_min_obj(bary.flatten())
                    if val > best_val:
                        best_val = val
                        best_bary = bary.copy()
    
    if best_bary is None:
        return generate_random_init(42)
    
    return best_bary

def generate_radial_init():
    """
    Generate points arranged in concentric rings from the centroid.
    """
    best_bary = None
    best_val = -np.inf
    
    center = np.array([1.0/3, 1.0/3])
    
    for n_rings in range(1, 4):
        for base_radius in [0.1, 0.15, 0.2, 0.25]:
            for n_per_ring in [3, 6]:
                pts = [center.copy()]
                
                for ring in range(1, n_rings + 1):
                    r = base_radius * ring
                    for k in range(n_per_ring):
                        angle = 2 * np.pi * k / n_per_ring + np.pi / 6
                        x_cart = 0.5 + r * np.cos(angle)
                        y_cart = H / 2 + r * np.sin(angle)
                        
                        w_bary = y_cart / H
                        v_bary = x_cart - 0.5 * (1 - w_bary)
                        u_bary = 1 - v_bary - w_bary
                        
                        if u_bary >= -1e-9 and v_bary >= -1e-9 and w_bary >= -1e-9 and u_bary + v_bary <= 1 + 1e-9:
                            pts.append(np.array([max(0, u_bary), max(0, v_bary)]))
                
                if len(pts) >= N:
                    rng = np.random.default_rng(123)
                    for trial in range(10):
                        if len(pts) == N:
                            bary = np.array(pts)
                        else:
                            idx = rng.choice(len(pts), N, replace=False)
                            bary = np.array(pts)[idx]
                        bary = clamp_bary(bary)
                        val = -hard_min_obj(bary.flatten())
                        if val > best_val:
                            best_val = val
                            best_bary = bary.copy()
    
    if best_bary is None:
        return generate_random_init(123)
    
    return best_bary

def generate_boundary_init():
    """
    Generate points with some on the boundary edges and some interior.
    """
    best_bary = None
    best_val = -np.inf
    
    # Vertices
    vertices = [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([0.0, 0.0])]
    
    for n_boundary in [3, 4, 5, 6]:
        for interior_fraction in [0.3, 0.4, 0.5]:
            n_interior = N - n_boundary
            if n_interior < 0:
                continue
            
            # Place boundary points evenly on edges
            boundary_pts = []
            pts_per_edge = n_boundary // 3
            remainder = n_boundary % 3
            
            for edge in range(3):
                n_on_edge = pts_per_edge + (1 if edge < remainder else 0)
                for k in range(n_on_edge):
                    if n_on_edge == 1:
                        t = 0.5
                    else:
                        t = (k + 0.5) / n_on_edge
                    
                    if edge == 0:  # Edge from (1,0) to (0,1) in bary: u=1-t, v=t, w=0
                        boundary_pts.append(np.array([1 - t, t]))
                    elif edge == 1:  # Edge from (0,1) to (0,0): u=0, v=1-t, w=t
                        boundary_pts.append(np.array([0, 1 - t]))
                    else:  # Edge from (0,0) to (1,0): u=t, v=0, w=1-t
                        boundary_pts.append(np.array([t, 0]))
            
            # Generate interior points
            rng = np.random.default_rng(456)
            interior_pts = []
            for _ in range(n_interior):
                u = rng.random() * (1 - interior_fraction)
                v = rng.random() * (1 - u) * (1 - interior_fraction)
                interior_pts.append(np.array([u, v]))
            
            pts = boundary_pts + interior_pts
            
            if len(pts) >= N:
                if len(pts) > N:
                    idx = rng.choice(len(pts), N, replace=False)
                    bary = np.array(pts)[idx]
                else:
                    bary = np.array(pts)
                
                bary = clamp_bary(bary)
                val = -hard_min_obj(bary.flatten())
                if val > best_val:
                    best_val = val
                    best_bary = bary.copy()
    
    if best_bary is None:
        return generate_random_init(456)
    
    return best_bary

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

    # 1. Try multiple structured initializations
    init_candidates = [
        ('hex', generate_hexagonal_init()),
        ('sym', generate_symmetric_init()),
        ('rad', generate_radial_init()),
        ('bnd', generate_boundary_init()),
    ]
    
    for name, init in init_candidates:
        if time.time() > deadline - 30:
            break
        refined = refine_local(init, time_limit=8.0)
        update_best(refined)
    
    # 2. Global Search with Differential Evolution
    if time.time() < deadline - 35:
        try:
            def de_obj(x):
                bary = x.reshape(N, 2)
                bary = clamp_bary(bary)
                areas = calc_areas_bary(bary)
                return -np.min(areas)
            
            bounds = [(0, 1)] * (N * 2)
            result_de = differential_evolution(
                de_obj,
                bounds,
                seed=42,
                maxiter=50,
                popsize=15,
                tol=1e-6,
                polish=False
            )
            
            de_bary = clamp_bary(result_de.x.reshape(N, 2))
            refined_de = refine_local(de_bary, time_limit=10.0)
            update_best(refined_de)
        except Exception:
            pass
    
    # 3. Multi-start: random + perturbed restarts with local refinement
    rng = np.random.default_rng(123)
    restart_count = 0
    while time.time() < deadline - 12:
        restart_count += 1
        if best_bary is not None and restart_count % 2 == 0:
            # Perturbation of best solution with varying sigma
            sigma = 0.03 + 0.02 * (restart_count % 3)
            init = generate_perturbed_init(best_bary, sigma=sigma, seed=2000 + restart_count)
        else:
            # Fresh random start
            init = generate_random_init(seed=3000 + restart_count)
        
        # Scale time limit based on remaining time
        remaining = deadline - 12 - time.time()
        time_per_restart = min(5.0, max(2.0, remaining / 8.0))
        refined = refine_local(init, time_limit=time_per_restart)
        update_best(refined)

    # 4. Final high-precision polish with epigraph and coordinated ascent
    if best_bary is not None and time.time() < deadline - 10:
        try:
            # Epigraph refinement
            init_areas = calc_areas_bary(best_bary)
            init_t = np.min(init_areas) * 0.95
            
            x_init = np.concatenate([best_bary.flatten(), [init_t]])
            bounds_full = [(0, 1)] * (N * 2) + [(0, 1)]
            
            result_epi = minimize(
                epigraph_obj,
                x_init,
                method='SLSQP',
                bounds=bounds_full,
                options={'maxiter': 500, 'ftol': 1e-13}
            )
            
            if result_epi.success:
                bary_candidate = clamp_bary(result_epi.x[:N*2].reshape(N, 2))
                update_best(bary_candidate)
        except Exception:
            pass
        
        if best_bary is not None and time.time() < deadline - 7:
            # Coordinated ascent on best solution
            try:
                bary_candidate = coordinated_ascent_step(best_bary, step_size=0.008, n_steps=8)
                update_best(bary_candidate)
            except Exception:
                pass
        
        if best_bary is not None and time.time() < deadline - 4:
            try:
                result_slsqp = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='SLSQP',
                    bounds=[(0, 1)] * (N * 2),
                    options={'maxiter': 1500, 'ftol': 1e-15}
                )
                update_best(clamp_bary(result_slsqp.x.reshape(N, 2)))
            except Exception:
                pass
        
        if best_bary is not None and time.time() < deadline - 2:
            try:
                result_nm = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 4000, 'xatol': 1e-14, 'fatol': 1e-18, 'adaptive': True}
                )
                update_best(clamp_bary(result_nm.x.reshape(N, 2)))
            except Exception:
                pass
        
        # Final SLSQP pass
        if best_bary is not None and time.time() < deadline - 0.5:
            try:
                result_final = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='SLSQP',
                    bounds=[(0, 1)] * (N * 2),
                    options={'maxiter': 2500, 'ftol': 1e-16}
                )
                update_best(clamp_bary(result_final.x.reshape(N, 2)))
            except Exception:
                pass

    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart

# EVOLVE-BLOCK-END
