import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import Delaunay, Voronoi
from itertools import combinations
import time

N = 11
TIME_LIMIT = 142.0  # Safety margin for 150s limit

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

def epigraph_obj(params):
    """
    Epigraph formulation: optimize over (bary, t) where t is the minimum area.
    Minimize -t subject to all areas >= t (penalty-augmented).
    params: [u0, v0, u1, v1, ..., u10, v10, t]
    """
    bary_flat = params[:N*2]
    t = params[N*2]
    
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    # Penalty for violating areas >= t
    violations = np.maximum(0, t - areas)
    penalty = 100.0 * np.sum(violations**2)
    
    # Objective: minimize -t + penalty
    return -t + penalty

def epigraph_constrained_obj(params):
    """
    Epigraph with stronger penalty and tighter constraints.
    """
    bary_flat = params[:N*2]
    t = params[N*2]
    
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    # Hard penalty for any area below t
    min_area = np.min(areas)
    if t > min_area:
        return -(min_area) + 1000.0 * (t - min_area)**2
    
    return -t

def generate_hexagonal_init():
    """
    Generate a hexagonal lattice-like initial configuration.
    """
    best_bary = None
    best_val = -np.inf
    
    # Try a range of scales to find a good starting point
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

def generate_symmetric_init():
    """
    Generate a symmetric configuration with 3-fold rotational symmetry.
    Place points on orbits of 3 (or 1 for center) under 120-degree rotation.
    """
    # 11 = 1 + 3*3 + 1*2 (center + 3 orbits of 3 + 2 extra)
    # Or: 1 center + 3 orbits of 3 = 10, plus 1 more
    # Let's do: center + 3 orbits of 3 = 10 points, then add 1 more
    
    # Barycentric center is (1/3, 1/3)
    center = np.array([1.0/3, 1.0/3])
    
    # Rotation by 120 degrees in barycentric coordinates
    # (u, v) -> (v, 1-u-v) -> (1-u-v, u)
    def rotate_bary(pt):
        u, v = pt
        return [v, 1.0 - u - v]
    
    def rotate_bary2(pt):
        u, v = pt
        return [1.0 - u - v, u]
    
    # Generate base points for orbits
    # Try different radial distances
    best_bary = None
    best_val = -np.inf
    
    for r1 in [0.1, 0.15, 0.2, 0.25, 0.3]:
        for r2 in [0.3, 0.35, 0.4, 0.45, 0.5]:
            for r3 in [0.4, 0.45, 0.5, 0.55, 0.6]:
                # Orbit 1: radius r1 from center
                base1 = center + r1 * np.array([0.5, 0.5/np.sqrt(3)])
                pts = [center.copy()]
                pts.append(base1.copy())
                pts.append(rotate_bary(base1).copy())
                pts.append(rotate_bary2(base1).copy())
                
                # Orbit 2: radius r2
                base2 = center + r2 * np.array([0.5, 0.5/np.sqrt(3)])
                pts.append(base2.copy())
                pts.append(rotate_bary(base2).copy())
                pts.append(rotate_bary2(base2).copy())
                
                # Orbit 3: radius r3
                base3 = center + r3 * np.array([0.5, 0.5/np.sqrt(3)])
                pts.append(base3.copy())
                pts.append(rotate_bary(base3).copy())
                pts.append(rotate_bary2(base3).copy())
                
                # 10 points so far, add 11th
                # Place 11th at a different angle
                base4 = center + 0.35 * np.array([0.5, -0.5/np.sqrt(3)])
                pts.append(base4.copy())
                
                bary = np.array(pts)
                bary = clamp_bary(bary)
                val = -hard_min_obj(bary.flatten())
                if val > best_val:
                    best_val = val
                    best_bary = bary.copy()
    
    return best_bary

def generate_voronoi_init():
    """
    Use Voronoi diagram to find well-spread initial points.
    """
    # Start with a grid-like pattern
    rng = np.random.default_rng(42)
    
    # Create a fine grid in barycentric coordinates
    n_grid = 10
    u_vals = np.linspace(0.05, 0.95, n_grid)
    v_vals = np.linspace(0.05, 0.95, n_grid)
    
    candidates = []
    for u in u_vals:
        for v in v_vals:
            if u + v <= 0.95:
                candidates.append([u, v])
    
    candidates = np.array(candidates)
    
    # Select N points that maximize minimum distance
    selected = [candidates[0].copy()]
    for _ in range(N - 1):
        best_idx = 0
        best_min_dist = -1
        for i in range(len(candidates)):
            if any(np.allclose(candidates[i], s) for s in selected):
                continue
            min_dist = min(np.linalg.norm(candidates[i] - s) for s in selected)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = i
        selected.append(candidates[best_idx].copy())
    
    bary = np.array(selected)
    return clamp_bary(bary)

def refine_local(bary_init, time_limit=5.0):
    """
    Refine a configuration using multiple optimization stages.
    Uses graduated soft-min and epigraph formulation.
    """
    start_time = time.time()
    bounds_bary = [(0, 1)] * (N * 2)
    
    bary_current = bary_init.copy()
    current_val = -hard_min_obj(bary_current.flatten())
    
    # Stage 1: Graduated soft-min with increasing beta
    betas = [15.0, 30.0, 60.0, 120.0, 250.0]
    for beta in betas:
        if time.time() - start_time > time_limit * 0.5:
            break
        try:
            result = minimize(
                soft_min_obj,
                bary_current.flatten(),
                args=(beta,),
                method='L-BFGS-B',
                bounds=bounds_bary,
                options={'maxiter': 150, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            new_bary = clamp_bary(result.x.reshape(N, 2))
            new_val = -hard_min_obj(new_bary.flatten())
            if new_val > current_val:
                bary_current = new_bary
                current_val = new_val
        except Exception:
            pass
    
    if time.time() - start_time > time_limit * 0.55:
        return bary_current
        
    # Stage 2: Epigraph formulation
    if time.time() - start_time < time_limit * 0.75:
        try:
            # Initialize t with current min area
            areas = calc_areas_bary(bary_current)
            t_init = np.min(areas) * 0.95
            params_init = np.concatenate([bary_current.flatten(), [t_init]])
            
            bounds_epigraph = bounds_bary + [(0.0, 1.0)]
            
            result_ep = minimize(
                epigraph_obj,
                params_init,
                method='L-BFGS-B',
                bounds=bounds_epigraph,
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            
            new_bary = clamp_bary(result_ep.x[:N*2].reshape(N, 2))
            new_val = -hard_min_obj(new_bary.flatten())
            if new_val > current_val:
                bary_current = new_bary
                current_val = new_val
        except Exception:
            pass
    
    if time.time() - start_time > time_limit * 0.8:
        return bary_current
    
    # Stage 3: Hard min refinement with Nelder-Mead
    try:
        result2 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-13, 'fatol': 1e-17, 'adaptive': True}
        )
        new_bary = clamp_bary(result2.x.reshape(N, 2))
        new_val = -hard_min_obj(new_bary.flatten())
        if new_val > current_val:
            bary_current = new_bary
            current_val = new_val
    except Exception:
        pass
        
    if time.time() - start_time > time_limit * 0.92:
        return bary_current
    
    # Stage 4: SLSQP polish
    try:
        result3 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='SLSQP',
            bounds=bounds_bary,
            options={'maxiter': 800, 'ftol': 1e-16}
        )
        new_bary = clamp_bary(result3.x.reshape(N, 2))
        new_val = -hard_min_obj(new_bary.flatten())
        if new_val > current_val:
            bary_current = new_bary
            current_val = new_val
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

    # 1. Try multiple initializations
    inits = [
        ('hex', generate_hexagonal_init()),
    ]
    
    # Add symmetric init
    try:
        sym_init = generate_symmetric_init()
        if sym_init is not None:
            inits.append(('sym', sym_init))
    except Exception:
        pass
    
    # Add Voronoi init
    try:
        vor_init = generate_voronoi_init()
        inits.append(('vor', vor_init))
    except Exception:
        pass
    
    # Add random init
    inits.append(('rand', generate_random_init(seed=42)))
    
    # Refine each initial point
    for name, init in inits:
        if time.time() > deadline - 30:
            break
        refined = refine_local(init, time_limit=6.0)
        update_best(refined)
    
    # 2. Global Search with Differential Evolution
    if time.time() < deadline - 30:
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
                popsize=18,
                tol=1e-7,
                polish=False
            )
            
            de_bary = clamp_bary(result_de.x.reshape(N, 2))
            refined_de = refine_local(de_bary, time_limit=8.0)
            update_best(refined_de)
        except Exception:
            pass
    
    # 3. Multi-start: random + perturbed restarts with local refinement
    restart_count = 0
    while time.time() < deadline - 15:
        restart_count += 1
        if best_bary is not None and restart_count % 3 == 0:
            # Perturbation of best solution with varying sigma
            sigma = 0.03 + 0.02 * (restart_count % 3)
            init = generate_perturbed_init(best_bary, sigma=sigma, seed=2000 + restart_count)
        elif best_bary is not None and restart_count % 3 == 1:
            # Larger perturbation
            init = generate_perturbed_init(best_bary, sigma=0.08, seed=3000 + restart_count)
        else:
            # Random restart
            init = generate_random_init(seed=4000 + restart_count)
        
        remaining = deadline - 15 - time.time()
        if remaining < 4:
            break
            
        t_limit = min(5.5, remaining)
        refined = refine_local(init, time_limit=t_limit)
        update_best(refined)

    # Final high-precision polish on the best candidate
    if best_bary is not None and time.time() < deadline - 8:
        try:
            bounds = [(0, 1)] * (N * 2)
            result_slsqp = minimize(
                hard_min_obj,
                best_bary.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 1500, 'ftol': 1e-16}
            )
            update_best(clamp_bary(result_slsqp.x.reshape(N, 2)))
        except Exception:
            pass
        
        if best_bary is not None and time.time() < deadline - 3:
            try:
                result_nm = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 4000, 'xatol': 1e-14, 'fatol': 1e-17, 'adaptive': True}
                )
                update_best(clamp_bary(result_nm.x.reshape(N, 2)))
            except Exception:
                pass
    
    # Final SLSQP polish
    if best_bary is not None and time.time() < deadline - 1:
        try:
            bounds = [(0, 1)] * (N * 2)
            result_final = minimize(
                hard_min_obj,
                best_bary.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-16}
            )
            update_best(clamp_bary(result_final.x.reshape(N, 2)))
        except Exception:
            pass

    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart
