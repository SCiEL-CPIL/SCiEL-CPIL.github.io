import numpy as np
from scipy.optimize import minimize, differential_evolution
from itertools import combinations
import time

N = 11
TIME_LIMIT = 145.0  # Safety margin for 150s limit

# Pre-compute hexagonal grid topology for 11 points
# 6 boundary + 5 interior in a hexagonal pattern
# Using a regular hexagonal lattice clipped to triangle
def generate_hex_topology():
    """
    Generate 11 points in a hexagonal grid pattern inside the unit equilateral triangle.
    Returns barycentric coordinates.
    """
    # Hexagonal grid with spacing s, centered in triangle
    # For 11 points: try different spacings and pick best subset
    best_pts = None
    best_score = -np.inf
    
    for s in [0.18, 0.20, 0.22, 0.25, 0.28, 0.30]:
        pts = []
        # Generate hexagonal grid points
        for i in range(-5, 6):
            for j in range(-5, 6):
                # Hexagonal coordinates to Cartesian (in barycentric-like space)
                x = s * (i + 0.5 * j)
                y = s * (0.866 * j)  # 0.866 ≈ sqrt(3)/2
                
                # Convert to barycentric: we want points in triangle with vertices (0,0), (1,0), (0.5, H)
                # In barycentric: point = u*V0 + v*V1 + w*V2 where u+v+w=1
                # x_cart = v + 0.5*w, y_cart = H*w
                # So: w = y/H, v = x - 0.5*w, u = 1 - v - w
                w_val = y / H
                v_val = x - 0.5 * w_val
                u_val = 1 - v_val - w_val
                
                if u_val >= -1e-9 and v_val >= -1e-9 and w_val >= -1e-9:
                    u_val = max(0, u_val)
                    v_val = max(0, v_val)
                    w_val = max(0, w_val)
                    # Normalize if needed
                    total = u_val + v_val + w_val
                    if total > 0:
                        u_val /= total
                        v_val /= total
                        w_val /= total
                    if u_val <= 1 + 1e-9 and v_val <= 1 + 1e-9 and w_val <= 1 + 1e-9:
                        pts.append([u_val, v_val])
        
        if len(pts) >= N:
            # Pick N points that maximize minimum area
            rng = np.random.default_rng(42)
            for trial in range(10):
                if len(pts) == N:
                    bary = np.array(pts)
                else:
                    idx = rng.choice(len(pts), N, replace=False)
                    bary = np.array(pts)[idx]
                bary = clamp_bary(bary)
                val = -hard_min_obj(bary.flatten())
                if val > best_score:
                    best_score = val
                    best_pts = bary.copy()
    
    if best_pts is None:
        # Fallback: simple grid
        rng = np.random.default_rng(42)
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        best_pts = np.column_stack([u, v])
    
    return best_pts

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

def epigraph_obj(x, beta=100.0):
    """
    Epigraph formulation: x = [bary_flat (22), t (1)]
    Minimize -t subject to soft constraints: all areas >= t (softly).
    """
    bary = x[:2*N].reshape(N, 2)
    t = x[2*N]
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    # Penalty for areas < t
    diff = areas - t
    if np.any(diff < -1e-10):
        # Large penalty if any area is significantly below t
        penalty = np.sum(np.maximum(0, -diff) ** 2) * 1e6
        return -t + penalty
    
    # Soft constraint via log-sum-exp
    if np.any(areas < 1e-15):
        return 1e10
    
    scaled = -beta * diff
    max_scaled = np.max(scaled)
    if max_scaled > 700:
        return -t + 1e6 * np.sum(np.maximum(0, -diff) ** 2)
        
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    # log_sum / beta should be <= 0 for feasibility; penalty if positive
    penalty = max(0, log_sum / beta) * 1e4
    return -t + penalty

def generate_hexagonal_init():
    """
    Generate a hexagonal lattice-like initial configuration.
    """
    return generate_hex_topology()

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

def generate_voronoi_init():
    """
    Generate an initial configuration based on a simplified Voronoi-like distribution.
    We place points on a grid and then slightly jitter them to break symmetry.
    """
    # A 4x3 grid inside the triangle roughly gives 12 points, we need 11.
    # Let's try a specific pattern that is known to be good for packing.
    # Vertices of a regular 11-gon? No, must be inside triangle.
    # Let's use a "flower" pattern: center + 6 around + 4 at corners?
    # Or just use a high-resolution grid and pick the best subset.
    
    best_bary = None
    best_val = -np.inf
    
    # Create a fine grid
    n_grid = 10
    u_grid, v_grid = np.meshgrid(np.linspace(0, 1, n_grid), np.linspace(0, 1, n_grid))
    u_grid = u_grid.flatten()
    v_grid = v_grid.flatten()
    
    # Filter valid points
    valid = (u_grid >= 0) & (v_grid >= 0) & (u_grid + v_grid <= 1)
    u_valid = u_grid[valid]
    v_valid = v_grid[valid]
    
    # We want to pick N points that are well-spread.
    # Use a greedy farthest-point sampling approach on the grid.
    rng = np.random.default_rng(42)
    
    # Start with a random point
    idxs = [rng.choice(len(u_valid))]
    for _ in range(N - 1):
        last_u, last_v = u_valid[idxs[-1]], v_valid[idxs[-1]]
        dists = (u_valid - last_u)**2 + (v_valid - last_v)**2
        # Exclude already chosen
        for i in idxs:
            dists[i] = -1
        # Pick the farthest
        next_idx = np.argmax(dists)
        idxs.append(next_idx)
        
    bary = np.column_stack([u_valid[idxs], v_valid[idxs]])
    return bary

def refine_local(bary_init, time_limit=5.0):
    """
    Refine a configuration using multiple optimization stages with graduated soft-min.
    Adaptive beta selection based on improvement.
    """
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    
    bary_current = bary_init.copy()
    current_val = -hard_min_obj(bary_current.flatten())

    def try_update(result):
        nonlocal bary_current, current_val
        if result is not None:
            new_bary = clamp_bary(result.x.reshape(N, 2))
            new_val = -hard_min_obj(new_bary.flatten())
            if new_val > current_val:
                bary_current = new_bary
                current_val = new_val

    # Stage 1: Coarse soft-min
    if time.time() < start_time + time_limit * 0.2:
        try:
            result1 = minimize(
                soft_min_obj,
                bary_current.flatten(),
                args=(20.0,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            try_update(result1)
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
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-13, 'gtol': 1e-9}
            )
            try_update(result2)
        except Exception:
            pass

    # Stage 3: Sharp soft-min
    if time.time() < start_time + time_limit * 0.6:
        try:
            result3 = minimize(
                soft_min_obj,
                bary_current.flatten(),
                args=(200.0,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            try_update(result3)
        except Exception:
            pass

    # Stage 4: Hard min refinement with Nelder-Mead
    if time.time() < start_time + time_limit * 0.8:
        try:
            result4 = minimize(
                hard_min_obj,
                bary_current.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-16, 'adaptive': True}
            )
            try_update(result4)
        except Exception:
            pass

    # Stage 5: Final NM polish
    if time.time() < start_time + time_limit:
        try:
            result5 = minimize(
                hard_min_obj,
                bary_current.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-14, 'fatol': 1e-18, 'adaptive': True}
            )
            try_update(result5)
        except Exception:
            pass
        
    return bary_current

def refine_epigraph(bary_init, time_limit=5.0):
    """
    Refine using epigraph formulation with auxiliary variable t.
    """
    start_time = time.time()
    bary_current = bary_init.copy()
    
    # Estimate initial t from current min area
    areas = calc_areas_bary(bary_current)
    t0 = np.min(areas) * 0.99
    
    # Epigraph optimization
    x0 = np.concatenate([bary_current.flatten(), [t0]])
    bounds = [(0, 1)] * (N * 2) + [(0, 0.5)]
    
    # Graduated beta for epigraph
    for beta in [50.0, 100.0, 200.0]:
        if time.time() - start_time > time_limit * 0.7:
            break
        try:
            result = minimize(
                epigraph_obj,
                x0,
                args=(beta,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            new_bary = clamp_bary(result.x[:2*N].reshape(N, 2))
            new_val = -hard_min_obj(new_bary.flatten())
            cur_val = -hard_min_obj(bary_current.flatten())
            if new_val > cur_val:
                bary_current = new_bary
            x0 = np.concatenate([bary_current.flatten(), [np.min(calc_areas_bary(bary_current)) * 0.99]])
        except Exception:
            pass
    
    # Final NM polish
    if time.time() - start_time < time_limit:
        try:
            result_nm = minimize(
                hard_min_obj,
                bary_current.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
            )
            bary_current = clamp_bary(result_nm.x.reshape(N, 2))
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

    # 1. Try Hexagonal Init
    hex_init = generate_hexagonal_init()
    refined_hex = refine_local(hex_init, time_limit=8.0)
    update_best(refined_hex)

    # 2. Try Voronoi-like Init
    if time.time() < deadline - 40:
        vor_init = generate_voronoi_init()
        refined_vor = refine_local(vor_init, time_limit=8.0)
        update_best(refined_vor)

    # 3. Global Search with Differential Evolution
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
                maxiter=30,
                popsize=15,  # Increased population for better coverage
                tol=1e-6,
                polish=False
            )

            de_bary = clamp_bary(result_de.x.reshape(N, 2))
            refined_de = refine_local(de_bary, time_limit=8.0)
            update_best(refined_de)
        except Exception:
            pass

    # 4. Multi-start: random + perturbed restarts with local refinement
    rng = np.random.default_rng(123)
    restart_count = 0
    while time.time() < deadline - 15:
        restart_count += 1
        if best_bary is not None and restart_count % 2 == 0:
            # Perturbation of best solution
            init = generate_perturbed_init(best_bary, sigma=0.04, seed=2000 + restart_count)
        else:
            # Fresh random start
            init = generate_random_init(seed=3000 + restart_count)
        
        refined = refine_local(init, time_limit=3.0)
        update_best(refined)

    # 5. Final high-precision polish
    if best_bary is not None and time.time() < deadline - 5:
        # Try epigraph on best
        refined_final_ep = refine_epigraph(best_bary, time_limit=4.0)
        update_best(refined_final_ep)
        
        if best_bary is not None and time.time() < deadline - 3:
            try:
                bounds = [(0, 1)] * (N * 2)
                result_slsqp = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': 1000, 'ftol': 1e-15}
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
                    options={'maxiter': 3000, 'xatol': 1e-14, 'fatol': 1e-18, 'adaptive': True}
                )
                update_best(clamp_bary(result_nm.x.reshape(N, 2)))
            except Exception:
                pass
        
        # One more SLSQP pass
        if best_bary is not None and time.time() < deadline - 0.5:
            try:
                bounds = [(0, 1)] * (N * 2)
                result_final = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': 2000, 'ftol': 1e-16}
                )
                update_best(clamp_bary(result_final.x.reshape(N, 2)))
            except Exception:
                pass

    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart
