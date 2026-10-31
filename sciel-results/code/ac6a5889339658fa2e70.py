import numpy as np
from scipy.optimize import minimize, differential_evolution
from itertools import combinations
import time

N = 11
TIME_LIMIT = 145.0

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)
N_TRI = len(tri_indices)

H = np.sqrt(3) / 2

def bary_to_cart(bary):
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    return np.column_stack([x, y])

def calc_areas_bary(bary):
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
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def soft_min_obj(bary_flat, beta=50.0):
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
    Minimize -t subject to soft constraints: all areas >= t.
    """
    bary = x[:2*N].reshape(N, 2)
    t = x[2*N]
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    # Hard penalty for infeasibility
    diff = areas - t
    neg_diff = np.maximum(0, -diff)
    penalty = np.sum(neg_diff**2) * 1e6
    
    if np.any(neg_diff > 1e-6):
        return -t + penalty
        
    # Smooth surrogate for the constraint
    if np.any(areas < 1e-15):
        return 1e10
        
    scaled = -beta * diff
    max_scaled = np.max(scaled)
    if max_scaled > 700:
        return -t + penalty
        
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    # log_sum/beta approximates the max(0, -min(diff)) = max(0, t - min(areas))
    # If min(areas) < t, this is positive. We want to minimize it.
    # Actually, log_sum_exp(-beta * (areas - t)) / beta approximates max(0, t - min(areas))
    # Wait, log_sum_exp(-beta * x) / beta approximates -min(x) for large beta.
    # So log_sum_exp(-beta * (areas - t)) / beta approximates -min(areas - t) = t - min(areas).
    # We want min(areas) >= t, so t - min(areas) <= 0.
    # The objective is -t + penalty.
    # If we add the smooth term: -t + max(0, t - min(areas)).
    # This is a standard epigraph relaxation.
    
    smooth_penalty = max(0, log_sum / beta) * 1e4
    return -t + smooth_penalty

def generate_hexagonal_init():
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
    rng = np.random.default_rng(seed)
    u = rng.random(N)
    v = rng.random(N) * (1 - u)
    return np.column_stack([u, v])

def generate_perturbed_init(base_bary, sigma=0.05, seed=None):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, base_bary.shape)
    return clamp_bary(base_bary + noise)

def generate_voronoi_init():
    best_bary = None
    best_val = -np.inf
    
    n_grid = 12
    u_grid, v_grid = np.meshgrid(np.linspace(0, 1, n_grid), np.linspace(0, 1, n_grid))
    u_grid = u_grid.flatten()
    v_grid = v_grid.flatten()
    
    valid = (u_grid >= 0) & (v_grid >= 0) & (u_grid + v_grid <= 1)
    u_valid = u_grid[valid]
    v_valid = v_grid[valid]
    
    if len(u_valid) < N:
        return generate_random_init(42)

    rng = np.random.default_rng(42)
    
    # Farthest point sampling
    idxs = [rng.choice(len(u_valid))]
    for _ in range(N - 1):
        last_u, last_v = u_valid[idxs[-1]], v_valid[idxs[-1]]
        dists = (u_valid - last_u)**2 + (v_valid - last_v)**2
        for i in idxs:
            dists[i] = -1
        next_idx = np.argmax(dists)
        idxs.append(next_idx)
        
    bary = np.column_stack([u_valid[idxs], v_valid[idxs]])
    return bary

def refine_local(bary_init, time_limit=5.0):
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

    # Stage 1: Graduated soft-min with L-BFGS-B
    try:
        if time.time() - start_time < time_limit * 0.4:
            total_iters = max(50, int(time_limit * 15))
            
            # Use epigraph formulation if possible? 
            # L-BFGS-B doesn't handle epigraph t variable easily without constraints.
            # Stick to soft_min for L-BFGS-B.
            def graduated_obj(bary_flat):
                state = {'iter': 0}
                def obj(bary_flat_inner):
                    state['iter'] += 1
                    progress = min(state['iter'] / total_iters, 1.0)
                    beta = 5.0 * (300.0 / 5.0) ** progress
                    
                    bary = bary_flat_inner.reshape(N, 2)
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
                return obj

            obj_fn = graduated_obj(bary_current.flatten())
            result1 = minimize(
                obj_fn,
                bary_current.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': total_iters, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            try_update(result1)
    except Exception:
        pass

    # Stage 2: Nelder-Mead on hard min
    if time.time() - start_time < time_limit * 0.7:
        try:
            nm_maxiter = max(500, int(time_limit * 200))
            result2 = minimize(
                hard_min_obj,
                bary_current.flatten(),
                method='Nelder-Mead',
                options={'maxiter': nm_maxiter, 'xatol': 1e-14, 'fatol': 1e-18, 'adaptive': True}
            )
            try_update(result2)
        except Exception:
            pass

    # Stage 3: SLSQP polish
    if time.time() - start_time < time_limit:
        try:
            result3 = minimize(
                hard_min_obj,
                bary_current.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-16, 'iprint': 0}
            )
            try_update(result3)
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

    # 1. Diverse Initializations
    inits = []
    inits.append(generate_hexagonal_init())
    inits.append(generate_voronoi_init())
    inits.append(generate_random_init(seed=0))
    
    # Add a specific lattice-based init
    K = 3
    pts = []
    for i in range(K + 1):
        for j in range(K + 1 - i):
            pts.append([i / K, j / K])
    # 10 points. Add centroid? (1/3, 1/3) is already there.
    # Add (0.35, 0.35)
    pts.append([0.35, 0.35])
    inits.append(np.array(pts))

    # Refine each
    for init in inits:
        if time.time() > deadline - 30:
            break
        refined = refine_local(init, time_limit=8.0)
        update_best(refined)

    # 2. Global Search with Differential Evolution
    if time.time() < deadline - 40:
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
                popsize=20,
                tol=1e-8,
                polish=False,
                init='sobol'
            )
            de_bary = clamp_bary(result_de.x.reshape(N, 2))
            refined_de = refine_local(de_bary, time_limit=10.0)
            update_best(refined_de)
        except Exception:
            pass

    # 3. Multi-start
    rng = np.random.default_rng(123)
    restart_count = 0
    while time.time() < deadline - 15:
        restart_count += 1
        if best_bary is not None:
            if restart_count % 3 == 0:
                init = generate_perturbed_init(best_bary, sigma=0.08, seed=2000 + restart_count)
            elif restart_count % 3 == 1:
                init = generate_perturbed_init(best_bary, sigma=0.03, seed=2000 + restart_count)
            else:
                init = generate_random_init(seed=3000 + restart_count)
        else:
            init = generate_random_init(seed=3000 + restart_count)
        
        remaining = deadline - time.time()
        if remaining > 30:
            tl = 4.0
        elif remaining > 20:
            tl = 3.0
        else:
            tl = 2.0
            
        refined = refine_local(init, time_limit=tl)
        update_best(refined)

    # 4. Final Polish
    if best_bary is not None and time.time() < deadline - 5:
        try:
            bounds = [(0, 1)] * (N * 2)
            result_slsqp = minimize(
                hard_min_obj,
                best_bary.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-17, 'iprint': 0}
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
                    options={'maxiter': 5000, 'xatol': 1e-18, 'fatol': 1e-22, 'adaptive': True}
                )
                update_best(clamp_bary(result_nm.x.reshape(N, 2)))
            except Exception:
                pass

    cart = bary_to_cart(best_bary)
    return cart
