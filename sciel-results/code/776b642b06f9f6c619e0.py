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

def soft_min_grad(bary_flat, beta=50.0):
    """
    Analytical gradient of the soft-min objective.
    Returns gradient of -soft_min w.r.t. bary_flat.
    """
    bary = bary_flat.reshape(N, 2)
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    areas = calc_areas_bary(bary)
    
    if np.any(areas < 1e-15):
        return np.zeros_like(bary_flat)
    
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    
    if max_scaled > 700:
        # Fall back to subgradient of hard min
        min_idx = np.argmin(areas)
        tri = tri_idx[min_idx]
        i, j, k = tri
        
        # Area = 0.5 * |det|, but we use det without 0.5 factor in calc_areas_bary
        # d(area_i)/d(u_a) for triangle (i,j,k)
        # det = u_i(v_j*w_k - v_k*w_j) - v_i(u_j*w_k - u_k*w_j) + w_i(u_j*v_k - u_k*v_j)
        # w = 1 - u - v, so dw/du = -1, dw/dv = -1
        
        grad = np.zeros(N * 2)
        
        # For point i (index i in triangle):
        # d(det)/du_i = (v_j*w_k - v_k*w_j) + (u_j*w_k - u_k*w_j)  [from -v_i term, dv_i/du_i=0 but w_i/du_i=-1]
        # Actually let me compute more carefully:
        # det = u_i*(v_j*w_k - v_k*w_j) - v_i*(u_j*w_k - u_k*w_j) + w_i*(u_j*v_k - u_k*v_j)
        # w_i = 1 - u_i - v_i
        # d(det)/du_i = (v_j*w_k - v_k*w_j) - v_i*0 + (-1)*(u_j*v_k - u_k*v_j)
        #             = (v_j*w_k - v_k*w_j) - (u_j*v_k - u_k*v_j)
        # d(det)/dv_i = u_i*0 - (u_j*w_k - u_k*w_j) + (-1)*(u_j*v_k - u_k*v_j)
        #             = -(u_j*w_k - u_k*w_j) - (u_j*v_k - u_k*v_j)
        
        # For point j:
        # d(det)/du_j = u_i*0 - v_i*w_k + w_i*v_k = -v_i*w_k + w_i*v_k
        # d(det)/dv_j = u_i*w_k - v_i*0 + w_i*u_k = u_i*w_k + w_i*u_k
        
        # For point k:
        # d(det)/du_k = u_i*(-w_j) - v_i*(-w_j) + w_i*(-v_j) = -u_i*w_j + v_i*w_j - w_i*v_j
        # d(det)/dv_k = u_i*(-v_j) - v_i*(-v_j) + w_i*(-u_j) = -u_i*v_j + v_i*v_j - w_i*u_j
        
        vi, vj, vk = v[i], v[j], v[k]
        ui, uj, uk = u[i], u[j], u[k]
        wi, wj, wk = w[i], w[j], w[k]
        
        d_det_dui = (vj*wk - vk*wj) - (uj*vk - uk*vj)
        d_det_dvi = -(uj*wk - uk*wj) - (uj*vk - uk*vj)
        d_det_duj = -vi*wk + wi*vk
        d_det_dvj = ui*wk + wi*uk
        d_det_duk = -ui*wj + vi*wj - wi*vj
        d_det_dvk = -ui*vj + vi*vj - wi*uj
        
        sign = np.sign(areas[min_idx])
        
        grad[i*2] = -beta * sign * d_det_dui
        grad[i*2+1] = -beta * sign * d_det_dvi
        grad[j*2] = -beta * sign * d_det_duj
        grad[j*2+1] = -beta * sign * d_det_dvj
        grad[k*2] = -beta * sign * d_det_duk
        grad[k*2+1] = -beta * sign * d_det_dvk
        
        return grad
    
    # Soft min gradient
    exp_scaled = np.exp(scaled - max_scaled)
    weights = exp_scaled / np.sum(exp_scaled)  # softmax weights
    
    # d(soft_min)/d(area_m) = -weights[m] / beta * (-1) ... let me be careful
    # soft_min = -log_sum / beta where log_sum = max_scaled + log(sum(exp(scaled - max_scaled)))
    # d(log_sum)/d(area_m) = d(max_scaled + log(sum(exp(-beta*areas - max_scaled))))/d(area_m)
    # If max_scaled doesn't depend on area_m (it's the max of all), then:
    # d(log_sum)/d(area_m) = 1/sum(exp(scaled-max)) * d(sum(exp(scaled-max)))/d(area_m)
    # = 1/sum(exp(scaled-max)) * exp(scaled_m - max) * (-beta)
    # = -beta * weights[m]
    # d(soft_min)/d(area_m) = -1/beta * d(log_sum)/d(area_m) = -1/beta * (-beta * weights[m]) = weights[m]
    # d(-soft_min)/d(area_m) = -weights[m]
    
    # Now d(area_m)/d(bary) for each triangle m
    grad = np.zeros(N * 2)
    
    for m in range(len(areas)):
        if weights[m] < 1e-15:
            continue
        tri = tri_idx[m]
        i, j, k = tri
        
        ui, uj, uk = u[i], u[j], u[k]
        vi, vj, vk = v[i], v[j], v[k]
        wi, wj, wk = w[i], w[j], w[k]
        
        sign = np.sign(areas[m])
        if sign == 0:
            continue
            
        d_det_dui = (vj*wk - vk*wj) - (uj*vk - uk*vj)
        d_det_dvi = -(uj*wk - uk*wj) - (uj*vk - uk*vj)
        d_det_duj = -vi*wk + wi*vk
        d_det_dvj = ui*wk + wi*uk
        d_det_duk = -ui*wj + vi*wj - wi*vj
        d_det_dvk = -ui*vj + vi*vj - wi*uj
        
        w_m = weights[m]
        
        grad[i*2] += -w_m * sign * d_det_dui
        grad[i*2+1] += -w_m * sign * d_det_dvi
        grad[j*2] += -w_m * sign * d_det_duj
        grad[j*2+1] += -w_m * sign * d_det_dvj
        grad[k*2] += -w_m * sign * d_det_duk
        grad[k*2+1] += -w_m * sign * d_det_dvk
    
    return grad

def epigraph_obj(x):
    """
    Epigraph formulation: minimize -t subject to t <= all areas.
    x = [bary_flat (22), t (1)]
    """
    bary_flat = x[:N*2]
    t = x[N*2]
    
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    # Penalize violations: t > areas[i]
    violations = np.maximum(0, t - areas)
    penalty = 1e6 * np.sum(violations**2)
    
    return -t + penalty

def generate_hexagonal_init():
    """
    Generate a hexagonal lattice-like initial configuration.
    """
    best_bary = None
    best_val = -np.inf
    
    scales = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    offsets = [0.0, 0.05, 0.1]
    
    for scale in scales:
        for offset_u in offsets:
            for offset_v in offsets:
                pts = []
                max_k = int(1.0 / scale) + 2
                for k in range(max_k + 1):
                    w_val = k * scale
                    if w_val > 1.0:
                        break
                    num_pts_in_row = int((1 - w_val) / scale) + 1
                    for j in range(num_pts_in_row):
                        u_val = j * scale + offset_u
                        v_val = (1 - w_val) - u_val + offset_v
                        if u_val >= -1e-9 and v_val >= -1e-9 and u_val + v_val <= 1 + 1e-9:
                            pts.append([u_val, v_val])
                
                if len(pts) >= N:
                    rng = np.random.default_rng(42 + int(scale*100) + int(offset_u*100) + int(offset_v*100))
                    for _ in range(3):
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

def refine_local(bary_init, time_limit=5.0):
    """
    Refine a configuration using fast local optimization stages with graduated soft-min.
    """
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    
    bary_current = bary_init.copy()
    current_val = -hard_min_obj(bary_current.flatten())
    
    # Stage 1: Graduated soft-min with L-BFGS-B using analytical gradients (beta = 20 -> 50 -> 200)
    for beta in [20.0, 50.0, 200.0]:
        if time.time() - start_time > time_limit * 0.4:
            break
        try:
            result = minimize(
                soft_min_obj,
                bary_current.flatten(),
                args=(beta,),
                jac=lambda x, b: soft_min_grad(x, b),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-11}
            )
            candidate = clamp_bary(result.x.reshape(N, 2))
            val = -hard_min_obj(candidate.flatten())
            if val > current_val:
                bary_current = candidate
                current_val = val
        except Exception:
            pass
    
    if time.time() - start_time > time_limit * 0.6:
        return bary_current
    
    # Stage 2: Epigraph formulation with auxiliary t variable
    if time.time() - start_time < time_limit * 0.8:
        try:
            # Initialize t with current best min area
            areas = calc_areas_bary(bary_current)
            t_init = np.min(areas) * 0.9
            x_init = np.concatenate([bary_current.flatten(), [t_init]])
            
            epigraph_bounds = [(0, 1)] * (N * 2) + [(0, 1)]
            
            result_ep = minimize(
                epigraph_obj,
                x_init,
                method='L-BFGS-B',
                bounds=epigraph_bounds,
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            candidate_bary = clamp_bary(result_ep.x[:N*2].reshape(N, 2))
            val = -hard_min_obj(candidate_bary.flatten())
            if val > current_val:
                bary_current = candidate_bary
                current_val = val
        except Exception:
            pass
    
    # Stage 3: Nelder-Mead on hard min (most effective for this problem)
    if time.time() - start_time < time_limit * 0.95:
        try:
            result_nm = minimize(
                hard_min_obj,
                bary_current.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-15, 'adaptive': True}
            )
            candidate = clamp_bary(result_nm.x.reshape(N, 2))
            val = -hard_min_obj(candidate.flatten())
            if val > current_val:
                bary_current = candidate
                current_val = val
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
    
    # 2. Global Search with Differential Evolution (primary global search)
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
                seed=123,
                maxiter=40,
                popsize=15,
                tol=1e-7,
                polish=False,
                mutation=(0.3, 1.0),
                recombination=0.7
            )
            
            de_bary = clamp_bary(result_de.x.reshape(N, 2))
            refined_de = refine_local(de_bary, time_limit=8.0)
            update_best(refined_de)
        except Exception:
            pass
    
    # 3. Multi-start: random + perturbed restarts with local refinement
    rng = np.random.default_rng(123)
    restart_count = 0
    while time.time() < deadline - 20:
        restart_count += 1
        if best_bary is not None and restart_count % 2 == 0:
            # Perturbation of best solution
            init = generate_perturbed_init(best_bary, sigma=0.04, seed=2000 + restart_count)
        else:
            # Random restart
            init = generate_random_init(seed=3000 + restart_count)
        
        remaining = deadline - 20 - time.time()
        if remaining < 4:
            break
            
        t_limit = min(6.0, remaining)
        refined = refine_local(init, time_limit=t_limit)
        update_best(refined)

    # Final high-precision polish on the best candidate
    if best_bary is not None and time.time() < deadline - 8:
        try:
            result_nm1 = minimize(
                hard_min_obj,
                best_bary.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-13, 'fatol': 1e-15, 'adaptive': True}
            )
            update_best(clamp_bary(result_nm1.x.reshape(N, 2)))
        except Exception:
            pass
        
        if best_bary is not None and time.time() < deadline - 4:
            try:
                bounds = [(0, 1)] * (N * 2)
                result_slsqp = minimize(
                    soft_min_obj,
                    best_bary.flatten(),
                    args=(100.0,),
                    jac=lambda x, b: soft_min_grad(x, b),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-13}
                )
                update_best(clamp_bary(result_slsqp.x.reshape(N, 2)))
            except Exception:
                pass
        
        if best_bary is not None and time.time() < deadline - 2:
            try:
                result_nm2 = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
                )
                update_best(clamp_bary(result_nm2.x.reshape(N, 2)))
            except Exception:
                pass

    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart
