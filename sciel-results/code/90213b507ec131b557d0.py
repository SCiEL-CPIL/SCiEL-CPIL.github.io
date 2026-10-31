import numpy as np
from scipy.optimize import minimize, differential_evolution
from itertools import combinations
import time

N = 11
TIME_LIMIT = 148.0  # Safety margin for 150s limit

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
    Uses numerically stable computation.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    if np.any(areas < 1e-15):
        return 1e10
    
    # Work with -areas for stability (we want to minimize -soft_min = maximize soft_min)
    neg_areas = -areas
    max_neg = np.max(neg_areas)
    
    # If max_neg is very large (areas very small), fall back to hard min
    if -max_neg < 1e-12:
        return -np.min(areas)
    
    # log-sum-exp of (-beta * areas) = beta * log-sum-exp(areas) with sign flip
    # We want: -log(sum(exp(-beta*areas))) / beta
    # = -(log(sum(exp(-beta*areas)))) / beta
    # Let z_i = -beta * areas_i
    # log(sum(exp(z_i))) = z_max + log(sum(exp(z_i - z_max)))
    # where z_max = max(z_i) = -beta * min(areas)
    
    z = -beta * areas
    z_max = np.max(z)
    
    if z_max > 700:
        # Numerical overflow risk; use hard min approximation
        return -np.min(areas)
        
    exp_terms = np.exp(z - z_max)
    log_sum = z_max + np.log(np.sum(exp_terms))
    soft_min_val = -log_sum / beta
    return -soft_min_val

def epigraph_obj(x, beta=100.0):
    """
    Epigraph formulation: x = [bary_flat (22), t (1)]
    Minimize -t subject to soft constraints: -log-sum-exp(-beta*(areas - t)) >= 0
    This is equivalent to: all areas >= t (softly).
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
    
    # Soft constraint: -log-sum-exp(-beta * diff) should be <= 0 (i.e., log-sum-exp >= 0)
    # Actually we want: sum(exp(-beta*(areas_i - t))) <= 1 for all to be >= t
    # Simpler: just use soft_min directly on (areas - t)
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

def energy_obj(bary_flat, alpha=2.0):
    """
    Smooth energy objective: sum of (1/area_i)^alpha for all triangles.
    Lower is better. This is smooth and differentiable (where areas > 0).
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    if np.any(areas < 1e-15):
        return 1e10
    return np.sum(areas ** (-alpha))

def energy_grad(bary_flat, alpha=2.0):
    """
    Analytical gradient of energy_obj with respect to barycentric coordinates.
    d/dx_j [sum_i area_i^{-alpha}] = -alpha * sum_i area_i^{-alpha-1} * d(area_i)/dx_j
    For triangle (i,j,k), area = 0.5 * |det|, and d(area)/dx_j = 0.5 * sign * cross(other_edge, e_j)
    In barycentric: det = u_i*(v_j*w_k - v_k*w_j) - v_i*(u_j*w_k - u_k*w_j) + w_i*(u_j*v_k - u_k*v_j)
    d(det)/du_i = v_j*w_k - v_k*w_j
    d(det)/du_j = -(u_i*w_k - u_k*w_i)
    d(det)/du_k = u_i*v_j - u_j*v_i
    (cyclic for v and w with w = 1-u-v)
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    areas = calc_areas_bary(bary)
    if np.any(areas < 1e-15):
        return np.zeros_like(bary_flat)
    
    # weight_i = alpha * area_i^{-alpha-1}
    weights = alpha * areas ** (-alpha - 1)
    
    grad = np.zeros((N, 2))
    
    ui = u[tri_idx[:, 0]]
    vi = v[tri_idx[:, 0]]
    wi = w[tri_idx[:, 0]]
    
    uj = u[tri_idx[:, 1]]
    vj = v[tri_idx[:, 1]]
    wj = w[tri_idx[:, 1]]
    
    uk = u[tri_idx[:, 2]]
    vk = v[tri_idx[:, 2]]
    wk = w[tri_idx[:, 2]]
    
    # det = ui*(vj*wk - vk*wj) - vi*(uj*wk - uk*wj) + wi*(uj*vk - uk*vj)
    det = ui * (vj * wk - vk * wj) - vi * (uj * wk - uk * wj) + wi * (uj * vk - uk * vj)
    
    # sign of det (for absolute value)
    sign = np.sign(det)
    sign[sign == 0] = 1.0
    
    # d(det)/du_i = vj*wk - vk*wj
    dud_i = vj * wk - vk * wj
    # d(det)/du_j = -(ui*wk - uk*wi)
    dud_j = -(ui * wk - uk * wi)
    # d(det)/du_k = ui*vj - uj*vi
    dud_k = ui * vj - uj * vi
    
    # d(det)/dv_i = -(uj*wk - uk*wj)
    dvd_i = -(uj * wk - uk * wj)
    # d(det)/dv_j = ui*wk - uk*wi
    dvd_j = ui * wk - uk * wi
    # d(det)/dv_k = -(ui*vj - uj*vi)
    dvd_k = -(ui * vj - uj * vi)
    
    # Since w = 1-u-v, d(det)/dw_i = dud_i + dvd_i, etc.
    # d(det)/du_i_total = dud_i - (dud_i + dvd_i) = -dvd_i
    # Wait, let me redo. w_i = 1 - u_i - v_i
    # d(det)/du_i = dud_i + dw_i/d u_i * d(det)/dw_i = dud_i - d(det)/dw_i
    # d(det)/dw_i = dud_i + dvd_i (from the expression)
    # So d(det)/du_i_effective = dud_i - (dud_i + dvd_i) = -dvd_i
    # d(det)/dv_i_effective = dvd_i - (dud_i + dvd_i) = -dud_i
    
    # Actually let me just compute directly:
    # det = ui*(vj*wk - vk*wj) - vi*(uj*wk - uk*wj) + wi*(uj*vk - uk*vj)
    # where wi = 1-ui-vi, wj = 1-uj-vj, wk = 1-uk-vk
    
    # d(det)/dui: 
    #   term1: (vj*wk - vk*wj) + ui*(0 - vk*dwj/d ui) - vi*(0 - uk*dwj/d ui) + (1-ui-vi)*(0) + (-1)*(uj*vk - uk*vj)
    #   dwj/d ui = 0, dwk/d ui = 0
    #   = (vj*wk - vk*wj) - (uj*vk - uk*vj)
    
    # This is getting complex. Let me use the simpler approach:
    # d(det)/du_i = vj*wk - vk*wj - (uj*vk - uk*vj)  [from expanding w_i = 1-u_i-v_i]
    # d(det)/dv_i = -(uj*wk - uk*wj) + (uj*vk - uk*vj)
    
    # Let me verify with a simple case. Actually, let me just use numerical differentiation
    # for safety and simplicity, or use a cleaner formulation.
    
    # Simpler: use the fact that in barycentric coords on a simplex,
    # the area of triangle (i,j,k) is 0.5*|det of [u_i,v_i,w_i; u_j,v_j,w_j; u_k,v_k,w_k]|
    # and the gradient with respect to (u_i, v_i) can be computed as:
    # dA/du_i = 0.5 * sign(det) * (vj*wk - vk*wj - (uj*vk - uk*vj))
    # dA/dv_i = 0.5 * sign(det) * (-(uj*wk - uk*wj) + (uj*vk - uk*vj))
    
    # But since we're minimizing sum of (2*area)^(-alpha) where area = 0.5*|det|,
    # d/dx [(0.5*|det|)^(-alpha)] = -alpha*(0.5*|det|)^(-alpha-1)*0.5*sign(det)*d(det)/dx
    # = -alpha*0.5^(1+alpha)*|det|^(-alpha-1)*sign(det)*d(det)/dx
    # = -alpha*0.5^(1+alpha)*det^(-alpha)*d(det)/dx  (since sign(det)*|det|^(-alpha-1) = det^(-alpha)/|det| ... )
    
    # Let me just use: d/dx [area^(-alpha)] where area = 0.5*|det|
    # = -alpha * area^(-alpha-1) * d(area)/dx
    # d(area)/dx = 0.5 * sign(det) * d(det)/dx
    
    # So d(energy)/dx = -alpha * sum_i area_i^(-alpha-1) * 0.5 * sign(det_i) * d(det_i)/dx
    
    # For point p in triangle (i,j,k):
    # d(det)/du_p = v_j*w_k - v_k*w_j - (u_j*v_k - u_k*v_j)  [when p=i]
    # d(det)/dv_p = -(u_j*w_k - u_k*w_j) + (u_j*v_k - u_k*v_j)  [when p=i]
    # (and cyclic for j and k)
    
    # Let me define helper terms:
    # For triangle (i,j,k):
    #   A_ik = v_j*w_k - v_k*w_j  (cofactor of u_i)
    #   A_jk = -(u_i*w_k - u_k*w_i)  (cofactor of u_j)  
    #   A_ij = u_i*v_j - u_j*v_i  (cofactor of u_k)
    
    # d(det)/du_i = A_ik - (u_j*v_k - u_k*v_j)  -- but u_j*v_k - u_k*v_j = A_ij
    # So d(det)/du_i = A_ik - A_ij
    # d(det)/dv_i = -(u_j*w_k - u_k*w_j) + (u_j*v_k - u_k*v_j) = -A_jk + A_ij
    # Wait, A_jk = -(u_i*w_k - u_k*w_i), so -A_jk = u_i*w_k - u_k*w_i
    # That's not right either.
    
    # Let me just compute numerically for safety.
    # Actually, for performance, let me use a cleaner derivation.
    
    # In barycentric coordinates, the determinant is:
    # det = u_i*v_j*w_k + u_j*v_k*w_i + u_k*v_i*w_j - u_i*v_k*w_j - u_j*v_i*w_k - u_k*v_j*w_i
    
    # d(det)/du_i = v_j*w_k - v_k*w_j
    # d(det)/dv_i = -(u_j*w_k - u_k*w_j)  ... wait
    # d(det)/dv_i = u_k*w_j - u_j*w_k
    # d(det)/dw_i = u_j*v_k - u_k*v_j
    
    # Since w_i = 1 - u_i - v_i:
    # d(det)/du_i (effective) = v_j*w_k - v_k*w_j - (u_j*v_k - u_k*v_j)
    # d(det)/dv_i (effective) = u_k*w_j - u_j*w_k - (u_j*v_k - u_k*v_j)
    
    # Let me define:
    #   T1 = v_j*w_k - v_k*w_j
    #   T2 = u_k*w_j - u_j*w_k  
    #   T3 = u_j*v_k - u_k*v_j
    
    # d(det)/du_i_eff = T1 - T3
    # d(det)/dv_i_eff = T2 - T3
    
    # Similarly for j:
    #   d(det)/du_j_eff = (u_i*w_k - u_k*w_i) - T3  ... no
    #   d(det)/du_j = -(u_i*w_k - u_k*w_i) = u_k*w_i - u_i*w_k
    #   d(det)/dv_j = u_i*w_k - u_k*w_i
    #   d(det)/dw_j = u_k*v_i - u_i*v_k
    #   d(det)/du_j_eff = (u_k*w_i - u_i*w_k) - (u_k*v_i - u_i*v_k)
    #   d(det)/dv_j_eff = (u_i*w_k - u_k*w_i) - (u_k*v_i - u_i*v_k)
    
    # This is getting messy. Let me just use numerical gradients for robustness.
    eps = 1e-7
    bary_num = bary.copy()
    grad = np.zeros((N, 2))
    
    for p in range(N):
        for d in range(2):
            bary_num[p, d] += eps
            bary_num = clamp_bary(bary_num)
            areas_p = calc_areas_bary(bary_num)
            if np.any(areas_p < 1e-15):
                bary_num[p, d] -= eps
                continue
            val_p = np.sum(areas_p ** (-alpha))
            bary_num[p, d] -= 2 * eps
            bary_num = clamp_bary(bary_num)
            areas_m = calc_areas_bary(bary_num)
            if np.any(areas_m < 1e-15):
                bary_num[p, d] += eps
                continue
            val_m = np.sum(areas_m ** (-alpha))
            grad[p, d] = (val_p - val_m) / (2 * eps)
            bary_num[p, d] += eps
    
    return grad.flatten()

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

def coordinated_ascent(bary, max_iter=50, step_size=0.001, time_budget=2.0):
    """
    Coordinated ascent: find the smallest triangle and push its vertices apart.
    This is a direct heuristic that can escape local minima of the min objective.
    """
    start_time = time.time()
    bary = bary.copy()
    cur_val = -hard_min_obj(bary.flatten())
    best_val = cur_val
    best_bary = bary.copy()
    
    for iteration in range(max_iter):
        if time.time() - start_time > time_budget:
            break
        
        areas = calc_areas_bary(bary)
        min_idx = np.argmin(areas)
        
        # Get the three points forming the smallest triangle
        i, j, k = tri_idx[min_idx]
        
        # Compute centroid of the smallest triangle
        centroid = (bary[i] + bary[j] + bary[k]) / 3.0
        
        # For each vertex, compute direction away from centroid
        # and add a small step in that direction
        new_bary = bary.copy()
        for idx in [i, j, k]:
            direction = bary[idx] - centroid
            norm = np.linalg.norm(direction)
            if norm > 1e-10:
                direction = direction / norm
                new_bary[idx] += step_size * direction
        
        new_bary = clamp_bary(new_bary)
        new_val = -hard_min_obj(new_bary.flatten())
        
        if new_val > cur_val:
            cur_val = new_val
            bary = new_bary.copy()
            if cur_val > best_val:
                best_val = cur_val
                best_bary = bary.copy()
        else:
            # Try smaller step
            new_bary2 = bary.copy()
            for idx in [i, j, k]:
                direction = bary[idx] - centroid
                norm = np.linalg.norm(direction)
                if norm > 1e-10:
                    direction = direction / norm
                    new_bary2[idx] += step_size * 0.5 * direction
            new_bary2 = clamp_bary(new_bary2)
            new_val2 = -hard_min_obj(new_bary2.flatten())
            if new_val2 > cur_val:
                cur_val = new_val2
                bary = new_bary2.copy()
                if cur_val > best_val:
                    best_val = cur_val
                    best_bary = bary.copy()
            else:
                # No improvement; break early
                break
    
    return best_bary

def refine_local(bary_init, time_limit=5.0):
    """
    Refine a configuration using energy-based optimization + soft-min + coordinated ascent + NM.
    """
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    
    bary_current = bary_init.copy()
    cur_val = -hard_min_obj(bary_current.flatten())
    
    # Stage 1: Energy-based optimization with L-BFGS-B (smooth, fast)
    try:
        result1 = minimize(
            energy_obj,
            bary_current.flatten(),
            jac=lambda x: energy_grad(x, alpha=2.0),
            args=(2.0,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-9}
        )
        new_bary = clamp_bary(result1.x.reshape(N, 2))
        new_val = -hard_min_obj(new_bary.flatten())
        if new_val > cur_val:
            bary_current = new_bary
            cur_val = new_val
    except Exception:
        pass
    
    if time.time() - start_time > time_limit * 0.25:
        return bary_current

    # Stage 2: Soft-min refinement (graduated beta)
    for beta in [30.0, 100.0, 300.0]:
        if time.time() - start_time > time_limit * 0.55:
            break
        try:
            result2 = minimize(
                soft_min_obj,
                bary_current.flatten(),
                args=(beta,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            new_bary = clamp_bary(result2.x.reshape(N, 2))
            new_val = -hard_min_obj(new_bary.flatten())
            if new_val > cur_val:
                bary_current = new_bary
                cur_val = new_val
        except Exception:
            pass
    
    if time.time() - start_time > time_limit * 0.65:
        return bary_current

    # Stage 3: Coordinated ascent to escape local minima
    try:
        ca_result = coordinated_ascent(bary_current, max_iter=150, step_size=0.002, time_budget=time_limit * 0.15)
        new_val = -hard_min_obj(ca_result.flatten())
        if new_val > cur_val:
            bary_current = ca_result
            cur_val = new_val
    except Exception:
        pass
        
    if time.time() - start_time > time_limit * 0.8:
        return bary_current

    # Stage 4: Hard min refinement with Nelder-Mead
    try:
        result4 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
        )
        bary_current = clamp_bary(result4.x.reshape(N, 2))
    except Exception:
        pass
        
    if time.time() - start_time > time_limit * 0.9:
        return bary_current

    # Stage 5: Final high-precision Nelder-Mead polish
    try:
        result5 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 1500, 'xatol': 1e-15, 'fatol': 1e-18, 'adaptive': True}
        )
        bary_current = clamp_bary(result5.x.reshape(N, 2))
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

    # 1. Try Hexagonal Init with energy-based refinement
    hex_init = generate_hexagonal_init()
    refined_hex = refine_local(hex_init, time_limit=10.0)
    update_best(refined_hex)

    # 2. Try epigraph refinement on the same hex init
    if time.time() < deadline - 45:
        refined_ep = refine_epigraph(hex_init, time_limit=8.0)
        update_best(refined_ep)

    # 3. Global Search with Differential Evolution
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
                maxiter=40,
                popsize=15,
                tol=1e-6,
                polish=False
            )

            de_bary = clamp_bary(result_de.x.reshape(N, 2))
            refined_de = refine_local(de_bary, time_limit=10.0)
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
            init = generate_perturbed_init(best_bary, sigma=0.05, seed=2000 + restart_count)
        else:
            # Fresh random start
            init = generate_random_init(seed=3000 + restart_count)
        
        refined = refine_local(init, time_limit=4.0)
        update_best(refined)

    # 5. Final high-precision polish
    if best_bary is not None and time.time() < deadline - 5:
        # Try epigraph on best
        refined_final_ep = refine_epigraph(best_bary, time_limit=5.0)
        update_best(refined_final_ep)
        
        # Coordinated ascent on best
        if best_bary is not None and time.time() < deadline - 3:
            try:
                ca_best = coordinated_ascent(best_bary, max_iter=300, step_size=0.001, time_budget=2.5)
                update_best(ca_best)
            except Exception:
                pass
        
        if best_bary is not None and time.time() < deadline - 2:
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
        
        if best_bary is not None and time.time() < deadline - 1.5:
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
        
        # Final coordinated ascent + NM polish
        if best_bary is not None and time.time() < deadline - 0.5:
            try:
                ca_final = coordinated_ascent(best_bary, max_iter=150, step_size=0.0005, time_budget=1.0)
                update_best(ca_final)
            except Exception:
                pass

    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart
