import numpy as np
from scipy.optimize import differential_evolution, minimize
from itertools import combinations
import time

N = 11
# The evaluator kills after 150s. We use a safety margin.
TIME_LIMIT = 140.0

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

def clamp_bary(bary, eps=1e-12):
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

def energy_obj(bary_flat, p=4.0):
    """
    Energy objective: Sum of inverse powers of areas.
    Minimizing this maximizes the minimum area.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    # Avoid division by zero
    areas_safe = np.maximum(areas, 1e-15)
    
    # Energy is sum(1/area^p)
    # We minimize this.
    return np.sum(areas_safe ** (-p))

def energy_grad(bary_flat, p=4.0):
    """
    Gradient of the energy objective.
    E = sum_j A_j^{-p}
    dE/dx = sum_j (-p) A_j^{-p-1} * dA_j/dx
    
    Note: A_j = |det_j|. dA_j/dx = sign(det_j) * d(det_j)/dx.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    a_idx = tri_idx[:, 0]
    b_idx = tri_idx[:, 1]
    c_idx = tri_idx[:, 2]
    
    ua = u[a_idx]; va = v[a_idx]; wa = w[a_idx]
    ub = u[b_idx]; vb = v[b_idx]; wb = w[b_idx]
    uc = u[c_idx]; vc = v[c_idx]; wc = w[c_idx]
    
    # Compute determinants (signed areas * 2)
    dets = ua * (vb * wc - vc * wb) - va * (ub * wc - uc * wb) + wa * (ub * vc - uc * vb)
    areas = np.abs(dets)
    areas_safe = np.maximum(areas, 1e-15)
    
    # Coefficient for each triangle in the gradient sum
    # dE/dA_j = -p * A_j^{-p-1}
    # dE/dx = sum_j (dE/dA_j) * (dA_j/dx)
    # dA_j/dx = sign(det_j) * d(det_j)/dx
    # So coefficient for d(det_j)/dx is: -p * A_j^{-p-1} * sign(det_j)
    
    signs = np.sign(dets)
    coeffs = -p * (areas_safe ** (-p - 1)) * signs
    
    # Initialize gradient
    grad_u = np.zeros(N)
    grad_v = np.zeros(N)
    
    # Derivatives of det w.r.t u and v for each point in the triangle
    # For point a:
    # d(det)/d(u_a) = (vb*wc - vc*wb) - (ub*vc - uc*vb)
    # d(det)/d(v_a) = -(ub*wc - uc*wb) - (ub*vc - uc*vb)
    
    term_ua = (vb * wc - vc * wb) - (ub * vc - uc * vb)
    term_va = -(ub * wc - uc * wb) - (ub * vc - uc * vb)
    
    # For point b:
    # d(det)/d(u_b) = (vc*wa - va*wc) - (uc*va - ua*vc)
    # d(det)/d(v_b) = -(uc*wa - ua*wc) - (uc*va - ua*vc)
    
    term_ub = (vc * wa - va * wc) - (uc * va - ua * vc)
    term_vb = -(uc * wa - ua * wc) - (uc * va - ua * vc)
    
    # For point c:
    # d(det)/d(u_c) = (va*wb - vb*wa) - (ua*vb - ub*va)
    # d(det)/d(v_c) = -(ua*wb - ub*wa) - (ua*vb - ub*va)
    
    term_uc = (va * wb - vb * wa) - (ua * vb - ub * va)
    term_vc = -(ua * wb - ub * wa) - (ua * vb - ub * va)
    
    # Accumulate gradients
    np.add.at(grad_u, a_idx, coeffs * term_ua)
    np.add.at(grad_v, a_idx, coeffs * term_va)
    
    np.add.at(grad_u, b_idx, coeffs * term_ub)
    np.add.at(grad_v, b_idx, coeffs * term_vb)
    
    np.add.at(grad_u, c_idx, coeffs * term_uc)
    np.add.at(grad_v, c_idx, coeffs * term_vc)
    
    # Return gradient as flat array [u0, v0, u1, v1, ...]
    grad_flat = np.empty(N * 2)
    grad_flat[0::2] = grad_u
    grad_flat[1::2] = grad_v
    
    return grad_flat

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
            for _ in range(10):
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

def refine_local(bary_init, time_limit=10.0):
    """
    Refine a configuration using gradient-based energy minimization.
    1. L-BFGS-B on Energy (p=4) with analytical gradient
    2. L-BFGS-B on Energy (p=10) with analytical gradient (sharper)
    3. L-BFGS-B on Hard Min (polish)
    """
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    
    bary_current = bary_init.copy()
    current_val = -hard_min_obj(bary_current.flatten())

    def check_time(frac):
        return time.time() - start_time > time_limit * frac

    # Stage 1: Coarse Energy Minimization (p=4)
    try:
        result1 = minimize(
            energy_obj,
            bary_current.flatten(),
            args=(4.0,),
            jac=energy_grad,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if result1.success or result1.fun < 1e10:
            bary_new = clamp_bary(result1.x.reshape(N, 2))
            val_new = -hard_min_obj(bary_new.flatten())
            if val_new > current_val:
                bary_current = bary_new
                current_val = val_new
    except Exception:
        pass
    
    if check_time(0.3):
        return bary_current

    # Stage 2: Sharper Energy Minimization (p=10)
    try:
        result2 = minimize(
            energy_obj,
            bary_current.flatten(),
            args=(10.0,),
            jac=energy_grad,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        if result2.success or result2.fun < 1e10:
            bary_new = clamp_bary(result2.x.reshape(N, 2))
            val_new = -hard_min_obj(bary_new.flatten())
            if val_new > current_val:
                bary_current = bary_new
                current_val = val_new
    except Exception:
        pass

    if check_time(0.7):
        return bary_current

    # Stage 3: Hard Min Polish (Nelder-Mead is robust here for final steps)
    try:
        result3 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        bary_new = clamp_bary(result3.x.reshape(N, 2))
        val_new = -hard_min_obj(bary_new.flatten())
        if val_new > current_val:
            bary_current = bary_new
            current_val = val_new
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
    refined_hex = refine_local(hex_init, time_limit=15.0)
    update_best(refined_hex)
    
    # 2. Global Search with Differential Evolution
    # Allocate ~40s for DE
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
                maxiter=30, 
                popsize=15,
                tol=1e-6,
                polish=False
            )
            
            de_bary = clamp_bary(result_de.x.reshape(N, 2))
            refined_de = refine_local(de_bary, time_limit=15.0)
            update_best(refined_de)
        except Exception:
            pass
    
    # 3. Multi-start: random + perturbed restarts with local refinement
    rng = np.random.default_rng(123)
    restart_count = 0
    # Allocate remaining time for multi-start, leaving 5s for final polish
    while time.time() < deadline - 10:
        restart_count += 1
        if best_bary is not None and restart_count % 2 == 0:
            # Perturbation of best solution
            init = generate_perturbed_init(best_bary, sigma=0.03, seed=2000 + restart_count)
        else:
            # Random restart
            init = generate_random_init(seed=3000 + restart_count)
        
        remaining = deadline - 10 - time.time()
        if remaining < 5:
            break
            
        t_limit = min(10.0, remaining)
        refined = refine_local(init, time_limit=t_limit)
        update_best(refined)

    # Final high-precision polish on the best candidate
    if best_bary is not None and time.time() < deadline - 2:
        try:
            # SLSQP with tight tolerance on hard min
            bounds = [(0, 1)] * (N * 2)
            result_slsqp = minimize(
                hard_min_obj,
                best_bary.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-14}
            )
            update_best(clamp_bary(result_slsqp.x.reshape(N, 2)))
        except Exception:
            pass
        
        if best_bary is not None and time.time() < deadline - 1:
            try:
                # Nelder-Mead with high iterations for final polish
                result_nm = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
                )
                update_best(clamp_bary(result_nm.x.reshape(N, 2)))
            except Exception:
                pass

    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart

if __name__ == "__main__":
    # Quick test
    start = time.time()
    res = construct()
    print(f"Time taken: {time.time() - start:.2f}s")
    print(f"Result shape: {res.shape}")
