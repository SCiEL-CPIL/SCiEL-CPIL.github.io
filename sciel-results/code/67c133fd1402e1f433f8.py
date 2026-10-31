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

def subgradient_push(bary, max_iters=200, lr_init=0.01, lr_decay=0.995):
    """
    Subgradient optimization: push points to increase the minimum triangle area.
    The subgradient of min_area w.r.t. each point is the average of gradients
    of the smallest-area triangles containing that point.
    """
    bary = bary.copy()
    lr = lr_init
    best_val = -np.min(calc_areas_bary(bary))
    best_bary = bary.copy()
    
    for it in range(max_iters):
        areas = calc_areas_bary(bary)
        min_area = np.min(areas)
        
        # Find all triangles with area close to minimum
        threshold = min_area * 1.05 + 1e-12
        active = areas <= threshold
        
        if not np.any(active):
            break
        
        # Compute subgradient for each point
        grad = np.zeros((N, 2))
        
        # For each active triangle, compute gradient contribution
        for idx in np.where(active)[0]:
            i, j, k = tri_idx[idx]
            u_i, v_i = bary[i, 0], bary[i, 1]
            u_j, v_j = bary[j, 0], bary[j, 1]
            u_k, v_k = bary[k, 0], bary[k, 1]
            
            # Area determinant (signed)
            det = (u_i * (v_j * (1-u_k-v_k) - v_k * (1-u_j-v_j)) -
                   v_i * (u_j * (1-u_k-v_k) - u_k * (1-u_j-v_j)) +
                   (1-u_i-v_i) * (u_j * v_k - u_k * v_j))
            
            sign = np.sign(det) if abs(det) > 1e-15 else 1.0
            
            # Gradient of |det| w.r.t. barycentric coords
            # d|det|/du_i = sign * (v_j * w_k - v_k * w_j)
            # d|det|/dv_i = sign * (w_j * u_k - w_k * u_j)
            # etc.
            
            w_i = 1 - u_i - v_i
            w_j = 1 - u_j - v_j
            w_k = 1 - u_k - v_k
            
            # Gradient w.r.t. point i
            gi_u = sign * (v_j * w_k - v_k * w_j)
            gi_v = sign * (w_j * u_k - w_k * u_j)
            
            # Gradient w.r.t. point j
            gj_u = sign * (v_k * w_i - v_i * w_k)
            gj_v = sign * (w_k * u_i - w_i * u_k)
            
            # Gradient w.r.t. point k
            gk_u = sign * (v_i * w_j - v_j * w_i)
            gk_v = sign * (w_i * u_j - w_j * u_i)
            
            # Accumulate (normalize by number of active triangles for stability)
            grad[i, 0] += gi_u
            grad[i, 1] += gi_v
            grad[j, 0] += gj_u
            grad[j, 1] += gj_v
            grad[k, 0] += gk_u
            grad[k, 1] += gk_v
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-15:
            grad /= grad_norm
        
        # Update
        bary = bary + lr * grad
        bary = clamp_bary(bary)
        
        # Check improvement
        current_val = -np.min(calc_areas_bary(bary))
        if current_val > best_val:
            best_val = current_val
            best_bary = bary.copy()
        
        lr *= lr_decay
    
    return best_bary

def generate_structured_inits():
    """
    Generate structured initial configurations based on known good patterns.
    """
    inits = []
    
    # Pattern 1: Hexagonal close packing subset
    center = np.array([1.0/3, 1.0/3])
    
    for r1 in [0.12, 0.15, 0.18, 0.22]:
        for r2 in [0.35, 0.45, 0.55]:
            pts = [center.copy()]
            
            for k in range(6):
                theta = np.pi/3 * k + np.pi/6
                dx = r1 * np.cos(theta)
                dy = r1 * np.sin(theta)
                du = -dx / 2 + dy / H
                dv = dx / 2 + dy / H
                pt = center + np.array([du, dv])
                if pt[0] > 0 and pt[1] > 0 and pt[0] + pt[1] < 1:
                    pts.append(pt.copy())
            
            if len(pts) < N:
                for k in range(6):
                    if len(pts) >= N:
                        break
                    theta = np.pi/3 * k
                    dx = r2 * np.cos(theta)
                    dy = r2 * np.sin(theta)
                    du = -dx / 2 + dy / H
                    dv = dx / 2 + dy / H
                    pt = center + np.array([du, dv])
                    if pt[0] > 0.01 and pt[1] > 0.01 and pt[0] + pt[1] < 0.98:
                        pts.append(pt.copy())
            
            if len(pts) == N:
                inits.append(np.array(pts))
    
    # Pattern 2: 3 near vertices, 3 near midpoints, 1 center, 4 distributed
    for vertex_offset in [0.05, 0.08, 0.12]:
        pts = []
        pts.append(np.array([vertex_offset, vertex_offset]))
        pts.append(np.array([1 - vertex_offset, vertex_offset]))
        pts.append(np.array([vertex_offset, 1 - vertex_offset]))
        pts.append(np.array([0.5, vertex_offset]))
        pts.append(np.array([0.5 * (1 - vertex_offset), 0.5 * (1 - vertex_offset)]))
        pts.append(np.array([vertex_offset, 0.5 * (1 - vertex_offset)]))
        pts.append(np.array([1/3, 1/3]))
        for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
            r = 0.25
            dx = r * np.cos(angle)
            dy = r * np.sin(angle)
            du = -dx / 2 + dy / H
            dv = dx / 2 + dy / H
            pt = np.array([1/3, 1/3]) + np.array([du, dv])
            if pt[0] > 0.01 and pt[1] > 0.01 and pt[0] + pt[1] < 0.98:
                pts.append(pt)
        
        if len(pts) == N:
            inits.append(np.array(pts))
    
    # Pattern 3: Grid-like arrangement
    for grid_size in [3, 4]:
        pts = []
        for i in range(grid_size):
            for j in range(grid_size - i):
                if len(pts) < N:
                    u = (i + 0.5) / grid_size * (1 - (j + 0.5) / grid_size)
                    v = (j + 0.5) / grid_size * (1 - (i + 0.5) / grid_size)
                    pts.append(np.array([u, v]))
        
        if len(pts) == N:
            inits.append(np.array(pts))
    
    return inits

def refine_local(bary_init, time_limit=5.0):
    """
    Refine a configuration using multiple optimization stages.
    """
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    
    bary_current = bary_init.copy()
    
    # Stage 1: Subgradient push
    if time.time() - start_time < time_limit * 0.3:
        try:
            bary_sg = subgradient_push(bary_current, max_iters=300, lr_init=0.02, lr_decay=0.99)
            if -hard_min_obj(bary_sg.flatten()) > -hard_min_obj(bary_current.flatten()):
                bary_current = bary_sg
        except Exception:
            pass
    
    # Stage 2: SLSQP on hard min
    if time.time() - start_time < time_limit * 0.6:
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
            
    # Stage 3: More subgradient push
    if time.time() - start_time < time_limit * 0.85:
        try:
            bary_sg2 = subgradient_push(bary_current, max_iters=200, lr_init=0.005, lr_decay=0.995)
            if -hard_min_obj(bary_sg2.flatten()) > -hard_min_obj(bary_current.flatten()):
                bary_current = bary_sg2
        except Exception:
            pass
    
    # Stage 4: Nelder-Mead for final polish
    if time.time() - start_time < time_limit:
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
    
    # 1. Random starts
    for i in range(20):
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        inits.append(np.column_stack([u, v]))
        
    # 2. Structured starts from known patterns
    structured_inits = generate_structured_inits()
    inits.extend(structured_inits)
    
    # 3. More structured: hexagonal grid
    for scale in [0.12, 0.15, 0.2]:
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
            for _ in range(3):
                idx = rng.choice(len(pts), N, replace=False)
                bary = np.array(pts)[idx]
                inits.append(bary)
    
    # 4. Symmetric configurations with perturbations
    base_symmetric = np.array([
        [1/3, 1/3],
        [0.1, 0.1],
        [0.8, 0.1],
        [0.1, 0.8],
        [0.5, 0.1],
        [0.5, 0.8],
        [0.1, 0.5],
        [0.3, 0.3],
        [0.6, 0.3],
        [0.3, 0.6],
        [0.5, 0.5]
    ])
    for perturb in [0.01, 0.02, 0.05]:
        perturbed = base_symmetric + rng.random((N, 2)) * perturb
        perturbed = clamp_bary(perturbed)
        inits.append(perturbed)
    
    # Deduplicate inits (approximately)
    unique_inits = []
    for init in inits:
        init_clamped = clamp_bary(init)
        is_unique = True
        for existing in unique_inits:
            if np.allclose(init_clamped, existing, atol=0.05):
                is_unique = False
                break
        if is_unique:
            unique_inits.append(init_clamped)
    
    inits = unique_inits
    
    # Refine each initial point with time budget
    time_per_init_base = 2.5
    for i, init in enumerate(inits):
        if time.time() > deadline - 15:
            break
            
        remaining_inits = len(inits) - i
        time_per_init = min(time_per_init_base, (deadline - time.time() - 15) / max(1, remaining_inits))
        if time_per_init < 0.3:
            time_per_init = 0.3
            
        refined = refine_local(init, time_limit=time_per_init)
        update_best(refined)
        
    # If we have time, do a global search with differential evolution
    if best_bary is not None and time.time() < deadline - 25:
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
                maxiter=40,
                tol=1e-10,
                popsize=12,
                mutation=(0.3, 1.0),
                recombination=0.7,
                polish=False,
                update_interval=1,
                init='sobol'
            )
            if result_de.success or result_de.fun < hard_min_obj(best_bary.flatten()):
                bary_de = clamp_bary(result_de.x.reshape(N, 2))
                if update_best(bary_de):
                    if time.time() < deadline - 8:
                        refined_de = refine_local(bary_de, time_limit=min(6.0, deadline - time.time() - 5))
                        update_best(refined_de)
        except Exception:
            pass
        
    # Final high-precision polish on the best found
    if best_bary is not None and time.time() < deadline - 6:
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
                bary_final_sg = subgradient_push(best_bary, max_iters=100, lr_init=0.003, lr_decay=0.998)
                if -hard_min_obj(bary_final_sg.flatten()) > -hard_min_obj(best_bary.flatten()):
                    update_best(bary_final_sg)
            except Exception:
                pass

    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart

# EVOLVE-BLOCK-END
