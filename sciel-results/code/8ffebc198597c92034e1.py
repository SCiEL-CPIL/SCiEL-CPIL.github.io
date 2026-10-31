import numpy as np
from scipy.optimize import differential_evolution, minimize
from itertools import combinations
import time

N = 11
TIME_LIMIT = 145.0

tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)

V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
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
    u = np.clip(u, eps, 1.0 - eps)
    v = np.clip(v, eps, 1.0 - eps)
    sums = u + v
    invalid = sums > (1.0 - 2 * eps)
    if np.any(invalid):
        scale = (1.0 - 2 * eps) / np.maximum(sums, 1e-15)
        u[invalid] *= scale[invalid]
        v[invalid] *= scale[invalid]
        u = np.clip(u, eps, 1.0 - eps)
        v = np.clip(v, eps, 1.0 - eps)
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
            for _ in range(10):
                if len(pts) == N:
                    bary = np.array(pts, dtype=float)
                else:
                    idx = rng.choice(len(pts), N, replace=False)
                    bary = np.array(pts, dtype=float)[idx]
                bary = clamp_bary(bary)
                val = -hard_min_obj(bary.flatten())
                if val > best_val:
                    best_val = val
                    best_bary = bary.copy()
    if best_bary is None:
        rng = np.random.default_rng(42)
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        best_bary = clamp_bary(np.column_stack([u, v]))
    return best_bary

def generate_random_init(seed=None):
    rng = np.random.default_rng(seed)
    u = rng.random(N)
    v = rng.random(N) * (1 - u)
    return clamp_bary(np.column_stack([u, v]))

def generate_perturbed_init(base_bary, sigma=0.04, seed=None):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, base_bary.shape)
    return clamp_bary(base_bary + noise)

def generate_boundary_interior_init(seed=None):
    rng = np.random.default_rng(seed)
    bary = np.zeros((N, 2))
    bary[0] = [0.08, 0.08]
    bary[1] = [0.92, 0.08]
    bary[2] = [0.08, 0.92]
    for i in range(3, 7):
        edge = (i - 3) % 3
        t = rng.uniform(0.25, 0.75)
        if edge == 0:
            bary[i] = [t, 0.03]
        elif edge == 1:
            bary[i] = [0.03, t]
        else:
            bary[i] = [1.0 - t - 0.03, 0.03]
    for i in range(7, 11):
        u = rng.uniform(0.15, 0.75)
        v = rng.uniform(0.15, 0.75 * (1 - u))
        bary[i] = [u, v]
    return clamp_bary(bary)

def refine_local(bary_init, time_limit=5.0):
    start_time = time.time()
    bounds = [(1e-9, 1 - 1e-9)] * (N * 2)
    bary_current = bary_init.copy()

    # Stage 1: Soft-min with L-BFGS-B
    try:
        result1 = minimize(
            soft_min_obj, bary_current.flatten(), args=(50.0,),
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        if result1.fun < -hard_min_obj(bary_current.flatten()):
            bary_current = clamp_bary(result1.x.reshape(N, 2))
    except Exception:
        pass

    if time.time() - start_time > time_limit * 0.35:
        return bary_current

    # Stage 2: Nelder-Mead on hard min
    try:
        result2 = minimize(
            hard_min_obj, bary_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 1500, 'xatol': 1e-12, 'fatol': 1e-16, 'adaptive': True}
        )
        bary_current = clamp_bary(result2.x.reshape(N, 2))
    except Exception:
        pass

    if time.time() - start_time > time_limit * 0.7:
        return bary_current

    # Stage 3: SLSQP polish
    try:
        result3 = minimize(
            hard_min_obj, bary_current.flatten(),
            method='SLSQP', bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-14}
        )
        bary_current = clamp_bary(result3.x.reshape(N, 2))
    except Exception:
        pass

    return bary_current

def coordinate_descent_polish(bary, time_limit=3.0):
    start_time = time.time()
    bary = bary.copy()
    for _ in range(80):
        if time.time() - start_time > time_limit:
            break
        improved = False
        for i in range(N):
            def obj_i(params):
                bary_temp = bary.copy()
                bary_temp[i] = params
                bary_temp = clamp_bary(bary_temp)
                areas = calc_areas_bary(bary_temp)
                return -np.min(areas)
            try:
                result = minimize(
                    obj_i, bary[i], method='Nelder-Mead',
                    options={'maxiter': 80, 'xatol': 1e-10, 'fatol': 1e-14}
                )
                new_pt = clamp_bary(result.x.reshape(1, 2))[0]
                if result.fun < obj_i(bary[i]) - 1e-15:
                    bary[i] = new_pt
                    improved = True
            except Exception:
                pass
        if not improved:
            break
    return bary

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

    # 1. Hexagonal init
    hex_init = generate_hexagonal_init()
    refined_hex = refine_local(hex_init, time_limit=8.0)
    update_best(refined_hex)

    # 2. Boundary-interior init
    bi_init = generate_boundary_interior_init(seed=123)
    refined_bi = refine_local(bi_init, time_limit=6.0)
    update_best(refined_bi)

    # 3. Differential Evolution
    if time.time() < deadline - 35:
        try:
            def de_obj(x):
                bary = x.reshape(N, 2)
                bary = clamp_bary(bary)
                areas = calc_areas_bary(bary)
                return -np.min(areas)

            bounds = [(1e-9, 1 - 1e-9)] * (N * 2)
            result_de = differential_evolution(
                de_obj, bounds, seed=42,
                maxiter=30, popsize=12, tol=1e-6, polish=False
            )
            de_bary = clamp_bary(result_de.x.reshape(N, 2))
            refined_de = refine_local(de_bary, time_limit=8.0)
            update_best(refined_de)
        except Exception:
            pass

    # 4. Multi-start with varied initializations
    restart_count = 0
    while time.time() < deadline - 18:
        restart_count += 1
        if best_bary is not None and restart_count % 2 == 0:
            init = generate_perturbed_init(best_bary, sigma=0.03, seed=2000 + restart_count)
        elif best_bary is not None and restart_count % 3 == 1:
            init = generate_boundary_interior_init(seed=4000 + restart_count)
        else:
            init = generate_random_init(seed=3000 + restart_count)

        refined = refine_local(init, time_limit=4.0)
        update_best(refined)

    # 5. Final high-precision polish
    if best_bary is not None and time.time() < deadline - 10:
        polished_cd = coordinate_descent_polish(best_bary, time_limit=5.0)
        update_best(polished_cd)

    if best_bary is not None and time.time() < deadline - 5:
        try:
            bounds = [(1e-9, 1 - 1e-9)] * (N * 2)
            result_slsqp = minimize(
                hard_min_obj, best_bary.flatten(),
                method='SLSQP', bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-15}
            )
            update_best(clamp_bary(result_slsqp.x.reshape(N, 2)))
        except Exception:
            pass

        if best_bary is not None and time.time() < deadline - 2:
            try:
                result_nm = minimize(
                    hard_min_obj, best_bary.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-14, 'fatol': 1e-18, 'adaptive': True}
                )
                update_best(clamp_bary(result_nm.x.reshape(N, 2)))
            except Exception:
                pass

        if best_bary is not None and time.time() < deadline - 1:
            try:
                bounds = [(1e-9, 1 - 1e-9)] * (N * 2)
                result_final = minimize(
                    hard_min_obj, best_bary.flatten(),
                    method='SLSQP', bounds=bounds,
                    options={'maxiter': 2000, 'ftol': 1e-16}
                )
                update_best(clamp_bary(result_final.x.reshape(N, 2)))
            except Exception:
                pass

    if best_bary is None:
        best_bary = generate_random_init(seed=0)

    cart = bary_to_cart(best_bary)
    return cart
