import numpy as np
from scipy.optimize import minimize
from itertools import combinations
import time

N = 11
TIME_LIMIT = 145.0  # safety margin for 150 s wall-clock kill

tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)

H = np.sqrt(3) / 2
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, H])


def bary_to_cart(bary):
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    return np.column_stack([x, y])


def clamp_bary(bary, eps=1e-10):
    u = np.clip(bary[:, 0].copy(), eps, 1.0 - eps)
    v = np.clip(bary[:, 1].copy(), eps, 1.0 - eps)
    sums = u + v
    invalid = sums > (1.0 - eps)
    if np.any(invalid):
        scale = (1.0 - eps) / np.where(sums > 0, sums, 1.0)
        u[invalid] *= scale[invalid]
        v[invalid] *= scale[invalid]
        u = np.clip(u, eps, 1.0 - eps)
        v = np.clip(v, eps, 1.0 - eps)
    return np.column_stack([u, v])


def _area_vec(bary):
    """Compute areas of all 165 triangles from 11 points in barycentric coords."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    ui = u[tri_idx[:, 0]]; vi = v[tri_idx[:, 0]]; wi = w[tri_idx[:, 0]]
    uj = u[tri_idx[:, 1]]; vj = v[tri_idx[:, 1]]; wj = w[tri_idx[:, 1]]
    uk = u[tri_idx[:, 2]]; vk = v[tri_idx[:, 2]]; wk = w[tri_idx[:, 2]]
    det = (ui * (vj * wk - vk * wj)
           - vi * (uj * wk - uk * wj)
           + wi * (uj * vk - uk * vj))
    return np.abs(det)


def calc_areas_bary(bary):
    return _area_vec(clamp_bary(bary))


def hard_min_obj(bary_flat):
    bary = clamp_bary(bary_flat.reshape(N, 2))
    return -np.min(_area_vec(bary))


def soft_min_obj(bary_flat, beta=100.0):
    bary = clamp_bary(bary_flat.reshape(N, 2))
    areas = _area_vec(bary)
    if np.any(areas < 1e-15):
        return 1e10
    scaled = -beta * areas
    m = np.max(scaled)
    if m > 700:
        return -np.min(areas)
    log_sum = m + np.log(np.sum(np.exp(scaled - m)))
    return log_sum / beta


def epigraph_obj(x):
    bary = clamp_bary(x[:2 * N].reshape(N, 2))
    t = x[2 * N]
    areas = _area_vec(bary)
    deficit = np.maximum(0.0, t - areas)
    return t + 200.0 * np.sum(deficit * deficit)


def epigraph_obj_linear(x):
    bary = clamp_bary(x[:2 * N].reshape(N, 2))
    t = x[2 * N]
    areas = _area_vec(bary)
    deficit = np.maximum(0.0, t - areas)
    return t + 1000.0 * np.max(deficit) + 50.0 * np.sum(deficit)


def _val(bary):
    bary = clamp_bary(bary)
    return float(np.min(_area_vec(bary)))


def generate_hexagonal_init():
    """Generate a hexagonal grid pattern init."""
    best_bary = None
    best_val = -np.inf
    for scale in [0.25, 0.3, 0.35, 0.4, 0.45]:
        pts = []
        for k in range(int(1.0 / scale) + 2):
            w_val = k * scale
            if w_val > 1.0 + 1e-9:
                break
            num = int((1 - w_val) / scale) + 1
            for j in range(num):
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
                val = _val(bary)
                if val > best_val:
                    best_val = val
                    best_bary = bary.copy()
    if best_bary is None:
        rng = np.random.default_rng(42)
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        best_bary = np.column_stack([u, v])
    return best_bary


def generate_lattice_k3_plus1():
    """K=3 lattice (10 pts) + 1 interior point."""
    K = 3
    pts = [[i / K, j / K] for i in range(K + 1) for j in range(K + 1 - i)]
    cands = [[0.35, 0.35], [0.45, 0.25], [0.25, 0.45], [0.3, 0.4], [0.4, 0.3],
             [0.5, 0.2], [0.2, 0.5], [0.33, 0.33], [0.4, 0.4]]
    best_bary, best_val = None, -np.inf
    for c in cands:
        b = clamp_bary(np.array(pts + [c]))
        v = _val(b)
        if v > best_val:
            best_val, best_bary = v, b.copy()
    return best_bary


def generate_lattice_k4_minus4():
    """K=4 lattice (15 pts), remove 4 interior to get 11."""
    K = 4
    pts = np.array([[i / K, j / K] for i in range(K + 1) for j in range(K + 1 - i)])
    interior = [i for i, p in enumerate(pts)
                if p[0] > 0.01 and p[1] > 0.01 and p[0] + p[1] < 0.99]
    # Try different removal strategies
    best_bary, best_val = None, -np.inf
    import itertools as it2
    for comb in it2.combinations(interior, 4):
        keep = [i for i in range(len(pts)) if i not in comb]
        b = clamp_bary(pts[keep])
        v = _val(b)
        if v > best_val:
            best_val, best_bary = v, b.copy()
    if best_bary is None:
        return generate_hexagonal_init()
    return best_bary


def generate_random_init(seed=None):
    rng = np.random.default_rng(seed)
    u = rng.random(N)
    v = rng.random(N) * (1 - u)
    return np.column_stack([u, v])


def generate_perturbed_init(base, sigma=0.05, seed=None):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, base.shape)
    return clamp_bary(base + noise)


def generate_sobolev_init():
    """
    Structural change: Use Sobolev optimal transport intuition.
    Minimize second moment (variance) relative to centroid as a proxy,
    then refine. Points should spread evenly.
    """
    # Start from a configuration that minimizes sum of squared distances to centroid
    # This is equivalent to maximizing spread (Fisher information proxy)
    best_bary = None
    best_val = -np.inf
    for seed in [0, 1, 2, 3, 4]:
        rng = np.random.default_rng(seed)
        # Initialize with points on a ring + center
        bary = np.zeros((N, 2))
        # Place points at vertices, edge midpoints, and interior
        bary[0] = [0.1, 0.1]
        bary[1] = [0.9, 0.1]
        bary[2] = [0.1, 0.9]
        # Ring of 8 points
        for i in range(8):
            angle = 2 * np.pi * i / 8
            r = 0.25
            u = 0.5 + r * np.cos(angle)
            v = 0.5 + r * np.sin(angle)
            # Map to barycentric: need u+v <= 1
            # Adjust: use a smaller radius
            u = 0.35 + 0.2 * np.cos(angle)
            v = 0.35 + 0.2 * np.sin(angle)
            if u < 0 or v < 0 or u + v > 1:
                u = max(0.05, min(0.95, u))
                v = max(0.05, min(0.95 - u, v))
            bary[i + 3] = [u, v]
        bary = clamp_bary(bary)
        val = _val(bary)
        if val > best_val:
            best_val, best_bary = val, bary.copy()
    return best_bary


def coordinate_descent(bary_init, time_limit=5.0):
    """
    Structural change: Coordinate descent exploiting piecewise-linear structure.
    For each point, the min area as a function of that point's coordinates is
    piecewise-linear (minimum of linear functions). We do line searches.
    """
    start = time.time()
    bary_cur = bary_init.copy()
    best_local = _val(bary_cur)
    n_iter = 0

    def try_update(b):
        nonlocal bary_cur, best_local
        b = clamp_bary(b)
        v = _val(b)
        if v > best_local:
            bary_cur, best_local = b.copy(), v

    # For each point, try moving it in various directions
    directions = []
    for dx in [-1, -0.5, 0, 0.5, 1]:
        for dy in [-1, -0.5, 0, 0.5, 1]:
            directions.append([dx, dy])
    directions = np.array(directions)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    step_sizes = [0.01, 0.02, 0.05, 0.1, 0.2]

    while time.time() < start + time_limit:
        improved = False
        for i in range(N):
            if time.time() > start + time_limit:
                break
            for d in directions:
                for step in step_sizes:
                    new_bary = bary_cur.copy()
                    new_bary[i] = bary_cur[i] + step * d
                    new_bary = clamp_bary(new_bary)
                    v = _val(new_bary)
                    if v > best_local:
                        try_update(new_bary)
                        improved = True
                        break
        n_iter += 1
        if not improved and n_iter > 5:
            break

    return bary_cur


def refine_local(bary_init, time_limit=5.0):
    start = time.time()
    bounds = [(0, 1)] * (2 * N)
    bary_cur = bary_init.copy()
    best_local = _val(bary_cur)

    def try_update(b):
        nonlocal bary_cur, best_local
        b = clamp_bary(b)
        v = _val(b)
        if v > best_local:
            bary_cur, best_local = b.copy(), v

    # Stage 1: graduated soft-min with L-BFGS-B
    if time_limit >= 3.0:
        betas = [20.0, 50.0, 150.0, 400.0]
        for beta in betas:
            if time.time() > start + time_limit * 0.4:
                break
            try:
                r = minimize(soft_min_obj, bary_cur.flatten(), args=(beta,),
                             method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-13})
                try_update(r.x.reshape(N, 2))
            except Exception:
                pass

    # Stage 2: Coordinate descent (structural addition)
    if time.time() < start + time_limit * 0.65:
        cd_tl = min(2.0, time_limit * 0.2)
        try:
            cd_result = coordinate_descent(bary_cur, time_limit=cd_tl)
            try_update(cd_result)
        except Exception:
            pass

    # Stage 3: Nelder-Mead on hard min
    if time.time() < start + time_limit * 0.85:
        try:
            r = minimize(hard_min_obj, bary_cur.flatten(), method='Nelder-Mead',
                         options={'maxiter': int(time_limit * 100),
                                  'xatol': 1e-14, 'fatol': 1e-18, 'adaptive': True})
            try_update(r.x.reshape(N, 2))
        except Exception:
            pass

    # Stage 4: SLSQP polish
    if time.time() < start + time_limit * 0.95:
        try:
            r = minimize(hard_min_obj, bary_cur.flatten(), method='SLSQP',
                         bounds=bounds,
                         options={'maxiter': 500, 'ftol': 1e-16, 'iprint': 0})
            try_update(r.x.reshape(N, 2))
        except Exception:
            pass

    return bary_cur


def refine_epigraph(bary_init, time_limit=5.0):
    start = time.time()
    bary_cur = bary_init.copy()
    best_local = _val(bary_cur)

    def try_update(b):
        nonlocal bary_cur, best_local
        b = clamp_bary(b)
        v = _val(b)
        if v > best_local:
            bary_cur, best_local = b.copy(), v

    t0 = best_local
    for obj, opts in [(epigraph_obj, {'maxiter': 500, 'ftol': 1e-15, 'iprint': 0}),
                      (epigraph_obj_linear, {'maxiter': 500, 'ftol': 1e-15, 'iprint': 0})]:
        if time.time() > start + time_limit:
            break
        try:
            x0 = np.concatenate([bary_cur.flatten(), [t0]])
            r = minimize(obj, x0, method='SLSQP',
                         bounds=[(0, 1)] * (2 * N) + [(-1, 1)],
                         options=opts)
            try_update(r.x[:2 * N].reshape(N, 2))
        except Exception:
            pass
    return bary_cur


def construct():
    start = time.time()
    deadline = start + TIME_LIMIT

    best_bary = None
    best_val = -np.inf

    def update_best(b):
        nonlocal best_bary, best_val
        b = clamp_bary(b)
        v = _val(b)
        if v > best_val:
            best_bary, best_val = b.copy(), v
            return True
        return False

    # Diverse structured inits
    inits = [generate_hexagonal_init(),
             generate_lattice_k3_plus1(),
             generate_lattice_k4_minus4(),
             generate_sobolev_init(),
             generate_random_init(0)]

    # Refine each init
    for init in inits:
        if time.time() > deadline - 25:
            break
        refined = refine_local(init, time_limit=6.0)
        update_best(refined)

    # Multi-start loop
    rng = np.random.default_rng(123)
    rc = 0
    while time.time() < deadline - 15:
        rc += 1
        if best_bary is not None:
            if rc % 3 == 0:
                init = generate_perturbed_init(best_bary, sigma=0.10, seed=2000 + rc)
            elif rc % 3 == 1:
                init = generate_perturbed_init(best_bary, sigma=0.04, seed=2000 + rc)
            else:
                init = generate_random_init(3000 + rc)
        else:
            init = generate_random_init(3000 + rc)

        remaining = deadline - time.time()
        tl = 3.5 if remaining > 25 else (2.5 if remaining > 15 else 1.5)
        refined = refine_local(init, time_limit=tl)
        update_best(refined)

        # Epigraph polish on best
        if rc % 2 == 0 and time.time() < deadline - 10:
            refined_e = refine_epigraph(best_bary, time_limit=min(3.0, max(1.0, deadline - time.time() - 8)))
            update_best(refined_e)

    # Final high-precision polish
    if best_bary is not None and time.time() < deadline - 4:
        try:
            r = minimize(hard_min_obj, best_bary.flatten(), method='SLSQP',
                         bounds=[(0, 1)] * (2 * N),
                         options={'maxiter': 1500, 'ftol': 1e-17, 'iprint': 0})
            update_best(clamp_bary(r.x.reshape(N, 2)))
        except Exception:
            pass

        if best_bary is not None and time.time() < deadline - 2:
            try:
                r = minimize(hard_min_obj, best_bary.flatten(), method='Nelder-Mead',
                             options={'maxiter': 4000, 'xatol': 1e-18, 'fatol': 1e-22, 'adaptive': True})
                update_best(clamp_bary(r.x.reshape(N, 2)))
            except Exception:
                pass

        # Final coordinate descent
        if best_bary is not None and time.time() < deadline - 1:
            try:
                cd_result = coordinate_descent(best_bary, time_limit=max(0.5, deadline - time.time() - 0.5))
                update_best(cd_result)
            except Exception:
                pass

        # Final epigraph
        if best_bary is not None and time.time() < deadline - 0.5:
            try:
                refined_e = refine_epigraph(best_bary, time_limit=max(0.5, deadline - time.time() - 0.3))
                update_best(refined_e)
            except Exception:
                pass

    cart = bary_to_cart(best_bary)
    return cart
