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

def get_min_area(bary):
    """Return the minimum triangle area for a configuration."""
    return np.min(calc_areas_bary(bary))

def epigraph_obj(x, mu=50.0):
    """
    Epigraph formulation: maximize t subject to areas >= t.
    x = [bary_flat (22), t]
    Minimize -t + mu * sum(violations^2) where violations = max(0, t - areas).
    """
    bary_flat = x[:N*2]
    t = x[N*2]
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)

    # Penalize violations of areas >= t
    violations = np.maximum(0.0, t - areas)
    penalty = np.sum(violations**2)

    return -t + mu * penalty

def epigraph_soft_obj(x, beta=30.0):
    """
    Smooth epigraph with log-sum-exp penalty for violations.
    """
    bary_flat = x[:N*2]
    t = x[N*2]
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)

    violations = t - areas
    max_viol = np.max(violations)
    if max_viol > 1e-12:
        scaled = beta * violations
        max_scaled = np.max(scaled)
        if max_scaled > 500:
            penalty = np.sum(np.maximum(0, violations)**2)
        else:
            # log-sum-exp of positive violations
            pos_viol = np.maximum(0, violations)
            if np.max(pos_viol) > 0:
                scaled_pos = beta * pos_viol
                max_sp = np.max(scaled_pos)
                log_sum = max_sp + np.log(np.sum(np.exp(scaled_pos - max_sp)))
                penalty = log_sum / beta
            else:
                penalty = 0.0
    else:
        penalty = 0.0

    return -t + penalty

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

def generate_symmetric_init():
    """
    Generate a symmetric configuration exploiting 3-fold symmetry.
    3 points near vertices, 6 near edge midpoints, 1 center, 1 additional.
    """
    bary = np.zeros((N, 2))

    # Near vertex 0: barycentric (1,0,0) -> (u=1,v=0)
    bary[0] = [0.95, 0.02]
    # Near vertex 1: barycentric (0,1,0) -> (u=0,v=1)
    bary[1] = [0.02, 0.95]
    # Near vertex 2: barycentric (0,0,1) -> (u=0,v=0)
    bary[2] = [0.02, 0.02]

    # Near edge midpoints (offset inward)
    bary[3] = [0.48, 0.03]  # edge 0-1
    bary[4] = [0.03, 0.48]  # edge 1-2
    bary[5] = [0.48, 0.48]  # edge 2-0

    # Center
    bary[6] = [1/3, 1/3]

    # Additional points: form a small hexagon around center
    r = 0.15
    for i in range(3):
        angle = 2 * np.pi * i / 3 + np.pi / 6
        du = r * np.cos(angle)
        dv = r * np.sin(angle)
        bary[7 + i] = [1/3 + du, 1/3 + dv]

    # One more point
    bary[10] = [0.5, 0.25]

    return clamp_bary(bary)

def generate_radial_init():
    """
    Generate points using radial/angular parameterization from centroid.
    Points are placed on concentric rings with 3-fold symmetry.
    """
    bary = np.zeros((N, 2))
    cx, cy = 1/3, 1/3

    # Ring 1: 3 points at radius r1, angles 0, 120, 240
    r1 = 0.35
    for i in range(3):
        angle = 2 * np.pi * i / 3
        bary[i] = [cx + r1 * np.cos(angle), cy + r1 * np.sin(angle)]

    # Ring 2: 3 points at radius r2, angles 60, 180, 300
    r2 = 0.25
    for i in range(3):
        angle = 2 * np.pi * i / 3 + np.pi / 3
        bary[3 + i] = [cx + r2 * np.cos(angle), cy + r2 * np.sin(angle)]

    # Ring 3: 3 points at radius r3, angles 0, 120, 240 (closer to center)
    r3 = 0.12
    for i in range(3):
        angle = 2 * np.pi * i / 3
        bary[6 + i] = [cx + r3 * np.cos(angle), cy + r3 * np.sin(angle)]

    # Center point
    bary[9] = [cx, cy]

    # Additional point offset
    bary[10] = [0.5, 0.25]

    return clamp_bary(bary)

def generate_boundary_init():
    """
    Generate points with some on/near the boundary of the triangle.
    """
    bary = np.zeros((N, 2))

    # 3 near vertices
    bary[0] = [0.97, 0.015]
    bary[1] = [0.015, 0.97]
    bary[2] = [0.015, 0.015]

    # 3 near edge midpoints
    bary[3] = [0.49, 0.01]
    bary[4] = [0.01, 0.49]
    bary[5] = [0.49, 0.49]

    # 3 inner ring
    cx, cy = 1/3, 1/3
    r = 0.2
    for i in range(3):
        angle = 2 * np.pi * i / 3 + np.pi / 6
        bary[6 + i] = [cx + r * np.cos(angle), cy + r * np.sin(angle)]

    # Center
    bary[9] = [cx, cy]

    # Additional
    bary[10] = [0.45, 0.35]

    return clamp_bary(bary)

def generate_heilbronn_init():
    """
    Generate initial configurations inspired by known Heilbronn triangle results.
    For n=11, use structured arrangements with boundary points.
    """
    candidates = []

    # Config 1: 3 corners, 4 edge quarters, 4 interior
    cfg1 = np.array([
        [0.0, 0.0],   # corner A
        [1.0, 0.0],   # corner B
        [0.0, 1.0],   # corner C
        [0.25, 0.0],  # edge AB quarter
        [0.75, 0.0],  # edge AB three-quarter
        [0.0, 0.25],  # edge AC quarter
        [0.0, 0.75],  # edge AC three-quarter
        [0.25, 0.5],  # edge BC
        [0.5, 0.25],  # edge BC
        [1/3, 1/3],   # centroid
        [0.3, 0.3],   # interior
    ])
    candidates.append(clamp_bary(cfg1[:11]))

    # Config 2: 3 corners, 3 edge midpoints, 5 interior (symmetric)
    cfg2 = np.array([
        [0.0, 0.0],   # corner
        [1.0, 0.0],   # corner
        [0.0, 1.0],   # corner
        [0.5, 0.0],   # edge midpoint AB
        [0.0, 0.5],   # edge midpoint AC
        [0.25, 0.75], # edge midpoint BC (in bary: u+v=1, so (0.25, 0.75))
        [1/3, 1/3],   # centroid
        [0.2, 0.3],   # interior
        [0.5, 0.2],   # interior
        [0.3, 0.5],   # interior
        [0.4, 0.4],   # interior
    ])
    candidates.append(clamp_bary(cfg2))

    # Config 3: Hexagonal-inspired
    cfg3 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.33, 0.0],
        [0.67, 0.0],
        [0.0, 0.33],
        [0.0, 0.67],
        [0.17, 0.5],
        [0.5, 0.17],
        [0.33, 0.33],
        [0.5, 0.33],
    ])
    candidates.append(clamp_bary(cfg3))

    # Config 4: Dense boundary
    cfg4 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.2, 0.0],
        [0.8, 0.0],
        [0.0, 0.2],
        [0.0, 0.8],
        [0.1, 0.9],
        [0.9, 0.1],
        [1/3, 1/3],
        [0.4, 0.35],
    ])
    candidates.append(clamp_bary(cfg4))

    return candidates

def generate_random_init(rng, n_candidates=5):
    """Generate random initial configurations."""
    candidates = []
    for _ in range(n_candidates):
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        candidates.append(np.column_stack([u, v]))
    return candidates

def generate_perturbed_best(rng, base_bary, n_perturb=5, scales=None):
    """Generate perturbed versions of a good configuration."""
    if scales is None:
        scales = [0.02, 0.03, 0.05, 0.08, 0.12]
    candidates = []
    for i in range(n_perturb):
        scale = scales[i % len(scales)]
        perturbed = base_bary + rng.normal(0, scale, size=base_bary.shape)
        candidates.append(clamp_bary(perturbed))
    return candidates

def refine_epigraph(bary_init, time_limit=5.0, mu_schedule=None):
    """
    Refine a configuration using the epigraph formulation.
    Iteratively increase mu (penalty weight) to tighten the constraint.
    """
    start_time = time.time()
    if mu_schedule is None:
        mu_schedule = [10.0, 50.0, 200.0, 1000.0]

    bary_current = bary_init.copy()
    current_val = get_min_area(bary_current)

    # Start with t = current min area
    t_current = current_val

    for mu in mu_schedule:
        if time.time() - start_time > time_limit * 0.9:
            break

        # Build the full state vector [bary_flat, t]
        x0 = np.concatenate([bary_current.flatten(), [t_current]])

        # Bounds: bary coords in [0, 1], t in [0, 0.1]
        bounds = [(0, 1)] * (N * 2) + [(0, 0.1)]

        try:
            result = minimize(
                epigraph_obj,
                x0,
                args=(mu,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 80, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            new_bary = clamp_bary(result.x[:N*2].reshape(N, 2))
            new_t = result.x[N*2]
            new_val = get_min_area(new_bary)

            if new_val > current_val:
                bary_current = new_bary
                current_val = new_val
                t_current = min(new_t, new_val)
            else:
                # Keep t at the actual min area for next iteration
                t_current = current_val
        except Exception:
            pass

    return bary_current, current_val

def refine_epigraph_soft(bary_init, time_limit=5.0):
    """
    Refine using the smooth log-sum-exp epigraph variant.
    """
    start_time = time.time()
    bary_current = bary_init.copy()
    current_val = get_min_area(bary_current)
    t_current = current_val

    for beta in [15.0, 30.0, 60.0]:
        if time.time() - start_time > time_limit * 0.9:
            break

        x0 = np.concatenate([bary_current.flatten(), [t_current]])
        bounds = [(0, 1)] * (N * 2) + [(0, 0.1)]

        try:
            result = minimize(
                epigraph_soft_obj,
                x0,
                args=(beta,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 60, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            new_bary = clamp_bary(result.x[:N*2].reshape(N, 2))
            new_val = get_min_area(new_bary)

            if new_val > current_val:
                bary_current = new_bary
                current_val = new_val
                t_current = min(result.x[N*2], new_val)
            else:
                t_current = current_val
        except Exception:
            pass

    return bary_current, current_val

def coordinated_ascent(bary_init, time_limit=5.0, max_passes=10):
    """
    Coordinated ascent: iteratively move each point to maximize the minimum area,
    focusing on active (near-minimum) triangles.
    """
    start_time = time.time()
    bary = bary_init.copy()
    current_min = get_min_area(bary)

    for pass_num in range(max_passes):
        if time.time() - start_time > time_limit:
            break

        improved = False
        for i in range(N):
            if time.time() - start_time > time_limit * 0.95:
                break

            # Find which triangles involve point i and are active
            areas = calc_areas_bary(bary)
            min_area = np.min(areas)
            active = np.where(areas <= min_area * 1.05)[0]

            # Triangles involving point i
            tri_with_i = np.where((tri_idx[:, 0] == i) | (tri_idx[:, 1] == i) | (tri_idx[:, 2] == i))[0]
            # Intersection: active triangles that involve i
            relevant = np.intersect1d(active, tri_with_i)

            # Objective: maximize min area considering only triangles involving i
            def point_obj(xi):
                bary_temp = bary.copy()
                bary_temp[i] = xi
                bary_temp = clamp_bary(bary_temp)
                areas_temp = calc_areas_bary(bary_temp)
                if len(relevant) > 0:
                    rel_min = np.min(areas_temp[relevant])
                    global_min = np.min(areas_temp)
                    return -(0.7 * rel_min + 0.3 * global_min)
                else:
                    return -np.min(areas_temp)

            try:
                res_i = minimize(
                    point_obj,
                    bary[i],
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                new_bary = bary.copy()
                new_bary[i] = clamp_bary(res_i.x.reshape(1, 2))[0]
                new_min = get_min_area(new_bary)
                if new_min > current_min + 1e-12:
                    bary = new_bary
                    current_min = new_min
                    improved = True
            except Exception:
                pass

        if not improved and pass_num > 2:
            break

    return bary, current_min

def refine_local(bary_init, time_limit=5.0):
    """
    Refine a configuration using multiple stages:
    1. Epigraph L-BFGS-B (primary)
    2. Soft epigraph
    3. Coordinated ascent (targeted)
    4. Nelder-Mead polish
    5. Final coordinated ascent
    """
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)

    bary_current = bary_init.copy()
    current_val = get_min_area(bary_current)

    # Stage 1: Epigraph formulation (primary)
    if time.time() - start_time < time_limit * 0.35:
        try:
            t_alloc = min(time_limit * 0.3, time_limit * 0.35 - (time.time() - start_time))
            bary_ep, val_ep = refine_epigraph(bary_current, time_limit=t_alloc)
            if val_ep > current_val:
                bary_current = bary_ep
                current_val = val_ep
        except Exception:
            pass

    # Stage 2: Soft epigraph
    if time.time() - start_time < time_limit * 0.55:
        try:
            t_alloc = min(time_limit * 0.15, time_limit * 0.55 - (time.time() - start_time))
            bary_es, val_es = refine_epigraph_soft(bary_current, time_limit=t_alloc)
            if val_es > current_val:
                bary_current = bary_es
                current_val = val_es
        except Exception:
            pass

    # Stage 3: Coordinated ascent
    if time.time() - start_time < time_limit * 0.75:
        try:
            t_alloc = min(time_limit * 0.15, time_limit * 0.75 - (time.time() - start_time))
            bary_ca, val_ca = coordinated_ascent(
                bary_current,
                time_limit=t_alloc,
                max_passes=5
            )
            if val_ca > current_val:
                bary_current = bary_ca
                current_val = val_ca
        except Exception:
            pass

    # Stage 4: Nelder-Mead on full configuration
    if time.time() - start_time < time_limit * 0.9:
        try:
            result2 = minimize(
                hard_min_obj,
                bary_current.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-11, 'fatol': 1e-14, 'adaptive': True}
            )
            new_bary = clamp_bary(result2.x.reshape(N, 2))
            new_val = get_min_area(new_bary)
            if new_val > current_val:
                bary_current = new_bary
                current_val = new_val
        except Exception:
            pass

    # Stage 5: Final coordinated ascent
    if time.time() - start_time < time_limit * 0.98:
        try:
            remaining = time_limit * 0.98 - (time.time() - start_time)
            if remaining > 0.5:
                bary_ca2, val_ca2 = coordinated_ascent(
                    bary_current,
                    time_limit=remaining,
                    max_passes=3
                )
                if val_ca2 > current_val:
                    bary_current = bary_ca2
                    current_val = val_ca2
        except Exception:
            pass

    return bary_current

def de_objective(x_flat):
    """Objective for differential evolution."""
    bary = x_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    return -get_min_area(bary)

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT

    best_bary = None
    best_val = -np.inf

    def update_best(bary):
        nonlocal best_bary, best_val
        bary = clamp_bary(bary)
        val = get_min_area(bary)
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    candidates = []

    # 1. Symmetric/radial/boundary inits from inspiration
    candidates.append(generate_symmetric_init())
    candidates.append(generate_radial_init())
    candidates.append(generate_boundary_init())

    # 2. Heilbronn-inspired configurations
    candidates.extend(generate_heilbronn_init())

    # 3. Random candidates
    rng = np.random.default_rng(42)
    candidates.extend(generate_random_init(rng, n_candidates=4))

    # Process candidates with time allocation
    # Prioritize structured inits, then random
    n_cands = len(candidates)
    for i, cand in enumerate(candidates):
        if time.time() > deadline - 25:
            break

        remaining = deadline - 25 - time.time()
        if remaining < 2:
            break

        # Allocate more time to structured inits (first 7)
        if i < 7:
            t_limit = min(6.0, remaining)
        else:
            t_limit = min(3.0, remaining)

        try:
            refined = refine_local(cand, time_limit=t_limit)
            update_best(refined)
        except Exception:
            continue

    # Iterative perturbation and refinement of best
    if best_bary is not None:
        rng2 = np.random.default_rng(999)
        for iter_num in range(10):
            if time.time() > deadline - 20:
                break

            remaining = deadline - 20 - time.time()
            if remaining < 3:
                break

            # Generate perturbations with decreasing scales
            scale = 0.15 * (0.7 ** iter_num)
            perturb_cands = generate_perturbed_best(
                rng2, best_bary, n_perturb=2, scales=[scale, scale * 0.5]
            )

            for pc in perturb_cands:
                if time.time() > deadline - 15:
                    break
                remaining2 = deadline - 15 - time.time()
                if remaining2 < 2:
                    break
                try:
                    refined_p = refine_local(pc, time_limit=min(4.0, remaining2))
                    update_best(refined_p)
                except Exception:
                    continue

    # Differential Evolution Global Search if time permits
    if time.time() < deadline - 35:
        try:
            de_bounds = [(0, 1)] * (N * 2)

            de_result = differential_evolution(
                de_objective,
                bounds=de_bounds,
                seed=42,
                maxiter=15,
                popsize=10,
                tol=1e-5,
                polish=True,
                init='latinhypercube'
            )

            if de_result is not None and de_result.x is not None:
                de_bary = clamp_bary(de_result.x.reshape(N, 2))
                de_val = get_min_area(de_bary)

                if de_val > best_val * 0.95 or de_val > best_val:
                    remaining_de = deadline - 20 - time.time()
                    if remaining_de > 5:
                        refined_de = refine_local(de_bary, time_limit=min(10, remaining_de))
                        update_best(refined_de)
                    else:
                        update_best(de_bary)
        except Exception:
            pass

    # Final polish on best solution
    if best_bary is not None and time.time() < deadline - 3:
        try:
            remaining = deadline - 3 - time.time()
            if remaining > 1:
                bary_final, val_final = coordinated_ascent(
                    best_bary,
                    time_limit=remaining,
                    max_passes=5
                )
                if val_final > best_val:
                    best_bary = bary_final
                    best_val = val_final
        except Exception:
            pass

        if best_bary is not None and time.time() < deadline - 1:
            try:
                result_nm = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 400, 'xatol': 1e-11, 'fatol': 1e-14, 'adaptive': True}
                )
                update_best(clamp_bary(result_nm.x.reshape(N, 2)))
            except Exception:
                pass

    if best_bary is None:
        best_bary = generate_symmetric_init()

    cart = bary_to_cart(best_bary)

    return cart
