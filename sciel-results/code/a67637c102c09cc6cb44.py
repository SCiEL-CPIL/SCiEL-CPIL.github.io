"""
Heilbronn triangles: place 11 points in an equilateral triangle to maximize
the minimum area of any triangle formed by three points.

Combines:
  - Barycentric coordinate representation with fast vectorized area computation
  - Multi-stage soft-min (log-sum-exp) refinement via L-BFGS-B
  - Differential Evolution for global exploration
  - Nelder-Mead local polish
  - Time-budgeted pipeline with explicit deadline guard
"""
import numpy as np
import time
from itertools import combinations
from scipy.optimize import minimize, differential_evolution

N = 11
TIME_LIMIT = 115.0  # Safety margin for 120s limit

# Pre-compute triangle indices (165 triangles for 11 points)
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
    """Calculate areas of all triangles formed by N points in barycentric coords.
    Returns array of 165 absolute determinants (proportional to area)."""
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
    """Clamp barycentric coordinates to the valid triangle region."""
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()

    u = np.clip(u, eps, 1.0 - eps)
    v = np.clip(v, eps, 1.0 - eps)

    sums = u + v
    invalid = sums > (1.0 - eps)
    if np.any(invalid):
        scale = (1.0 - eps) / np.where(sums > 0, sums, 1.0)
        u[invalid] *= scale[invalid]
        v[invalid] *= scale[invalid]

    u = np.clip(u, eps, 1.0 - eps)
    v = np.clip(v, eps, 1.0 - eps)

    return np.column_stack([u, v])


def hard_min_obj(bary_flat):
    """Objective using hard min: minimize negative of minimum area."""
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


def generate_hexagonal_init():
    """Generate a hexagonal lattice-like initial configuration in barycentric coords."""
    best_bary = None
    best_val = -np.inf

    for scale in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]:
        pts = []
        max_k = int(1.0 / scale) + 1
        for k in range(max_k + 1):
            w_val = k * scale
            if w_val > 1.0:
                break
            num_pts_in_row = int((1.0 - w_val) / scale) + 1
            for j in range(num_pts_in_row):
                u_val = j * scale
                v_val = (1.0 - w_val) - u_val
                if u_val >= -1e-9 and v_val >= -1e-9 and u_val + v_val <= 1.0 + 1e-9:
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
        v = rng.random(N) * (1.0 - u)
        best_bary = np.column_stack([u, v])

    return best_bary


def generate_random_init(seed=None):
    """Generate random valid barycentric coordinates."""
    rng = np.random.default_rng(seed)
    u = rng.random(N)
    v = rng.random(N) * (1.0 - u)
    return np.column_stack([u, v])


def generate_perturbed_init(base_bary, sigma=0.05, seed=None):
    """Generate a perturbed version of a base configuration."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, base_bary.shape)
    return clamp_bary(base_bary + noise)


def refine_local(bary_init, time_limit=5.0):
    """
    Refine a configuration using multiple optimization stages with graduated soft-min.
    5-stage approach: coarse -> medium -> sharp -> NM -> NM polish
    """
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)

    bary_current = bary_init.copy()

    # Stage 1: Coarse soft-min (low beta)
    try:
        result1 = minimize(
            soft_min_obj,
            bary_current.flatten(),
            args=(20.0,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if result1.fun < -hard_min_obj(bary_current.flatten()):
            bary_current = clamp_bary(result1.x.reshape(N, 2))
    except Exception:
        pass

    if time.time() - start_time > time_limit * 0.25:
        return bary_current

    # Stage 2: Medium soft-min
    try:
        result2 = minimize(
            soft_min_obj,
            bary_current.flatten(),
            args=(50.0,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-13, 'gtol': 1e-9}
        )
        if result2.fun < -hard_min_obj(bary_current.flatten()):
            bary_current = clamp_bary(result2.x.reshape(N, 2))
    except Exception:
        pass

    if time.time() - start_time > time_limit * 0.45:
        return bary_current

    # Stage 3: Sharp soft-min (high beta, close to hard min)
    try:
        result3 = minimize(
            soft_min_obj,
            bary_current.flatten(),
            args=(200.0,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        if result3.fun < -hard_min_obj(bary_current.flatten()):
            bary_current = clamp_bary(result3.x.reshape(N, 2))
    except Exception:
        pass

    if time.time() - start_time > time_limit * 0.65:
        return bary_current

    # Stage 4: Nelder-Mead on hard min
    try:
        result4 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-10}
        )
        if result4.fun < -hard_min_obj(bary_current.flatten()):
            bary_current = clamp_bary(result4.x.reshape(N, 2))
    except Exception:
        pass

    if time.time() - start_time > time_limit * 0.85:
        return bary_current

    # Stage 5: Final NM polish
    try:
        result5 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-12}
        )
        if result5.fun < -hard_min_obj(bary_current.flatten()):
            bary_current = clamp_bary(result5.x.reshape(N, 2))
    except Exception:
        pass

    return bary_current


def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT

    best_bary = None
    best_score = -np.inf

    def _evaluate(bary):
        nonlocal best_bary, best_score
        bary = clamp_bary(bary)
        areas = calc_areas_bary(bary)
        score = np.min(areas)
        if score > best_score:
            best_score = score
            best_bary = bary.copy()
        return score

    # --- Phase 1: Generate diverse initial candidates ---
    candidates = []

    # Hexagonal lattice
    hex_bary = generate_hexagonal_init()
    candidates.append(hex_bary)

    # Random inits with various seeds
    for seed in range(30):
        candidates.append(generate_random_init(seed))

    # Perturbed hex
    for sigma in [0.03, 0.05, 0.08, 0.1]:
        for seed in range(5):
            candidates.append(generate_perturbed_init(hex_bary, sigma=sigma, seed=seed))

    # Boundary-like distribution (points near edges)
    def _boundary_bary():
        pts = []
        # Along edge u=0 (from V2 to V1 in bary)
        for i in range(4):
            t = i / 3.0
            pts.append([0.0, t])
        # Along edge v=0 (from V0 to V2)
        for i in range(4):
            t = i / 3.0
            pts.append([t, 0.0])
        # Along edge u+v=1 (from V0 to V1)
        for i in range(3):
            t = i / 2.0
            pts.append([t, 1.0 - t])
        return np.array(pts)

    candidates.append(_boundary_bary())

    # Centroid + ring
    def _ring_bary():
        pts = []
        pts.append([1.0/3, 1.0/3])  # centroid
        for i in range(10):
            angle = 2 * np.pi * i / 10
            r = 0.3
            u = 1.0/3 + r * np.cos(angle)
            v = 1.0/3 + r * np.sin(angle)
            pts.append([u, v])
        return np.array(pts)

    candidates.append(clamp_bary(_ring_bary()))

    # Evaluate all candidates
    for cand in candidates:
        if len(cand) != N:
            continue
        _evaluate(cand)

    if best_bary is None:
        best_bary = generate_random_init(0)
        _evaluate(best_bary)

    # --- Phase 2: Local refinement of top candidates ---
    # Sort candidates by score, refine the best few
    scored = []
    for cand in candidates:
        if len(cand) != N:
            continue
        bary = clamp_bary(cand)
        areas = calc_areas_bary(bary)
        scored.append((np.min(areas), bary))
    scored.sort(key=lambda x: -x[0])

    # Refine top 5 with local optimizer
    refine_budget = 3.0  # seconds per refinement
    for score_val, bary_init in scored[:5]:
        if time.time() > deadline - 20:
            break
        refined = refine_local(bary_init, time_limit=refine_budget)
        _evaluate(refined)

    # --- Phase 3: Differential Evolution for global search ---
    if time.time() < deadline - 40:
        try:
            bounds = [(0, 1)] * (N * 2)

            def _de_objective(params):
                bary = params.reshape(N, 2)
                bary = clamp_bary(bary)
                areas = calc_areas_bary(bary)
                return -np.min(areas)

            de_result = differential_evolution(
                _de_objective,
                bounds,
                maxiter=80,
                popsize=15,
                tol=1e-6,
                seed=42,
                init='latinhypercube'
            )

            if de_result.fun < -best_score:
                de_bary = clamp_bary(de_result.x.reshape(N, 2))
                _evaluate(de_bary)

                # Refine the DE result
                if time.time() < deadline - 15:
                    refined_de = refine_local(de_bary, time_limit=5.0)
                    _evaluate(refined_de)
        except Exception:
            pass

    # --- Phase 4: More DE restarts if time permits ---
    if time.time() < deadline - 30:
        try:
            bounds = [(0, 1)] * (N * 2)

            def _de_objective2(params):
                bary = params.reshape(N, 2)
                bary = clamp_bary(bary)
                areas = calc_areas_bary(bary)
                return -np.min(areas)

            for de_seed in [7, 123, 999]:
                if time.time() > deadline - 15:
                    break
                de_result2 = differential_evolution(
                    _de_objective2,
                    bounds,
                    maxiter=50,
                    popsize=12,
                    tol=1e-5,
                    seed=de_seed,
                    init='latinhypercube'
                )
                if de_result2.fun < -best_score:
                    de_bary2 = clamp_bary(de_result2.x.reshape(N, 2))
                    _evaluate(de_bary2)
                    if time.time() < deadline - 10:
                        refined_de2 = refine_local(de_bary2, time_limit=3.0)
                        _evaluate(refined_de2)
        except Exception:
            pass

    # --- Phase 5: Final local polish on the best ---
    if time.time() < deadline - 5:
        try:
            final_refined = refine_local(best_bary, time_limit=4.0)
            _evaluate(final_refined)
        except Exception:
            pass

    # Convert to Cartesian coordinates
    best_bary = clamp_bary(best_bary)
    points_cart = bary_to_cart(best_bary)

    # Verify all points are inside the triangle (with small tolerance)
    for p in points_cart:
        if p[0] < -1e-6 or p[0] > 1.0 + 1e-6 or p[1] < -1e-6 or p[1] > H + 1e-6:
            # Fallback to safe configuration
            best_bary = generate_random_init(0)
            points_cart = bary_to_cart(best_bary)
            break

    return np.array(points_cart)
