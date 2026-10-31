"""Heilbronn triangles: maximize minimum triangle area for 11 points in unit equilateral triangle."""
import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

N = 11

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2

def bary_to_cart(bary):
    """
    Convert barycentric coords (u, v) to Cartesian (x, y).
    bary: (N, 2) array of (u, v). w = 1 - u - v.
    Point = u*A + v*B + w*C
    A=(0,0), B=(1,0), C=(0.5, H)
    x = v + 0.5*w
    y = w*H
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    return np.column_stack([x, y])

def calc_areas_bary(bary):
    """
    Calculate areas of all triangles formed by N points given in barycentric coords (u, v).
    Returns array of shape (165,) containing the area ratios.
    """
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

def clamp_bary(bary, eps=1e-6):
    """
    Clamp barycentric coordinates to the strictly valid triangle region.
    u >= eps, v >= eps, u+v <= 1 - eps
    """
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

def soft_min_obj(bary_flat, beta=100.0):
    """
    Objective function for optimization.
    Minimizes the negative soft-min of the areas.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)

    areas = calc_areas_bary(bary)

    scaled = -beta * areas
    max_scaled = np.max(scaled)
    if max_scaled > 700:
        return -np.min(areas)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min = -log_sum / beta

    return -soft_min

def hard_min_obj(bary_flat):
    """
    Objective function using hard min.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def generate_random_init(seed=None):
    """Generate random valid barycentric coordinates."""
    rng = np.random.default_rng(seed)
    u = rng.random(N)
    v = rng.random(N) * (1 - u)
    return np.column_stack([u, v])

def generate_perturbed_init(base, noise_scale=0.05, seed=None):
    """Generate perturbed version of a base configuration."""
    rng = np.random.default_rng(seed)
    bary = base.copy()
    bary += rng.normal(0, noise_scale, bary.shape)
    return clamp_bary(bary)

def generate_structured_init():
    """
    Generate a structured initial configuration using corners, midpoints, and interior points.
    """
    eps = 0.02
    points = []

    # Corners (pushed in)
    points.append([eps, eps])
    points.append([1 - eps, eps])
    points.append([eps, 1 - eps])

    # Edge midpoints
    points.append([0.0, 0.5])    # bottom edge midpoint
    points.append([0.5, 0.0])    # right edge midpoint
    points.append([0.5, 0.5])    # top edge midpoint

    # Centroid
    points.append([1/3, 1/3])

    # Interior points
    candidates = [
        [0.25, 0.25],
        [0.75, 0.15],
        [0.15, 0.75],
        [0.4, 0.35],
        [0.35, 0.4],
        [0.55, 0.3],
        [0.3, 0.55],
    ]

    for p in candidates:
        if len(points) >= N:
            break
        is_new = True
        for existing in points:
            if np.linalg.norm(np.array(p) - np.array(existing)) < 0.08:
                is_new = False
                break
        if is_new:
            points.append(p)

    rng = np.random.default_rng(123)
    while len(points) < N:
        u = rng.random()
        v = rng.random() * (1 - u)
        points.append([u, v])

    return np.array(points[:N])

def generate_maxspread_init(seed=None, iterations=2000):
    """
    Generate initial configuration by maximizing minimum pairwise distance.
    This is a structural change: use pairwise distances as a proxy for triangle area.
    Points that are far apart tend to form larger triangles.
    """
    rng = np.random.default_rng(seed)

    def pairwise_min_dist(bary_flat):
        bary = bary_flat.reshape(N, 2)
        bary = clamp_bary(bary)
        cart = bary_to_cart(bary)
        # Compute all pairwise distances
        diffs = cart[:, np.newaxis, :] - cart[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        # Get minimum distance (excluding diagonal)
        np.fill_diagonal(dists, np.inf)
        return -np.min(dists)

    # Start from multiple random configurations and optimize min pairwise distance
    best_bary = None
    best_val = -np.inf

    for attempt in range(5):
        init = generate_random_init(seed=seed + attempt * 100)
        try:
            result = minimize(
                pairwise_min_dist,
                init.flatten(),
                method='L-BFGS-B',
                bounds=[(0, 1)] * (N * 2),
                options={'maxiter': iterations, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            val = -pairwise_min_dist(result.x)
            if val > best_val:
                best_val = val
                best_bary = result.x.reshape(N, 2)
        except Exception:
            pass

    if best_bary is None:
        best_bary = generate_structured_init()

    return clamp_bary(best_bary)

def simulated_annealing(bary, n_iter=5000, t_start=0.05, t_end=0.001, seed=None):
    """
    Simulated annealing for global search.
    This is a structural change: use stochastic search to escape local minima.
    """
    rng = np.random.default_rng(seed)
    current = bary.copy()
    best = bary.copy()
    best_val = -hard_min_obj(best.flatten())
    current_val = -hard_min_obj(current.flatten())

    n = n_iter
    for i in range(n):
        # Temperature schedule (geometric cooling)
        t = t_start * (t_end / t_start) ** (i / max(n - 1, 1))

        # Generate random perturbation
        noise_scale = t * 0.5
        perturbed = current.copy()
        # Randomly select a subset of points to perturb
        n_perturb = max(1, int(rng.integers(1, 4)))
        idx = rng.choice(N, size=n_perturb, replace=False)
        perturbed[idx] += rng.normal(0, noise_scale, size=(n_perturb, 2))
        perturbed = clamp_bary(perturbed)

        perturbed_val = -hard_min_obj(perturbed.flatten())

        # Accept if better or with probability exp(-delta/t)
        delta = current_val - perturbed_val  # positive means worse
        if delta < 0 or rng.random() < np.exp(-delta / max(t, 1e-10)):
            current = perturbed
            current_val = perturbed_val
            if current_val > best_val:
                best_val = current_val
                best = current.copy()

    return best

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill

    best_bary = None
    best_val = -np.inf

    def update_best(bary):
        """Update best if improved."""
        nonlocal best_bary, best_val
        bary = clamp_bary(bary)
        val = -hard_min_obj(bary.flatten())
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    def try_optimize_multi_stage(bary_init, stages=None):
        """
        Multi-stage optimization with increasing beta (annealing).
        """
        if time.time() > deadline - 10:
            return

        if stages is None:
            stages = [
                (20.0, 100),
                (100.0, 200),
                (500.0, 300),
                (2000.0, 500)
            ]

        current_x = bary_init.flatten()

        for beta, maxiter in stages:
            if time.time() > deadline - 5:
                break

            try:
                def obj(x):
                    return soft_min_obj(x, beta=beta)

                bounds = [(0, 1)] * (N * 2)

                result = minimize(
                    obj,
                    current_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8}
                )

                current_x = result.x
                update_best(result.x.reshape(N, 2))

            except Exception:
                pass

    # Phase 1: Generate diverse initial configurations
    # 1a. Structured initialization
    struct_init = generate_structured_init()
    update_best(struct_init)

    # 1b. Max-spread initialization (new structural approach)
    if time.time() < deadline - 80:
        maxspread_init = generate_maxspread_init(seed=42, iterations=1000)
        update_best(maxspread_init)

    # Phase 2: Multi-stage optimization from each good start
    # 2a. From structured init
    if time.time() < deadline - 60:
        try_optimize_multi_stage(struct_init)

    # 2b. From max-spread init
    if best_bary is not None and time.time() < deadline - 50:
        try_optimize_multi_stage(best_bary)

    # 2c. From random inits
    num_random_starts = 6
    for i in range(num_random_starts):
        if time.time() > deadline - 40:
            break
        rng_seed = 5000 + i * 7
        rand_init = generate_random_init(seed=rng_seed)
        stages = [
            (50.0, 100),
            (200.0, 150),
            (1000.0, 200)
        ]
        try_optimize_multi_stage(rand_init, stages=stages)

    # Phase 3: Simulated annealing (new structural approach)
    if best_bary is not None and time.time() < deadline - 30:
        sa_result = simulated_annealing(best_bary, n_iter=3000, t_start=0.03, t_end=0.0005, seed=777)
        update_best(sa_result)

    # Phase 4: Perturb best and re-optimize
    if best_bary is not None:
        for i in range(8):
            if time.time() > deadline - 15:
                break
            noise = 0.01 + 0.005 * i
            perturbed = generate_perturbed_init(best_bary, noise_scale=noise, seed=8000 + i)
            stages = [
                (100.0, 100),
                (500.0, 150),
                (2000.0, 200)
            ]
            try_optimize_multi_stage(perturbed, stages=stages)

    # Phase 5: Final polish with Nelder-Mead on hard min
    if best_bary is not None and time.time() < deadline - 10:
        try:
            bary_nm = best_bary.copy()
            bary_nm[:, 0] = np.clip(bary_nm[:, 0], 1e-6, 1 - 1e-6)
            bary_nm[:, 1] = np.clip(bary_nm[:, 1], 1e-6, 1 - bary_nm[:, 0] - 1e-6)

            result = minimize(
                hard_min_obj,
                bary_nm.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-16}
            )

            final_bary = clamp_bary(result.x.reshape(N, 2))
            update_best(final_bary)
        except Exception:
            pass

    # Phase 6: Additional local search with smaller steps
    if best_bary is not None and time.time() < deadline - 5:
        try:
            for i in range(20):
                if time.time() > deadline - 3:
                    break
                perturbed = generate_perturbed_init(best_bary, noise_scale=0.005, seed=9000 + i)
                if update_best(perturbed):
                    pass
        except Exception:
            pass

    # If all failed, generate a random valid set
    if best_bary is None:
        best_bary = generate_random_init(seed=42)

    # Convert to Cartesian
    bary_final = clamp_bary(best_bary, eps=1e-5)
    cart = bary_to_cart(bary_final)

    # Final safety check: ensure all points are strictly inside
    for i in range(N):
        x, y = cart[i]
        if y < 1e-9:
            cart[i, 1] = 1e-9
        if y > np.sqrt(3) * x - 1e-9:
            cart[i, 0] = max(cart[i, 0], y / np.sqrt(3) + 1e-6)
        if y + np.sqrt(3) * x > np.sqrt(3) - 1e-9:
            cart[i, 0] = min(cart[i, 0], (np.sqrt(3) - y) / np.sqrt(3) - 1e-6)

    for i in range(N):
        x, y = cart[i]
        if y < 1e-9:
            cart[i, 1] = 1e-9

    return cart
