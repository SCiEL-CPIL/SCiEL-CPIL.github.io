"""Heilbronn triangles: maximize minimum triangle area for 11 points in unit equilateral triangle.

Strategy: Use Lissajous curve parametrization to reduce search space, combined with
direct coordinate optimization, multi-start, and local refinement.
"""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from itertools import combinations

N = 11
H = np.sqrt(3) / 2

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, H])

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)


def cart_to_bary(pts):
    """Convert Cartesian points to barycentric (u, v) where w = 1-u-v."""
    x = pts[:, 0]
    y = pts[:, 1]
    w = y / H
    v = x - 0.5 * w
    u = 1.0 - v - w
    return np.column_stack([u, v])


def bary_to_cart(bary):
    """Convert barycentric coords (u, v) to Cartesian (x, y)."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    return np.column_stack([x, y])


def calc_areas_bary(bary):
    """Calculate absolute areas of all triangles formed by N points in barycentric coords."""
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


def calc_min_area_bary(bary):
    """Return the minimum triangle area for a barycentric configuration."""
    return np.min(calc_areas_bary(bary))


def clamp_bary(bary, eps=1e-8):
    """Clamp barycentric coordinates to valid triangle region."""
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


def hard_min_obj(x):
    """Objective: minimize negative min area. x is flat (N*2,) barycentric coords."""
    bary = x.reshape(N, 2)
    bary = clamp_bary(bary)
    return -calc_min_area_bary(bary)


def soft_min_obj(x, beta=50.0):
    """Smooth approximation of min area."""
    bary = x.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    scaled = -beta * areas
    max_s = np.max(scaled)
    if max_s > 700:
        return -np.min(areas)
    log_sum = max_s + np.log(np.sum(np.exp(scaled - max_s)))
    return -log_sum / beta


def lissajous_bary(fx, fy, px, py, t_vals):
    """
    Generate barycentric coordinates from Lissajous curve parameters.
    fx, fy: frequencies (integer-like)
    px, py: phases
    t_vals: N distinct time parameters in [0, 1]
    Returns barycentric (u, v) for N points.
    """
    t = t_vals
    x_raw = np.sin(fx * t + px)  # in [-1, 1]
    y_raw = np.sin(fy * t + py)  # in [-1, 1]

    # Map to barycentric coordinates
    # We want points well-distributed in the triangle
    # x_raw -> u direction, y_raw -> v direction, with constraints
    u = 0.5 + 0.45 * x_raw
    v = 0.5 + 0.45 * y_raw

    # Ensure u + v <= 1 - margin
    # If u + v > 1, scale back
    margin = 0.05
    sums = u + v
    exceed = sums > (1.0 - margin)
    if np.any(exceed):
        scale = (1.0 - margin) / np.maximum(sums[exceed], 1e-10)
        u[exceed] *= scale
        v[exceed] *= scale

    return clamp_bary(np.column_stack([u, v]))


def lissajous_obj(params, t_vals):
    """Objective for Lissajous parameters: minimize negative min area."""
    fx, fy, px, py = params
    bary = lissajous_bary(fx, fy, px, py, t_vals)
    return -calc_min_area_bary(bary)


def generate_hexagonal_init():
    """Generate a hexagonal/triangular lattice initial configuration."""
    best_bary = None
    best_val = -np.inf

    for s in np.linspace(0.08, 0.45, 30):
        for offset_u in np.linspace(0.0, s * 0.5, 5):
            for offset_v in np.linspace(0.0, s * 0.3, 5):
                points = []
                for i in range(6):
                    for j in range(6):
                        u = offset_u + i * s
                        v = offset_v + j * s
                        if u + v <= 1.0 and len(points) < N:
                            points.append([u, v])

                if len(points) == N:
                    bary = np.array(points)
                    val = -hard_min_obj(bary.flatten())
                    if val > best_val:
                        best_val = val
                        best_bary = bary.copy()

    if best_bary is None:
        # Fallback: grid
        bary = np.array([[i / (N - 1), j / (N - 1) * (1 - i / (N - 1))]
                         for i in range(N) for j in range(N)][:N])
        best_bary = clamp_bary(bary)

    return best_bary


def generate_random_init(seed=None):
    """Generate random valid barycentric coordinates."""
    rng = np.random.default_rng(seed)
    u = rng.random(N)
    v = rng.random(N) * (1.0 - u)
    return clamp_bary(np.column_stack([u, v]))


def generate_grid_init():
    """Generate a structured grid initialization."""
    # Try a few different grid patterns
    best_bary = None
    best_val = -np.inf

    # Pattern 1: regular grid
    for n_cols in [3, 4, 5]:
        n_rows = int(np.ceil(N / n_cols))
        for du in np.linspace(0.1, 0.3, 5):
            for dv in np.linspace(0.1, 0.3, 5):
                pts = []
                for i in range(n_cols):
                    for j in range(n_rows):
                        u = 0.1 + i * du
                        v = 0.1 + j * dv
                        if u + v < 0.9 and len(pts) < N:
                            pts.append([u, v])
                if len(pts) >= N:
                    pts = pts[:N]
                    bary = clamp_bary(np.array(pts))
                    val = -hard_min_obj(bary.flatten())
                    if val > best_val:
                        best_val = val
                        best_bary = bary

    if best_bary is None:
        best_bary = generate_random_init(42)

    return best_bary


def _local_polish_bary(bary, max_iter=100, deadline=None):
    """
    Local polish using coordinate descent with adaptive steps.
    """
    if deadline is not None and time.time() > deadline - 2:
        return bary

    current_bary = clamp_bary(bary.copy())
    current_val = -hard_min_obj(current_bary.flatten())

    for it in range(max_iter):
        if deadline is not None and time.time() > deadline - 2:
            break

        improved = False
        for i in range(N):
            # Try steps in u direction
            for step in [0.002, -0.002, 0.005, -0.005, 0.001, -0.001, 0.01, -0.01]:
                trial = current_bary.copy()
                trial[i, 0] += step
                trial = clamp_bary(trial)
                trial_val = -hard_min_obj(trial.flatten())
                if trial_val > current_val:
                    current_bary = trial
                    current_val = trial_val
                    improved = True
                    break

            # Try steps in v direction
            for step in [0.002, -0.002, 0.005, -0.005, 0.001, -0.001, 0.01, -0.01]:
                trial = current_bary.copy()
                trial[i, 1] += step
                trial = clamp_bary(trial)
                trial_val = -hard_min_obj(trial.flatten())
                if trial_val > current_val:
                    current_bary = trial
                    current_val = trial_val
                    improved = True
                    break

        if not improved:
            break

    return current_bary


def _nelder_mead_polish(bary, maxiter=2000, deadline=None):
    """Nelder-Mead local optimization."""
    if deadline is not None and time.time() > deadline - 5:
        return bary

    try:
        result = minimize(
            hard_min_obj,
            bary.flatten(),
            method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        return clamp_bary(result.x.reshape(N, 2))
    except Exception:
        return bary


def _lbfgs_optimize(bary_init, beta1=50.0, beta2=200.0, maxiter1=200, maxiter2=300, deadline=None):
    """Two-stage L-BFGS-B optimization."""
    if deadline is not None and time.time() > deadline - 10:
        return bary_init

    try:
        bounds = [(0, 1)] * (N * 2)

        # Stage 1: smooth
        result1 = minimize(
            lambda x: soft_min_obj(x, beta=beta1),
            bary_init.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': maxiter1, 'ftol': 1e-12, 'gtol': 1e-8}
        )

        if deadline is not None and time.time() > deadline - 5:
            return clamp_bary(result1.x.reshape(N, 2))

        # Stage 2: sharper
        result2 = minimize(
            lambda x: soft_min_obj(x, beta=beta2),
            result1.x,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': maxiter2, 'ftol': 1e-14, 'gtol': 1e-10}
        )

        return clamp_bary(result2.x.reshape(N, 2))
    except Exception:
        return bary_init


def _differential_evolution_search(deadline):
    """Run differential evolution on Lissajous parameters."""
    if time.time() > deadline - 10:
        return None

    # Fixed time parameters for the 11 points
    t_vals = np.linspace(0.15, 0.85, N)

    # Bounds for Lissajous parameters
    # Frequencies: 1 to 6 (try integer-ish values)
    # Phases: 0 to 2*pi
    bounds_de = [(1, 6), (1, 6), (0, 2 * np.pi), (0, 2 * np.pi)]

    try:
        result = differential_evolution(
            lissajous_obj,
            bounds=bounds_de,
            args=(t_vals,),
            maxiter=50,
            popsize=15,
            tol=1e-8,
            seed=42,
            polish=False,
            update_interval=1,
            callback=lambda x, convergence: None
        )

        # Check time
        if time.time() > deadline - 5:
            return None

        # Convert best Lissajous params to barycentric
        bary = lissajous_bary(result.x[0], result.x[1], result.x[2], result.x[3], t_vals)
        return bary

    except Exception:
        return None


def _coordinate_descent_polish(bary, n_rounds=50, deadline=None):
    """
    More aggressive coordinate descent: for each point, try to move it
    in the direction that most improves the minimum area.
    """
    if deadline is not None and time.time() > deadline - 2:
        return bary

    current = clamp_bary(bary.copy())

    for _ in range(n_rounds):
        if deadline is not None and time.time() > deadline - 2:
            break

        for i in range(N):
            best_trial = current.copy()
            best_val = -hard_min_obj(current.flatten())

            # Try multiple step sizes and directions
            for du in np.array([-0.02, -0.01, -0.005, -0.002, 0.002, 0.005, 0.01, 0.02]):
                for dv in np.array([-0.02, -0.01, -0.005, -0.002, 0.002, 0.005, 0.01, 0.02]):
                    trial = current.copy()
                    trial[i, 0] += du
                    trial[i, 1] += dv
                    trial = clamp_bary(trial)
                    val = -hard_min_obj(trial.flatten())
                    if val > best_val:
                        best_val = val
                        best_trial = trial.copy()

            if best_val > -hard_min_obj(current.flatten()):
                current = best_trial

    return current


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin from 150s limit

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

    # Phase 1: Generate diverse initial candidates
    inits = []

    # Hexagonal init
    hex_init = generate_hexagonal_init()
    inits.append(hex_init)

    # Grid init
    grid_init = generate_grid_init()
    inits.append(grid_init)

    # Random inits
    for seed in [1, 7, 13, 42, 99, 123, 256, 512]:
        inits.append(generate_random_init(seed))

    # Lissajous inits with different parameter sets
    t_vals = np.linspace(0.15, 0.85, N)
    for fx, fy in [(2, 3), (3, 2), (3, 4), (4, 3), (2, 5), (5, 2)]:
        for px, py in [(0, 0), (np.pi / 2, 0), (0, np.pi / 2), (np.pi / 4, np.pi / 4)]:
            bary = lissajous_bary(fx, fy, px, py, t_vals)
            inits.append(bary)

    # Phase 2: Optimize each initial candidate
    for i, init in enumerate(inits):
        if time.time() > deadline - 30:
            break

        # L-BFGS optimization
        result = _lbfgs_optimize(init, beta1=50.0, beta2=200.0,
                                 maxiter1=150, maxiter2=200, deadline=deadline)
        update_best(result)

    # Phase 3: Differential evolution on Lissajous parameters
    de_result = _differential_evolution_search(deadline)
    if de_result is not None:
        # Polish the DE result
        polished = _lbfgs_optimize(de_result, beta1=100.0, beta2=300.0,
                                   maxiter1=100, maxiter2=150, deadline=deadline)
        update_best(polished)

    # Phase 4: Local refinement on best
    if best_bary is not None:
        # Coordinate descent polish
        cd_result = _coordinate_descent_polish(best_bary, n_rounds=30, deadline=deadline)
        update_best(cd_result)

        # Nelder-Mead polish
        nm_result = _nelder_mead_polish(best_bary, maxiter=2000, deadline=deadline)
        update_best(nm_result)

    # Phase 5: Perturb and re-optimize
    if best_bary is not None:
        for i in range(6):
            if time.time() > deadline - 20:
                break

            rng = np.random.default_rng(1000 + i * 17)
            perturbed = best_bary.copy()
            perturbed += rng.normal(0, 0.02, perturbed.shape)
            perturbed = clamp_bary(perturbed)

            result = _lbfgs_optimize(perturbed, beta1=100.0, beta2=300.0,
                                     maxiter1=150, maxiter2=200, deadline=deadline)
            update_best(result)

            # Quick coordinate descent
            cd = _coordinate_descent_polish(result, n_rounds=20, deadline=deadline)
            update_best(cd)

    # Phase 6: Final polish
    if best_bary is not None and time.time() < deadline - 10:
        # Aggressive coordinate descent
        cd = _coordinate_descent_polish(best_bary, n_rounds=50, deadline=deadline)
        update_best(cd)

    # Final Nelder-Mead
    if best_bary is not None and time.time() < deadline - 5:
        nm = _nelder_mead_polish(best_bary, maxiter=3000, deadline=deadline)
        update_best(nm)

    # Final coordinate descent
    if best_bary is not None and time.time() < deadline - 3:
        cd = _local_polish_bary(best_bary, max_iter=100, deadline=deadline)
        update_best(cd)

    # Ensure valid result
    if best_bary is None:
        best_bary = generate_random_init(1)

    best_bary = clamp_bary(best_bary)
    cart_pts = bary_to_cart(best_bary)

    return cart_pts


if __name__ == "__main__":
    pts = construct()
    print("Points:")
    print(pts)

    # Validate
    for p in pts:
        x, y = p
        if x < -1e-9 or y < -1e-9:
            print(f"Invalid point (outside): {p}")
        # Check triangle bounds
        if y > H + 1e-9:
            print(f"Invalid point (above triangle): {p}")
        if y > H * x + 1e-9:
            print(f"Invalid point (above left edge): {p}")
        if y > H * (1 - x) + 1e-9:
            print(f"Invalid point (above right edge): {p}")

    # Calculate score
    bary = cart_to_bary(pts)
    areas = calc_areas_bary(bary)
    print("Min Area (normalized):", np.min(areas))
    print("Max Area (normalized):", np.max(areas))
    print("Number of points:", len(pts))
