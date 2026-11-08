import numpy as np
import time
from scipy.optimize import minimize

N = 21
DEADLINE = 145.0  # safety margin from 150s limit


def _compute_radii_fast(centers, W, H):
    """Vectorized computation of radii using pairwise distances."""
    n = len(centers)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_nn = np.min(dists, axis=1) / 2.0
    cx, cy = centers[:, 0], centers[:, 1]
    boundary_dist = np.minimum.reduce([cx, W - cx, cy, H - cy])
    radii = np.maximum(np.minimum(min_nn, boundary_dist), 1e-12)
    return radii


def _sum_radii(centers, W, H):
    """Objective: sum of radii."""
    return np.sum(_compute_radii_fast(centers, W, H))


def _objective_soft(x, W, H, beta):
    """Differentiable soft objective for L-BFGS-B - vectorized."""
    n = len(x) // 2
    c = x.reshape(n, 2)
    cx, cy = c[:, 0], c[:, 1]

    # Distance to boundaries (n,)
    d_bound = np.minimum.reduce([cx, W - cx, cy, H - cy])

    # Pairwise distances (n, n)
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    half_dists = dists / 2.0  # (n, n)

    # Stack: (n, n+1)
    all_vals = np.column_stack([d_bound, half_dists])  # (n, n+1)

    # Soft-min along axis=1
    min_vals = np.min(all_vals, axis=1)  # (n,)
    exp_terms = np.exp(-beta * (all_vals - min_vals[:, np.newaxis]))  # (n, n+1)
    sum_exp = np.sum(exp_terms, axis=1)  # (n,)
    softmins = min_vals - np.log(sum_exp) / beta  # (n,)

    return -np.sum(softmins)


def _local_refine(centers, W, H, time_limit=10.0, max_iters=500):
    """Local refinement using L-BFGS-B with soft-min objective, multiple beta levels."""
    n = len(centers)
    if n < 2:
        return centers, _sum_radii(centers, W, H)

    best_x = centers.flatten().copy()
    best_val = _sum_radii(centers, W, H)

    start_time = time.time()
    betas = [20.0, 50.0, 100.0, 200.0, 400.0, 800.0]

    bounds = []
    for i in range(n):
        bounds.append((0, W))
        bounds.append((0, H))

    for beta in betas:
        if time.time() - start_time > time_limit - 1:
            break

        try:
            res = minimize(
                _objective_soft,
                best_x,
                args=(W, H, beta),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iters, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if res.fun < -best_val:
                best_val = -res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    # Final exact evaluation
    final_centers = best_x.reshape(-1, 2)
    final_val = _sum_radii(final_centers, W, H)

    return final_centers, final_val


def _sa_optimize(centers, W, H, time_limit=15.0, seed=42):
    """Simulated annealing for circle packing."""
    n = len(centers)
    rng = np.random.default_rng(seed)

    best_centers = centers.copy()
    best_val = _sum_radii(best_centers, W, H)
    current_centers = centers.copy()
    current_val = best_val

    start_time = time.time()
    T = 0.02  # initial temperature
    T_min = 1e-6
    cooling = 0.995

    while time.time() - start_time < time_limit:
        # Generate perturbation
        step_size = T * 2.0
        delta = rng.normal(0, step_size, size=(n, 2))
        new_centers = current_centers + delta

        # Clamp to bounds
        new_centers[:, 0] = np.clip(new_centers[:, 0], 0, W)
        new_centers[:, 1] = np.clip(new_centers[:, 1], 0, H)

        new_val = _sum_radii(new_centers, W, H)
        delta_val = new_val - current_val

        # Accept if better or with probability exp(delta/T)
        if delta_val > 0 or rng.random() < np.exp(delta_val / max(T, T_min)):
            current_centers = new_centers
            current_val = new_val
            if current_val > best_val:
                best_val = current_val
                best_centers = new_centers.copy()

        T *= cooling
        if T < T_min:
            T = T_min

    return best_centers, best_val


def _initial_grid(N, W, H, rows=5, cols=5):
    """Create a grid placement."""
    centers = []
    for i in range(rows):
        for j in range(cols):
            if len(centers) >= N:
                break
            cx = (j + 0.5) * W / cols
            cy = (i + 0.5) * H / rows
            centers.append([cx, cy])
    return np.array(centers[:N])


def _initial_hexagonal(N, W, H):
    """Hexagonal packing initialization."""
    centers = []
    for n_rows in range(3, 8):
        n_cols = int(np.ceil(N / n_rows))
        for i in range(n_rows):
            for j in range(n_cols):
                if len(centers) >= N:
                    break
                x_spacing = W / (n_cols + 0.5)
                y_spacing = H / (n_rows + 0.5)
                cx = (j + 0.25 + (0.25 if i % 2 == 1 else 0)) * x_spacing
                cy = (i + 0.25) * y_spacing
                cx = min(max(cx, 0.02 * W), 0.98 * W)
                cy = min(max(cy, 0.02 * H), 0.98 * H)
                centers.append([cx, cy])
        if len(centers) >= N:
            break
    return np.array(centers[:N])


def _initial_random(N, W, H, seed=None):
    """Create a random initial placement."""
    rng = np.random.default_rng(seed)
    margin = 0.03
    centers = rng.uniform(margin, 1 - margin, size=(N, 2))
    centers[:, 0] *= W
    centers[:, 1] *= H
    return centers


def _optimize_for_aspect(W, H, time_limit=20.0, seed=42):
    """Optimize for a given aspect ratio (W x H)."""
    start_time = time.time()
    best_centers = None
    best_val = -np.inf

    def try_optimize(init_centers):
        nonlocal best_centers, best_val
        if time.time() - start_time > time_limit - 1:
            return

        val_init = _sum_radii(init_centers, W, H)
        if val_init > best_val:
            best_val = val_init
            best_centers = init_centers.copy()

        remaining = time_limit - (time.time() - start_time) - 0.5
        if remaining < 2:
            return

        # Try SA first (explore), then L-BFGS-B (exploit)
        sa_time = min(remaining * 0.4, 8.0)
        sa_centers, sa_val = _sa_optimize(init_centers, W, H, time_limit=sa_time, seed=seed)

        if sa_val > best_val:
            best_val = sa_val
            best_centers = sa_centers.copy()

        remaining = time_limit - (time.time() - start_time) - 0.5
        if remaining < 2:
            return

        # L-BFGS-B refinement from SA result
        lbfgs_time = min(remaining * 0.8, 12.0)
        refined, val_ref = _local_refine(sa_centers, W, H,
                                         time_limit=lbfgs_time,
                                         max_iters=800)
        if val_ref > best_val:
            best_val = val_ref
            best_centers = refined.copy()

    # Try multiple initial configurations
    # 1. 5x5 grid
    try_optimize(_initial_grid(N, W, H, 5, 5))

    # 2. 7x3 grid
    try_optimize(_initial_grid(N, W, H, 3, 7))

    # 3. 4x6 grid
    try_optimize(_initial_grid(N, W, H, 4, 6))

    # 4. Hexagonal packing
    try_optimize(_initial_hexagonal(N, W, H))

    # 5. Random starts
    rng = np.random.default_rng(seed)
    n_random = 3
    for k in range(n_random):
        if time.time() - start_time > time_limit - 2:
            break
        try_optimize(_initial_random(N, W, H, seed=seed + k * 1000))

    return best_centers, best_val


def construct():
    start_time = time.time()
    deadline = DEADLINE

    # Search over aspect ratios W + H = 2 (perimeter 4)
    best_value = 0
    best_pos = None
    best_r = None
    best_W = None
    best_H = None

    # Try a range of aspect ratios
    # Focus on square and slightly elongated rectangles
    W_fracs = np.linspace(0.6, 1.4, 17)

    for W_frac in W_fracs:
        if time.time() > deadline - 5:
            break

        W_test = W_frac
        H_test = 2.0 - W_frac
        if W_test <= 0.1 or H_test <= 0.1:
            continue

        # Allocate time based on how close to square (square is often optimal)
        dist_from_square = abs(W_frac - 1.0)
        # More time for aspect ratios near 1
        time_frac = max(0.15, 1.0 - 1.5 * dist_from_square)
        per_aspect_time = min(20.0, (deadline - (time.time() - start_time)) * time_frac / max(1, len(W_fracs) - 1))
        per_aspect_time = max(3.0, per_aspect_time)

        centers, val = _optimize_for_aspect(W_test, H_test, time_limit=per_aspect_time, seed=42)

        if centers is not None and val > best_value:
            best_value = val
            best_pos = centers
            best_r = _compute_radii_fast(centers, W_test, H_test)
            best_W = W_test
            best_H = H_test

    # Final polish: high-precision L-BFGS-B pass on the best result
    if best_pos is not None and time.time() < deadline - 3:
        remaining = deadline - (time.time() - start_time) - 2
        if remaining > 2:
            refined, val_ref = _local_refine(best_pos, best_W, best_H,
                                             time_limit=remaining,
                                             max_iters=2000)
            if val_ref > best_value:
                best_value = val_ref
                best_pos = refined
                best_r = _compute_radii_fast(refined, best_W, best_H)

    # Build result
    if best_pos is None:
        # Fallback: simple grid
        W_f, H_f = 1.0, 1.0
        best_pos = _initial_grid(N, W_f, H_f, 5, 5)
        best_r = _compute_radii_fast(best_pos, W_f, H_f)
        best_W, best_H = W_f, H_f

    result = np.zeros((N, 3))
    result[:, 0] = best_pos[:, 0]
    result[:, 1] = best_pos[:, 1]
    result[:, 2] = best_r

    return result
