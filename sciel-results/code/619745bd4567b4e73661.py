"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

DEADLINE = 118.0  # seconds, safety margin before 150s kill

# Precompute pair indices for vectorized overlap computation
_PAIR_I, _PAIR_J = np.triu_indices(N, k=1)
_PAIR_I = _PAIR_I.astype(np.int32)
_PAIR_J = _PAIR_J.astype(np.int32)
_N_PAIRS = len(_PAIR_I)


def _check_validity(circles, W, H):
    """Check if circles are valid: inside rectangle, no overlaps."""
    if len(circles) != N:
        return False
    for i in range(N):
        x, y, r = circles[i]
        if r <= 0:
            return False
        if x - r < -1e-9 or y - r < -1e-9 or x + r > W + 1e-9 or y + r > H + 1e-9:
            return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = circles[i][0] - circles[j][0]
            dy = circles[i][1] - circles[j][1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < circles[i][2] + circles[j][2] - 1e-9:
                return False
    return True


def _separate_circles(circles, W, H, max_iter=5000, tol=1e-12):
    """
    Directly resolve overlaps and boundary violations using pairwise repulsion.
    Returns (circles, is_valid).
    """
    n = len(circles)
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()

    for it in range(max_iter):
        moved = False

        # Resolve boundary violations by clamping
        for i in range(n):
            if r[i] <= 0:
                r[i] = 1e-4
            x[i] = np.clip(x[i], r[i], W - r[i])
            y[i] = np.clip(y[i], r[i], H - r[i])

        # Resolve overlaps via pairwise separation
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = r[i] + r[j]
                if dist < min_dist - tol:
                    if dist < 1e-12:
                        angle = np.random.uniform(0, 2 * np.pi)
                        dx = np.cos(angle) * 1e-6
                        dy = np.sin(angle) * 1e-6
                        dist = 1e-6
                    overlap = min_dist - dist
                    total_r = r[i] + r[j]
                    wi = r[i] / total_r if total_r > 0 else 0.5
                    wj = r[j] / total_r if total_r > 0 else 0.5
                    push = overlap + 1e-9
                    x[i] += wi * push * dx / dist
                    y[i] += wi * push * dy / dist
                    x[j] -= wj * push * dx / dist
                    y[j] -= wj * push * dy / dist
                    moved = True

        # Re-clamp after separation
        for i in range(n):
            max_r_x = min(x[i], W - x[i])
            max_r_y = min(y[i], H - y[i])
            max_r = max(1e-4, min(max_r_x, max_r_y))
            if r[i] > max_r:
                r[i] = max_r
            x[i] = np.clip(x[i], r[i], W - r[i])
            y[i] = np.clip(y[i], r[i], H - r[i])

        if not moved:
            break

    result = np.column_stack([x, y, r])
    valid = _check_validity(result, W, H)
    return result, valid


def _smooth_max_grad(v, eps=1e-4):
    """Returns (smooth_max(0,v), d/dv smooth_max(0,v))."""
    smax = 0.5 * (v + np.sqrt(v ** 2 + eps ** 2))
    grad_smax = 0.5 * (1.0 + v / np.sqrt(v ** 2 + eps ** 2))
    return smax, grad_smax


def _objective_and_grad(params, W, H, eps=1e-4):
    """
    Objective: minimize -sum(r) + penalties for violations.
    Returns (obj, grad) with analytical gradient.
    """
    n = N
    circles = params.reshape(n, 3)
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]

    obj = -np.sum(r)
    grad = np.zeros_like(params)
    grad[2::3] = -1.0

    boundary_weight = 100.0

    # Boundary penalties
    v = r - x
    smax, gsmax = _smooth_max_grad(v, eps)
    obj += boundary_weight * np.sum(smax ** 2)
    grad[0::3] += -2.0 * boundary_weight * smax * gsmax
    grad[2::3] += 2.0 * boundary_weight * smax * gsmax

    v = r - y
    smax, gsmax = _smooth_max_grad(v, eps)
    obj += boundary_weight * np.sum(smax ** 2)
    grad[1::3] += -2.0 * boundary_weight * smax * gsmax
    grad[2::3] += 2.0 * boundary_weight * smax * gsmax

    v = x + r - W
    smax, gsmax = _smooth_max_grad(v, eps)
    obj += boundary_weight * np.sum(smax ** 2)
    grad[0::3] += 2.0 * boundary_weight * smax * gsmax
    grad[2::3] += 2.0 * boundary_weight * smax * gsmax

    v = y + r - H
    smax, gsmax = _smooth_max_grad(v, eps)
    obj += boundary_weight * np.sum(smax ** 2)
    grad[1::3] += 2.0 * boundary_weight * smax * gsmax
    grad[2::3] += 2.0 * boundary_weight * smax * gsmax

    v = -r
    smax, gsmax = _smooth_max_grad(v, eps)
    obj += boundary_weight * np.sum(smax ** 2)
    grad[2::3] += -2.0 * boundary_weight * smax * gsmax

    # Overlap penalties: vectorized
    overlap_weight = 100.0
    dx = x[_PAIR_I] - x[_PAIR_J]
    dy = y[_PAIR_I] - y[_PAIR_J]
    dist_sq = dx * dx + dy * dy
    dist_sq = np.maximum(dist_sq, 1e-24)
    dist = np.sqrt(dist_sq)
    v = r[_PAIR_I] + r[_PAIR_J] - dist
    smax, gsmax = _smooth_max_grad(v, eps)
    obj += overlap_weight * np.sum(smax ** 2)

    d_obj = 2.0 * overlap_weight * smax * gsmax
    inv_dist = 1.0 / dist
    gx = dx * inv_dist
    gy = dy * inv_dist

    np.add.at(grad, 3 * _PAIR_I, -d_obj * gx)
    np.add.at(grad, 3 * _PAIR_I + 1, -d_obj * gy)
    np.add.at(grad, 3 * _PAIR_I + 2, d_obj)
    np.add.at(grad, 3 * _PAIR_J, d_obj * gx)
    np.add.at(grad, 3 * _PAIR_J + 1, d_obj * gy)
    np.add.at(grad, 3 * _PAIR_J + 2, d_obj)

    return obj, grad


def _local_optimize(circles, W, H, max_iter=2000, eps=1e-4):
    """Use L-BFGS-B with analytical gradients to locally optimize."""
    params = circles.flatten()
    r_max = min(W, H) / 2
    bounds = []
    for i in range(N):
        bounds.append((1e-6, W - 1e-6))
        bounds.append((1e-6, H - 1e-6))
        bounds.append((1e-6, r_max))

    result = minimize(
        _objective_and_grad, params, args=(W, H, eps),
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
    )
    return result.x.reshape(N, 3)


def _polish(circles, W, H, max_iter=15000, eps=1e-5):
    """High-precision polish pass with L-BFGS-B and tight tolerances."""
    params = circles.flatten()
    r_max = min(W, H) / 2
    bounds = []
    for i in range(N):
        bounds.append((1e-6, W - 1e-6))
        bounds.append((1e-6, H - 1e-6))
        bounds.append((1e-6, r_max))

    result = minimize(
        _objective_and_grad, params, args=(W, H, eps),
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 300}
    )
    return result.x.reshape(N, 3)


def _hexagonal_initialization(W, H):
    """
    Create a hexagonal cluster initial configuration.
    21 circles: 1 center + 6 in first ring + 12 in second ring + 2 in third ring.
    """
    positions = [(0.0, 0.0)]  # center

    # Ring 1: 6 positions at distance 2
    for k in range(6):
        angle = k * np.pi / 3
        positions.append((2.0 * np.cos(angle), 2.0 * np.sin(angle)))

    # Ring 2: 12 positions at distance 4
    for k in range(12):
        angle = k * np.pi / 6
        positions.append((4.0 * np.cos(angle), 4.0 * np.sin(angle)))

    # Ring 3: need 2 more
    for k in range(2):
        angle = k * np.pi / 3 + np.pi / 6
        positions.append((6.0 * np.cos(angle), 6.0 * np.sin(angle)))

    positions = positions[:N]

    xs = np.array([p[0] for p in positions])
    ys = np.array([p[1] for p in positions])
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    width = x_max - x_min
    height = y_max - y_min

    scale_x = W / (width + 2.0)
    scale_y = H / (height + 2.0)
    scale = min(scale_x, scale_y)

    cx_offset = W / 2.0 - scale * (x_min + x_max) / 2.0
    cy_offset = H / 2.0 - scale * (y_min + y_max) / 2.0

    circles = []
    for px, py in positions:
        x = scale * px + cx_offset
        y = scale * py + cy_offset
        r = scale * 0.98
        circles.append([x, y, r])

    return np.array(circles)


def _grid_initialization(W, H):
    """Create a grid-based initial configuration."""
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= N:
                break
            cx = (i + 0.5) * W / side
            cy = (j + 0.5) * H / side
            r = min(W, H) / (2 * side) - 1e-4
            cells.append([cx, cy, r])
    return np.array(cells[:N])


def _random_initialization(W, H, rng):
    """Create a random initial configuration."""
    circles = []
    max_r = min(W, H) / 4
    for _ in range(N):
        r = rng.uniform(0.02, max_r)
        x = rng.uniform(r, W - r)
        y = rng.uniform(r, H - r)
        circles.append([x, y, r])
    return np.array(circles)


def _shake_and_settle(circles, W, H, rng, n_shakes=5, shake_scale=0.01):
    """
    Perturb circles slightly and re-optimize to escape local minima.
    """
    best = circles.copy()
    best_val = np.sum(circles[:, 2])

    for _ in range(n_shakes):
        perturbed = circles.copy()
        perturbed[:, 0] += rng.normal(0, shake_scale, N)
        perturbed[:, 1] += rng.normal(0, shake_scale, N)

        perturbed[:, 0] = np.clip(perturbed[:, 0], perturbed[:, 2], W - perturbed[:, 2])
        perturbed[:, 1] = np.clip(perturbed[:, 1], perturbed[:, 2], H - perturbed[:, 2])
        perturbed[:, 2] = np.clip(perturbed[:, 2], 0.001, min(W, H) / 2)

        optimized = _local_optimize(perturbed, W, H, max_iter=1000)
        optimized[:, 0] = np.clip(optimized[:, 0], optimized[:, 2], W - optimized[:, 2])
        optimized[:, 1] = np.clip(optimized[:, 1], optimized[:, 2], H - optimized[:, 2])
        optimized[:, 2] = np.clip(optimized[:, 2], 0.001, min(W, H) / 2)
        optimized, valid = _separate_circles(optimized, W, H)

        if valid:
            val = np.sum(optimized[:, 2])
            if val > best_val:
                best_val = val
                best = optimized.copy()

    return best


def _force_directed(circles, W, H, max_iter=20000, dt=0.01):
    """
    Force-directed relaxation: apply repulsive forces between overlapping circles
    and attractive forces toward the center, with boundary constraints.
    """
    n = len(circles)
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()

    cx = W / 2.0
    cy = H / 2.0

    for it in range(max_iter):
        # Compute forces
        fx = np.zeros(n)
        fy = np.zeros(n)

        # Pairwise repulsion (stronger when overlapping)
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    if dist < 1e-12:
                        angle = np.random.uniform(0, 2 * np.pi)
                        dx = np.cos(angle) * 1e-6
                        dy = np.sin(angle) * 1e-6
                        dist = 1e-6
                    overlap = min_dist - dist
                    force = 10.0 * overlap
                    fx[i] += force * dx / dist
                    fy[i] += force * dy / dist
                    fx[j] -= force * dx / dist
                    fy[j] -= force * dy / dist

        # Mild attraction toward center to keep circles together
        for i in range(n):
            fx[i] += 0.1 * (cx - x[i])
            fy[i] += 0.1 * (cy - y[i])

        # Apply forces
        for i in range(n):
            x[i] += dt * fx[i]
            y[i] += dt * fy[i]

        # Clamp to boundaries
        for i in range(n):
            x[i] = np.clip(x[i], r[i], W - r[i])
            y[i] = np.clip(y[i], r[i], H - r[i])

        # Check convergence
        if np.max(np.abs(fx)) < 1e-10 and np.max(np.abs(fy)) < 1e-10:
            break

    result = np.column_stack([x, y, r])
    return result


def _best_known(W, H, time_budget=30.0):
    """Return the best configuration found so far for given W, H."""
    start = time.time()

    # Start with hexagonal cluster
    circles_hex = _hexagonal_initialization(W, H)
    circles_grid = _grid_initialization(W, H)

    if np.sum(circles_hex[:, 2]) >= np.sum(circles_grid[:, 2]):
        circles = circles_hex.copy()
    else:
        circles = circles_grid.copy()

    best_val = np.sum(circles[:, 2])
    best_circles = circles.copy()

    # Local optimization
    circles_opt = _local_optimize(circles, W, H, max_iter=3000)
    circles_opt[:, 0] = np.clip(circles_opt[:, 0], circles_opt[:, 2], W - circles_opt[:, 2])
    circles_opt[:, 1] = np.clip(circles_opt[:, 1], circles_opt[:, 2], H - circles_opt[:, 2])
    circles_opt[:, 2] = np.clip(circles_opt[:, 2], 0.001, min(W, H) / 2)
    circles_opt, valid_opt = _separate_circles(circles_opt, W, H)
    if valid_opt:
        s = np.sum(circles_opt[:, 2])
        if s > best_val:
            best_val = s
            best_circles = circles_opt.copy()

    # Force-directed relaxation
    if time.time() - start < time_budget - 10:
        fd_circles = _force_directed(best_circles, W, H, max_iter=10000)
        fd_circles[:, 0] = np.clip(fd_circles[:, 0], fd_circles[:, 2], W - fd_circles[:, 2])
        fd_circles[:, 1] = np.clip(fd_circles[:, 1], fd_circles[:, 2], H - fd_circles[:, 2])
        fd_circles[:, 2] = np.clip(fd_circles[:, 2], 0.001, min(W, H) / 2)
        fd_circles, valid_fd = _separate_circles(fd_circles, W, H)
        if valid_fd:
            s = np.sum(fd_circles[:, 2])
            if s > best_val:
                best_val = s
                best_circles = fd_circles.copy()

    # Shake and settle
    if time.time() - start < time_budget - 8:
        rng_local = np.random.default_rng(int(time.time() * 1000) % 100000)
        shaken = _shake_and_settle(best_circles, W, H, rng_local, n_shakes=4, shake_scale=0.005)
        shaken[:, 0] = np.clip(shaken[:, 0], shaken[:, 2], W - shaken[:, 2])
        shaken[:, 1] = np.clip(shaken[:, 1], shaken[:, 2], H - shaken[:, 2])
        shaken[:, 2] = np.clip(shaken[:, 2], 0.001, min(W, H) / 2)
        shaken, valid_shake = _separate_circles(shaken, W, H)
        if valid_shake:
            s = np.sum(shaken[:, 2])
            if s > best_val:
                best_val = s
                best_circles = shaken.copy()

    # Final polish
    if time.time() - start < time_budget - 3:
        polished = _polish(best_circles, W, H, max_iter=8000)
        polished[:, 0] = np.clip(polished[:, 0], polished[:, 2], W - polished[:, 2])
        polished[:, 1] = np.clip(polished[:, 1], polished[:, 2], H - polished[:, 2])
        polished[:, 2] = np.clip(polished[:, 2], 0.001, min(W, H) / 2)
        polished, valid_polish = _separate_circles(polished, W, H)
        if valid_polish:
            s = np.sum(polished[:, 2])
            if s > best_val:
                best_val = s
                best_circles = polished.copy()

    return best_circles, best_val


def construct():
    start_time = time.time()
    best_score = -np.inf
    best_result = None
    best_W, best_H = 1.0, 1.0

    # Aspect ratios: focus on near-square and slightly elongated
    aspect_ratios = [0.85, 0.9, 0.93, 0.95, 0.97, 1.0, 1.03, 1.05, 1.08, 1.1, 1.15, 1.2]

    rng = np.random.default_rng(42)

    for ar in aspect_ratios:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 18:
            break

        # Allocate more time to promising aspect ratios
        if 0.95 <= ar <= 1.05:
            time_budget = 22.0
        elif 0.9 <= ar <= 1.1:
            time_budget = 16.0
        elif 0.85 <= ar <= 1.15:
            time_budget = 12.0
        else:
            time_budget = 8.0

        W = 2.0 * ar / (ar + 1.0)
        H = 2.0 / (ar + 1.0)

        circles, score = _best_known(W, H, time_budget=time_budget)
        if score > best_score:
            best_score = score
            best_result = circles.copy()
            best_W, best_H = W, H

        # Random restarts if time permits
        remaining = DEADLINE - 18 - (time.time() - start_time)
        if remaining > 8:
            n_restarts = max(1, min(3, int(remaining / 5)))
            for _ in range(n_restarts):
                if time.time() - start_time > DEADLINE - 18:
                    break
                circ_rand = _random_initialization(W, H, rng)
                circ_opt = _local_optimize(circ_rand, W, H, max_iter=1500)
                circ_opt[:, 0] = np.clip(circ_opt[:, 0], circ_opt[:, 2], W - circ_opt[:, 2])
                circ_opt[:, 1] = np.clip(circ_opt[:, 1], circ_opt[:, 2], H - circ_opt[:, 2])
                circ_opt[:, 2] = np.clip(circ_opt[:, 2], 0.001, min(W, H) / 2)
                circ_opt, valid_rand = _separate_circles(circ_opt, W, H)
                if valid_rand:
                    s = np.sum(circ_opt[:, 2])
                    if s > best_score:
                        best_score = s
                        best_result = circ_opt.copy()
                        best_W, best_H = W, H

    # Final polishing on the best result
    if best_result is not None and time.time() - start_time < DEADLINE - 15:
        rng_final = np.random.default_rng(int(time.time() * 1000) % 100000)
        shaken = _shake_and_settle(best_result, best_W, best_H, rng_final, n_shakes=4, shake_scale=0.003)
        shaken[:, 0] = np.clip(shaken[:, 0], shaken[:, 2], best_W - shaken[:, 2])
        shaken[:, 1] = np.clip(shaken[:, 1], shaken[:, 2], best_H - shaken[:, 2])
        shaken[:, 2] = np.clip(shaken[:, 2], 0.001, min(best_W, best_H) / 2)
        shaken, valid_shake = _separate_circles(shaken, best_W, best_H)
        if valid_shake:
            s = np.sum(shaken[:, 2])
            if s > best_score:
                best_score = s
                best_result = shaken.copy()

        # High-precision polish
        if time.time() - start_time < DEADLINE - 10:
            polished = _polish(best_result, best_W, best_H, max_iter=15000, eps=1e-5)
            polished[:, 0] = np.clip(polished[:, 0], polished[:, 2], best_W - polished[:, 2])
            polished[:, 1] = np.clip(polished[:, 1], polished[:, 2], best_H - polished[:, 2])
            polished[:, 2] = np.clip(polished[:, 2], 0.001, min(best_W, best_H) / 2)
            polished, valid_final = _separate_circles(polished, best_W, best_H)
            if valid_final:
                s = np.sum(polished[:, 2])
                if s > best_score:
                    best_score = s
                    best_result = polished.copy()

        # Second polish pass
        if time.time() - start_time < DEADLINE - 6:
            polished2 = _polish(best_result, best_W, best_H, max_iter=10000, eps=1e-6)
            polished2[:, 0] = np.clip(polished2[:, 0], polished2[:, 2], best_W - polished2[:, 2])
            polished2[:, 1] = np.clip(polished2[:, 1], polished2[:, 2], best_H - polished2[:, 2])
            polished2[:, 2] = np.clip(polished2[:, 2], 0.001, min(best_W, best_H) / 2)
            polished2, valid_final2 = _separate_circles(polished2, best_W, best_H)
            if valid_final2:
                s2 = np.sum(polished2[:, 2])
                if s2 > best_score:
                    best_score = s2
                    best_result = polished2.copy()

    # Fallback: simple grid in square
    if best_result is None:
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                cx = (i + 0.5) / side
                cy = (j + 0.5) / side
                r = 0.5 / side - 1e-3
                cells.append([cx, cy, r])
        best_result = np.array(cells[:N])
        best_W, best_H = 1.0, 1.0

    # Ensure validity
    if not _check_validity(best_result, best_W, best_H):
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                cx = (i + 0.5) / side
                cy = (j + 0.5) / side
                r = 0.5 / side - 1e-3
                cells.append([cx, cy, r])
        best_result = np.array(cells[:N])
        best_W, best_H = 1.0, 1.0

    return best_result
# EVOLVE-BLOCK-END
