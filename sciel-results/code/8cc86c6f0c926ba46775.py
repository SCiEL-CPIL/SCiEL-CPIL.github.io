"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

DEADLINE = 130.0  # seconds, safety margin before 150s kill


def _check_validity(circles, W, H, tol=1e-8):
    """Check if circles are valid: inside rectangle, no overlaps."""
    if len(circles) != N:
        return False
    for i in range(N):
        x, y, r = circles[i]
        if r <= 0:
            return False
        if x - r < -tol or y - r < -tol or x + r > W + tol or y + r > H + tol:
            return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = circles[i][0] - circles[j][0]
            dy = circles[i][1] - circles[j][1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < circles[i][2] + circles[j][2] - tol:
                return False
    return True


def _separate_circles(circles, W, H, max_iter=3000):
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
            max_r = min(x[i], W - x[i], y[i], H - y[i])
            if r[i] > max_r and max_r > 0:
                r[i] = max_r
            x[i] = np.clip(x[i], r[i], W - r[i])
            y[i] = np.clip(y[i], r[i], H - r[i])

        # Resolve overlaps via pairwise separation
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx * dx + dy * dy
                min_dist = r[i] + r[j]
                if dist_sq < min_dist * min_dist:
                    dist = np.sqrt(dist_sq)
                    if dist < 1e-12:
                        angle = np.random.uniform(0, 2 * np.pi)
                        dx = np.cos(angle) * 1e-6
                        dy = np.sin(angle) * 1e-6
                        dist = 1e-6
                    overlap = min_dist - dist
                    total_r = r[i] + r[j]
                    wi = r[i] / total_r if total_r > 0 else 0.5
                    wj = r[j] / total_r if total_r > 0 else 0.5
                    push = overlap + 1e-8
                    x[i] += wi * push * dx / dist
                    y[i] += wi * push * dy / dist
                    x[j] -= wj * push * dx / dist
                    y[j] -= wj * push * dy / dist
                    moved = True

        # Re-clamp
        for i in range(n):
            max_r = min(x[i], W - x[i], y[i], H - y[i])
            if r[i] > max_r and max_r > 0:
                r[i] = max_r
            x[i] = np.clip(x[i], r[i], W - r[i])
            y[i] = np.clip(y[i], r[i], H - r[i])

        if not moved:
            break

    result = np.column_stack([x, y, r])
    valid = _check_validity(result, W, H)
    return result, valid


def _objective_and_grad(params, W, H, penalty_weight, eps=1e-4):
    """
    Objective and gradient: minimize -sum(r) + penalties for violations.
    Vectorized for speed with analytical gradient.
    """
    circles = params.reshape(N, 3)
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]

    obj = -np.sum(r)
    grad = np.zeros_like(params)
    grad[2::3] = -1.0  # d(-sum r)/dr = -1

    # Boundary penalties
    # v1 = r - x  (violation if x - r < 0)
    # v2 = r - y  (violation if y - r < 0)
    # v3 = x + r - W  (violation if x + r > W)
    # v4 = y + r - H  (violation if y + r > H)
    # v5 = -r  (violation if r < 0)

    def _smooth_max_and_grad(v):
        sv = 0.5 * (v + np.sqrt(v ** 2 + eps ** 2))
        dv = 0.5 * (1 + v / np.sqrt(v ** 2 + eps ** 2))
        return sv, dv

    for vi, xi in [(0, 'x'), (1, 'y')]:
        pass

    # v1 = r - x
    v1 = r - x
    s1, d1 = _smooth_max_and_grad(v1)
    obj += penalty_weight * np.sum(s1 ** 2)
    # grad wrt x: d(s1^2)/dx = 2*s1*d1*d(v1)/dx = 2*s1*d1*(-1)
    grad[0::3] += penalty_weight * 2 * s1 * d1 * (-1)
    # grad wrt r: d(s1^2)/dr = 2*s1*d1*d(v1)/dr = 2*s1*d1*(1)
    grad[2::3] += penalty_weight * 2 * s1 * d1 * (1)

    # v2 = r - y
    v2 = r - y
    s2, d2 = _smooth_max_and_grad(v2)
    obj += penalty_weight * np.sum(s2 ** 2)
    grad[1::3] += penalty_weight * 2 * s2 * d2 * (-1)
    grad[2::3] += penalty_weight * 2 * s2 * d2 * (1)

    # v3 = x + r - W
    v3 = x + r - W
    s3, d3 = _smooth_max_and_grad(v3)
    obj += penalty_weight * np.sum(s3 ** 2)
    grad[0::3] += penalty_weight * 2 * s3 * d3 * (1)
    grad[2::3] += penalty_weight * 2 * s3 * d3 * (1)

    # v4 = y + r - H
    v4 = y + r - H
    s4, d4 = _smooth_max_and_grad(v4)
    obj += penalty_weight * np.sum(s4 ** 2)
    grad[1::3] += penalty_weight * 2 * s4 * d4 * (1)
    grad[2::3] += penalty_weight * 2 * s4 * d4 * (1)

    # v5 = -r
    v5 = -r
    s5, d5 = _smooth_max_and_grad(v5)
    obj += penalty_weight * np.sum(s5 ** 2)
    grad[2::3] += penalty_weight * 2 * s5 * d5 * (-1)

    # Overlap penalties
    dx_mat = x[:, np.newaxis] - x[np.newaxis, :]
    dy_mat = y[:, np.newaxis] - y[np.newaxis, :]
    dist_mat = np.sqrt(dx_mat ** 2 + dy_mat ** 2)
    np.fill_diagonal(dist_mat, 1e12)

    r_sum_mat = r[:, np.newaxis] + r[np.newaxis, :]
    v_overlap = r_sum_mat - dist_mat

    iu, ju = np.triu_indices(N, k=1)
    v_vec = v_overlap[iu, ju]
    v_pos = np.maximum(v_vec, 0)

    obj += penalty_weight * np.sum(v_pos ** 2)

    # Gradient for overlap terms
    # For each pair (i,j) with violation v = r_i + r_j - dist
    # d(obj)/dr_i += penalty_weight * 2 * v * (1)  if v > 0
    # d(obj)/dr_j += penalty_weight * 2 * v * (1)  if v > 0
    # d(obj)/dx_i += penalty_weight * 2 * v * d(-dist)/dx_i = penalty_weight * 2 * v * (dx_i - dx_j)/dist
    # d(obj)/dx_j += penalty_weight * 2 * v * d(-dist)/dx_j = penalty_weight * 2 * v * (dx_j - dx_i)/dist
    # Similarly for y

    for k in range(len(iu)):
        i, j = iu[k], ju[k]
        if v_vec[k] > 0:
            factor = penalty_weight * 2 * v_vec[k]
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = dist_mat[i, j]
            if dist > 1e-12:
                grad[i * 3] += factor * dx / dist
                grad[j * 3] += factor * (-dx) / dist
                grad[i * 3 + 1] += factor * dy / dist
                grad[j * 3 + 1] += factor * (-dy) / dist
            grad[i * 3 + 2] += factor
            grad[j * 3 + 2] += factor

    return obj, grad


def _local_optimize(W, H, initial_circles, time_budget=15.0):
    """
    Optimize the configuration using L-BFGS-B with annealed penalties.
    """
    start = time.time()
    params = initial_circles.flatten()
    r_max = min(W, H) / 2
    bounds = []
    for i in range(N):
        bounds.append((1e-6, W - 1e-6))
        bounds.append((1e-6, H - 1e-6))
        bounds.append((1e-6, r_max))

    best_circles = initial_circles.copy()
    best_score = -np.sum(best_circles[:, 2])

    # Anneal penalty weight
    weights = [10, 50, 200, 1000, 5000, 20000]
    current_params = params.copy()

    for w in weights:
        if time.time() - start > time_budget - 2:
            break
        try:
            res = minimize(
                _objective_and_grad, current_params, args=(W, H, w),
                method='L-BFGS-B',
                bounds=bounds,
                jac=True,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
            )
            current_params = res.x
        except Exception:
            continue

        circles = current_params.reshape(N, 3)
        # Repair step
        circles, valid = _separate_circles(circles, W, H, max_iter=500)
        if valid:
            score = np.sum(circles[:, 2])
            if score > best_score:
                best_score = score
                best_circles = circles.copy()
                current_params = best_circles.flatten()

    return best_circles, best_score


def _hexagonal_start(W, H):
    """Hexagonal packing initialization."""
    circles = []
    # Try different hexagonal configurations
    best_score = -np.inf
    best_circles = None

    for rows in range(3, 8):
        for cols in range(3, 10):
            if rows * cols < N:
                continue
            if rows * cols > N + 5:
                continue

            # Hexagonal spacing
            r = min(W / (cols + 0.5), H / (rows + 0.866)) / 2 - 1e-3
            if r <= 0:
                continue

            dx = 2 * r
            dy = r * np.sqrt(3)

            count = 0
            for i in range(rows):
                for j in range(cols):
                    if count >= N:
                        break
                    x = (j + 0.5) * dx + (i % 2) * dx / 2
                    y = (i + 0.5) * dy
                    if x + r > W or y + r > H:
                        continue
                    circles.append([x, y, r])
                    count += 1
            if len(circles) >= N:
                arr = np.array(circles[:N])
                circles, valid = _separate_circles(arr, W, H, max_iter=1000)
                if valid:
                    score = np.sum(circles[:, 2])
                    if score > best_score:
                        best_score = score
                        best_circles = circles.copy()
            circles = []

    if best_circles is not None:
        return best_circles

    # Fallback: grid
    side = 5
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= N:
                break
            cx = (i + 0.5) * W / side
            cy = (j + 0.5) * H / side
            r = min(W, H) / (2 * side) - 1e-3
            cells.append([cx, cy, r])
    return np.array(cells[:N])


def _grid_initialization(W, H, rows=3, cols=7):
    """Grid initialization."""
    cells = []
    for i in range(rows):
        for j in range(cols):
            if len(cells) >= N:
                break
            cx = (j + 0.5) * W / cols
            cy = (i + 0.5) * H / rows
            r = min(W / cols, H / rows) / 2 - 1e-3
            cells.append([cx, cy, r])
    return np.array(cells[:N])


def _random_initialization(W, H, rng):
    """Random initialization."""
    circles = []
    max_r = min(W, H) / 4
    for _ in range(N):
        r = rng.uniform(0.05, max_r)
        x = rng.uniform(r, W - r)
        y = rng.uniform(r, H - r)
        circles.append([x, y, r])
    return np.array(circles)


def _force_directed_layout(W, H, initial_circles, time_budget=10.0):
    """
    Force-directed layout: circles repel each other and are attracted to
    a target radius. This is a simple iterative method that can escape
    local minima better than gradient-based methods.
    """
    start = time.time()
    circles = initial_circles.copy()
    n = N

    # Target radius: aim for a reasonable packing
    target_r = min(W, H) / 5.0

    alpha = 0.1  # Learning rate
    for it in range(5000):
        if time.time() - start > time_budget:
            break

        # Compute forces
        forces_x = np.zeros(n)
        forces_y = np.zeros(n)
        forces_r = np.zeros(n)

        for i in range(n):
            x, y, r = circles[i]

            # Attraction to target radius
            forces_r[i] = (target_r - r) * 0.5

            # Boundary forces
            if x - r < 0:
                forces_x[i] += (r - x) * 2
            if y - r < 0:
                forces_y[i] += (r - y) * 2
            if x + r > W:
                forces_x[i] -= (x + r - W) * 2
            if y + r > H:
                forces_y[i] -= (y + r - H) * 2

        # Repulsion between circles
        for i in range(n):
            for j in range(i + 1, n):
                dx = circles[i][0] - circles[j][0]
                dy = circles[i][1] - circles[j][1]
                dist_sq = dx * dx + dy * dy
                min_dist = circles[i][2] + circles[j][2]
                if dist_sq < min_dist * min_dist:
                    dist = np.sqrt(dist_sq)
                    if dist < 1e-12:
                        angle = rng_uniform(0, 2 * np.pi)
                        dx = np.cos(angle) * 1e-6
                        dy = np.sin(angle) * 1e-6
                        dist = 1e-6
                    overlap = min_dist - dist
                    total_r = circles[i][2] + circles[j][2]
                    wi = circles[i][2] / total_r if total_r > 0 else 0.5
                    wj = circles[j][2] / total_r if total_r > 0 else 0.5
                    fx = overlap * dx / dist
                    fy = overlap * dy / dist
                    forces_x[i] += wi * fx
                    forces_y[i] += wi * fy
                    forces_x[j] -= wj * fx
                    forces_y[j] -= wj * fy
                    # Slightly shrink radii when overlapping
                    forces_r[i] -= overlap * 0.1
                    forces_r[j] -= overlap * 0.1

        # Update positions
        for i in range(n):
            circles[i][0] += alpha * forces_x[i]
            circles[i][1] += alpha * forces_y[i]
            circles[i][2] += alpha * forces_r[i]

            # Clamp
            circles[i][0] = np.clip(circles[i][0], circles[i][2], W - circles[i][2])
            circles[i][1] = np.clip(circles[i][1], circles[i][2], H - circles[i][2])
            circles[i][2] = max(1e-4, circles[i][2])

        # Decay learning rate
        alpha *= 0.999

    # Final separation
    circles, valid = _separate_circles(circles, W, H, max_iter=1000)
    return circles, valid


def rng_uniform(low, high):
    """Simple random uniform for force-directed layout."""
    return np.random.uniform(low, high)


def _pack_for_aspect(W, H, time_budget=20.0):
    """Try multiple initializations and optimization strategies for a given aspect ratio."""
    start = time.time()
    best_score = -np.inf
    best_circles = None

    rng = np.random.default_rng(42)

    # Generate initializations
    inits = []
    inits.append(_hexagonal_start(W, H))
    inits.append(_grid_initialization(W, H, 3, 7))
    inits.append(_grid_initialization(W, H, 5, 5))
    inits.append(_random_initialization(W, H, rng))

    for init in inits:
        if time.time() - start > time_budget - 3:
            break
        remaining = time_budget - (time.time() - start)
        if remaining < 3:
            break
        tb = min(10.0, remaining / 2.0)

        # Try L-BFGS-B optimization
        circles, score = _local_optimize(W, H, init, time_budget=tb)
        if score > best_score:
            best_score = score
            best_circles = circles.copy()

        # Try force-directed layout
        if time.time() - start < time_budget - 3:
            circles, valid = _force_directed_layout(W, H, init, time_budget=5.0)
            if valid:
                score = np.sum(circles[:, 2])
                if score > best_score:
                    best_score = score
                    best_circles = circles.copy()

    return best_circles, best_score


def construct():
    start_time = time.time()
    best_score = -np.inf
    best_result = None
    best_W, best_H = 1.0, 1.0

    # Aspect ratios to try: W/H ratio
    aspect_ratios = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6, 2.0]

    for ar in aspect_ratios:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 15:
            break

        W = 2.0 * ar / (ar + 1.0)
        H = 2.0 / (ar + 1.0)

        remaining = DEADLINE - 10 - (time.time() - start_time)
        if remaining < 5:
            break
        time_budget = min(20.0, remaining / 3.0)

        circles, score = _pack_for_aspect(W, H, time_budget=time_budget)
        if score > best_score:
            best_score = score
            best_result = circles.copy()
            best_W, best_H = W, H

    # Final polish on the best result
    if best_result is not None and time.time() - start_time < DEADLINE - 5:
        polished, valid = _separate_circles(best_result, best_W, best_H, max_iter=5000)
        if valid:
            score = np.sum(polished[:, 2])
            if score > best_score:
                best_score = score
                best_result = polished.copy()

    # Fallback
    if best_result is None:
        best_result = _grid_initialization(1.0, 1.0, 5, 5)
        best_result, _ = _separate_circles(best_result, 1.0, 1.0)
        best_W, best_H = 1.0, 1.0

    if not _check_validity(best_result, best_W, best_H):
        # Emergency fallback
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
