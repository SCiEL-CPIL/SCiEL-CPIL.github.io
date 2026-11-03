"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

DEADLINE = 140.0  # seconds, safety margin before 150s kill


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


def _separate_circles(circles, W, H, max_iter=2000, tol=1e-10):
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
                    push = overlap + 1e-8
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


def _force_directed_layout(circles, W, H, max_iter=5000, dt=0.01, cooling=0.999):
    """
    Force-directed layout: circles repel when overlapping, attract when too far,
    walls repel. This settles into a contact graph structure.
    """
    n = len(circles)
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()

    # Adaptive step size
    current_dt = dt
    best_energy = np.inf

    for it in range(max_iter):
        fx = np.zeros(n)
        fy = np.zeros(n)

        # Pairwise forces
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < 1e-12:
                    angle = np.random.uniform(0, 2 * np.pi)
                    dx = np.cos(angle) * 1e-6
                    dy = np.sin(angle) * 1e-6
                    dist = 1e-6

                min_dist = r[i] + r[j]
                # Strong repulsion when overlapping, weak attraction when close
                if dist < min_dist:
                    # Repulsive force proportional to overlap
                    force = 10.0 * (min_dist - dist)
                    fx[i] += force * dx / dist
                    fy[i] += force * dy / dist
                    fx[j] -= force * dx / dist
                    fy[j] -= force * dy / dist
                elif dist < min_dist * 1.5:
                    # Weak attractive force to encourage contact
                    force = 0.1 * (dist - min_dist)
                    fx[i] -= force * dx / dist
                    fy[i] -= force * dy / dist
                    fx[j] += force * dx / dist
                    fy[j] += force * dy / dist

        # Wall forces
        for i in range(n):
            # Left wall
            if x[i] < r[i]:
                fx[i] += 10.0 * (r[i] - x[i])
            # Right wall
            if x[i] > W - r[i]:
                fx[i] -= 10.0 * (x[i] - (W - r[i]))
            # Bottom wall
            if y[i] < r[i]:
                fy[i] += 10.0 * (r[i] - y[i])
            # Top wall
            if y[i] > H - r[i]:
                fy[i] -= 10.0 * (y[i] - (H - r[i]))

        # Update positions
        x += current_dt * fx
        y += current_dt * fy

        # Clamp to rectangle
        for i in range(n):
            x[i] = np.clip(x[i], r[i], W - r[i])
            y[i] = np.clip(y[i], r[i], H - r[i])

        # Cool down
        current_dt *= cooling

    result = np.column_stack([x, y, r])
    return result


def _objective_and_grad(params, W, H):
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

    eps = 1e-4

    def _smooth_max_grad(v):
        smax = 0.5 * (v + np.sqrt(v ** 2 + eps ** 2))
        gsmax = 0.5 * (1.0 + v / np.sqrt(v ** 2 + eps ** 2))
        return smax, gsmax

    # Boundary penalties
    v = r - x
    smax, gsmax = _smooth_max_grad(v)
    obj += 100.0 * np.sum(smax ** 2)
    grad[0::3] += -200.0 * smax * gsmax
    grad[2::3] += 200.0 * smax * gsmax

    v = r - y
    smax, gsmax = _smooth_max_grad(v)
    obj += 100.0 * np.sum(smax ** 2)
    grad[1::3] += -200.0 * smax * gsmax
    grad[2::3] += 200.0 * smax * gsmax

    v = x + r - W
    smax, gsmax = _smooth_max_grad(v)
    obj += 100.0 * np.sum(smax ** 2)
    grad[0::3] += 200.0 * smax * gsmax
    grad[2::3] += 200.0 * smax * gsmax

    v = y + r - H
    smax, gsmax = _smooth_max_grad(v)
    obj += 100.0 * np.sum(smax ** 2)
    grad[1::3] += 200.0 * smax * gsmax
    grad[2::3] += 200.0 * smax * gsmax

    v = -r
    smax, gsmax = _smooth_max_grad(v)
    obj += 100.0 * np.sum(smax ** 2)
    grad[2::3] += -200.0 * smax * gsmax

    # Overlap penalties
    overlap_weight = 100.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < 1e-12:
                dist = 1e-12
                dx = 1e-12
                dy = 0.0
            v = r[i] + r[j] - dist
            smax, gsmax = _smooth_max_grad(v)
            obj += overlap_weight * smax ** 2

            d_obj_dx_i = overlap_weight * 2.0 * smax * gsmax * (-dx / dist)
            d_obj_dy_i = overlap_weight * 2.0 * smax * gsmax * (-dy / dist)
            d_obj_dx_j = overlap_weight * 2.0 * smax * gsmax * (dx / dist)
            d_obj_dy_j = overlap_weight * 2.0 * smax * gsmax * (dy / dist)
            d_obj_dr_i = overlap_weight * 2.0 * smax * gsmax * 1.0
            d_obj_dr_j = overlap_weight * 2.0 * smax * gsmax * 1.0

            grad[3 * i] += d_obj_dx_i
            grad[3 * i + 1] += d_obj_dy_i
            grad[3 * i + 2] += d_obj_dr_i
            grad[3 * j] += d_obj_dx_j
            grad[3 * j + 1] += d_obj_dy_j
            grad[3 * j + 2] += d_obj_dr_j

    return obj, grad


def _local_optimize(circles, W, H, max_iter=2000):
    """Use L-BFGS-B with analytical gradients to locally optimize."""
    params = circles.flatten()
    r_max = min(W, H) / 2
    bounds = []
    for i in range(N):
        bounds.append((1e-6, W - 1e-6))
        bounds.append((1e-6, H - 1e-6))
        bounds.append((1e-6, r_max))

    result = minimize(
        _objective_and_grad, params, args=(W, H),
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
    )
    return result.x.reshape(N, 3)


def _polish(circles, W, H, max_iter=5000):
    """High-precision polish pass with L-BFGS-B and tight tolerances."""
    params = circles.flatten()
    r_max = min(W, H) / 2
    bounds = []
    for i in range(N):
        bounds.append((1e-6, W - 1e-6))
        bounds.append((1e-6, H - 1e-6))
        bounds.append((1e-6, r_max))

    # First pass
    result = minimize(
        _objective_and_grad, params, args=(W, H),
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={'maxiter': max_iter // 2, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 100}
    )
    params = result.x

    # Second pass: tighten further
    result = minimize(
        _objective_and_grad, params, args=(W, H),
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={'maxiter': max_iter // 2, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 200}
    )
    return result.x.reshape(N, 3)


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
    """Create a random initial configuration with moderate circles."""
    circles = []
    max_r = min(W, H) / 4
    for _ in range(N):
        r = rng.uniform(0.02, max_r)
        x = rng.uniform(r, W - r)
        y = rng.uniform(r, H - r)
        circles.append([x, y, r])
    return np.array(circles)


def _best_for_aspect(W, H, rng, time_budget=20.0):
    """Find best packing for given W, H within time budget."""
    start = time.time()
    best_val = -np.inf
    best_circles = None

    # Start with grid
    circles = _grid_initialization(W, H)
    
    # Local optimize
    circles_opt = _local_optimize(circles, W, H, max_iter=1500)
    circles_opt[:, 0] = np.clip(circles_opt[:, 0], circles_opt[:, 2], W - circles_opt[:, 2])
    circles_opt[:, 1] = np.clip(circles_opt[:, 1], circles_opt[:, 2], H - circles_opt[:, 2])
    circles_opt[:, 2] = np.clip(circles_opt[:, 2], 0.001, min(W, H) / 2)
    circles_opt, valid = _separate_circles(circles_opt, W, H)
    if valid:
        s = np.sum(circles_opt[:, 2])
        if s > best_val:
            best_val = s
            best_circles = circles_opt.copy()

    # Force-directed layout from grid
    if time.time() - start < time_budget * 0.5:
        fd_circles = _force_directed_layout(circles, W, H, max_iter=3000, dt=0.02)
        fd_circles[:, 0] = np.clip(fd_circles[:, 0], fd_circles[:, 2], W - fd_circles[:, 2])
        fd_circles[:, 1] = np.clip(fd_circles[:, 1], fd_circles[:, 2], H - fd_circles[:, 2])
        fd_circles, valid_fd = _separate_circles(fd_circles, W, H)
        if valid_fd:
            s = np.sum(fd_circles[:, 2])
            if s > best_val:
                best_val = s
                best_circles = fd_circles.copy()
                # Polish the force-directed result
                if time.time() - start < time_budget * 0.8:
                    polished = _polish(fd_circles, W, H, max_iter=3000)
                    polished[:, 0] = np.clip(polished[:, 0], polished[:, 2], W - polished[:, 2])
                    polished[:, 1] = np.clip(polished[:, 1], polished[:, 2], H - polished[:, 2])
                    polished[:, 2] = np.clip(polished[:, 2], 0.001, min(W, H) / 2)
                    polished, valid_pol = _separate_circles(polished, W, H)
                    if valid_pol:
                        s2 = np.sum(polished[:, 2])
                        if s2 > best_val:
                            best_val = s2
                            best_circles = polished.copy()

    # Random restarts
    remaining = time_budget - (time.time() - start)
    if remaining > 3 and best_circles is not None:
        n_restarts = max(1, int(remaining / 4))
        for _ in range(n_restarts):
            if time.time() - start > time_budget - 2:
                break
            circ_rand = _random_initialization(W, H, rng)
            circ_opt = _local_optimize(circ_rand, W, H, max_iter=600)
            circ_opt[:, 0] = np.clip(circ_opt[:, 0], circ_opt[:, 2], W - circ_opt[:, 2])
            circ_opt[:, 1] = np.clip(circ_opt[:, 1], circ_opt[:, 2], H - circ_opt[:, 2])
            circ_opt[:, 2] = np.clip(circ_opt[:, 2], 0.001, min(W, H) / 2)
            circ_opt, valid_rand = _separate_circles(circ_opt, W, H)
            if valid_rand:
                s = np.sum(circ_opt[:, 2])
                if s > best_val:
                    best_val = s
                    best_circles = circ_opt.copy()

    return best_circles, best_val


def construct():
    start_time = time.time()
    best_score = -np.inf
    best_result = None
    best_W, best_H = 1.0, 1.0

    # Try different aspect ratios
    aspect_ratios = [0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.4, 1.6, 2.0]

    rng = np.random.default_rng(42)

    for ar in aspect_ratios:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 15:
            break

        # Allocate time based on promise
        if 0.95 <= ar <= 1.05:
            time_budget = 22.0
        elif 0.9 <= ar <= 1.1:
            time_budget = 18.0
        elif 0.85 <= ar <= 1.2:
            time_budget = 14.0
        else:
            time_budget = 10.0

        W = 2.0 * ar / (ar + 1.0)
        H = 2.0 / (ar + 1.0)

        circles, score = _best_for_aspect(W, H, rng, time_budget=time_budget)
        if circles is not None and score > best_score:
            best_score = score
            best_result = circles.copy()
            best_W, best_H = W, H

    # Final high-precision polish on the best result
    if best_result is not None and time.time() - start_time < DEADLINE - 10:
        polished = _polish(best_result, best_W, best_H, max_iter=8000)
        polished[:, 0] = np.clip(polished[:, 0], polished[:, 2], best_W - polished[:, 2])
        polished[:, 1] = np.clip(polished[:, 1], polished[:, 2], best_H - polished[:, 2])
        polished[:, 2] = np.clip(polished[:, 2], 0.001, min(best_W, best_H) / 2)
        polished, valid_final = _separate_circles(polished, best_W, best_H, max_iter=2000)
        if valid_final:
            s = np.sum(polished[:, 2])
            if s > best_score:
                best_score = s
                best_result = polished.copy()

    # Second polish pass if time permits
    if best_result is not None and time.time() - start_time < DEADLINE - 5:
        polished2 = _polish(best_result, best_W, best_H, max_iter=4000)
        polished2[:, 0] = np.clip(polished2[:, 0], polished2[:, 2], best_W - polished2[:, 2])
        polished2[:, 1] = np.clip(polished2[:, 1], polished2[:, 2], best_H - polished2[:, 2])
        polished2[:, 2] = np.clip(polished2[:, 2], 0.001, min(best_W, best_H) / 2)
        polished2, valid_final2 = _separate_circles(polished2, best_W, best_H, max_iter=2000)
        if valid_final2:
            s = np.sum(polished2[:, 2])
            if s > best_score:
                best_score = s
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
