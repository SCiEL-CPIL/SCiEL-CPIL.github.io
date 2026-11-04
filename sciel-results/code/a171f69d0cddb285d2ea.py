"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
DEADLINE_MARGIN = 10.0  # seconds of safety margin before 120s kill


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
    Directly resolve overlaps and boundary violations using pairwise repulsion
    and strict projection. Returns (circles, is_valid).
    """
    n = len(circles)
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()

    for it in range(max_iter):
        moved = False

        # Resolve boundary violations by clamping and shrinking if needed
        for i in range(n):
            if r[i] <= 0:
                r[i] = 1e-4
            max_r_x = min(x[i], W - x[i])
            max_r_y = min(y[i], H - y[i])
            max_r = max(1e-4, min(max_r_x, max_r_y))
            if r[i] > max_r:
                r[i] = max_r
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


def _smooth_max_grad(v, eps=1e-4):
    """Returns (smooth_max(0,v), d/dv smooth_max(0,v))."""
    smax = 0.5 * (v + np.sqrt(v ** 2 + eps ** 2))
    grad_smax = 0.5 * (1.0 + v / np.sqrt(v ** 2 + eps ** 2))
    return smax, grad_smax


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
    grad[2::3] = -1.0  # d(-sum(r))/dr

    # Boundary penalties (smooth)
    # x - r >= 0  => violation = r - x
    v = r - x
    smax, gsmax = _smooth_max_grad(v)
    obj += 100.0 * np.sum(smax ** 2)
    grad[0::3] += -200.0 * smax * gsmax
    grad[2::3] += 200.0 * smax * gsmax

    # y - r >= 0
    v = r - y
    smax, gsmax = _smooth_max_grad(v)
    obj += 100.0 * np.sum(smax ** 2)
    grad[1::3] += -200.0 * smax * gsmax
    grad[2::3] += 200.0 * smax * gsmax

    # x + r <= W  => violation = x + r - W
    v = x + r - W
    smax, gsmax = _smooth_max_grad(v)
    obj += 100.0 * np.sum(smax ** 2)
    grad[0::3] += 200.0 * smax * gsmax
    grad[2::3] += 200.0 * smax * gsmax

    # y + r <= H  => violation = y + r - H
    v = y + r - H
    smax, gsmax = _smooth_max_grad(v)
    obj += 100.0 * np.sum(smax ** 2)
    grad[1::3] += 200.0 * smax * gsmax
    grad[2::3] += 200.0 * smax * gsmax

    # r >= 0
    v = -r
    smax, gsmax = _smooth_max_grad(v)
    obj += 100.0 * np.sum(smax ** 2)
    grad[2::3] += -200.0 * smax * gsmax

    # Overlap penalties: for each pair, violation = r_i + r_j - dist_ij
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


def _local_optimize(circles, W, H, max_iter=3000, ftol=1e-12, gtol=1e-8):
    """Use L-BFGS-B with analytical gradients to locally optimize."""
    params = circles.flatten()
    r_max = min(W, H) / 2
    bounds = []
    for i in range(N):
        bounds.append((1e-6, W - 1e-6))  # x
        bounds.append((1e-6, H - 1e-6))  # y
        bounds.append((1e-6, r_max))     # r

    result = minimize(
        _objective_and_grad, params, args=(W, H),
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol, 'maxls': 50}
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

    result = minimize(
        _objective_and_grad, params, args=(W, H),
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 100}
    )
    return result.x.reshape(N, 3)


def _projected_gauss_seidel(circles, W, H, max_iter=1000, tol=1e-12):
    """
    Iterative projected Gauss-Seidel method that strictly enforces non-overlap
    and boundary constraints at each step without penalty terms.
    """
    n = len(circles)
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()

    for it in range(max_iter):
        max_change = 0.0

        for i in range(n):
            # Enforce boundary constraints
            max_r_x = min(x[i], W - x[i])
            max_r_y = min(y[i], H - y[i])
            max_r = max(1e-6, min(max_r_x, max_r_y))
            if r[i] > max_r:
                r[i] = max_r
            x[i] = np.clip(x[i], r[i], W - r[i])
            y[i] = np.clip(y[i], r[i], H - r[i])

            # Resolve overlaps with all other circles
            for j in range(n):
                if i == j:
                    continue
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
                    # Move circle i to resolve overlap
                    x[i] += overlap * dx / dist
                    y[i] += overlap * dy / dist
                    max_change = max(max_change, overlap)

        if max_change < tol:
            break

    result = np.column_stack([x, y, r])
    return result


def _pack_for_aspect(W, H, deadline):
    """Pack circles in W x H rectangle. Returns (circles, W, H, sum_radii)."""
    # Initial grid placement
    cols = max(1, int(np.round(np.sqrt(N * W / H))))
    rows = max(1, int(np.ceil(N / cols)))

    cell_w = W / cols
    cell_h = H / rows

    centers = np.zeros((N, 2))
    for i in range(N):
        c = i % cols
        r = i // cols
        centers[i, 0] = (c + 0.5) * cell_w
        centers[i, 1] = (r + 0.5) * cell_h

    # Estimate initial radius
    r_est = min(cell_w, cell_h) / 2.0 * 0.95
    circles = np.column_stack([centers, np.full(N, r_est)])

    # Step 1: Separate circles to resolve overlaps
    circles, valid = _separate_circles(circles, W, H, max_iter=500)

    # Step 2: Local optimization with L-BFGS-B
    if time.time() < deadline - 15:
        circles = _local_optimize(circles, W, H, max_iter=2000)

    # Step 3: Polish with high-precision L-BFGS-B
    if time.time() < deadline - 5:
        circles = _polish(circles, W, H, max_iter=3000)

    # Step 4: Projected Gauss-Seidel for final strict feasibility
    circles = _projected_gauss_seidel(circles, W, H, max_iter=500)

    # Final validation
    circles, valid = _separate_circles(circles, W, H, max_iter=200)

    sum_radii = np.sum(circles[:, 2])
    return circles, W, H, sum_radii


def construct():
    """Construct 21 circles in a perimeter-4 rectangle, maximizing sum of radii."""
    start_time = time.time()
    deadline = start_time + 120.0 - DEADLINE_MARGIN

    best_sum = 0.0
    best_result = None

    # Try different aspect ratios
    # Perimeter = 2(W + H) = 4, so W + H = 2
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.43, 0.6, 1.67, 0.5, 2.0, 0.4, 1.1, 0.9, 1.3, 0.75, 1.5]

    for ar in aspect_ratios:
        if time.time() > deadline:
            break
        # W * H with W + H = 2
        # W = ar * H, so H = 2/(1+ar), W = 2*ar/(1+ar)
        H = 2.0 / (1.0 + ar)
        W = 2.0 * ar / (1.0 + ar)

        circles, W, H, s = _pack_for_aspect(W, H, deadline)

        if s > best_sum:
            best_sum = s
            best_result = (circles, W, H)

    if best_result is None:
        # Fallback: square grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    circles, W, H = best_result

    # Final validation and cleanup
    circles, valid = _separate_circles(circles, W, H, max_iter=500)

    # Build output: (x, y, r) for each circle
    result = np.zeros((N, 3))
    result[:, 0] = circles[:, 0]
    result[:, 1] = circles[:, 1]
    result[:, 2] = circles[:, 2]

    return result
