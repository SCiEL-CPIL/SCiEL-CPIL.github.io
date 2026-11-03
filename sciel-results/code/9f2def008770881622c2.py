"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

DEADLINE = 110.0  # seconds, safety margin before 120s kill


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


def _smooth_overlap_penalty(violations, beta=10.0):
    """Smooth approximation of sum(max(0, v)^2) using log-sum-exp style."""
    # Use soft-plus: max(0, v) ≈ v + log(1 + exp(-beta*v))/beta for large beta
    # Then square it. Alternatively, use a smooth quadratic:
    # f(v) = 0.5 * (v + sqrt(v^2 + eps^2))  gives smooth max(0,v)
    # Let's use: for each v, smooth_max(0, v) = 0.5*(v + sqrt(v^2 + eps^2))
    eps = 1e-4
    smooth_v = 0.5 * (violations + np.sqrt(violations ** 2 + eps ** 2))
    return np.sum(smooth_v ** 2)


def _objective(params, W, H, beta=10.0):
    """Minimize negative sum of radii + smooth penalties for overlaps and boundary violations."""
    circles = params.reshape(N, 3)
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]

    # Smooth penalty for negative radii
    penalty = _smooth_overlap_penalty(-r, beta=50.0) * 100.0

    # Smooth penalty for going outside rectangle
    penalty += _smooth_overlap_penalty(-x + r, beta=50.0) * 100.0
    penalty += _smooth_overlap_penalty(-y + r, beta=50.0) * 100.0
    penalty += _smooth_overlap_penalty(x - W + r, beta=50.0) * 100.0
    penalty += _smooth_overlap_penalty(y - H + r, beta=50.0) * 100.0

    # Smooth penalty for overlaps
    overlap_violations = []
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            overlap = r[i] + r[j] - dist
            overlap_violations.append(overlap)
    overlap_violations = np.array(overlap_violations)
    penalty += _smooth_overlap_penalty(overlap_violations, beta=beta) * 10.0

    return -np.sum(r) + penalty


def _local_optimize(circles, W, H, max_iter=5000, xatol=1e-8, fatol=1e-10):
    """Use Nelder-Mead to locally optimize the circle positions and radii."""
    params = circles.flatten()
    result = minimize(
        _objective, params, args=(W, H, 10.0),
        method='Nelder-Mead',
        options={'maxiter': max_iter, 'xatol': xatol, 'fatol': fatol, 'adaptive': True}
    )
    return result.x.reshape(N, 3)


def _polish(circles, W, H, max_iter=10000):
    """High-precision polish pass with tighter tolerances."""
    # First, ensure we start from a valid or near-valid configuration
    params = circles.flatten()
    
    # Use a very tight tolerance for the final polish
    result = minimize(
        _objective, params, args=(W, H, 20.0),
        method='Nelder-Mead',
        options={'maxiter': max_iter, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
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
    """Create a random initial configuration with small circles."""
    circles = []
    for _ in range(N):
        r = rng.uniform(0.01, min(W, H) / 6)
        x = rng.uniform(r, W - r)
        y = rng.uniform(r, H - r)
        circles.append([x, y, r])
    return np.array(circles)


def _successive_convex_relaxation(circles, W, H, max_sweeps=50, tol=1e-6):
    """
    Successive Convex Relaxation for circle packing with smooth penalties.
    """
    n = len(circles)
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()

    best_val = -np.sum(r)
    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()

    for sweep in range(max_sweeps):
        def lin_objective(params):
            p = params.reshape(n, 3)
            px = p[:, 0]
            py = p[:, 1]
            pr = p[:, 2]

            obj = -np.sum(pr)

            # Smooth boundary penalties
            obj += _smooth_overlap_penalty(-px + pr, beta=50.0) * 100.0
            obj += _smooth_overlap_penalty(-py + pr, beta=50.0) * 100.0
            obj += _smooth_overlap_penalty(px - W + pr, beta=50.0) * 100.0
            obj += _smooth_overlap_penalty(py - H + pr, beta=50.0) * 100.0
            obj += _smooth_overlap_penalty(-pr, beta=50.0) * 100.0

            # Linearized non-overlap penalties with smooth surrogate
            violations = []
            for i in range(n):
                for j in range(i + 1, n):
                    dx0 = x[i] - x[j]
                    dy0 = y[i] - y[j]
                    dist0 = np.sqrt(dx0 * dx0 + dy0 * dy0)
                    if dist0 < 1e-10:
                        dist0 = 1e-10
                    gx_i = dx0 / dist0
                    gy_i = dy0 / dist0
                    gx_j = -dx0 / dist0
                    gy_j = -dy0 / dist0

                    lin_dist = dist0 + gx_i * (px[i] - x[i]) + gy_i * (py[i] - y[i]) \
                                 + gx_j * (px[j] - x[j]) + gy_j * (py[j] - y[j])
                    violation = (pr[i] + pr[j]) - lin_dist
                    violations.append(violation)
            
            violations = np.array(violations)
            obj += _smooth_overlap_penalty(violations, beta=10.0) * 10.0

            return obj

        params = np.column_stack([x, y, r]).flatten()
        result = minimize(
            lin_objective, params,
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        new_params = result.x.reshape(n, 3)
        new_x = new_params[:, 0]
        new_y = new_params[:, 1]
        new_r = new_params[:, 2]

        # Clip to valid range
        new_x = np.clip(new_x, new_r, W - new_r)
        new_y = np.clip(new_y, new_r, H - new_r)
        new_r = np.clip(new_r, 0.001, min(W, H) / 2)

        new_val = -np.sum(new_r)
        if new_val < best_val - 1e-8:
            best_val = new_val
            best_x, best_y, best_r = new_x.copy(), new_y.copy(), new_r.copy()

        if np.max(np.abs(new_x - x)) < tol and np.max(np.abs(new_y - y)) < tol and np.max(np.abs(new_r - r)) < tol:
            break

        x, y, r = new_x, new_y, new_r

    return np.column_stack([best_x, best_y, best_r]), -best_val


def _best_known(W, H, time_budget=30.0):
    """Return the best configuration found so far for given W, H."""
    start = time.time()
    # Start with grid
    circles = _grid_initialization(W, H)
    best_val = -np.sum(circles[:, 2])
    best_circles = circles.copy()

    # Try local optimization
    circles_opt = _local_optimize(circles, W, H, max_iter=3000)
    circles_opt[:, 0] = np.clip(circles_opt[:, 0], circles_opt[:, 2], W - circles_opt[:, 2])
    circles_opt[:, 1] = np.clip(circles_opt[:, 1], circles_opt[:, 2], H - circles_opt[:, 2])
    circles_opt[:, 2] = np.clip(circles_opt[:, 2], 0.001, min(W, H) / 2)

    val = -np.sum(circles_opt[:, 2])
    if _check_validity(circles_opt, W, H) and val < best_val:
        best_val = val
        best_circles = circles_opt.copy()

    # Try successive convex relaxation if time permits
    if time.time() - start < time_budget - 5:
        sc_circles, sc_score = _successive_convex_relaxation(circles_opt, W, H, max_sweeps=20)
        sc_circles[:, 0] = np.clip(sc_circles[:, 0], sc_circles[:, 2], W - sc_circles[:, 2])
        sc_circles[:, 1] = np.clip(sc_circles[:, 1], sc_circles[:, 2], H - sc_circles[:, 2])
        sc_circles[:, 2] = np.clip(sc_circles[:, 2], 0.001, min(W, H) / 2)
        if _check_validity(sc_circles, W, H) and sc_score > -best_val:
            best_val = -sc_score
            best_circles = sc_circles.copy()

    # Final polish if time permits
    if time.time() - start < time_budget - 2:
        polished = _polish(best_circles, W, H, max_iter=5000)
        polished[:, 0] = np.clip(polished[:, 0], polished[:, 2], W - polished[:, 2])
        polished[:, 1] = np.clip(polished[:, 1], polished[:, 2], H - polished[:, 2])
        polished[:, 2] = np.clip(polished[:, 2], 0.001, min(W, H) / 2)
        if _check_validity(polished, W, H):
            s = np.sum(polished[:, 2])
            if s > -best_val:
                best_val = -s
                best_circles = polished.copy()

    return best_circles, -best_val


def construct():
    start_time = time.time()
    best_score = -np.inf
    best_result = None
    best_W, best_H = 1.0, 1.0

    # Try different aspect ratios
    aspect_ratios = [0.5, 0.7, 1.0, 1.2, 1.5, 2.0, 3.0]

    rng = np.random.default_rng(42)

    for ar in aspect_ratios:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 15:
            break
        
        # Allocate more time to promising aspect ratios (near square)
        if ar == 1.0:
            time_budget = 25.0
        elif 0.7 <= ar <= 1.5:
            time_budget = 20.0
        else:
            time_budget = 12.0
        
        W = 2.0 * ar / (ar + 1.0)
        H = 2.0 / (ar + 1.0)

        circles, score = _best_known(W, H, time_budget=time_budget)
        if score > best_score:
            best_score = score
            best_result = circles.copy()
            best_W, best_H = W, H

        # Try random restarts if time permits
        remaining = DEADLINE - 10 - (time.time() - start_time)
        if remaining > 5:
            n_restarts = max(1, int(remaining / 5))
            for _ in range(n_restarts):
                if time.time() - start_time > DEADLINE - 10:
                    break
                circ_rand = _random_initialization(W, H, rng)
                circ_opt = _local_optimize(circ_rand, W, H, max_iter=1500)
                circ_opt[:, 0] = np.clip(circ_opt[:, 0], circ_opt[:, 2], W - circ_opt[:, 2])
                circ_opt[:, 1] = np.clip(circ_opt[:, 1], circ_opt[:, 2], H - circ_opt[:, 2])
                circ_opt[:, 2] = np.clip(circ_opt[:, 2], 0.001, min(W, H) / 2)

                if _check_validity(circ_opt, W, H):
                    s = np.sum(circ_opt[:, 2])
                    if s > best_score:
                        best_score = s
                        best_result = circ_opt.copy()
                        best_W, best_H = W, H

    # Final polish on the best result
    if best_result is not None and time.time() - start_time < DEADLINE - 5:
        polished = _polish(best_result, best_W, best_H, max_iter=8000)
        polished[:, 0] = np.clip(polished[:, 0], polished[:, 2], best_W - polished[:, 2])
        polished[:, 1] = np.clip(polished[:, 1], polished[:, 2], best_H - polished[:, 2])
        polished[:, 2] = np.clip(polished[:, 2], 0.001, min(best_W, best_H) / 2)
        if _check_validity(polished, best_W, best_H):
            s = np.sum(polished[:, 2])
            if s > best_score:
                best_score = s
                best_result = polished.copy()

    # Final fallback: simple grid in square
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        best_result = np.array(cells[:N])
        best_W, best_H = 1.0, 1.0

    # Ensure the result is valid; if not, fall back to a safe grid
    if not _check_validity(best_result, best_W, best_H):
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        best_result = np.array(cells[:N])

    return best_result
# EVOLVE-BLOCK-END
