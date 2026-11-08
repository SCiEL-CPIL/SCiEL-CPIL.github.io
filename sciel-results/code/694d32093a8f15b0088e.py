"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 26
DEADLINE = 115.0  # seconds, safety margin before 120s kill

# EVOLVE-BLOCK-START

def _make_initial_grid(n, rng=None):
    """Create a slightly perturbed grid initial guess."""
    if rng is None:
        rng = np.random.default_rng(42)
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= n:
                break
            x = (i + 0.5) / side + (rng.random() - 0.5) * 0.02
            y = (j + 0.5) / side + (rng.random() - 0.5) * 0.02
            r = 0.5 / side - 0.005
            cells.append([x, y, r])
    return np.array(cells[:n])


def _objective(params, n):
    """
    Objective: minimize negative sum of radii plus penalty for violations.
    params: flat array of length 3*n: [x0,y0,r0, x1,y1,r1, ...]
    """
    pts = params.reshape(n, 3)
    x, y, r = pts[:, 0], pts[:, 1], pts[:, 2]

    # Boundary penalties: circles must be inside [0,1]^2
    # x >= r, x <= 1-r, y >= r, y <= 1-r
    bound_pen = 0.0
    for i in range(n):
        if x[i] < r[i]:
            bound_pen += (r[i] - x[i]) ** 2
        if x[i] > 1 - r[i]:
            bound_pen += (x[i] - (1 - r[i])) ** 2
        if y[i] < r[i]:
            bound_pen += (r[i] - y[i]) ** 2
        if y[i] > 1 - r[i]:
            bound_pen += (y[i] - (1 - r[i])) ** 2

    # Overlap penalties: distance between centers >= r_i + r_j
    overlap_pen = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = r[i] + r[j]
            if dist < min_dist:
                overlap_pen += (min_dist - dist) ** 2

    # Negative sum of radii (we maximize sum of radii)
    neg_sum_r = -np.sum(r)

    # Total penalty weight
    penalty_weight = 100.0
    total = neg_sum_r + penalty_weight * (bound_pen + overlap_pen)
    return total


def _objective_fast(params, n):
    """Vectorized objective for speed."""
    pts = params.reshape(n, 3)
    x, y, r = pts[:, 0], pts[:, 1], pts[:, 2]

    # Boundary penalties
    bound_pen = 0.0
    # x lower: x < r -> (r - x)^2
    mask_xl = x < r
    if np.any(mask_xl):
        bound_pen += np.sum((r[mask_xl] - x[mask_xl]) ** 2)
    # x upper: x > 1-r -> (x - (1-r))^2
    mask_xu = x > 1 - r
    if np.any(mask_xu):
        bound_pen += np.sum((x[mask_xu] - (1 - r[mask_xu])) ** 2)
    # y lower
    mask_yl = y < r
    if np.any(mask_yl):
        bound_pen += np.sum((r[mask_yl] - y[mask_yl]) ** 2)
    # y upper
    mask_yu = y > 1 - r
    if np.any(mask_yu):
        bound_pen += np.sum((y[mask_yu] - (1 - r[mask_yu])) ** 2)

    # Overlap penalties - vectorized
    overlap_pen = 0.0
    for i in range(n):
        dx = x[i] - x[i + 1:]
        dy = y[i] - y[i + 1:]
        dist = np.sqrt(dx * dx + dy * dy)
        min_dist = r[i] + r[i + 1:]
        diff = min_dist - dist
        mask = diff > 0
        if np.any(mask):
            overlap_pen += np.sum(diff[mask] ** 2)

    neg_sum_r = -np.sum(r)
    penalty_weight = 100.0
    total = neg_sum_r + penalty_weight * (bound_pen + overlap_pen)
    return total


def _check_validity(pts, n, tol=1e-6):
    """Check if all circles are inside unit square and non-overlapping."""
    x, y, r = pts[:, 0], pts[:, 1], pts[:, 2]
    # Boundary check
    if np.any(x < r - tol) or np.any(x > 1 - r + tol):
        return False
    if np.any(y < r - tol) or np.any(y > 1 - r + tol):
        return False
    # Overlap check
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - tol:
                return False
    return True


def _score(pts, n):
    """Compute sum of radii if valid, else -inf."""
    if _check_validity(pts, n, tol=1e-5):
        return np.sum(pts[:, 2])
    return -np.inf


def _fix_and_validate(params, n):
    """Fix boundary violations by clamping, then check overlaps."""
    pts = params.reshape(n, 3).copy()
    x, y, r = pts[:, 0], pts[:, 1], pts[:, 2]

    # Clamp radii to be positive
    r = np.maximum(r, 1e-8)
    pts[:, 2] = r

    # Clamp positions to satisfy boundary
    x = np.clip(x, r, 1 - r)
    y = np.clip(y, r, 1 - r)
    pts[:, 0] = x
    pts[:, 1] = y

    return pts


def construct():
    start_time = time.time()
    best_pts = None
    best_score = -np.inf

    # Try multiple random restarts
    n_restarts = 8
    for restart in range(n_restarts):
        if time.time() - start_time > DEADLINE - 5:
            break

        rng = np.random.default_rng(42 + restart)
        side = int(np.ceil(np.sqrt(N)))

        # Different initializations for diversity
        if restart == 0:
            # Grid
            pts_init = _make_initial_grid(N, rng)
        elif restart == 1:
            # Slightly different grid
            pts_init = _make_initial_grid(N, rng)
        elif restart == 2:
            # Random positions
            pts_init = np.zeros((N, 3))
            pts_init[:, 0] = rng.uniform(0.1, 0.9, N)
            pts_init[:, 1] = rng.uniform(0.1, 0.9, N)
            pts_init[:, 2] = 0.5 / side - 0.01
        elif restart == 3:
            # Hexagonal-like packing
            spacing = 1.0 / (side + 0.5)
            pts_init = np.zeros((N, 3))
            idx = 0
            for row in range(side + 1):
                for col in range(side + 1):
                    if idx >= N:
                        break
                    x = (col + 0.5 * (row % 2)) * spacing
                    y = row * spacing * 0.866
                    if x < 1 and y < 1:
                        pts_init[idx, 0] = x
                        pts_init[idx, 1] = y
                        pts_init[idx, 2] = spacing * 0.5 - 0.005
                        idx += 1
            if idx < N:
                # Fill remaining with random
                for k in range(idx, N):
                    pts_init[k, 0] = rng.uniform(0.1, 0.9)
                    pts_init[k, 1] = rng.uniform(0.1, 0.9)
                    pts_init[k, 2] = 0.5 / side - 0.01
        else:
            # Random with varying radii
            pts_init = np.zeros((N, 3))
            pts_init[:, 0] = rng.uniform(0.15, 0.85, N)
            pts_init[:, 1] = rng.uniform(0.15, 0.85, N)
            pts_init[:, 2] = rng.uniform(0.03, 0.08, N)

        # Ensure initial validity by clamping
        pts_init = _fix_and_validate(pts_init.flatten(), N)

        # Optimization
        x0 = pts_init.flatten()
        bounds = []
        for i in range(N):
            bounds.append((1e-8, 1 - 1e-8))  # x
            bounds.append((1e-8, 1 - 1e-8))  # y
            bounds.append((1e-8, 0.5))        # r

        try:
            result = minimize(
                _objective_fast,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 2000}
            )
        except Exception:
            continue

        if time.time() - start_time > DEADLINE - 3:
            break

        # Extract and validate
        pts_opt = _fix_and_validate(result.x, N)
        s = _score(pts_opt, N)

        if s > best_score:
            best_score = s
            best_pts = pts_opt.copy()

        # If we have a valid solution, try to improve with local refinement
        if s > 0 and time.time() - start_time < DEADLINE - 10:
            # Second pass with tighter tolerances
            try:
                result2 = minimize(
                    _objective_fast,
                    pts_opt.flatten(),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 1500}
                )
                pts_opt2 = _fix_and_validate(result2.x, N)
                s2 = _score(pts_opt2, N)
                if s2 > best_score:
                    best_score = s2
                    best_pts = pts_opt2.copy()
            except Exception:
                pass

    # If no valid solution found, fall back to grid
    if best_pts is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        best_pts = np.array(cells[:N])

    return best_pts

# EVOLVE-BLOCK-END
