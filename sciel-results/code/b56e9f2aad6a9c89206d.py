"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

DEADLINE = 140.0  # Safety margin before 150s kill

def _time_ok(start, margin=10.0):
    return (time.time() - start) < (DEADLINE - margin)

def _force_directed(pos, radii, w, h, iters=60, k_rep=1.0, k_att=0.05, lr=0.1):
    """Force-directed relaxation: repel overlapping pairs, attract to center.
    Returns updated positions (radii unchanged)."""
    pos = pos.copy()
    for _ in range(iters):
        forces = np.zeros_like(pos)
        n = len(pos)
        for i in range(n):
            # Attraction to center
            forces[i] += k_att * (np.array([w / 2.0, h / 2.0]) - pos[i])
            for j in range(i + 1, n):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < 1e-9:
                    dx, dy = 1e-6, 1e-6
                    dist = np.sqrt(dx * dx + dy * dy)
                # Repulsion if overlapping
                overlap = radii[i] + radii[j] - dist
                if overlap > 0:
                    f = k_rep * overlap
                    fx = f * dx / dist
                    fy = f * dy / dist
                    forces[i] += np.array([fx, fy])
                    forces[j] -= np.array([fx, fy])
        pos += lr * forces
        # Clamp to rectangle (keep inside, with radius margin)
        for i in range(n):
            r = radii[i]
            pos[i, 0] = np.clip(pos[i, 0], r, w - r)
            pos[i, 1] = np.clip(pos[i, 1], r, h - r)
    return pos

def _validate(x, w, h, tol=1e-6):
    """Check strict validity of a flat vector x = [x1,y1,r1, x2,y2,r2, ...]."""
    n = len(x) // 3
    xs = x[0::3]
    ys = x[1::3]
    rs = x[2::3]
    if np.any(rs < -tol):
        return False
    if np.any(xs < rs - tol) or np.any(xs > w - rs + tol):
        return False
    if np.any(ys < rs - tol) or np.any(ys > h - rs + tol):
        return False
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt((xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2)
            if dist < rs[i] + rs[j] - tol:
                return False
    return True

def _slsqp_solve(x0, w, h, maxiter=300, ftol=1e-9):
    """Run SLSQP to maximize sum(r) with constraints. Returns (x, valid)."""
    n = len(x0) // 3
    xs = x0[0::3]
    ys = x0[1::3]
    rs = x0[2::3]

    def objective(v):
        return -np.sum(v[2::3])

    def cons_inside(v):
        x = v[0::3]
        y = v[1::3]
        r = v[2::3]
        return np.concatenate([x - r, w - x - r, y - r, h - y - r])

    def cons_non_overlap(v):
        x = v[0::3]
        y = v[1::3]
        r = v[2::3]
        cons = []
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                cons.append(dist - r[i] - r[j])
        return np.array(cons)

    def cons_positive_r(v):
        return v[2::3]

    constraints = [
        {'type': 'ineq', 'fun': cons_inside},
        {'type': 'ineq', 'fun': cons_non_overlap},
        {'type': 'ineq', 'fun': cons_positive_r},
    ]

    try:
        result = minimize(
            objective, x0, method='SLSQP',
            constraints=constraints,
            options={'maxiter': maxiter, 'ftol': ftol},
        )
        x_res = result.x
        if _validate(x_res, w, h):
            return x_res, True
        else:
            return x_res, False
    except Exception:
        return x0, False

def construct():
    start_time = time.time()
    best_score = -np.inf
    best_result = None

    np.random.seed(42)

    # Initial aspect ratio guesses: a = w/h, with w+h=2
    # Square is 1.0; we'll let SLSQP refine, but start from a spread.
    aspect_guesses = [1.0, 0.85, 1.15, 0.7, 1.3, 1.5, 0.6]

    for a in aspect_guesses:
        if not _time_ok(start_time, margin=15.0):
            break

        w = 2.0 * a / (1.0 + a)
        h = 2.0 - w
        if h < 0.15 or w < 0.15:
            continue

        # Grid initialization matching aspect ratio
        cols = max(2, int(np.sqrt(N * w / h)))
        rows = max(2, int(np.ceil(N / cols)))
        while cols * rows < N:
            if cols <= rows:
                cols += 1
            else:
                rows += 1

        for restart in range(3):
            if not _time_ok(start_time, margin=5.0):
                break

            positions = np.zeros((N, 2))
            idx = 0
            for r_idx in range(rows):
                for c_idx in range(cols):
                    if idx >= N:
                        break
                    x = (c_idx + 0.5) * w / cols
                    y = (r_idx + 0.5) * h / rows
                    if restart > 0:
                        jitter = 0.15 * min(w / cols, h / rows)
                        x += np.random.uniform(-jitter, jitter)
                        y += np.random.uniform(-jitter, jitter)
                    x = np.clip(x, 0.01, w - 0.01)
                    y = np.clip(y, 0.01, h - 0.01)
                    positions[idx] = [x, y]
                    idx += 1

            # Initial radius estimate
            min_dist = np.inf
            for i in range(N):
                for j in range(i + 1, N):
                    d = np.sqrt((positions[i, 0] - positions[j, 0]) ** 2 +
                                (positions[i, 1] - positions[j, 1]) ** 2)
                    min_dist = min(min_dist, d)
            for i in range(N):
                min_dist = min(min_dist, positions[i, 0], w - positions[i, 0],
                               positions[i, 1], h - positions[i, 1])
            r0 = 0.35 * min_dist
            radii = np.full(N, r0)

            # Force-directed pre-polish to escape local minima
            positions = _force_directed(positions, radii, w, h, iters=50)

            # Build flat initial vector
            x0 = np.zeros(3 * N)
            for i in range(N):
                x0[3 * i] = positions[i, 0]
                x0[3 * i + 1] = positions[i, 1]
                x0[3 * i + 2] = radii[i]

            # SLSQP precise optimization
            x_res, valid = _slsqp_solve(x0, w, h, maxiter=300, ftol=1e-9)
            if valid:
                score = np.sum(x_res[2::3])
                if score > best_score:
                    best_score = score
                    best_result = np.column_stack((
                        x_res[0::3], x_res[1::3], x_res[2::3]
                    )).copy()

    # Fallback: simple valid grid packing in a square (1x1, perimeter 4)
    if best_result is None or best_score < 0:
        side = 1.0
        cols, rows = 5, 5
        cells = []
        idx = 0
        for r_idx in range(rows):
            for c_idx in range(cols):
                if idx >= N:
                    break
                x = (c_idx + 0.5) * side / cols
                y = (r_idx + 0.5) * side / rows
                r = 0.5 / cols - 1e-3
                cells.append([x, y, r])
                idx += 1
        best_result = np.array(cells[:N])

    return best_result
# EVOLVE-BLOCK-END
