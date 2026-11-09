"""Circle packing in unit square: 26 circles, maximize sum of radii.
Two-phase: (1) vectorized force-directed relaxation, (2) constraint-aware
local search with radius maximization and pairwise separation repair.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 26
DEADLINE = 130.0  # safety margin before 150s kill


def _vectorized_relax(x, y, r, lr=0.008, lam_ov=8.0, lam_bd=8.0,
                      max_steps=4000, deadline=None, start_time=None):
    """Fast vectorized force-directed relaxation with adaptive learning rate.
    Uses stronger overlap repulsion and boundary forces for faster convergence."""
    for step in range(max_steps):
        if deadline is not None and time.time() > deadline:
            break
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d = np.sqrt(dx**2 + dy**2)
        d_safe = np.where(d < 1e-8, 1e-8, d)

        r_sum = r[:, None] + r[None, :]
        overlap = np.maximum(0, r_sum - d)
        np.fill_diagonal(overlap, 0)

        # Overlap forces: push apart
        gx_ov = -2 * lam_ov * np.sum(overlap * (dx / d_safe), axis=1)
        gy_ov = -2 * lam_ov * np.sum(overlap * (dy / d_safe), axis=1)
        gr_ov = 2 * lam_ov * np.sum(overlap, axis=1)

        # Boundary forces: keep circles inside
        bxl = np.maximum(0, r - x)
        bxr = np.maximum(0, r - (1 - x))
        byl = np.maximum(0, r - y)
        byr = np.maximum(0, r - (1 - y))

        gx_bd = -2 * lam_bd * bxl + 2 * lam_bd * bxr
        gy_bd = -2 * lam_bd * byl + 2 * lam_bd * byr
        gr_bd = 2 * lam_bd * (bxl + bxr + byl + byr)

        # Objective: maximize radii
        gr_obj = -np.ones(N)

        gx = gx_ov + gx_bd
        gy = gy_ov + gy_bd
        gr = gr_ov + gr_bd + gr_obj

        # Adaptive learning rate: scale inversely with gradient magnitude
        grad_mag = np.sqrt(np.sum(gx**2) + np.sum(gy**2) + np.sum(gr**2))
        if grad_mag > 1e-8:
            eff_lr = lr * min(1.0, 5.0 / grad_mag)
        else:
            eff_lr = lr

        x -= eff_lr * gx
        y -= eff_lr * gy
        r -= eff_lr * gr

        r = np.clip(r, 0.01, 0.4)
        x = np.clip(x, 0, 1)
        y = np.clip(y, 0, 1)
        lr *= 0.999

    return x, y, r


def _max_radius_for_position(x, y, r, exclude=None):
    """For each circle i, compute max radius allowed by boundary and
    distance to all other circles (given their current radii). Vectorized."""
    # Boundary constraint
    r_max = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
    
    # Pairwise constraints: for each i, r_i <= d_ij - r_j for all j != i
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(d, np.inf)
    
    # r_max[i] = min over j of (d[i,j] - r[j])
    r_constraint = d - r[None, :]  # shape (N, N): r_constraint[i,j] = d_ij - r_j
    # For each i, take min over j (excluding j=i which is inf)
    r_max = np.minimum(r_max, np.min(r_constraint, axis=1))
    
    r_max = np.maximum(r_max, 1e-6)
    return r_max


def _local_search(x, y, r, deadline, start_time):
    """Constraint-aware local search with multiple phases:
    1. Enforce validity by shrinking.
    2. Iterative radius maximization.
    3. Simulated annealing for position perturbation.
    4. Pairwise separation moves.
    5. Scipy SLSQP optimization on the full configuration.
    """
    x = x.copy()
    y = y.copy()
    r = r.copy()

    best_sum = np.sum(r)
    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()

    # Phase 0: Enforce validity by shrinking
    for _ in range(100):
        if time.time() > deadline:
            break
        r_max_b = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        r = np.minimum(r, r_max_b)
        r = np.maximum(r, 1e-6)

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(d, np.inf)
        r_sum = r[:, None] + r[None, :]
        overlaps = d < r_sum - 1e-9
        np.fill_diagonal(overlaps, False)

        if not np.any(overlaps):
            break

        overlap_count = np.sum(overlaps, axis=1)
        for i in range(N):
            if overlap_count[i] > 0:
                r[i] *= (1 - 0.03 * overlap_count[i])
        r = np.maximum(r, 1e-6)

    # Phase 1: Iterative radius maximization
    for _ in range(150):
        if time.time() > deadline:
            break
        r_max = _max_radius_for_position(x, y, r)
        r = r_max
        s = np.sum(r)
        if s > best_sum:
            best_sum = s
            best_x, best_y, best_r = x.copy(), y.copy(), r.copy()

    # Phase 2: Simulated Annealing on positions
    # Perturb positions, recompute max radii, accept if sum improves or with prob exp(-delta/T)
    T = 0.01  # initial temperature
    T_min = 0.0001
    cooling = 0.97
    sa_steps = 500
    for step in range(sa_steps):
        if time.time() > deadline:
            break
        if T < T_min:
            break

        # Pick a random circle to perturb
        i = np.random.randint(N)
        old_xi, old_yi = x[i], y[i]
        # Gaussian perturbation
        x[i] += np.random.normal(0, 0.01)
        y[i] += np.random.normal(0, 0.01)
        x[i] = np.clip(x[i], 0.01, 0.99)
        y[i] = np.clip(y[i], 0.01, 0.99)

        # Recompute max radii
        r_new = _max_radius_for_position(x, y, r)
        s_new = np.sum(r_new)
        s_old = np.sum(r)
        delta = s_new - s_old

        # Accept if better, or with probability exp(-delta/T) if worse
        if delta > 0 or np.random.rand() < np.exp(-delta / T):
            r = r_new
            if s_new > best_sum:
                best_sum = s_new
                best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
        else:
            # Revert
            x[i], y[i] = old_xi, old_yi

        T *= cooling

    # Phase 3: Pairwise separation moves
    for iteration in range(200):
        if time.time() > deadline:
            break

        r = _max_radius_for_position(x, y, r)

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(d, np.inf)
        r_sum = r[:, None] + r[None, :]
        tight = np.abs(d - r_sum) < 0.015
        np.fill_diagonal(tight, False)

        if not np.any(tight):
            break

        moved = False
        for i in range(N):
            for j in range(i + 1, N):
                if not tight[i, j]:
                    continue
                dirx = x[i] - x[j]
                diry = y[i] - y[j]
                norm = np.sqrt(dirx**2 + diry**2)
                if norm < 1e-8:
                    continue
                dirx /= norm
                diry /= norm

                delta = 0.003
                new_xi = x[i] + delta * dirx
                new_yi = y[i] + delta * diry
                new_xj = x[j] - delta * dirx
                new_yj = y[j] - delta * diry

                if (new_xi < 0 or new_xi > 1 or new_yi < 0 or new_yi > 1 or
                        new_xj < 0 or new_xj > 1 or new_yj < 0 or new_yj > 1):
                    continue

                old_xi, old_yi = x[i], y[i]
                old_xj, old_yj = x[j], y[j]
                x[i], y[i] = new_xi, new_yi
                x[j], y[j] = new_xj, new_yj

                r_new = _max_radius_for_position(x, y, r)
                s_new = np.sum(r_new)
                if s_new > np.sum(r) + 1e-8:
                    r = r_new
                    moved = True
                else:
                    x[i], y[i] = old_xi, old_yi
                    x[j], y[j] = old_xj, old_yj

        if not moved:
            break

        s = np.sum(r)
        if s > best_sum:
            best_sum = s
            best_x, best_y, best_r = x.copy(), y.copy(), r.copy()

    # Phase 4: Scipy SLSQP optimization
    if time.time() < deadline - 5:
        try:
            # Pack all 3*N variables: x[0:N], y[0:N], r[0:N]
            def _neg_sum(vars):
                return -np.sum(vars[2*N:3*N])

            def _constraint_overlap(vars):
                """All pairwise separation constraints: d_ij >= r_i + r_j - 1e-9"""
                xx = vars[:N]
                yy = vars[N:2*N]
                rr = vars[2*N:3*N]
                dx = xx[:, None] - xx[None, :]
                dy = yy[:, None] - yy[None, :]
                d = np.sqrt(dx**2 + dy**2)
                r_sum = rr[:, None] + rr[None, :]
                # Return d - r_sum for all pairs (excluding diagonal)
                np.fill_diagonal(d, np.inf)
                np.fill_diagonal(r_sum, 0)
                return d - r_sum - 1e-9

            def _constraint_boundary(vars):
                """x >= r, 1-x >= r, y >= r, 1-y >= r"""
                xx = vars[:N]
                yy = vars[N:2*N]
                rr = vars[2*N:3*N]
                return np.concatenate([
                    xx - rr, 1 - xx - rr, yy - rr, 1 - yy - rr
                ])

            constraints = [
                {'type': 'ineq', 'fun': lambda v: np.min(_constraint_overlap(v))},
                {'type': 'ineq', 'fun': lambda v: np.min(_constraint_boundary(v))},
            ]

            bounds = [(0, 1) for _ in range(N)] + [(0, 1) for _ in range(N)] + [(0.001, 0.4) for _ in range(N)]

            x0 = np.concatenate([best_x, best_y, best_r])
            result = minimize(_neg_sum, x0, method='SLSQP', bounds=bounds,
                              constraints=constraints, options={'maxiter': 200, 'ftol': 1e-8})

            if result.success and _validate(result.x[:N], result.x[N:2*N], result.x[2*N:3*N]):
                s_opt = np.sum(result.x[2*N:3*N])
                if s_opt > best_sum:
                    best_sum = s_opt
                    best_x, best_y, best_r = (result.x[:N].copy(), result.x[N:2*N].copy(),
                                               result.x[2*N:3*N].copy())
        except Exception:
            pass

    # Final validation and return best
    if _validate(best_x, best_y, best_r):
        return best_x, best_y, best_r
    return x, y, r


def _validate(x, y, r, tol=1e-9):
    """Check validity: boundary and no overlaps."""
    if np.any(r < 0):
        return False
    if np.any(x < r - tol) or np.any(x > 1 - r + tol) or \
       np.any(y < r - tol) or np.any(y > 1 - r + tol):
        return False
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(d, np.inf)
    r_sum = r[:, None] + r[None, :]
    if np.any(d < r_sum - tol):
        return False
    return True


def _hex_init(n, scale=1.0):
    """Place n circles on a hexagonal lattice scaled to fit in unit square."""
    s = scale
    pts = []
    cols = int(np.ceil(1.0 / s)) + 2
    rows = int(np.ceil(1.0 / (s * np.sqrt(3) / 2))) + 2
    for j in range(rows):
        y = j * s * np.sqrt(3) / 2
        offset = 0 if j % 2 == 0 else s / 2
        for i in range(cols):
            x = i * s + offset
            if 0 <= x <= 1 and 0 <= y <= 1:
                pts.append((x, y))
    pts = np.array(pts)
    if len(pts) < n:
        extra = np.random.rand(n - len(pts), 2) * 0.8 + 0.1
        pts = np.vstack([pts, extra])
    return pts[:n]


def construct():
    t_start = time.time()
    deadline = t_start + DEADLINE

    best_score = -np.inf
    best_config = None

    def _try_config(x, y, r):
        nonlocal best_score, best_config
        if _validate(x, y, r):
            s = np.sum(r)
            if s > best_score:
                best_score = s
                best_config = (x.copy(), y.copy(), r.copy())
        else:
            # Try to repair
            for _ in range(50):
                if time.time() > deadline:
                    return
                r_max_b = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
                r = np.minimum(r, r_max_b)
                r = np.maximum(r, 1e-6)
                dx = x[:, None] - x[None, :]
                dy = y[:, None] - y[None, :]
                d = np.sqrt(dx**2 + dy**2)
                np.fill_diagonal(d, np.inf)
                r_sum = r[:, None] + r[None, :]
                overlaps = d < r_sum - 1e-9
                np.fill_diagonal(overlaps, False)
                if not np.any(overlaps):
                    break
                overlap_count = np.sum(overlaps, axis=1)
                for i in range(N):
                    if overlap_count[i] > 0:
                        r[i] *= (1 - 0.03 * overlap_count[i])
                r = np.maximum(r, 1e-6)
            if _validate(x, y, r):
                s = np.sum(r)
                if s > best_score:
                    best_score = s
                    best_config = (x.copy(), y.copy(), r.copy())

    # Multiple diverse initializations
    num_restarts = 12
    for restart in range(num_restarts):
        if time.time() > deadline:
            break

        np.random.seed(1000 + restart * 137)

        if restart % 4 == 0:
            # Hex lattice
            scale = 0.18 - 0.01 * (restart // 4)
            pts = _hex_init(N, scale)
            x = pts[:, 0].copy()
            y = pts[:, 1].copy()
            r = np.full(N, 0.03)
        elif restart % 4 == 1:
            # Jittered grid
            side = int(np.ceil(np.sqrt(N)))
            xs, ys, rs = [], [], []
            for i in range(side):
                for j in range(side):
                    if len(xs) < N:
                        px = (i + 0.5) / side + np.random.uniform(-0.05, 0.05)
                        py = (j + 0.5) / side + np.random.uniform(-0.05, 0.05)
                        px = np.clip(px, 0.1, 0.9)
                        py = np.clip(py, 0.1, 0.9)
                        xs.append(px)
                        ys.append(py)
                        rs.append(0.05)
            x = np.array(xs)
            y = np.array(ys)
            r = np.array(rs)
        elif restart % 4 == 2:
            # Random
            x = np.random.uniform(0.15, 0.85, N)
            y = np.random.uniform(0.15, 0.85, N)
            r = np.random.uniform(0.03, 0.07, N)
        else:
            # Clustered: few large, many small
            xs, ys, rs = [], [], []
            for _ in range(6):
                xs.append(np.random.uniform(0.2, 0.8))
                ys.append(np.random.uniform(0.2, 0.8))
                rs.append(np.random.uniform(0.07, 0.11))
            for _ in range(N - 6):
                xs.append(np.random.uniform(0.1, 0.9))
                ys.append(np.random.uniform(0.1, 0.9))
                rs.append(np.random.uniform(0.02, 0.05))
            x = np.array(xs)
            y = np.array(ys)
            r = np.array(rs)

        # Phase 1: Force-directed relaxation
        x, y, r = _vectorized_relax(x, y, r, lr=0.008, lam_ov=8.0, lam_bd=8.0,
                                    max_steps=2500, deadline=deadline, start_time=t_start)
        _try_config(x, y, r)

        if time.time() > deadline:
            break

        # Phase 2: Local search with SA + SLSQP
        x2, y2, r2 = _local_search(x, y, r, deadline, t_start)
        _try_config(x2, y2, r2)

    # Fallback: simple grid
    if best_config is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        best_config = (np.array([c[0] for c in cells[:N]]),
                       np.array([c[1] for c in cells[:N]]),
                       np.array([c[2] for c in cells[:N]]))

    x, y, r = best_config
    return np.column_stack((x, y, r))
