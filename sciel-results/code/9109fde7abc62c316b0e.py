import numpy as np
import time

N = 21
TOL = 1e-7

def _validate_solution(pos_r, w, h, n_circles, tol=TOL):
    """pos_r: (n,3) array of [x,y,r]. Check bounds and non-overlap."""
    if pos_r is None or pos_r.shape[0] != n_circles:
        return False
    xs = pos_r[:, 0]
    ys = pos_r[:, 1]
    rs = pos_r[:, 2]
    if np.any(rs < 1e-9):
        return False
    if np.any(xs - rs < -tol) or np.any(xs + rs > w + tol):
        return False
    if np.any(ys - rs < -tol) or np.any(ys + rs > h + tol):
        return False
    for i in range(n_circles):
        for j in range(i + 1, n_circles):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < (rs[i] + rs[j]) - tol:
                return False
    return True


def _resolve_collisions(circles, w, h, max_iters=2000, tol=1e-8):
    """Physics-based relaxation: push circles to resolve overlaps and boundary violations."""
    n = circles.shape[0]
    pos = circles[:, :2].copy()
    rad = circles[:, 2].copy()

    for _ in range(max_iters):
        moved = False
        # Boundary constraints
        for i in range(n):
            x, y, r = pos[i, 0], pos[i, 1], rad[i]
            if x - r < 0:
                pos[i, 0] = r
                moved = True
            if x + r > w:
                pos[i, 0] = w - r
                moved = True
            if y - r < 0:
                pos[i, 1] = r
                moved = True
            if y + r > h:
                pos[i, 1] = h - r
                moved = True
        # Pairwise collisions
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = rad[i] + rad[j]
                if dist < min_dist:
                    if dist < 1e-9:
                        angle = np.random.uniform(0, 2 * np.pi)
                        push = (min_dist - dist) * 0.5
                        pos[i, 0] -= np.cos(angle) * push
                        pos[i, 1] -= np.sin(angle) * push
                        pos[j, 0] += np.cos(angle) * push
                        pos[j, 1] += np.sin(angle) * push
                        moved = True
                    else:
                        overlap = min_dist - dist
                        nx = dx / dist
                        ny = dy / dist
                        pos[i, 0] -= nx * overlap * 0.5
                        pos[i, 1] -= ny * overlap * 0.5
                        pos[j, 0] += nx * overlap * 0.5
                        pos[j, 1] += ny * overlap * 0.5
                        moved = True
        if not moved:
            break

    for i in range(n):
        pos[i, 0] = np.clip(pos[i, 0], rad[i], w - rad[i])
        pos[i, 1] = np.clip(pos[i, 1], rad[i], h - rad[i])
    return np.column_stack((pos, rad))


def _project_feasible(circles, w, h, max_iters=500):
    """Projection onto the feasible set: enforce non-overlap and boundaries via relaxation."""
    return _resolve_collisions(circles, w, h, max_iters=max_iters)


def _gradient_flow(circles, w, h, deadline, dt_init=0.005, dt_min=1e-5, cool=0.9995):
    """
    Differential-inclusion / gradient-flow optimization.
    Each circle's radius grows (gradient of sum of radii), and positions are
    adjusted by a projection step that enforces non-overlap and boundary
    constraints. Semi-implicit Euler: grow r, then project positions.
    """
    n = circles.shape[0]
    pos = circles[:, :2].copy()
    rad = circles[:, 2].copy()
    best_score = float(np.sum(rad))
    best_state = circles.copy()

    dt = dt_init
    start = time.time()

    while time.time() < deadline - 1.0 and dt > dt_min:
        # Gradient step: grow all radii slightly (gradient of sum r is +1 in r)
        rad_new = rad + dt * 0.5
        # Cap radii so they can still fit
        max_r = min(w, h) * 0.49
        rad_new = np.minimum(rad_new, max_r)
        rad_new = np.maximum(rad_new, 1e-6)

        # Build candidate circles
        cand = np.column_stack((pos, rad_new))

        # Projection: resolve overlaps and boundary violations
        proj = _project_feasible(cand, w, h, max_iters=150)

        # Check if the projected radii are still >= the proposed radii (i.e., growth was feasible)
        # If projection had to shrink radii (it doesn't in _resolve_collisions, it only moves positions),
        # the radii are preserved. We accept the step if it improves the score.
        score = float(np.sum(proj[:, 2]))
        if score > best_score:
            best_score = score
            best_state = proj.copy()

        pos = proj[:, :2]
        rad = proj[:, 2]

        # Cool down the step size
        dt *= cool

    # Final projection to ensure validity
    final = _project_feasible(best_state, w, h, max_iters=200)
    if _validate_solution(final, w, h, n):
        return final, float(np.sum(final[:, 2]))
    return best_state, best_score


def _slsqp_optimize(circles, w, h, deadline):
    """SLSQP refinement on top of the gradient-flow result."""
    if time.time() > deadline - 5.0:
        return None
    n = len(circles)
    x0 = circles.flatten()

    def objective(x):
        return -np.sum(x[2::3])

    constraints = []
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3] - x[i*3+2]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: w - x[i*3] - x[i*3+2]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3+1] - x[i*3+2]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: h - x[i*3+1] - x[i*3+2]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3+2] - 1e-6})
    for i in range(n):
        for j in range(i + 1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: np.sqrt((x[i*3]-x[j*3])**2 + (x[i*3+1]-x[j*3+1])**2) - x[i*3+2] - x[j*3+2]
            })

    bounds = []
    for i in range(n):
        bounds.append((1e-6, w))
        bounds.append((1e-6, h))
        bounds.append((1e-6, min(w, h) / 2))

    try:
        from scipy.optimize import minimize
        res = minimize(
            objective, x0, method='SLSQP', bounds=bounds, constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-9, 'disp': False}
        )
        if res.success or res.x is not None:
            new_circles = res.x.reshape(-1, 3)
            if _validate_solution(new_circles, w, h, n):
                return new_circles
            fixed = _resolve_collisions(new_circles, w, h, max_iters=100)
            if _validate_solution(fixed, w, h, n):
                return fixed
    except Exception:
        pass
    return None


def _generate_grid_init(w, h, n_circles, cols, rows):
    circles = np.zeros((n_circles, 3))
    count = 0
    for i in range(cols):
        for j in range(rows):
            if count >= n_circles:
                break
            x = (i + 0.5) * w / cols
            y = (j + 0.5) * h / rows
            cell_w = w / cols
            cell_h = h / rows
            r = 0.45 * min(cell_w, cell_h)
            circles[count, 0] = x
            circles[count, 1] = y
            circles[count, 2] = r
            count += 1
    return circles


def _solve_aspect(w, h, deadline):
    """Solve for a specific aspect ratio: grid init -> relax -> gradient flow -> SLSQP."""
    best_state = None
    best_score = 0.0

    ar = w / h
    if ar > 1.5:
        grid_configs = [(4, 5), (3, 7), (4, 6)]
    elif ar < 0.67:
        grid_configs = [(5, 4), (7, 3), (6, 4)]
    else:
        grid_configs = [(4, 5), (5, 4), (5, 5), (4, 6), (6, 4)]

    # Pick the best grid initialization
    for cols, rows in grid_configs:
        if cols * rows < N:
            continue
        init = _generate_grid_init(w, h, N, cols, rows)
        relaxed = _resolve_collisions(init, w, h, max_iters=200)
        if _validate_solution(relaxed, w, h, N):
            score = float(np.sum(relaxed[:, 2]))
            if score > best_score:
                best_score = score
                best_state = relaxed

    if best_state is None:
        best_state = _generate_grid_init(w, h, N, 5, 5)
        best_state = _resolve_collisions(best_state, w, h, max_iters=200)
        if not _validate_solution(best_state, w, h, N):
            return None, 0.0

    # Gradient flow (primary engine)
    if time.time() < deadline - 10:
        gf_state, gf_score = _gradient_flow(best_state, w, h, deadline - 10)
        if gf_state is not None and gf_score > best_score:
            best_state = gf_state
            best_score = gf_score

    # SLSQP polish (if time remains)
    if time.time() < deadline - 5:
        slsqp_state = _slsqp_optimize(best_state, w, h, deadline - 1)
        if slsqp_state is not None:
            s = float(np.sum(slsqp_state[:, 2]))
            if s > best_score:
                best_state = slsqp_state
                best_score = s

    if _validate_solution(best_state, w, h, N):
        return best_state, best_score
    return None, 0.0


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    best_result = None
    best_score = 0.0

    # Aspect ratios with perimeter 2(w+h)=4 => w+h=2
    aspects = [
        (1.0, 1.0),
        (1.2, 0.8),
        (1.3, 0.7),
        (1.4, 0.6),
        (1.5, 0.5),
        (1.1, 0.9),
        (1.6, 0.4),
    ]

    # Deterministic order (no shuffle) to ensure the best aspect is always tried
    for w, h in aspects:
        if time.time() > deadline:
            break
        time_remaining = deadline - time.time()
        if time_remaining < 15:
            continue
        t_alloc = min(time_remaining - 5, 60)

        x_res, score = _solve_aspect(w, h, start_time + t_alloc)
        if x_res is not None and score > best_score:
            best_score = score
            best_result = x_res.copy()

    # Fallback
    if best_result is None:
        w, h = 1.0, 1.0
        init = _generate_grid_init(w, h, N, 5, 5)
        best_result = _resolve_collisions(init, w, h, max_iters=200)
        if not _validate_solution(best_result, w, h, N):
            # Last resort: tiny circles in a grid
            positions = []
            radii = []
            count = 0
            for i in range(5):
                for j in range(5):
                    if count >= N:
                        break
                    positions.append([(i + 0.5) * w / 5, (j + 0.5) * h / 5])
                    radii.append(0.5 * min(w / 5, h / 5) * 0.9)
                    count += 1
            positions = np.array(positions)
            radii = np.array(radii)
            best_result = np.column_stack([positions, radii])

    return best_result
