import numpy as np
import time

N = 21
TOL = 1e-5

def _validate_solution(pos_r, w, h, n_circles, tol=TOL):
    if pos_r is None or len(pos_r) != n_circles * 3:
        return False
    for i in range(n_circles):
        xi = pos_r[2 * i]
        yi = pos_r[2 * i + 1]
        ri = pos_r[2 * i + 2]
        if ri < 0:
            return False
        if xi - ri < -tol or xi + ri > w + tol:
            return False
        if yi - ri < -tol or yi + ri > h + tol:
            return False
    for i in range(n_circles):
        for j in range(i + 1, n_circles):
            dist = np.hypot(pos_r[2 * i] - pos_r[2 * j], pos_r[2 * i + 1] - pos_r[2 * j + 1])
            if dist < pos_r[2 * i + 2] + pos_r[2 * j + 2] - tol:
                return False
    return True


def _slsqp_optimize(x0, w, h, time_budget):
    from scipy.optimize import minimize

    n = N
    x = x0.copy()
    best_score = np.sum(x[2::3])
    best_x = x.copy()

    def neg_obj(vec):
        nonlocal x, best_score, best_x
        # Check validity and update best
        # We check wall constraints and pairwise overlap
        # For SLSQP, we can use penalties to keep it in bounds if we don't use constraints
        # But let's use SLSQP constraints properly
        return -np.sum(vec[2::3])

    def obj_grad(vec):
        # Gradient of -sum(r) is just 0,0,-1,0,0,-1...
        grad = np.zeros(3 * n)
        grad[2::3] = -1.0
        return grad

    def make_wall_constraint(ci, side):
        def con(vec):
            xi = vec[2 * ci]
            yi = vec[2 * ci + 1]
            ri = vec[2 * ci + 2]
            if side == 0:
                return xi - ri
            elif side == 1:
                return w - xi - ri
            elif side == 2:
                return yi - ri
            else:
                return h - yi - ri
        return con

    def make_overlap_constraint(ci, cj):
        def con(vec):
            xi = vec[2 * ci]
            yi = vec[2 * ci + 1]
            ri = vec[2 * ci + 2]
            xj = vec[2 * cj]
            yj = vec[2 * cj + 1]
            rj = vec[2 * cj + 2]
            dx = xi - xj
            dy = yi - yj
            dist = np.hypot(dx, dy)
            if dist < 1e-12:
                return -1e6  # Strong penalty if exactly same location
            return dist - ri - rj
        return con

    # Build constraints
    constraints = []
    for i in range(n):
        for side in range(4):
            constraints.append({'type': 'ineq', 'fun': make_wall_constraint(i, side)})
    for i in range(n):
        for j in range(i + 1, n):
            constraints.append({'type': 'ineq', 'fun': make_overlap_constraint(i, j)})

    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((0, w))
        bounds.append((0, h))
        bounds.append((0, min(w, h) * 0.5)) # Upper bound on radius

    start_time = time.time()
    try:
        res = minimize(
            neg_obj,
            x,
            jac=obj_grad,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 300, 'ftol': 1e-10, 'disp': False, 'maxls': 50}
        )
        if res.x is not None:
            if _validate_solution(res.x, w, h, n):
                score = np.sum(res.x[2::3])
                if score > best_score:
                    best_score = score
                    best_x = res.x.copy()
    except Exception:
        pass

    # Post-process to ensure validity
    if not _validate_solution(best_x, w, h, n):
        # Fallback to x0 if SLSQP failed badly
        if _validate_solution(x, w, h, n):
            best_x = x
            best_score = np.sum(x[2::3])
        else:
            return None, 0.0

    return best_x, best_score


def _sa_pack(w, h, time_limit):
    n_circles = N
    x0 = np.zeros(n_circles * 3)
    # Grid initialization
    # 21 circles. 5x5 grid, take first 21?
    # Or 4x5 + 1?
    # Let's use a dense grid and take 21
    # 21 = 3 * 7.
    # Or just scatter.
    # Grid 5x5 is 25.
    for i in range(n_circles):
        col = i % 5
        row = i // 5
        x0[2 * i] = (col + 0.5) * w / 5.0
        x0[2 * i + 1] = (row + 0.5) * h / 5.0
        x0[2 * i + 2] = 0.05

    # Fix initial overlaps
    # Simple relaxation
    for _ in range(100):
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                dx = x0[2 * i] - x0[2 * j]
                dy = x0[2 * i + 1] - x0[2 * j + 1]
                dist = np.hypot(dx, dy)
                min_dist = x0[2 * i + 2] + x0[2 * j + 2]
                if dist < min_dist:
                    if dist < 1e-9:
                        dx = 0.01
                        dy = 0.01
                        dist = np.hypot(dx, dy)
                    overlap = min_dist - dist
                    # Move apart
                    x0[2 * i] += dx / dist * overlap / 2
                    x0[2 * i + 1] += dy / dist * overlap / 2
                    x0[2 * j] -= dx / dist * overlap / 2
                    x0[2 * j + 1] -= dy / dist * overlap / 2
            # Fix walls
            x0[2 * i] = np.clip(x0[2 * i], x0[2 * i + 2], w - x0[2 * i + 2])
            x0[2 * i + 1] = np.clip(x0[2 * i + 1], x0[2 * i + 2], h - x0[2 * i + 2])

    best_x = x0.copy()
    best_score = np.sum(x0[2::3])

    start_time = time.time()
    T = 0.2
    cooling = 0.999

    while time.time() - start_time < time_limit and T > 1e-6:
        i = np.random.randint(n_circles)
        if np.random.rand() < 0.7:
            # Move
            step = T * 0.2
            dx = np.random.uniform(-step, step)
            dy = np.random.uniform(-step, step)
            new_x = x0[2 * i] + dx
            new_y = x0[2 * i + 1] + dy
            new_r = x0[2 * i + 2]
        else:
            # Resize
            step = T * 0.1
            new_r = x0[2 * i + 2] + np.random.uniform(-step, step)
            new_x = x0[2 * i]
            new_y = x0[2 * i + 1]

        if new_r < 0:
            T *= cooling
            continue

        # Check validity
        valid = True
        if new_x < new_r or new_x > w - new_r or new_y < new_r or new_y > h - new_r:
            valid = False
        else:
            for j in range(n_circles):
                if j == i:
                    continue
                dx = new_x - x0[2 * j]
                dy = new_y - x0[2 * j + 1]
                dist = np.hypot(dx, dy)
                if dist < new_r + x0[2 * j + 2] - 1e-9:
                    valid = False
                    break

        if not valid:
            T *= cooling
            continue

        old_r = x0[2 * i + 2]
        new_r = max(0, new_r)
        delta = new_r - old_r

        if delta > 0 or np.random.rand() < np.exp(-delta / T):
            x0[2 * i] = new_x
            x0[2 * i + 1] = new_y
            x0[2 * i + 2] = new_r
            score = np.sum(x0[2::3])
            if score > best_score:
                best_score = score
                best_x = x0.copy()

        T *= cooling

    return best_x, best_score


def _solve_aspect(w, h, time_budget):
    best_x = None
    best_score = 0.0

    # 1. SA
    if time_budget > 5:
        t_sa = time_budget * 0.6
        x_sa, s_sa = _sa_pack(w, h, t_sa)
        if x_sa is not None and _validate_solution(x_sa, w, h, N):
            best_x = x_sa
            best_score = s_sa

    # 2. SLSQP
    if best_x is not None and time_budget > 10:
        t_slsqp = time_budget * 0.4
        x_opt, s_opt = _slsqp_optimize(best_x, w, h, t_slsqp)
        if x_opt is not None and s_opt > best_score:
            best_x = x_opt
            best_score = s_opt

    return best_x, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    best_result = None
    best_score = 0.0

    # Aspect ratios to try
    # Perimeter = 2(w+h) = 4 => w+h=2
    aspects = [
        (1.0, 1.0),
        (1.2, 0.8),
        (1.3, 0.7),
        (1.4, 0.6),
        (1.5, 0.5),
    ]

    # Shuffle to prioritize diversity?
    np.random.shuffle(aspects)

    for w, h in aspects:
        if time.time() > deadline:
            break

        time_remaining = deadline - time.time()
        if time_remaining < 15:
            continue

        # Allocate time
        t_alloc = min(time_remaining - 5, 60) # Max 60s per aspect

        x_res, score = _solve_aspect(w, h, t_alloc)

        if x_res is not None and score > best_score:
            best_score = score
            # Format result
            positions = np.column_stack([x_res[0::3], x_res[1::3]])
            radii = x_res[2::3]
            best_result = np.column_stack([positions, radii])

    # If no valid result, fallback
    if best_result is None:
        # Simple grid in 1x1
        w, h = 1.0, 1.0
        positions = []
        radii = []
        count = 0
        for i in range(5):
            for j in range(5):
                if count >= N:
                    break
                pos = [(i + 0.5) * w / 5, (j + 0.5) * h / 5]
                r = 0.5 * min(w / 5, h / 5) * 0.9
                positions.append(pos)
                radii.append(r)
                count += 1
        positions = np.array(positions)
        radii = np.array(radii)
        best_result = np.column_stack([positions, radii])

    return best_result
