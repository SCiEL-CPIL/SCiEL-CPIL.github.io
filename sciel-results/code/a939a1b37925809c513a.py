<<<<<<< SEARCH
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
=======
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


def _hexagonal_init(w, h, n_circles):
    """Initialize circles in a hexagonal close-packed pattern."""
    x0 = np.zeros(n_circles * 3)
    # Determine grid dimensions
    cols = int(np.ceil(np.sqrt(n_circles * 1.1547)))  # hex packing factor
    rows = int(np.ceil(n_circles / cols))
    
    # Hexagonal spacing
    dx = w / (cols + 0.5)
    dy = h / (rows + 0.2887)  # hex vertical spacing is sqrt(3)/2 * dx
    
    idx = 0
    for row in range(rows):
        for col in range(cols):
            if idx >= n_circles:
                break
            x = (col + 0.5) * dx
            y = (row + 0.5) * dy
            if row % 2 == 1:
                x += dx / 2
            # Clamp to bounds
            r = 0.45 * min(dx, dy * 1.1547)
            x = np.clip(x, r, w - r)
            y = np.clip(y, r, h - r)
            x0[2 * idx] = x
            x0[2 * idx + 1] = y
            x0[2 * idx + 2] = r
            idx += 1
    return x0


def _grid_init(w, h, n_circles):
    """Initialize circles in a rectangular grid."""
    x0 = np.zeros(n_circles * 3)
    cols = int(np.ceil(np.sqrt(n_circles)))
    rows = int(np.ceil(n_circles / cols))
    
    dx = w / cols
    dy = h / rows
    
    for i in range(n_circles):
        col = i % cols
        row = i // cols
        x = (col + 0.5) * dx
        y = (row + 0.5) * dy
        r = 0.48 * min(dx, dy)
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)
        x0[2 * i] = x
        x0[2 * i + 1] = y
        x0[2 * i + 2] = r
    return x0


def _penalty_objective(vec, w, h, penalty_weight, beta):
    """
    Compute objective with smooth penalty for constraints.
    vec: [x0, y0, r0, x1, y1, r1, ...]
    Returns: (objective_value, gradient)
    """
    n = N
    xs = vec[0::3]
    ys = vec[1::3]
    rs = vec[2::3]
    
    # Base objective: maximize sum of radii
    obj = -np.sum(rs)
    grad = np.zeros_like(vec)
    grad[2::3] = -1.0
    
    # Wall constraints (smooth)
    # Left: x_i >= r_i  => violation = r_i - x_i
    viol_left = np.maximum(0, rs - xs)
    # Right: x_i + r_i <= w  => violation = x_i + r_i - w
    viol_right = np.maximum(0, xs + rs - w)
    # Bottom: y_i >= r_i
    viol_bottom = np.maximum(0, rs - ys)
    # Top: y_i + r_i <= h
    viol_top = np.maximum(0, ys + rs - h)
    
    # Smooth penalty: use softplus-like function
    def smooth_penalty(v, b):
        return np.where(v > 0, v + np.log(1 + np.exp(-b * v)) / b, np.log(1 + np.exp(b * v)) / b)
    
    # Actually, let's use a simpler quadratic penalty with smooth max
    # penalty = sum(max(0, violation)^2)
    # Gradient of max(0, v)^2 is 2*max(0, v)*sign(v)
    
    wall_pen = (viol_left**2 + viol_right**2 + viol_bottom**2 + viol_top**2)
    obj += penalty_weight * wall_pen
    
    # Gradients for wall penalties
    # d/dx_i of (r_i - x_i)^2 when r_i > x_i: -2(r_i - x_i)
    # d/dx_i of (x_i + r_i - w)^2 when x_i + r_i > w: 2(x_i + r_i - w)
    # d/dr_i of (r_i - x_i)^2 when r_i > x_i: 2(r_i - x_i)
    # d/dr_i of (x_i + r_i - w)^2 when x_i + r_i > w: 2(x_i + r_i - w)
    
    grad[0::3] += penalty_weight * (-2 * viol_left + 2 * viol_right)
    grad[1::3] += penalty_weight * (-2 * viol_bottom + 2 * viol_top)
    grad[2::3] += penalty_weight * (2 * viol_left + 2 * viol_right + 2 * viol_bottom + 2 * viol_top)
    
    # Overlap constraints (vectorized)
    # For each pair (i,j), violation = r_i + r_j - dist(i,j)
    # Use broadcasting for efficiency
    xs_mat = xs[:, np.newaxis]
    ys_mat = ys[:, np.newaxis]
    rs_mat = rs[:, np.newaxis]
    
    dx = xs_mat - xs_mat.T
    dy = ys_mat - ys_mat.T
    dist = np.sqrt(dx**2 + dy**2 + 1e-12)
    
    # Avoid self-interaction
    np.fill_diagonal(dist, np.inf)
    
    sum_r = rs_mat + rs_mat.T
    overlap_viol = np.maximum(0, sum_r - dist)
    
    # Only count upper triangle once
    triu_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    overlap_pen = np.sum(overlap_viol[triu_mask]**2)
    
    obj += penalty_weight * overlap_pen
    
    # Gradients for overlap penalties
    # d/dx_i of (r_i + r_j - dist_ij)^2 when overlap
    # = 2*(r_i + r_j - dist_ij) * (-dx_ij / dist_ij) for x_i
    # = 2*(r_i + r_j - dist_ij) * (dx_ij / dist_ij) for x_j
    
    overlap_viol_full = overlap_viol  # full matrix, but we only use upper triangle
    # Gradient contribution for each pair
    # For x_i: sum over j>i of 2*viol_ij * (-dx_ij/dist_ij) + sum over j<i of 2*viol_ji * (dx_ji/dist_ji)
    # Since viol is symmetric and dx changes sign, we can compute:
    
    # grad_x[i] += 2 * sum_j viol_ij * (-dx_ij / dist_ij) for j != i
    # But viol_ij is zero when no overlap, so we only sum over overlapping pairs
    
    # Let's compute the gradient matrix
    # G[i,j] = 2 * viol_ij * (-dx_ij / dist_ij) for i != j
    # Then grad_x[i] = sum_j G[i,j]
    # G[i,j] = 2 * viol_ij * (-dx_ij / dist_ij)
    # Note: viol_ij = viol_ji, dx_ji = -dx_ij, dist_ji = dist_ij
    # So G[i,j] = -G[j,i]
    
    # Compute G
    with np.errstate(divide='ignore', invalid='ignore'):
        G = 2 * overlap_viol * (-dx / dist)
    np.fill_diagonal(G, 0)
    
    grad[0::3] += penalty_weight * np.sum(G, axis=1)
    grad[1::3] += penalty_weight * np.sum(2 * overlap_viol * (-dy / dist), axis=1)
    
    # For radii: d/dr_i of (r_i + r_j - dist)^2 = 2*(r_i + r_j - dist)
    # grad_r[i] += 2 * sum_j viol_ij
    grad[2::3] += penalty_weight * 2 * np.sum(overlap_viol, axis=1)
    
    return obj, grad


def _lbfgs_optimize(x0, w, h, time_budget):
    """Optimize using L-BFGS-B with penalty method."""
    from scipy.optimize import minimize
    
    n = N
    x = x0.copy()
    best_score = np.sum(x[2::3])
    best_x = x.copy()
    
    # Start with moderate penalty, increase if needed
    penalty_weight = 10.0
    
    start_time = time.time()
    
    # Multiple restarts with different penalty weights
    for pw in [5.0, 10.0, 20.0, 50.0]:
        if time.time() - start_time > time_budget * 0.8:
            break
        
        def obj_func(vec):
            val, _ = _penalty_objective(vec, w, h, pw, 10.0)
            return val
        
        def grad_func(vec):
            _, g = _penalty_objective(vec, w, h, pw, 10.0)
            return g
        
        bounds = [(0, w), (0, h), (0, min(w, h) * 0.5)] * n
        
        try:
            res = minimize(
                obj_func,
                x,
                jac=grad_func,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
            )
            if res.x is not None:
                if _validate_solution(res.x, w, h, n):
                    score = np.sum(res.x[2::3])
                    if score > best_score:
                        best_score = score
                        best_x = res.x.copy()
                        x = res.x.copy()  # Use this as starting point for next penalty
        except Exception:
            pass
    
    if not _validate_solution(best_x, w, h, n):
        if _validate_solution(x0, w, h, n):
            best_x = x0
            best_score = np.sum(x0[2::3])
        else:
            return None, 0.0
    
    return best_x, best_score


def _sa_pack(w, h, time_limit):
    """Simulated annealing for local refinement."""
    n_circles = N
    # Use hexagonal init
    x0 = _hexagonal_init(w, h, n_circles)
    
    # Fix initial overlaps with relaxation
    for _ in range(50):
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
                    x0[2 * i] += dx / dist * overlap / 2
                    x0[2 * i + 1] += dy / dist * overlap / 2
                    x0[2 * j] -= dx / dist * overlap / 2
                    x0[2 * j + 1] -= dy / dist * overlap / 2
            x0[2 * i] = np.clip(x0[2 * i], x0[2 * i + 2], w - x0[2 * i + 2])
            x0[2 * i + 1] = np.clip(x0[2 * i + 1], x0[2 * i + 2], h - x0[2 * i + 2])
    
    best_x = x0.copy()
    best_score = np.sum(x0[2::3])
    
    start_time = time.time()
    T = 0.1
    cooling = 0.9995
    
    while time.time() - start_time < time_limit and T > 1e-5:
        i = np.random.randint(n_circles)
        if np.random.rand() < 0.6:
            # Move
            step = T * 0.15
            dx = np.random.uniform(-step, step)
            dy = np.random.uniform(-step, step)
            new_x = x0[2 * i] + dx
            new_y = x0[2 * i + 1] + dy
            new_r = x0[2 * i + 2]
        else:
            # Resize
            step = T * 0.08
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
    """Solve for given aspect ratio."""
    best_x = None
    best_score = 0.0
    
    # 1. Try L-BFGS-B with penalty from hexagonal init
    x_hex = _hexagonal_init(w, h, N)
    x_opt, s_opt = _lbfgs_optimize(x_hex, w, h, time_budget * 0.5)
    if x_opt is not None and _validate_solution(x_opt, w, h, N):
        best_x = x_opt
        best_score = s_opt
    
    # 2. Try SA if time remains
    if time_budget > 10:
        t_sa = time_budget * 0.3
        x_sa, s_sa = _sa_pack(w, h, t_sa)
        if x_sa is not None and _validate_solution(x_sa, w, h, N):
            if s_sa > best_score:
                best_x = x_sa
                best_score = s_sa
    
    # 3. Final L-BFGS-B polish if time remains and we have a good start
    if best_x is not None and time_budget > 15:
        t_polish = time_budget * 0.2
        x_polish, s_polish = _lbfgs_optimize(best_x, w, h, t_polish)
        if x_polish is not None and _validate_solution(x_polish, w, h, N):
            if s_polish > best_score:
                best_x = x_polish
                best_score = s_polish
    
    return best_x, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    best_result = None
    best_score = 0.0
    
    # Aspect ratios to try (w + h = 2)
    # Try more diverse aspects
    aspects = [
        (1.0, 1.0),      # square
        (1.2, 0.8),
        (1.3, 0.7),
        (1.4, 0.6),
        (1.5, 0.5),
        (1.6, 0.4),
        (0.9, 1.1),
        (0.8, 1.2),
    ]
    
    # Sort by expected promise: square and near-square first
    np.random.shuffle(aspects)
    # But put square first
    aspects.sort(key=lambda x: abs(x[0] - 1.0))
    
    for w, h in aspects:
        if time.time() > deadline:
            break
        
        time_remaining = deadline - time.time()
        if time_remaining < 10:
            continue
        
        # Allocate time based on remaining
        t_alloc = min(time_remaining - 5, 40)  # Max 40s per aspect
        
        x_res, score = _solve_aspect(w, h, t_alloc)
        
        if x_res is not None and score > best_score:
            best_score = score
            positions = np.column_stack([x_res[0::3], x_res[1::3]])
            radii = x_res[2::3]
            best_result = np.column_stack([positions, radii])
    
    # If no valid result, fallback to simple grid
    if best_result is None:
        w, h = 1.0, 1.0
        x0 = _grid_init(w, h, N)
        positions = np.column_stack([x0[0::3], x0[1::3]])
        radii = x0[2::3]
        best_result = np.column_stack([positions, radii])
    
    return best_result
>>>>>>> REPLACE
