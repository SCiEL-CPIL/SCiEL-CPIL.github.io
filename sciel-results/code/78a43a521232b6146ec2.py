"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize, linprog

N = 26

# EVOLVE-BLOCK-START

def _softplus(x, beta=50.0):
    """Numerically stable softplus."""
    x = np.asarray(x, dtype=float)
    return np.where(x > 20.0 / beta, x, np.log1p(np.exp(beta * x)) / beta)

def _soft_max(v, beta=50.0):
    """Smooth approximation of max(0, v) using softplus."""
    return _softplus(v, beta)

def _soft_max_lse(v, beta=50.0):
    """Log-sum-exp based smooth max(0, v)."""
    v = np.asarray(v, dtype=float)
    bv = beta * v
    result = np.maximum(bv, 0.0) + np.log1p(np.exp(-np.abs(bv)))
    return result / beta

def _distance_field_smooth(x, penalty_weight=5000.0, beta=50.0):
    """
    Smooth distance-field objective using softplus for the max(0, -d) terms.
    """
    pos = x[:2 * N].reshape(N, 2)
    radii = np.maximum(x[2 * N:], 1e-6)

    d_left = pos[:, 0] - radii
    d_right = (1.0 - pos[:, 0]) - radii
    d_bottom = pos[:, 1] - radii
    d_top = (1.0 - pos[:, 1]) - radii

    boundary_pen = np.sum(_soft_max(-d_left, beta)) + np.sum(_soft_max(-d_right, beta)) + \
                   np.sum(_soft_max(-d_bottom, beta)) + np.sum(_soft_max(-d_top, beta))

    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist = np.sqrt(dx * dx + dy * dy + 1e-12)
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    signed_dist = dist - min_dist
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)

    overlap_pen = np.sum(_soft_max(-signed_dist[mask], beta))

    return -np.sum(radii) + penalty_weight * (boundary_pen + overlap_pen)

def _objective_smooth_lse(x, penalty_weight=20000.0, beta=80.0):
    """
    Objective with log-sum-exp smooth penalties (tighter approximation).
    """
    pos = x[:2 * N].reshape(N, 2)
    radii = np.maximum(x[2 * N:], 1e-6)

    left_v = radii - pos[:, 0]
    right_v = pos[:, 0] - (1.0 - radii)
    bottom_v = radii - pos[:, 1]
    top_v = pos[:, 1] - (1.0 - radii)

    boundary_pen = np.sum(_soft_max_lse(left_v, beta)) + np.sum(_soft_max_lse(right_v, beta)) + \
                   np.sum(_soft_max_lse(bottom_v, beta)) + np.sum(_soft_max_lse(top_v, beta))

    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    min_dist_sq = min_dist * min_dist
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)

    viol_raw = min_dist_sq - dist_sq
    overlap_pen = np.sum(_soft_max_lse(viol_raw[mask], beta))

    return -np.sum(radii) + penalty_weight * (boundary_pen + overlap_pen)

def _epigraph_objective(x, penalty_weight=10000.0, beta=60.0):
    """
    Epigraph-style smooth objective: minimize t + penalty * violations,
    where t >= sum(radii) is handled via smooth max.
    """
    pos = x[:2 * N].reshape(N, 2)
    radii = np.maximum(x[2 * N:], 1e-6)

    # Boundary violations (smooth)
    left_v = radii - pos[:, 0]
    right_v = pos[:, 0] - (1.0 - radii)
    bottom_v = radii - pos[:, 1]
    top_v = pos[:, 1] - (1.0 - radii)

    boundary_pen = np.sum(_soft_max(left_v, beta)) + np.sum(_soft_max(right_v, beta)) + \
                   np.sum(_soft_max(bottom_v, beta)) + np.sum(_soft_max(top_v, beta))

    # Overlap violations (smooth)
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist = np.sqrt(dx * dx + dy * dy + 1e-12)
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    signed_dist = dist - min_dist
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)

    overlap_pen = np.sum(_soft_max(-signed_dist[mask], beta))

    return -np.sum(radii) + penalty_weight * (boundary_pen + overlap_pen)

def _is_valid(pos, radii, tol=1e-7):
    """Strict validity check."""
    n = len(radii)
    if np.any(pos[:, 0] - radii < -tol) or np.any(pos[:, 0] + radii > 1.0 + tol):
        return False
    if np.any(pos[:, 1] - radii < -tol) or np.any(pos[:, 1] + radii > 1.0 + tol):
        return False

    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)

    if np.any(dist_sq[mask] < (min_dist[mask] - tol) ** 2):
        return False
    return True

def _post_process(pos, radii):
    """Shrink radii to ensure strict validity."""
    pos = pos.copy()
    radii = radii.copy()

    for _ in range(300):
        valid = True

        for i in range(N):
            r = radii[i]
            x, y = pos[i]
            max_r = min(x, 1.0 - x, y, 1.0 - y)
            if r > max_r:
                radii[i] = max(max_r * 0.99999, 1e-5)
                valid = False

        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r:
                    k = dist / (sum_r + 1e-12) * 0.99999
                    if k < 1.0:
                        radii[i] = max(radii[i] * k, 1e-5)
                        radii[j] = max(radii[j] * k, 1e-5)
                        valid = False

        if valid:
            break

    return pos, radii

def _lp_radius_optimization(pos, radii_init, deadline, max_iter=50):
    """
    LP-based radius optimization: given fixed positions, solve for maximum radii
    subject to non-overlap and boundary constraints.
    
    Variables: r_0, ..., r_{N-1}
    Maximize: sum(r_i)
    Subject to:
      r_i <= min(x_i, 1-x_i, y_i, 1-y_i)  for each i  (boundary)
      r_i + r_j <= dist(i,j)  for each pair (i,j)  (non-overlap)
      r_i >= 0
    """
    pos = pos.copy()
    radii = radii_init.copy()
    
    # Precompute distances and wall limits
    wall_limits = np.minimum(np.minimum(pos[:, 0], 1.0 - pos[:, 0]),
                             np.minimum(pos[:, 1], 1.0 - pos[:, 1]))
    
    # Build constraint matrix for linprog (minimize -sum(r) subject to A_ub @ r <= b_ub)
    # Boundary: r_i <= wall_limits[i]  =>  -r_i >= -wall_limits[i]  =>  r_i <= wall_limits[i]
    # Overlap: r_i + r_j <= dist(i,j)
    
    n = N
    # A_ub rows: for each constraint, coefficients of r
    # Boundary constraints: e_i (unit vector) <= wall_limits[i]
    # Overlap constraints: e_i + e_j <= dist(i,j)
    
    constraints = []
    bounds_list = []
    
    # Build A_ub and b_ub
    # Each row is one constraint: A_ub[k] @ r <= b_ub[k]
    A_rows = []
    b_vals = []
    
    # Boundary constraints
    for i in range(n):
        row = np.zeros(n)
        row[i] = 1.0
        A_rows.append(row)
        b_vals.append(wall_limits[i])
    
    # Overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_rows.append(row)
            b_vals.append(dist)
    
    A_ub = np.array(A_rows)
    b_ub = np.array(b_vals)
    
    # Objective: minimize -sum(r)
    c_obj = -np.ones(n)
    
    # Bounds: r_i >= 0
    bounds = [(0.0, None) for _ in range(n)]
    
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    for iteration in range(max_iter):
        if time.time() > deadline - 5:
            break
        
        try:
            result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            
            if result.success:
                new_radii = result.x
                # Clip to ensure non-negative
                new_radii = np.maximum(new_radii, 1e-6)
                new_score = np.sum(new_radii)
                
                if new_score > best_score + 1e-10:
                    best_score = new_score
                    best_radii = new_radii.copy()
                else:
                    # No improvement, try to perturb positions slightly
                    break
            else:
                break
        except Exception:
            break
        
        # If we got a good solution, do a small position perturbation to escape
        # local optima and re-solve
        if iteration > 0 and iteration % 5 == 0:
            # Small random perturbation of positions
            rng = np.random.RandomState(iteration * 100 + 42)
            pos_pert = pos + rng.uniform(-0.001, 0.001, (n, 2))
            pos_pert = np.clip(pos_pert, 0.01, 0.99)
            # Rebuild constraints with perturbed positions
            wall_limits_p = np.minimum(np.minimum(pos_pert[:, 0], 1.0 - pos_pert[:, 0]),
                                       np.minimum(pos_pert[:, 1], 1.0 - pos_pert[:, 1]))
            A_rows_p = []
            b_vals_p = []
            for i in range(n):
                row = np.zeros(n)
                row[i] = 1.0
                A_rows_p.append(row)
                b_vals_p.append(wall_limits_p[i])
            for i in range(n):
                for j in range(i + 1, n):
                    dx = pos_pert[i, 0] - pos_pert[j, 0]
                    dy = pos_pert[i, 1] - pos_pert[j, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    row = np.zeros(n)
                    row[i] = 1.0
                    row[j] = 1.0
                    A_rows_p.append(row)
                    b_vals_p.append(dist)
            A_ub_p = np.array(A_rows_p)
            b_ub_p = np.array(b_vals_p)
            try:
                result_p = linprog(c_obj, A_ub=A_ub_p, b_ub=b_ub_p, bounds=bounds, method='highs')
                if result_p.success:
                    new_radii_p = np.maximum(result_p.x, 1e-6)
                    new_score_p = np.sum(new_radii_p)
                    if new_score_p > best_score + 1e-10:
                        best_score = new_score_p
                        best_radii = new_radii_p.copy()
                        pos = pos_pert  # Accept perturbed position
            except Exception:
                pass
    
    # Final post-processing to ensure validity
    pos, best_radii = _post_process(pos, best_radii)
    return pos, best_radii

def _initial_best_fit():
    """Greedy best-fit initialization."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)

    pos[0] = [0.5, 0.5]
    radii[0] = 0.25

    for i in range(1, N):
        best_x, best_y, best_r = 0.5, 0.5, 0.0
        found = False
        rng = np.random.RandomState(i * 1000 + 42)

        for _ in range(8000):
            x = rng.uniform(0.02, 0.98)
            y = rng.uniform(0.02, 0.98)

            max_r_wall = min(x, 1.0 - x, y, 1.0 - y)
            max_r_circ = max_r_wall
            for j in range(i):
                dx = x - pos[j, 0]
                dy = y - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                possible_r = dist - radii[j]
                if possible_r < max_r_circ:
                    max_r_circ = possible_r

            if max_r_circ > best_r and max_r_circ > 0.003:
                best_r = max_r_circ
                best_x = x
                best_y = y
                found = True

        if not found or best_r < 0.003:
            best_x, best_y = 0.5, 0.5
            best_r = 0.003

        pos[i] = [best_x, best_y]
        radii[i] = best_r * 0.97

    return pos, radii

def _initial_hexagonal():
    """Hexagonal lattice initialization."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)

    r = 0.11
    dx = r * np.sqrt(3)
    dy = r * 1.5

    k = 0
    for row in range(6):
        offset = dx / 2 if row % 2 == 1 else 0
        start_x = r + offset
        y = row * dy + r

        for col in range(5):
            if k >= N:
                break
            x = start_x + col * dx

            if x + r <= 1.0 and x - r >= 0.0 and y + r <= 1.0 and y - r >= 0.0:
                pos[k] = [x, y]
                radii[k] = r
                k += 1

        if k >= N:
            break

    while k < N:
        rng = np.random.RandomState(k * 7 + 13)
        for _ in range(200):
            x = rng.uniform(0.1, 0.9)
            y = rng.uniform(0.1, 0.9)
            min_dist = min(x, 1 - x, y, 1 - y)
            for j in range(k):
                d = np.sqrt((x - pos[j, 0]) ** 2 + (y - pos[j, 1]) ** 2)
                min_dist = min(min_dist, d - radii[j])
            if min_dist > 0.02:
                pos[k] = [x, y]
                radii[k] = min_dist * 0.95
                k += 1
                break
        if k < N:
            pos[k] = [0.5, 0.5]
            radii[k] = 0.01
            k += 1

    return pos, radii

def _initial_corner_biased(seed=7):
    """Corner-biased initialization."""
    rng = np.random.RandomState(seed)
    pos = np.zeros((N, 2))
    radii = np.zeros(N)

    corners = [(0.15, 0.15), (0.85, 0.15), (0.15, 0.85), (0.85, 0.85)]
    for i, (cx, cy) in enumerate(corners):
        pos[i] = [cx + rng.uniform(-0.03, 0.03), cy + rng.uniform(-0.03, 0.03)]
        radii[i] = 0.18

    for i in range(4, N):
        if rng.random() < 0.4:
            edge = rng.randint(4)
            t = rng.uniform(0.1, 0.9)
            if edge == 0:
                x, y = t, 0.08
            elif edge == 1:
                x, y = t, 0.92
            elif edge == 2:
                x, y = 0.08, t
            else:
                x, y = 0.92, t
        else:
            x, y = rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)

        max_r = min(x, 1.0 - x, y, 1.0 - y)
        for j in range(i):
            dx = x - pos[j, 0]
            dy = y - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            max_r = min(max_r, dist - radii[j])

        if max_r < 0.01:
            for attempt in range(300):
                x2 = rng.uniform(0.05, 0.95)
                y2 = rng.uniform(0.05, 0.95)
                max_r2 = min(x2, 1.0 - x2, y2, 1.0 - y2)
                for j in range(i):
                    dx = x2 - pos[j, 0]
                    dy = y2 - pos[j, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    max_r2 = min(max_r2, dist - radii[j])
                if max_r2 > max_r:
                    max_r = max_r2
                    x, y = x2, y2

        pos[i] = [x, y]
        radii[i] = max(max_r, 0.01) * 0.95

    return pos, radii

def _initial_random(seed):
    """Random initialization with post-processing."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, (N, 2))
    radii = rng.uniform(0.04, 0.12, N)
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _try_radius_expansion(pos, radii, max_iterations=500):
    """Try to expand each circle's radius while maintaining validity using bisection."""
    pos = pos.copy()
    radii = radii.copy()

    for _ in range(max_iterations):
        improved = False
        for i in range(N):
            x, y = pos[i]
            r = radii[i]

            max_r_wall = min(x, 1.0 - x, y, 1.0 - y)

            max_r_circ = max_r_wall
            for j in range(N):
                if i == j:
                    continue
                dx = x - pos[j, 0]
                dy = y - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                max_r_circ = min(max_r_circ, dist - radii[j])

            if max_r_circ <= r * 1.000001:
                continue

            lo, hi = r, max_r_circ * 0.9999999
            for _ in range(30):
                mid = (lo + hi) / 2.0
                radii[i] = mid
                if _is_valid(pos, radii):
                    lo = mid
                else:
                    hi = mid
            radii[i] = lo
            if lo > r * 1.000001:
                improved = True

        if not improved:
            break

    return pos, radii

def _coordinate_descent_positions(pos, radii, deadline, max_rounds=50):
    """
    Coordinate descent: fix radii, optimize each position one at a time
    using L-BFGS-B on a 2D subproblem.
    """
    pos = pos.copy()
    radii = radii.copy()

    for rnd in range(max_rounds):
        if time.time() > deadline - 10:
            break
        improved = False
        for i in range(N):
            if time.time() > deadline - 10:
                break

            def obj_pos_i(xi, i=i):
                x, y = xi
                bv = np.maximum(0.0, radii[i] - x) + np.maximum(0.0, x - (1.0 - radii[i])) + \
                     np.maximum(0.0, radii[i] - y) + np.maximum(0.0, y - (1.0 - radii[i]))
                boundary_pen = bv * bv

                overlap_pen = 0.0
                for j in range(N):
                    if i == j:
                        continue
                    dx = x - pos[j, 0]
                    dy = y - pos[j, 1]
                    dist_sq = dx * dx + dy * dy
                    min_dist = radii[i] + radii[j]
                    viol = np.maximum(0.0, min_dist * min_dist - dist_sq)
                    overlap_pen += viol * viol

                return 1000.0 * (boundary_pen + overlap_pen)

            x0 = np.array([pos[i, 0], pos[i, 1]])
            bounds_i = [(0.005, 0.995), (0.005, 0.995)]

            try:
                res = minimize(obj_pos_i, x0, method='L-BFGS-B', bounds=bounds_i,
                               options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12})
                if res.fun < obj_pos_i(x0) - 1e-12:
                    pos[i] = res.x
                    improved = True
            except Exception:
                pass

        if not improved:
            break

    return pos, radii

def _alternating_optimization(pos, radii, deadline, max_iter=20):
    """
    Alternate between:
    1. Fix positions, expand radii (bisection)
    2. Fix radii, optimize positions (coordinate descent)
    """
    pos = pos.copy()
    radii = radii.copy()

    for it in range(max_iter):
        if time.time() > deadline - 15:
            break

        pos, radii = _try_radius_expansion(pos, radii, max_iterations=200)
        pos, radii = _post_process(pos, radii)

        pos, radii = _coordinate_descent_positions(pos, radii, deadline, max_rounds=30)
        pos, radii = _post_process(pos, radii)

    return pos, radii

def _slsqp_constraints():
    """Build SLSQP inequality constraints: g(x) >= 0 for feasibility."""
    constraints = []
    n = N

    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i] - x[2*N + i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i] - x[2*N + i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i + 1] - x[2*N + i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i + 1] - x[2*N + i]})

    for i in range(n):
        for j in range(i + 1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: (x[2*i] - x[2*j])**2 + (x[2*i+1] - x[2*j+1])**2 - (x[2*N+i] + x[2*N+j])**2
            })

    return constraints

def _force_directed_relax(pos, radii, deadline, max_iter=2000, lr=0.001):
    """
    Force-directed relaxation to untangle configurations.
    """
    pos = pos.copy()
    radii = radii.copy()
    
    for it in range(max_iter):
        if time.time() > deadline - 5:
            break
        
        forces = np.zeros_like(pos)
        
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy + 1e-12)
                min_dist = radii[i] + radii[j]
                
                if dist < min_dist:
                    overlap = min_dist - dist
                    force_mag = 10.0 * overlap + 1.0
                    fx = force_mag * dx / dist
                    fy = force_mag * dy / dist
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
        
        for i in range(N):
            if pos[i, 0] < radii[i]:
                forces[i, 0] += 5.0 * (radii[i] - pos[i, 0])
            if pos[i, 0] > 1.0 - radii[i]:
                forces[i, 0] -= 5.0 * (pos[i, 0] - (1.0 - radii[i]))
            if pos[i, 1] < radii[i]:
                forces[i, 1] += 5.0 * (radii[i] - pos[i, 1])
            if pos[i, 1] > 1.0 - radii[i]:
                forces[i, 1] -= 5.0 * (pos[i, 1] - (1.0 - radii[i]))
        
        step = lr * np.exp(-it / 500.0)
        pos += step * forces
        
        for i in range(N):
            pos[i, 0] = np.clip(pos[i, 0], 0.001, 0.999)
            pos[i, 1] = np.clip(pos[i, 1], 0.001, 0.999)
        
        max_overlap = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                overlap = radii[i] + radii[j] - dist
                max_overlap = max(max_overlap, overlap)
        
        if max_overlap < 1e-6:
            break
    
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _polish(pos, radii, deadline):
    """High-precision polishing with multi-stage local search including LP radius optimization."""
    x = np.concatenate([pos.flatten(), radii])

    bounds = []
    for i in range(N):
        bounds.append((0.005, 0.995))
        bounds.append((0.005, 0.995))
    for i in range(N):
        bounds.append((0.005, 0.5))

    # Stage 1: Force-directed relaxation to untangle
    if time.time() < deadline - 45:
        pos_tmp = x[:2*N].reshape(N, 2)
        radii_tmp = np.maximum(x[2*N:], 1e-5)
        pos_fd, radii_fd = _force_directed_relax(pos_tmp, radii_tmp, deadline, max_iter=1000)
        if np.sum(radii_fd) > np.sum(x[2*N:]):
            x = np.concatenate([pos_fd.flatten(), radii_fd])

    # Stage 2: L-BFGS-B with distance-field smooth objective
    if time.time() < deadline - 40:
        try:
            result = minimize(_distance_field_smooth, x, method='L-BFGS-B', bounds=bounds,
                              args=(5000.0, 50.0),
                              options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12})
            if result.fun < _distance_field_smooth(x, 5000.0, 50.0) - 1e-13:
                x = result.x
        except Exception:
            pass

    # Stage 3: SLSQP with hard constraints
    if time.time() < deadline - 35:
        try:
            pos_tmp = x[:2*N].reshape(N, 2)
            radii_tmp = np.maximum(x[2*N:], 1e-5)
            pos_tmp, radii_tmp = _post_process(pos_tmp, radii_tmp)
            x_hard = np.concatenate([pos_tmp.flatten(), radii_tmp])
            cons = _slsqp_constraints()
            result = minimize(lambda v: -np.sum(v[2*N:]), x_hard, method='SLSQP', bounds=bounds,
                              constraints=cons,
                              options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12})
            if result.success or result.fun < -np.sum(x_hard[2*N:]):
                x = result.x
        except Exception:
            pass

    # Stage 4: Alternating optimization
    if time.time() < deadline - 30:
        pos_tmp = x[:2*N].reshape(N, 2)
        radii_tmp = np.maximum(x[2*N:], 1e-5)
        pos_tmp, radii_tmp = _post_process(pos_tmp, radii_tmp)
        pos_alt, radii_alt = _alternating_optimization(pos_tmp, radii_tmp, deadline, max_iter=10)
        if np.sum(radii_alt) > np.sum(x[2*N:]):
            x = np.concatenate([pos_alt.flatten(), radii_alt])

    # Stage 5: L-BFGS-B with LSE smooth at higher penalty
    if time.time() < deadline - 25:
        try:
            result = minimize(_objective_smooth_lse, x, method='L-BFGS-B', bounds=bounds,
                              args=(20000.0, 80.0),
                              options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-13})
            if result.fun < _objective_smooth_lse(x, 20000.0, 80.0) - 1e-13:
                x = result.x
        except Exception:
            pass

    pos_out = x[:2*N].reshape(N, 2)
    radii_out = np.maximum(x[2*N:], 1e-5)
    pos_out, radii_out = _post_process(pos_out, radii_out)

    # Stage 6: LP-based radius optimization (key improvement)
    if time.time() < deadline - 15:
        pos_lp, radii_lp = _lp_radius_optimization(pos_out, radii_out, deadline, max_iter=30)
        if np.sum(radii_lp) > np.sum(radii_out) + 1e-10:
            pos_out = pos_lp
            radii_out = radii_lp

    # Stage 7: SLSQP with hard constraints (second pass)
    if time.time() < deadline - 10:
        try:
            x_hard = np.concatenate([pos_out.flatten(), radii_out])
            cons = _slsqp_constraints()
            result = minimize(lambda v: -np.sum(v[2*N:]), x_hard, method='SLSQP', bounds=bounds,
                              constraints=cons,
                              options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14})
            if result.fun < -np.sum(x_hard[2*N:]):
                x = result.x
                pos_out = x[:2*N].reshape(N, 2)
                radii_out = np.maximum(x[2*N:], 1e-5)
                pos_out, radii_out = _post_process(pos_out, radii_out)
        except Exception:
            pass

    # Stage 8: Aggressive radius expansion
    if time.time() < deadline - 5:
        pos_out, radii_out = _try_radius_expansion(pos_out, radii_out, max_iterations=500)

    # Stage 9: Final LP-based radius optimization
    if time.time() < deadline - 2:
        pos_lp2, radii_lp2 = _lp_radius_optimization(pos_out, radii_out, deadline, max_iter=20)
        if np.sum(radii_lp2) > np.sum(radii_out) + 1e-10:
            pos_out = pos_lp2
            radii_out = radii_lp2

    # Stage 10: Final aggressive radius expansion
    if time.time() < deadline - 0.5:
        pos_out, radii_out = _try_radius_expansion(pos_out, radii_out, max_iterations=300)

    return pos_out, radii_out

def construct():
    start_time = time.time()
    deadline = start_time + 146.0  # 4s safety margin

    inits = []
    pos_bf, rad_bf = _initial_best_fit()
    inits.append((pos_bf, rad_bf))

    pos_hex, rad_hex = _initial_hexagonal()
    inits.append((pos_hex, rad_hex))

    pos_cb, rad_cb = _initial_corner_biased(seed=7)
    inits.append((pos_cb, rad_cb))

    pos_cb2, rad_cb2 = _initial_corner_biased(seed=13)
    inits.append((pos_cb2, rad_cb2))

    for i in range(2):
        pos_r, rad_r = _initial_random(100 + i)
        inits.append((pos_r, rad_r))

    best_pos = pos_bf.copy()
    best_radii = rad_bf.copy()
    best_score = np.sum(best_radii)

    bounds = []
    for i in range(N):
        bounds.append((0.01, 0.99))
        bounds.append((0.01, 0.99))
    for i in range(N):
        bounds.append((0.01, 0.5))

    for idx, (pos_init, rad_init) in enumerate(inits):
        if time.time() > deadline - 35:
            break

        x0 = np.concatenate([pos_init.flatten(), rad_init])
        current_x = x0.copy()

        # Stage 1: L-BFGS-B with distance-field smooth objective (progressive penalty)
        for p_w in [100.0, 1000.0, 5000.0]:
            if time.time() > deadline - 20:
                break
            try:
                result = minimize(_distance_field_smooth, current_x, method='L-BFGS-B', bounds=bounds,
                                  args=(p_w, 40.0),
                                  options={'maxiter': 800, 'ftol': 1e-12, 'gtol': 1e-8})
                if result.fun < _distance_field_smooth(current_x, p_w, 40.0) - 1e-10:
                    current_x = result.x
            except Exception:
                pass

            pos_opt = current_x[:2*N].reshape(N, 2)
            radii_opt = np.maximum(current_x[2*N:], 1e-4)
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
            score = np.sum(radii_pp)
            if _is_valid(pos_pp, radii_pp) and score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()

        # Stage 2: Alternating optimization
        if time.time() < deadline - 18:
            pos_tmp = current_x[:2*N].reshape(N, 2)
            radii_tmp = np.maximum(current_x[2*N:], 1e-4)
            pos_tmp, radii_tmp = _post_process(pos_tmp, radii_tmp)
            pos_alt, radii_alt = _alternating_optimization(pos_tmp, radii_tmp, deadline, max_iter=8)
            score_alt = np.sum(radii_alt)
            if _is_valid(pos_alt, radii_alt) and score_alt > best_score:
                best_score = score_alt
                best_pos = pos_alt.copy()
                best_radii = radii_alt.copy()

        # Stage 3: SLSQP with hard constraints
        if time.time() < deadline - 15:
            try:
                pos_tmp = current_x[:2*N].reshape(N, 2)
                radii_tmp = np.maximum(current_x[2*N:], 1e-4)
                pos_tmp, radii_tmp = _post_process(pos_tmp, radii_tmp)
                x_hard = np.concatenate([pos_tmp.flatten(), radii_tmp])
                cons = _slsqp_constraints()
                result = minimize(lambda v: -np.sum(v[2*N:]), x_hard, method='SLSQP', bounds=bounds,
                                  constraints=cons,
                                  options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-11})
                if result.fun < -np.sum(x_hard[2*N:]):
                    current_x = result.x
                    pos_opt = current_x[:2*N].reshape(N, 2)
                    radii_opt = np.maximum(current_x[2*N:], 1e-4)
                    pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
                    score = np.sum(radii_pp)
                    if _is_valid(pos_pp, radii_pp) and score > best_score:
                        best_score = score
                        best_pos = pos_pp.copy()
                        best_radii = radii_pp.copy()
            except Exception:
                pass

        # Stage 4: LP-based radius optimization on this branch
        if time.time() < deadline - 12:
            pos_lp, radii_lp = _lp_radius_optimization(best_pos, best_radii, deadline, max_iter=15)
            score_lp = np.sum(radii_lp)
            if _is_valid(pos_lp, radii_lp) and score_lp > best_score:
                best_score = score_lp
                best_pos = pos_lp.copy()
                best_radii = radii_lp.copy()

    # Final polish with the best found so far
    if time.time() < deadline - 5:
        try:
            pos_pol, rad_pol = _polish(best_pos, best_radii, deadline)
            pos_pp, radii_pp = _post_process(pos_pol, rad_pol)
            score = np.sum(radii_pp)
            if _is_valid(pos_pp, radii_pp) and score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
        except Exception:
            pass

    # Extra polish pass if time allows
    if time.time() < deadline - 2:
        try:
            pos_pol2, rad_pol2 = _polish(best_pos, best_radii, deadline)
            pos_pp2, radii_pp2 = _post_process(pos_pol2, rad_pol2)
            score2 = np.sum(radii_pp2)
            if _is_valid(pos_pp2, radii_pp2) and score2 > best_score:
                best_score = score2
                best_pos = pos_pp2.copy()
                best_radii = radii_pp2.copy()
        except Exception:
            pass

    # Final validation
    best_pos, best_radii = _post_process(best_pos, best_radii)

    if not _is_valid(best_pos, best_radii):
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        k = 0
        for i in range(5):
            for j in range(5):
                if k < N - 1:
                    pos[k] = [(i + 0.5) / 5.0, (j + 0.5) / 5.0]
                    radii[k] = 0.095
                    k += 1
        pos[k] = [0.5, 0.02]
        radii[k] = 0.01
        best_pos, best_radii = pos, radii
        best_pos, best_radii = _post_process(best_pos, best_radii)

    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
