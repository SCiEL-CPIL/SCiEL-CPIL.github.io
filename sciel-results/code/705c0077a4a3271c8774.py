"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _softplus(x, beta=50.0):
    """Numerically stable softplus."""
    x = np.asarray(x, dtype=float)
    return np.where(x > 20.0 / beta, x, np.log1p(np.exp(beta * x)) / beta)

def _soft_max(v, beta=50.0):
    """Smooth approximation of max(0, v) using softplus."""
    return _softplus(v, beta)

def _objective(x, penalty_weight=100.0):
    """
    Objective: minimize -sum(radii) + penalty * violations.
    x = [x1, y1, ..., xN, yN, r1, ..., rN]
    """
    pos = x[:2 * N].reshape(N, 2)
    radii = np.maximum(x[2 * N:], 1e-6)

    # Boundary violations
    bv = np.maximum(0.0, radii - pos[:, 0]) + np.maximum(0.0, pos[:, 0] - (1.0 - radii)) + \
         np.maximum(0.0, radii - pos[:, 1]) + np.maximum(0.0, pos[:, 1] - (1.0 - radii))
    boundary_pen = np.sum(bv * bv)

    # Overlap violations (vectorized)
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    min_dist_sq = min_dist * min_dist
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)

    viol = np.maximum(0.0, min_dist_sq - dist_sq)
    overlap_pen = np.sum(viol[mask] * viol[mask])

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

def _try_radius_expansion(pos, radii, max_iterations=300):
    """Try to expand each circle's radius while maintaining validity using bisection."""
    pos = pos.copy()
    radii = radii.copy()

    for _ in range(max_iterations):
        improved = False
        for i in range(N):
            x, y = pos[i]
            r = radii[i]

            # Max radius from walls
            max_r_wall = min(x, 1.0 - x, y, 1.0 - y)

            # Max radius from other circles
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

            # Bisection to find exact max valid radius
            lo, hi = r, max_r_circ * 0.9999999
            for _ in range(25):
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

def _slsqp_constraints(pos, radii):
    """Build SLSQP inequality constraints: g(x) >= 0 for feasibility."""
    constraints = []
    n = len(radii)

    # Boundary constraints: x_i - r_i >= 0, 1 - x_i - r_i >= 0, etc.
    for i in range(n):
        # Left
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i] - x[2*N + i]})
        # Right
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i] - x[2*N + i]})
        # Bottom
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i + 1] - x[2*N + i]})
        # Top
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i + 1] - x[2*N + i]})

    # Overlap constraints: dist_ij >= r_i + r_j
    # dist_ij^2 - (r_i + r_j)^2 >= 0
    for i in range(n):
        for j in range(i + 1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: (x[2*i] - x[2*j])**2 + (x[2*i+1] - x[2*j+1])**2 - (x[2*N+i] + x[2*N+j])**2
            })

    return constraints

def _polish(pos, radii, deadline):
    """High-precision polishing with multi-stage local search using SLSQP and expansion."""
    x = np.concatenate([pos.flatten(), radii])

    bounds = []
    for i in range(N):
        bounds.append((0.005, 0.995))
        bounds.append((0.005, 0.995))
    for i in range(N):
        bounds.append((0.005, 0.5))

    # Stage 1: SLSQP with hard constraints
    if time.time() < deadline - 20:
        try:
            pos_tmp = x[:2*N].reshape(N, 2)
            radii_tmp = np.maximum(x[2*N:], 1e-5)
            pos_tmp, radii_tmp = _post_process(pos_tmp, radii_tmp)
            x_hard = np.concatenate([pos_tmp.flatten(), radii_tmp])
            cons = _slsqp_constraints(pos_tmp, radii_tmp)
            result = minimize(lambda v: -np.sum(v[2*N:]), x_hard, method='SLSQP', bounds=bounds,
                              constraints=cons,
                              options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12})
            if result.success or result.fun < -np.sum(x_hard[2*N:]):
                x = result.x
        except Exception:
            pass

    pos_out = x[:2*N].reshape(N, 2)
    radii_out = np.maximum(x[2*N:], 1e-5)
    pos_out, radii_out = _post_process(pos_out, radii_out)

    # Stage 2: Aggressive radius expansion
    if time.time() < deadline - 10:
        pos_out, radii_out = _try_radius_expansion(pos_out, radii_out, max_iterations=300)

    # Stage 3: Another SLSQP pass after expansion
    if time.time() < deadline - 5:
        try:
            x_hard = np.concatenate([pos_out.flatten(), radii_out])
            cons = _slsqp_constraints(pos_out, radii_out)
            result = minimize(lambda v: -np.sum(v[2*N:]), x_hard, method='SLSQP', bounds=bounds,
                              constraints=cons,
                              options={'maxiter': 1500, 'ftol': 1e-16, 'gtol': 1e-14})
            if result.fun < -np.sum(x_hard[2*N:]):
                pos_out = result.x[:2*N].reshape(N, 2)
                radii_out = np.maximum(result.x[2*N:], 1e-5)
                pos_out, radii_out = _post_process(pos_out, radii_out)
        except Exception:
            pass

    # Stage 4: Final radius expansion
    if time.time() < deadline - 1:
        pos_out, radii_out = _try_radius_expansion(pos_out, radii_out, max_iterations=200)

    return pos_out, radii_out

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    # Generate multiple initializations
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
        if time.time() > deadline - 30:
            break

        x0 = np.concatenate([pos_init.flatten(), rad_init])
        current_x = x0.copy()

        # Stage 1: SLSQP with hard constraints (Primary optimizer)
        if time.time() < deadline - 15:
            try:
                pos_tmp = current_x[:2*N].reshape(N, 2)
                radii_tmp = np.maximum(current_x[2*N:], 1e-4)
                pos_tmp, radii_tmp = _post_process(pos_tmp, radii_tmp)
                x_hard = np.concatenate([pos_tmp.flatten(), radii_tmp])
                cons = _slsqp_constraints(pos_tmp, radii_tmp)
                result = minimize(lambda v: -np.sum(v[2*N:]), x_hard, method='SLSQP', bounds=bounds,
                                  constraints=cons,
                                  options={'maxiter': 1000, 'ftol': 1e-13, 'gtol': 1e-10})
                if result.success or result.fun < -np.sum(x_hard[2*N:]):
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

        # Stage 2: Radius Expansion
        if time.time() < deadline - 10:
            pos_exp, rad_exp = _try_radius_expansion(current_x[:2*N].reshape(N, 2), np.maximum(current_x[2*N:], 1e-4), max_iterations=100)
            score = np.sum(rad_exp)
            if _is_valid(pos_exp, rad_exp) and score > best_score:
                best_score = score
                best_pos = pos_exp.copy()
                best_radii = rad_exp.copy()
                current_x = np.concatenate([pos_exp.flatten(), rad_exp])

    # Final polish with the best found so far
    if time.time() < deadline - 3:
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
    if time.time() < deadline - 1:
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
