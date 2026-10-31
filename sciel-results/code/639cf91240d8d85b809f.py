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

def _objective(x, penalty_weight=100.0, beta=50.0):
    """
    Objective: minimize -sum(radii) + penalty * smooth_violations.
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

def _objective_smooth(x, penalty_weight=100.0, beta=30.0):
    """
    Objective with softplus-based smooth penalties.
    """
    pos = x[:2 * N].reshape(N, 2)
    radii = np.maximum(x[2 * N:], 1e-6)

    # Smooth boundary violations
    left_v = radii - pos[:, 0]
    right_v = pos[:, 0] - (1.0 - radii)
    bottom_v = radii - pos[:, 1]
    top_v = pos[:, 1] - (1.0 - radii)

    boundary_pen = np.sum(_soft_max(left_v, beta)) + np.sum(_soft_max(right_v, beta)) + \
                   np.sum(_soft_max(bottom_v, beta)) + np.sum(_soft_max(top_v, beta))

    # Smooth overlap violations
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    min_dist_sq = min_dist * min_dist
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)

    viol_raw = min_dist_sq - dist_sq
    overlap_pen = np.sum(_soft_max(viol_raw[mask], beta))

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
    """Shrink radii to ensure strict validity, preserving as much radius as possible."""
    pos = pos.copy()
    radii = radii.copy()

    for _ in range(200):
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

        for _ in range(5000):
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
        for _ in range(100):
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
            for attempt in range(200):
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

def _simulated_annealing(pos, radii, n_steps=3000, t0=0.05, t_end=0.001, seed=0):
    """
    Simulated annealing: perturb positions and radii, accept based on energy.
    """
    rng = np.random.RandomState(seed)
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    n = len(radii)

    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)

    current_energy = _objective(np.concatenate([pos.flatten(), radii]), 50.0)

    for step in range(n_steps):
        frac = step / max(n_steps - 1, 1)
        temp = t0 * (t_end / t0) ** frac

        i = rng.randint(n)
        dx = rng.normal(0, temp * 0.02)
        dy = rng.normal(0, temp * 0.02)
        dr = rng.normal(0, temp * 0.01)

        new_x = pos[i, 0] + dx
        new_y = pos[i, 1] + dy
        new_r = radii[i] + dr

        new_x = np.clip(new_x, 0.01, 0.99)
        new_y = np.clip(new_y, 0.01, 0.99)
        new_r = np.clip(new_r, 0.01, 0.4)

        old_x, old_y, old_r = pos[i, 0], pos[i, 1], radii[i]
        pos[i, 0] = new_x
        pos[i, 1] = new_y
        radii[i] = new_r

        x_new = np.concatenate([pos.flatten(), radii])
        new_energy = _objective(x_new, 50.0)

        delta_e = new_energy - current_energy
        if delta_e < 0 or rng.random() < np.exp(-delta_e / max(temp, 1e-8)):
            current_energy = new_energy
        else:
            pos[i, 0] = old_x
            pos[i, 1] = old_y
            radii[i] = old_r

        if step % 100 == 0:
            pos_pp, radii_pp = _post_process(pos, radii)
            if _is_valid(pos_pp, radii_pp):
                score = np.sum(radii_pp)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()

    return best_pos, best_radii

def _polish(pos, radii, deadline):
    """High-precision polishing with SLSQP and L-BFGS-B."""
    x = np.concatenate([pos.flatten(), radii])

    bounds = []
    for i in range(N):
        bounds.append((0.01, 0.99))
        bounds.append((0.01, 0.99))
    for i in range(N):
        bounds.append((0.01, 0.5))

    try:
        result = minimize(_objective, x, method='SLSQP', bounds=bounds,
                          args=(2000.0,),
                          options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10})
        if result.fun < _objective(x, 2000.0) - 1e-12:
            x = result.x
    except Exception:
        pass

    if time.time() < deadline - 5:
        try:
            result = minimize(_objective, x, method='L-BFGS-B', bounds=bounds,
                              args=(5000.0,),
                              options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10})
            if result.fun < _objective(x, 5000.0) - 1e-12:
                x = result.x
        except Exception:
            pass

    pos_out = x[:2 * N].reshape(N, 2)
    radii_out = np.maximum(x[2 * N:], 1e-4)

    return pos_out, radii_out

def _slsqp_hard_constraints(x0, deadline):
    """
    SLSQP with hard inequality constraints for non-overlap and boundary.
    """
    x = x0.copy()

    pair_i, pair_j = np.triu_indices(N, k=1)

    def constraint_boundary(x):
        pos = x[:2 * N].reshape(N, 2)
        radii = x[2 * N:]
        c1 = pos[:, 0] - radii
        c2 = 1.0 - pos[:, 0] - radii
        c3 = pos[:, 1] - radii
        c4 = 1.0 - pos[:, 1] - radii
        return np.concatenate([c1, c2, c3, c4])

    def constraint_overlap(x):
        pos = x[:2 * N].reshape(N, 2)
        radii = x[2 * N:]
        dx = pos[pair_i, 0] - pos[pair_j, 0]
        dy = pos[pair_i, 1] - pos[pair_j, 1]
        dist_sq = dx * dx + dy * dy
        min_dist = radii[pair_i] + radii[pair_j]
        return dist_sq - min_dist * min_dist

    constraints = [
        {'type': 'ineq', 'fun': constraint_boundary},
        {'type': 'ineq', 'fun': constraint_overlap}
    ]

    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
    for i in range(N):
        bounds.append((0.0, 0.5))

    def neg_sum_radii(x):
        return -np.sum(x[2 * N:])

    try:
        result = minimize(neg_sum_radii, x, method='SLSQP', bounds=bounds,
                          constraints=constraints,
                          options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10})
        if result.success or (not result.success and result.fun < neg_sum_radii(x)):
            x = result.x
    except Exception:
        pass

    pos_out = x[:2 * N].reshape(N, 2)
    radii_out = np.maximum(x[2 * N:], 1e-4)
    return pos_out, radii_out

def _contact_graph(pos, radii, tol=1e-6):
    """Build contact graph: pairs of circles that are nearly touching."""
    n = len(radii)
    contacts = set()
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if abs(dist - (radii[i] + radii[j])) < tol:
                contacts.add((i, j))
    return contacts

def _expand_circles_lp(pos, radii, max_iter=50, tol=1e-8):
    """
    Iteratively expand radii using LP-style local optimization.
    """
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    n = len(radii)

    for _ in range(max_iter):
        max_delta = 1.0
        for i in range(n):
            max_r_wall = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
            max_delta = min(max_delta, max_r_wall - radii[i])

        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                slack = dist - radii[i] - radii[j]
                max_delta = min(max_delta, slack / 2.0)

        max_delta = max(max_delta, 0.0)
        if max_delta < tol:
            break

        delta = max_delta * 0.999
        radii += delta

        valid = True
        for i in range(n):
            if pos[i, 0] - radii[i] < -tol or pos[i, 0] + radii[i] > 1.0 + tol:
                valid = False
                break
            if pos[i, 1] - radii[i] < -tol or pos[i, 1] + radii[i] > 1.0 + tol:
                valid = False
                break
        if valid:
            for i in range(n):
                for j in range(i + 1, n):
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    dist_sq = dx * dx + dy * dy
                    min_dist = radii[i] + radii[j]
                    if dist_sq < (min_dist - tol) ** 2:
                        valid = False
                        break
                if not valid:
                    break

        if not valid:
            radii -= delta
            break

    return pos, radii

def _solve_qp_radii(pos, max_iter=200, tol=1e-12):
    """
    Solve for optimal radii given fixed positions using iterative water-filling.
    Maximize sum(r_i) subject to:
      r_i <= min(x_i, 1-x_i, y_i, 1-y_i)  (boundary)
      r_i + r_j <= dist(i,j)               (non-overlap)
      r_i >= 0
    """
    pos = np.asarray(pos, dtype=float)
    n = len(pos)
    radii = np.zeros(n)
    
    for i in range(n):
        radii[i] = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1]) * 0.5

    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dists = np.sqrt(dx * dx + dy * dy)
    
    bnd = np.array([min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1]) for i in range(n)])
    
    for it in range(max_iter):
        old_radii = radii.copy()
        
        radii = np.minimum(radii, bnd)
        radii = np.maximum(radii, 0.0)
        
        for i in range(n):
            for j in range(i + 1, n):
                d = dists[i, j]
                if radii[i] + radii[j] > d + tol:
                    total = radii[i] + radii[j]
                    if total > 0:
                        scale = d / total
                        radii[i] *= scale
                        radii[j] *= scale
                    else:
                        radii[i] = d / 2
                        radii[j] = d / 2
        
        radii = np.minimum(radii, bnd)
        radii = np.maximum(radii, 0.0)
        
        if np.max(np.abs(radii - old_radii)) < tol:
            break
    
    for i in range(n):
        radii[i] = min(radii[i], bnd[i])
        radii[i] = max(radii[i], 0.0)
    
    for i in range(n):
        for j in range(i + 1, n):
            d = dists[i, j]
            if radii[i] + radii[j] > d:
                scale = d / (radii[i] + radii[j] + 1e-15) * 0.99999
                radii[i] *= scale
                radii[j] *= scale
    
    return radii

def _solve_radii_lp(pos):
    """
    Solve exact LP for maximum radii given fixed positions.
    Maximize sum(r_i) subject to:
      r_i <= bnd_i (boundary)
      r_i + r_j <= dist_ij (non-overlap)
      r_i >= 0
    Uses scipy.optimize.linprog with HiGHS solver.
    """
    pos = np.asarray(pos, dtype=float)
    n = len(pos)
    
    # Compute boundary limits
    bnd = np.array([min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1]) for i in range(n)])
    
    # Compute pairwise distances
    pair_i, pair_j = np.triu_indices(n, k=1)
    dx = pos[pair_i, 0] - pos[pair_j, 0]
    dy = pos[pair_i, 1] - pos[pair_j, 1]
    dists = np.sqrt(dx * dx + dy * dy)
    
    n_pairs = len(pair_i)
    
    # Objective: minimize -sum(r_i)
    c = -np.ones(n)
    
    # Inequality constraints: A_ub @ r <= b_ub
    # Boundary: r_i <= bnd_i  =>  1*r_i <= bnd_i
    # Overlap: r_i + r_j <= dist_ij
    
    A_ub_rows = []
    b_ub_vals = []
    
    # Boundary constraints
    for i in range(n):
        row = np.zeros(n)
        row[i] = 1.0
        A_ub_rows.append(row)
        b_ub_vals.append(bnd[i])
    
    # Overlap constraints
    for k in range(n_pairs):
        row = np.zeros(n)
        row[pair_i[k]] = 1.0
        row[pair_j[k]] = 1.0
        A_ub_rows.append(row)
        b_ub_vals.append(dists[k])
    
    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_vals)
    
    # Bounds: 0 <= r_i <= bnd_i
    bounds = [(0.0, bnd[i]) for i in range(n)]
    
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if result.success:
            radii = np.maximum(result.x, 0.0)
            # Final safety check
            for i in range(n):
                radii[i] = min(radii[i], bnd[i])
            for k in range(n_pairs):
                i, j = pair_i[k], pair_j[k]
                if radii[i] + radii[j] > dists[k]:
                    scale = dists[k] / (radii[i] + radii[j] + 1e-15) * 0.99999
                    radii[i] *= scale
                    radii[j] *= scale
            return radii
    except Exception:
        pass
    
    # Fallback to iterative method
    return _solve_qp_radii(pos)

def _refine_positions_and_radii(pos, radii, deadline, n_rounds=3):
    """
    Alternating refinement: optimize radii for fixed positions (LP), then optimize
    positions with fixed radii (SLSQP), repeat.
    """
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    best_score = np.sum(radii)
    best_pos = pos.copy()
    best_radii = radii.copy()
    
    for round_idx in range(n_rounds):
        if time.time() > deadline - 10:
            break
        
        # Step 1: Optimize radii for fixed positions using LP
        radii_new = _solve_radii_lp(pos)
        if _is_valid(pos, radii_new):
            score = np.sum(radii_new)
            if score > best_score:
                best_score = score
                best_pos = pos.copy()
                best_radii = radii_new.copy()
            radii = radii_new
        
        if time.time() > deadline - 15:
            break
        
        # Step 2: Optimize positions with fixed radii using SLSQP
        x0 = np.concatenate([pos.flatten(), radii])
        
        pair_i, pair_j = np.triu_indices(N, k=1)
        
        def constraint_boundary(x):
            p = x[:2 * N].reshape(N, 2)
            r = x[2 * N:]
            c1 = p[:, 0] - r
            c2 = 1.0 - p[:, 0] - r
            c3 = p[:, 1] - r
            c4 = 1.0 - p[:, 1] - r
            return np.concatenate([c1, c2, c3, c4])
        
        def constraint_overlap(x):
            p = x[:2 * N].reshape(N, 2)
            r = x[2 * N:]
            dx = p[pair_i, 0] - p[pair_j, 0]
            dy = p[pair_i, 1] - p[pair_j, 1]
            dist_sq = dx * dx + dy * dy
            min_dist = r[pair_i] + r[pair_j]
            return dist_sq - min_dist * min_dist
        
        constraints = [
            {'type': 'ineq', 'fun': constraint_boundary},
            {'type': 'ineq', 'fun': constraint_overlap}
        ]
        
        bounds = []
        for i in range(N):
            bounds.append((0.0, 1.0))
            bounds.append((0.0, 1.0))
        for i in range(N):
            bounds.append((radii[i] * 0.999, radii[i] * 1.001))
        
        def neg_sum_radii(x):
            return -np.sum(x[2 * N:])
        
        try:
            result = minimize(neg_sum_radii, x0, method='SLSQP', bounds=bounds,
                              constraints=constraints,
                              options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10})
            if result.fun < neg_sum_radii(x0) - 1e-12:
                pos_new = result.x[:2 * N].reshape(N, 2)
                rad_new = result.x[2 * N:]
                pos_pp, rad_pp = _post_process(pos_new, rad_new)
                if _is_valid(pos_pp, rad_pp):
                    score = np.sum(rad_pp)
                    if score > best_score:
                        best_score = score
                        best_pos = pos_pp.copy()
                        best_radii = rad_pp.copy()
                    pos = pos_pp
                    radii = rad_pp
        except Exception:
            pass
    
    return best_pos, best_radii

def construct():
    start_time = time.time()
    deadline = start_time + 148.0

    inits = []
    pos_bf, rad_bf = _initial_best_fit()
    inits.append((pos_bf, rad_bf))

    pos_hex, rad_hex = _initial_hexagonal()
    inits.append((pos_hex, rad_hex))

    pos_cb, rad_cb = _initial_corner_biased(seed=7)
    inits.append((pos_cb, rad_cb))

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

    penalty_stages = [50.0, 500.0, 2000.0]

    for idx, (pos_init, rad_init) in enumerate(inits):
        if time.time() > deadline - 20:
            break

        x0 = np.concatenate([pos_init.flatten(), rad_init])
        current_x = x0.copy()

        for p_w in penalty_stages:
            if time.time() > deadline - 10:
                break
            try:
                result = minimize(_objective, current_x, method='L-BFGS-B', bounds=bounds,
                                  args=(p_w,),
                                  options={'maxiter': 800, 'ftol': 1e-12, 'gtol': 1e-8})
                if result.fun < _objective(current_x, p_w) - 1e-10:
                    current_x = result.x
            except Exception:
                pass

            pos_opt = current_x[:2 * N].reshape(N, 2)
            radii_opt = np.maximum(current_x[2 * N:], 1e-4)
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
            score = np.sum(radii_pp)
            if _is_valid(pos_pp, radii_pp) and score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()

        if time.time() < deadline - 10:
            for p_w in [500.0, 2000.0]:
                if time.time() > deadline - 5:
                    break
                try:
                    result = minimize(_objective_smooth, current_x, method='L-BFGS-B', bounds=bounds,
                                      args=(p_w, 30.0),
                                      options={'maxiter': 600, 'ftol': 1e-12, 'gtol': 1e-8})
                    if result.fun < _objective_smooth(current_x, p_w, 30.0) - 1e-10:
                        current_x = result.x
                except Exception:
                    pass

                pos_opt = current_x[:2 * N].reshape(N, 2)
                radii_opt = np.maximum(current_x[2 * N:], 1e-4)
                pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
                score = np.sum(radii_pp)
                if _is_valid(pos_pp, radii_pp) and score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()

        if time.time() < deadline - 8:
            try:
                pos_sa, rad_sa = _simulated_annealing(
                    current_x[:2 * N].reshape(N, 2),
                    np.maximum(current_x[2 * N:], 1e-4),
                    n_steps=2000, t0=0.03, t_end=0.0005,
                    seed=idx * 100
                )
                pos_pp, radii_pp = _post_process(pos_sa, rad_sa)
                score = np.sum(radii_pp)
                if _is_valid(pos_pp, radii_pp) and score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
            except Exception:
                pass

        if time.time() < deadline - 12:
            try:
                pos_hc, rad_hc = _slsqp_hard_constraints(current_x, deadline)
                pos_pp, radii_pp = _post_process(pos_hc, rad_hc)
                score = np.sum(radii_pp)
                if _is_valid(pos_pp, radii_pp) and score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
            except Exception:
                pass

        if time.time() < deadline - 5:
            try:
                pos_exp, rad_exp = _expand_circles_lp(
                    current_x[:2 * N].reshape(N, 2),
                    np.maximum(current_x[2 * N:], 1e-4),
                    max_iter=50)
                pos_pp, radii_pp = _post_process(pos_exp, rad_exp)
                score = np.sum(radii_pp)
                if _is_valid(pos_pp, radii_pp) and score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
            except Exception:
                pass

    # Final polish
    if time.time() < deadline - 8:
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

    # LP-based radius optimization for fixed positions
    if time.time() < deadline - 5:
        try:
            rad_lp = _solve_radii_lp(best_pos)
            if _is_valid(best_pos, rad_lp):
                score = np.sum(rad_lp)
                if score > best_score:
                    best_score = score
                    best_radii = rad_lp.copy()
        except Exception:
            pass

    # Alternating position+radius refinement
    if time.time() < deadline - 15:
        try:
            pos_ref, rad_ref = _refine_positions_and_radii(best_pos, best_radii, deadline, n_rounds=2)
            pos_pp, radii_pp = _post_process(pos_ref, rad_ref)
            score = np.sum(radii_pp)
            if _is_valid(pos_pp, radii_pp) and score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
        except Exception:
            pass

    # Final SLSQP hard constraints on best
    if time.time() < deadline - 8:
        try:
            pos_hc, rad_hc = _slsqp_hard_constraints(
                np.concatenate([best_pos.flatten(), best_radii]), deadline)
            pos_pp, radii_pp = _post_process(pos_hc, rad_hc)
            score = np.sum(radii_pp)
            if _is_valid(pos_pp, radii_pp) and score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
        except Exception:
            pass

    # LP-based radius expansion
    if time.time() < deadline - 3:
        try:
            pos_exp, rad_exp = _expand_circles_lp(best_pos, best_radii, max_iter=100)
            pos_pp, radii_pp = _post_process(pos_exp, rad_exp)
            score = np.sum(radii_pp)
            if _is_valid(pos_pp, radii_pp) and score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
        except Exception:
            pass

    # LP radius optimization again
    if time.time() < deadline - 2:
        try:
            rad_lp2 = _solve_radii_lp(best_pos)
            if _is_valid(best_pos, rad_lp2):
                score = np.sum(rad_lp2)
                if score > best_score:
                    best_score = score
                    best_radii = rad_lp2.copy()
        except Exception:
            pass

    # One more SLSQP pass after LP expansion
    if time.time() < deadline - 3:
        try:
            pos_hc2, rad_hc2 = _slsqp_hard_constraints(
                np.concatenate([best_pos.flatten(), best_radii]), deadline)
            pos_pp, radii_pp = _post_process(pos_hc2, rad_hc2)
            score = np.sum(radii_pp)
            if _is_valid(pos_pp, radii_pp) and score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
        except Exception:
            pass

    # Final LP expansion after last SLSQP
    if time.time() < deadline - 1:
        try:
            pos_exp2, rad_exp2 = _expand_circles_lp(best_pos, best_radii, max_iter=100)
            pos_pp, radii_pp = _post_process(pos_exp2, rad_exp2)
            score = np.sum(radii_pp)
            if _is_valid(pos_pp, radii_pp) and score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
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
