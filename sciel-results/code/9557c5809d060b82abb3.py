"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _soft_max_lse(v, beta=50.0):
    """Log-sum-exp based smooth max(0, v)."""
    v = np.asarray(v, dtype=float)
    bv = beta * v
    result = np.maximum(bv, 0.0) + np.log1p(np.exp(-np.abs(bv)))
    return result / beta

def _objective_smooth_lse(x, penalty_weight=100.0, beta=40.0):
    """
    Objective with log-sum-exp smooth penalties.
    x = [x1, y1, ..., xN, yN, r1, ..., rN]
    """
    pos = x[:2 * N].reshape(N, 2)
    radii = np.maximum(x[2 * N:], 1e-6)

    left_v = radii - pos[:, 0]
    right_v = pos[:, 0] - (1.0 - radii)
    bottom_v = radii - pos[:, 1]
    top_v = pos[:, 1] - (1.0 - radii)

    boundary_pen = (np.sum(_soft_max_lse(left_v, beta)) + np.sum(_soft_max_lse(right_v, beta)) +
                    np.sum(_soft_max_lse(bottom_v, beta)) + np.sum(_soft_max_lse(top_v, beta)))

    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    min_dist_sq = min_dist * min_dist
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)

    viol_raw = min_dist_sq - dist_sq
    overlap_pen = np.sum(_soft_max_lse(viol_raw[mask], beta))

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

def _simulated_annealing(pos, radii, n_steps=3000, t0=0.05, t_end=0.001, seed=0):
    """Simulated annealing."""
    rng = np.random.RandomState(seed)
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    n = len(radii)

    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)

    current_energy = _objective_smooth_lse(np.concatenate([pos.flatten(), radii]), 50.0)

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
        new_energy = _objective_smooth_lse(x_new, 50.0)

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

def _try_radius_expansion(pos, radii, max_iterations=300):
    """Try to expand each circle's radius while maintaining validity using bisection."""
    pos = pos.copy()
    radii = radii.copy()
    n = N

    for _ in range(max_iterations):
        improved = False
        for i in range(n):
            x, y = pos[i]
            r = radii[i]

            # Max radius from walls
            max_r_wall = min(x, 1.0 - x, y, 1.0 - y)

            # Max radius from other circles
            dx = pos[:, 0] - x
            dy = pos[:, 1] - y
            dists = np.sqrt(dx * dx + dy * dy)
            dists[i] = np.inf
            max_r_circ = max_r_wall
            for j in range(n):
                if j == i:
                    continue
                possible = dists[j] - radii[j]
                if possible < max_r_circ:
                    max_r_circ = possible

            if max_r_circ <= r * 1.000001:
                continue

            # Bisection: find max valid radius for circle i
            lo, hi = r, max_r_circ
            for _ in range(18):
                mid = (lo + hi) / 2.0
                radii[i] = mid
                # Quick validity check: only need to check circle i against others and walls
                if (mid > x or mid > 1.0 - x or mid > y or mid > 1.0 - y):
                    hi = mid
                    continue
                # Check against all other circles
                valid = True
                for j in range(n):
                    if j == i:
                        continue
                    dxj = pos[i, 0] - pos[j, 0]
                    dyj = pos[i, 1] - pos[j, 1]
                    d = np.sqrt(dxj * dxj + dyj * dyj)
                    if d < mid + radii[j] - 1e-9:
                        valid = False
                        break
                if valid:
                    lo = mid
                else:
                    hi = mid

            radii[i] = lo
            if lo > r * 1.000001:
                improved = True

        if not improved:
            break

    return pos, radii

def _coordinate_descent_tighten(pos, radii, n_iterations=200, deadline=None):
    """
    Alternate between expanding radii (fixing positions) and adjusting positions (fixing radii) to maximize sum of radii.
    """
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    n = N
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    for it in range(n_iterations):
        if deadline is not None and time.time() > deadline - 1:
            break
            
        # Step 1: Expand radii
        pos, radii = _try_radius_expansion(pos, radii, max_iterations=20)
        
        # Step 2: Adjust positions to allow further expansion
        improved = False
        for i in range(n):
            x, y = pos[i]
            r = radii[i]
            
            # Try 8 directions + current
            best_move = (0, 0)
            best_new_r = r
            
            # Use slightly larger steps for initial moves, smaller later
            step = 0.01 if it < 20 else 0.002
            for dx in [-step, 0, step]:
                for dy in [-step, 0, step]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if nx < 0.01 or nx > 0.99 or ny < 0.01 or ny > 0.99:
                        continue
                    
                    # Check if this position allows a larger radius
                    max_r_new = min(nx, 1-nx, ny, 1-ny)
                    for j in range(n):
                        if i == j: continue
                        dxj = nx - pos[j, 0]
                        dyj = ny - pos[j, 1]
                        dist = np.sqrt(dxj*dxj + dyj*dyj)
                        max_r_new = min(max_r_new, dist - radii[j])
                    
                    if max_r_new > best_new_r:
                        best_new_r = max_r_new
                        best_move = (dx, dy)
                        
            if best_move != (0, 0) and best_new_r > r * 1.001:
                pos[i, 0] += best_move[0]
                pos[i, 1] += best_move[1]
                improved = True
                
        # Check validity and score
        pos_pp, radii_pp = _post_process(pos, radii)
        if _is_valid(pos_pp, radii_pp):
            score = np.sum(radii_pp)
            if score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
                pos = best_pos.copy()
                radii = best_radii.copy()
            elif not improved:
                # If no improvement in position moves, we might be stuck.
                pass

    return best_pos, best_radii

def _exact_feasibility_shrink(pos, radii, tol=1e-8):
    """Shrink radii minimally to ensure strict validity. Returns (pos, radii, was_shrunk)."""
    pos = pos.copy()
    radii = radii.copy()
    n = N
    was_shrunk = False

    for _ in range(100):
        all_ok = True

        # Wall constraints
        for i in range(n):
            x, y, r = pos[i, 0], pos[i, 1], radii[i]
            max_r = min(x, 1.0 - x, y, 1.0 - y)
            if r > max_r - tol:
                radii[i] = max(max_r - tol, 1e-6)
                was_shrunk = True
                all_ok = False

        # Pairwise constraints
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r - tol:
                    # Shrink both proportionally
                    excess = sum_r - dist
                    # Need sum_r_new <= dist - tol
                    scale = (dist - tol) / sum_r
                    if scale < 1.0:
                        radii[i] *= scale
                        radii[j] *= scale
                        was_shrunk = True
                        all_ok = False

        if all_ok:
            break

    return pos, radii, was_shrunk

def _perturbation_local_search(pos, radii, n_perturbations=500, max_step=0.02, seed=0, deadline=None):
    """Local search with random perturbations."""
    rng = np.random.RandomState(seed)
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    n = len(radii)

    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)

    for p in range(n_perturbations):
        if deadline is not None and time.time() > deadline - 2:
            break

        n_pert = rng.randint(1, 4)
        indices = rng.choice(n, size=n_pert, replace=False)

        old_pos = pos[indices].copy()
        old_radii = radii[indices].copy()

        for idx in indices:
            dx = rng.normal(0, max_step * rng.uniform(0.5, 1.0))
            dy = rng.normal(0, max_step * rng.uniform(0.5, 1.0))
            dr = rng.normal(0, max_step * 0.3)

            new_x = pos[idx, 0] + dx
            new_y = pos[idx, 1] + dy
            new_r = radii[idx] + dr

            new_x = np.clip(new_x, 0.005, 0.995)
            new_y = np.clip(new_y, 0.005, 0.995)
            new_r = np.clip(new_r, 0.005, 0.4)

            pos[idx, 0] = new_x
            pos[idx, 1] = new_y
            radii[idx] = new_r

        pos_pp, radii_pp = _post_process(pos, radii)
        if _is_valid(pos_pp, radii_pp):
            score = np.sum(radii_pp)
            if score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
                pos = best_pos.copy()
                radii = best_radii.copy()
        else:
            pos[indices] = old_pos
            radii[indices] = old_radii

    return best_pos, best_radii

def _slsqp_constraints(pos, radii):
    """Build SLSQP inequality constraints."""
    constraints = []
    n = len(radii)

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

def _polish(pos, radii, deadline):
    """High-precision polishing with multi-stage local search."""
    x = np.concatenate([pos.flatten(), radii])

    bounds = []
    for i in range(N):
        bounds.append((0.005, 0.995))
        bounds.append((0.005, 0.995))
    for i in range(N):
        bounds.append((0.005, 0.5))

    # Stage 1: L-BFGS-B with LSE smooth objective
    if time.time() < deadline - 25:
        try:
            result = minimize(_objective_smooth_lse, x, method='L-BFGS-B', bounds=bounds,
                              args=(5000.0, 50.0),
                              options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12})
            if result.fun < _objective_smooth_lse(x, 5000.0, 50.0) - 1e-13:
                x = result.x
        except Exception:
            pass

    # Stage 2: SLSQP with hard constraints
    if time.time() < deadline - 20:
        try:
            pos_tmp = x[:2*N].reshape(N, 2)
            radii_tmp = np.maximum(x[2*N:], 1e-5)
            pos_tmp, radii_tmp = _post_process(pos_tmp, radii_tmp)
            x_hard = np.concatenate([pos_tmp.flatten(), radii_tmp])
            cons = _slsqp_constraints(pos_tmp, radii_tmp)
            result = minimize(lambda v: -np.sum(v[2*N:]), x_hard, method='SLSQP', bounds=bounds,
                              constraints=cons,
                              options={'maxiter': 1500, 'ftol': 1e-15, 'gtol': 1e-12})
            if result.success or result.fun < -np.sum(x_hard[2*N:]):
                x = result.x
        except Exception:
            pass

    # Stage 3: L-BFGS-B with LSE smooth at higher penalty
    if time.time() < deadline - 16:
        try:
            result = minimize(_objective_smooth_lse, x, method='L-BFGS-B', bounds=bounds,
                              args=(20000.0, 80.0),
                              options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-13})
            if result.fun < _objective_smooth_lse(x, 20000.0, 80.0) - 1e-13:
                x = result.x
        except Exception:
            pass

    pos_out = x[:2*N].reshape(N, 2)
    radii_out = np.maximum(x[2*N:], 1e-5)
    pos_out, radii_out = _post_process(pos_out, radii_out)

    # Stage 4: SLSQP with hard constraints (second pass)
    if time.time() < deadline - 12:
        try:
            x_hard = np.concatenate([pos_out.flatten(), radii_out])
            cons = _slsqp_constraints(pos_out, radii_out)
            result = minimize(lambda v: -np.sum(v[2*N:]), x_hard, method='SLSQP', bounds=bounds,
                              constraints=cons,
                              options={'maxiter': 1500, 'ftol': 1e-16, 'gtol': 1e-14})
            if result.fun < -np.sum(x_hard[2*N:]):
                x = result.x
                pos_out = x[:2*N].reshape(N, 2)
                radii_out = np.maximum(x[2*N:], 1e-5)
                pos_out, radii_out = _post_process(pos_out, radii_out)
        except Exception:
            pass

    # Stage 5: Aggressive radius expansion
    if time.time() < deadline - 8:
        pos_out, radii_out = _try_radius_expansion(pos_out, radii_out, max_iterations=300)

    # Stage 6: Another SLSQP hard-constraint pass
    if time.time() < deadline - 5:
        try:
            x_hard = np.concatenate([pos_out.flatten(), radii_out])
            cons = _slsqp_constraints(pos_out, radii_out)
            result = minimize(lambda v: -np.sum(v[2*N:]), x_hard, method='SLSQP', bounds=bounds,
                              constraints=cons,
                              options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14})
            if result.fun < -np.sum(x_hard[2*N:]):
                pos_out = result.x[:2*N].reshape(N, 2)
                radii_out = np.maximum(result.x[2*N:], 1e-5)
                pos_out, radii_out = _post_process(pos_out, radii_out)
        except Exception:
            pass

    # Stage 7: Coordinate descent tightening
    if time.time() < deadline - 5:
        pos_out, radii_out = _coordinate_descent_tighten(
            pos_out, radii_out,
            n_iterations=100,
            deadline=deadline
        )

    # Stage 8: Perturbation local search
    if time.time() < deadline - 2:
        pos_out, radii_out = _perturbation_local_search(
            pos_out, radii_out,
            n_perturbations=200, max_step=0.005, seed=42,
            deadline=deadline
        )

    # Stage 9: Final aggressive radius expansion
    if time.time() < deadline - 0.5:
        pos_out, radii_out = _try_radius_expansion(pos_out, radii_out, max_iterations=200)

    # Stage 9: Final L-BFGS-B with very high penalty for tightness
    if time.time() < deadline - 0.3:
        try:
            x_final = np.concatenate([pos_out.flatten(), radii_out])
            result = minimize(_objective_smooth_lse, x_final, method='L-BFGS-B', bounds=bounds,
                              args=(100000.0, 100.0),
                              options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-14})
            if result.fun < _objective_smooth_lse(x_final, 100000.0, 100.0) - 1e-14:
                pos_out = result.x[:2*N].reshape(N, 2)
                radii_out = np.maximum(result.x[2*N:], 1e-5)
                pos_out, radii_out = _post_process(pos_out, radii_out)
        except Exception:
            pass

    # Stage 10: Exact feasibility shrink (minimal radius loss)
    pos_out, radii_out, _ = _exact_feasibility_shrink(pos_out, radii_out, tol=1e-9)

    return pos_out, radii_out

def _grid_based_init(resolution=40, seed=0):
    """
    Grid-based initialization: place circles at grid points with maximum possible radius.
    Uses a greedy approach: iterate over grid points in order of potential radius,
    placing circles where they fit.
    """
    rng = np.random.RandomState(seed)
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # Generate grid points
    grid_points = []
    for i in range(resolution + 1):
        for j in range(resolution + 1):
            x = i / resolution
            y = j / resolution
            grid_points.append((x, y))
    
    # Shuffle for diversity
    rng.shuffle(grid_points)
    
    placed = 0
    for x, y in grid_points:
        if placed >= N:
            break
        
        # Calculate max radius at this position
        max_r = min(x, 1.0 - x, y, 1.0 - y)
        if max_r < 0.01:
            continue
        
        # Check against already placed circles
        valid = True
        for j in range(placed):
            dx = x - pos[j, 0]
            dy = y - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < radii[j] + max_r * 0.99:
                valid = False
                break
        
        if valid and max_r > 0.01:
            pos[placed] = [x, y]
            radii[placed] = max_r * 0.95
            placed += 1
    
    # Fill remaining with random placement
    while placed < N:
        x = rng.uniform(0.05, 0.95)
        y = rng.uniform(0.05, 0.95)
        max_r = min(x, 1.0 - x, y, 1.0 - y)
        for j in range(placed):
            dx = x - pos[j, 0]
            dy = y - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            max_r = min(max_r, dist - radii[j])
        
        if max_r > 0.01:
            pos[placed] = [x, y]
            radii[placed] = max_r * 0.95
            placed += 1
        else:
            # Try again with smaller radius
            pos[placed] = [x, y]
            radii[placed] = 0.01
            placed += 1
    
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _force_directed_relax(pos, radii, n_iterations=1000, deadline=None):
    """
    Force-directed relaxation: circles repel each other and are attracted to expand.
    This helps escape local optima by applying physical-like forces.
    """
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    n = len(radii)
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    for it in range(n_iterations):
        if deadline is not None and time.time() > deadline - 2:
            break
        
        # Compute forces
        forces = np.zeros((n, 2))
        
        # Repulsion between overlapping or close circles
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy) + 1e-10
                min_dist = radii[i] + radii[j]
                
                if dist < min_dist * 1.1:  # Only apply force when close
                    overlap = min_dist - dist
                    force_mag = overlap * 10.0
                    if dist > 1e-8:
                        fx = (dx / dist) * force_mag
                        fy = (dy / dist) * force_mag
                    else:
                        fx = force_mag
                        fy = 0.0
                    forces[i] += [fx, fy]
                    forces[j] -= [fx, fy]
        
        # Wall repulsion
        for i in range(n):
            x, y = pos[i]
            r = radii[i]
            
            # Left wall
            if x - r < 0.05:
                forces[i, 0] += (0.05 - (x - r)) * 50.0
            # Right wall
            if (1.0 - x) - r < 0.05:
                forces[i, 0] -= (0.05 - ((1.0 - x) - r)) * 50.0
            # Bottom wall
            if y - r < 0.05:
                forces[i, 1] += (0.05 - (y - r)) * 50.0
            # Top wall
            if (1.0 - y) - r < 0.05:
                forces[i, 1] -= (0.05 - ((1.0 - y) - r)) * 50.0
        
        # Apply forces with damping
        damping = 0.9 ** (it / 100.0)
        step_size = 0.001 * damping
        
        for i in range(n):
            pos[i, 0] += forces[i, 0] * step_size
            pos[i, 1] += forces[i, 1] * step_size
            pos[i, 0] = np.clip(pos[i, 0], 0.01, 0.99)
            pos[i, 1] = np.clip(pos[i, 1], 0.01, 0.99)
        
        # Periodically try to expand radii
        if it % 50 == 0:
            pos_pp, radii_pp = _try_radius_expansion(pos, radii, max_iterations=10)
            pos_pp, radii_pp = _post_process(pos_pp, radii_pp)
            if _is_valid(pos_pp, radii_pp):
                score = np.sum(radii_pp)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
                    pos = best_pos.copy()
                    radii = best_radii.copy()
    
    return best_pos, best_radii

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

    pos_grid, rad_grid = _grid_based_init(resolution=40, seed=0)
    inits.append((pos_grid, rad_grid))

    pos_grid2, rad_grid2 = _grid_based_init(resolution=50, seed=1)
    inits.append((pos_grid2, rad_grid2))

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

        # Stage 1: L-BFGS-B with LSE smooth objective (progressive penalty)
        for p_w in [100.0, 1000.0, 5000.0]:
            if time.time() > deadline - 15:
                break
            try:
                result = minimize(_objective_smooth_lse, current_x, method='L-BFGS-B', bounds=bounds,
                                  args=(p_w, 40.0),
                                  options={'maxiter': 800, 'ftol': 1e-12, 'gtol': 1e-8})
                if result.fun < _objective_smooth_lse(current_x, p_w, 40.0) - 1e-10:
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

        # Stage 2: SLSQP with hard constraints
        if time.time() < deadline - 12:
            try:
                pos_tmp = current_x[:2*N].reshape(N, 2)
                radii_tmp = np.maximum(current_x[2*N:], 1e-4)
                pos_tmp, radii_tmp = _post_process(pos_tmp, radii_tmp)
                x_hard = np.concatenate([pos_tmp.flatten(), radii_tmp])
                cons = _slsqp_constraints(pos_tmp, radii_tmp)
                result = minimize(lambda v: -np.sum(v[2*N:]), x_hard, method='SLSQP', bounds=bounds,
                                  constraints=cons,
                                  options={'maxiter': 600, 'ftol': 1e-13, 'gtol': 1e-10})
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

        # Stage 2.5: Coordinate descent tightening
        if time.time() < deadline - 10:
            try:
                pos_cd, rad_cd = _coordinate_descent_tighten(
                    current_x[:2*N].reshape(N, 2),
                    np.maximum(current_x[2*N:], 1e-4),
                    n_iterations=50,
                    deadline=deadline
                )
                pos_pp, radii_pp = _post_process(pos_cd, rad_cd)
                score = np.sum(radii_pp)
                if _is_valid(pos_pp, radii_pp) and score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
                    # Update current_x to reflect this improvement
                    current_x = np.concatenate([pos_pp.flatten(), radii_pp])
            except Exception:
                pass

        # Stage 3: Simulated annealing (shorter, only for first 3 inits)
        if idx < 3 and time.time() < deadline - 10:
            try:
                pos_sa, rad_sa = _simulated_annealing(
                    current_x[:2*N].reshape(N, 2),
                    np.maximum(current_x[2*N:], 1e-4),
                    n_steps=1500, t0=0.025, t_end=0.0005,
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

        # Stage 4: Force-directed relaxation for first 4 inits
        if idx < 4 and time.time() < deadline - 8:
            try:
                pos_fd, rad_fd = _force_directed_relax(
                    current_x[:2*N].reshape(N, 2),
                    np.maximum(current_x[2*N:], 1e-4),
                    n_iterations=500,
                    deadline=deadline
                )
                pos_pp, radii_pp = _post_process(pos_fd, rad_fd)
                score = np.sum(radii_pp)
                if _is_valid(pos_pp, radii_pp) and score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
            except Exception:
                pass

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

    # Final validation: use exact feasibility shrink for minimal radius loss
    best_pos, best_radii, _ = _exact_feasibility_shrink(best_pos, best_radii, tol=1e-9)

    if not _is_valid(best_pos, best_radii, tol=1e-8):
        # Fallback: conservative grid placement
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
        best_pos, best_radii, _ = _exact_feasibility_shrink(best_pos, best_radii, tol=1e-9)

    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
