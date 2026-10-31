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

def _objective_epigraph(x, penalty_weight=100.0, beta=50.0):
    """
    Epigraph formulation: minimize -sum(radii) + penalty * sum(t_i)
    where t_i >= 0 are auxiliary variables for each constraint.
    x = [x1, y1, ..., xN, yN, r1, ..., rN, t1, ..., tM]
    M = 4*N (boundary) + N*(N-1)/2 (overlap)
    """
    pos = x[:2 * N].reshape(N, 2)
    radii = np.maximum(x[2 * N:], 1e-6)
    
    # Number of constraints
    n_boundary = 4 * N
    n_overlap = N * (N - 1) // 2
    n_constraints = n_boundary + n_overlap
    t = x[2 * N + N:2 * N + N + n_constraints]
    t = np.maximum(t, 0.0)

    # Boundary constraint violations (should be <= t_i)
    left_v = radii - pos[:, 0]
    right_v = pos[:, 0] - (1.0 - radii)
    bottom_v = radii - pos[:, 1]
    top_v = pos[:, 1] - (1.0 - radii)
    
    boundary_viols = np.concatenate([left_v, right_v, bottom_v, top_v])
    t_boundary = t[:n_boundary]
    boundary_pen = np.sum(np.maximum(0.0, boundary_viols - t_boundary))

    # Overlap constraint violations
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    min_dist_sq = min_dist * min_dist
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    overlap_viols = min_dist_sq[mask] - dist_sq[mask]
    t_overlap = t[n_boundary:]
    overlap_pen = np.sum(np.maximum(0.0, overlap_viols - t_overlap))

    return -np.sum(radii) + penalty_weight * (np.sum(t) + boundary_pen + overlap_pen)

def _slsqp_hard_constraints(pos, radii, deadline):
    """
    SLSQP optimization with explicit hard constraints for non-overlap and boundaries.
    """
    x = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((0.001, 0.999))
        bounds.append((0.001, 0.999))
    for i in range(N):
        bounds.append((0.001, 0.5))

    # Define constraints
    constraints = []
    
    # Boundary constraints: x_i - r_i >= 0, 1 - x_i - r_i >= 0, etc.
    for i in range(N):
        xi, yi, ri = 2*i, 2*i+1, 2*N+i
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i] - x[2*N+i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i] - x[2*N+i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i+1] - x[2*N+i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i+1] - x[2*N+i]})
    
    # Non-overlap constraints: (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
    for i in range(N):
        for j in range(i+1, N):
            xi, yi, ri = 2*i, 2*i+1, 2*N+i
            xj, yj, rj = 2*j, 2*j+1, 2*N+j
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, xi=xi, yi=yi, ri=ri, xj=xj, yj=yj, rj=rj: 
                    (x[xi] - x[xj])**2 + (x[yi] - x[yj])**2 - (x[ri] + x[rj])**2
            })

    # Try multiple SLSQP runs with different tolerances
    best_x = x.copy()
    best_fun = _objective(x, 1.0)
    
    for ftol, gtol, maxiter in [(1e-10, 1e-8, 400), (1e-12, 1e-10, 500), (1e-14, 1e-12, 600)]:
        if time.time() > deadline - 5:
            break
        try:
            result = minimize(_objective, best_x, method='SLSQP', bounds=bounds,
                              args=(1.0,),  # Low penalty since we have hard constraints
                              constraints=constraints,
                              options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol})
            if result.success:
                fun_val = _objective(result.x, 1.0)
                if fun_val < best_fun - 1e-12:
                    best_fun = fun_val
                    best_x = result.x
        except Exception:
            pass

    x = best_x
    pos_out = x[:2 * N].reshape(N, 2)
    radii_out = np.maximum(x[2 * N:], 1e-5)
    pos_out, radii_out = _post_process(pos_out, radii_out)
    return pos_out, radii_out

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
    """Shrink radii to ensure strict validity. Prefer shrinking the smaller circle in conflicts."""
    pos = pos.copy()
    radii = radii.copy()

    for _ in range(300):
        valid = True

        for i in range(N):
            r = radii[i]
            x, y = pos[i]
            max_r = min(x, 1.0 - x, y, 1.0 - y)
            if r > max_r:
                radii[i] = max(max_r * 0.9999, 1e-5)
                valid = False

        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r:
                    # Shrink the smaller circle more to preserve total radius
                    if radii[i] <= radii[j]:
                        max_allowed = dist - radii[j]
                        if max_allowed > 0:
                            radii[i] = max(max_allowed * 0.9999, 1e-5)
                        else:
                            k = dist / (sum_r + 1e-12) * 0.9999
                            radii[i] = max(radii[i] * k, 1e-5)
                            radii[j] = max(radii[j] * k, 1e-5)
                    else:
                        max_allowed = dist - radii[i]
                        if max_allowed > 0:
                            radii[j] = max(max_allowed * 0.9999, 1e-5)
                        else:
                            k = dist / (sum_r + 1e-12) * 0.9999
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

def _try_radius_expansion(pos, radii, max_iterations=300):
    """Try to expand each circle's radius while maintaining validity."""
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

            new_r = max_r_circ * 0.99999
            if new_r > r * 1.0001:
                radii[i] = new_r
                # Quick check: only verify circle i against others
                valid = True
                for j in range(N):
                    if i == j:
                        continue
                    dx = x - pos[j, 0]
                    dy = y - pos[j, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    if dist < radii[i] + radii[j] - 1e-10:
                        valid = False
                        break
                if valid:
                    improved = True
                else:
                    radii[i] = r

        if not improved:
            break

    # Additional pass: try to expand radii by moving circles slightly
    for _ in range(100):
        improved = False
        for i in range(N):
            x, y = pos[i]
            r = radii[i]
            
            # Try small moves in 16 directions with varying magnitudes
            for angle in np.linspace(0, 2*np.pi, 16, endpoint=False):
                for mag in [0.005, 0.01, 0.015, 0.02]:
                    dx = mag * np.cos(angle)
                    dy = mag * np.sin(angle)
                    new_x = np.clip(x + dx, 0.01, 0.99)
                    new_y = np.clip(y + dy, 0.01, 0.99)
                    
                    max_r_wall = min(new_x, 1.0 - new_x, new_y, 1.0 - new_y)
                    max_r_circ = max_r_wall
                    for j in range(N):
                        if i == j:
                            continue
                        dx2 = new_x - pos[j, 0]
                        dy2 = new_y - pos[j, 1]
                        dist = np.sqrt(dx2 * dx2 + dy2 * dy2)
                        max_r_circ = min(max_r_circ, dist - radii[j])
                    
                    new_r = max_r_circ * 0.99999
                    if new_r > r * 1.001:
                        # Test validity - only check circle i
                        old_pos_i = pos[i].copy()
                        old_r_i = radii[i]
                        pos[i] = [new_x, new_y]
                        radii[i] = new_r
                        valid = True
                        for j in range(N):
                            if i == j:
                                continue
                            dx = new_x - pos[j, 0]
                            dy = new_y - pos[j, 1]
                            dist = np.sqrt(dx * dx + dy * dy)
                            if dist < radii[i] + radii[j] - 1e-10:
                                valid = False
                                break
                        if valid:
                            improved = True
                            break
                        else:
                            pos[i] = old_pos_i
                            radii[i] = old_r_i
                    if improved:
                        break
                if improved:
                    break
        
        if not improved:
            break

    return pos, radii

def _perturbation_local_search(pos, radii, n_perturbations=500, max_step=0.02, seed=0, deadline=None):
    """Local search with random perturbations and adaptive step size."""
    rng = np.random.RandomState(seed)
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    n = len(radii)

    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    # Adaptive step size
    current_step = max_step
    no_improve_count = 0
    max_no_improve = 50

    for p in range(n_perturbations):
        if deadline is not None and time.time() > deadline - 2:
            break

        # Reduce step size if no improvement for a while
        if no_improve_count > max_no_improve:
            current_step *= 0.8
            no_improve_count = 0
            if current_step < 0.001:
                break

        n_pert = rng.randint(1, 4)
        indices = rng.choice(n, size=n_pert, replace=False)

        old_pos = pos[indices].copy()
        old_radii = radii[indices].copy()

        for idx in indices:
            dx = rng.normal(0, current_step * rng.uniform(0.5, 1.5))
            dy = rng.normal(0, current_step * rng.uniform(0.5, 1.5))
            dr = rng.normal(0, current_step * 0.3)

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
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            pos[indices] = old_pos
            radii[indices] = old_radii
            no_improve_count += 1

    return best_pos, best_radii

def _final_refinement(pos, radii, deadline):
    """
    Multi-stage SLSQP hard-constraint refinement with decreasing ftol,
    followed by high-penalty L-BFGS-B polish.
    """
    x = np.concatenate([pos.flatten(), radii])

    bounds = []
    for i in range(N):
        bounds.append((0.005, 0.995))
        bounds.append((0.005, 0.995))
    for i in range(N):
        bounds.append((0.005, 0.5))

    # Stage 1: SLSQP with moderate penalty and loose tolerance
    try:
        result = minimize(_objective, x, method='SLSQP', bounds=bounds,
                          args=(2000.0,),
                          options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
        if result.fun < _objective(x, 2000.0) - 1e-12:
            x = result.x
    except Exception:
        pass

    # Stage 2: SLSQP with higher penalty and tighter tolerance
    if time.time() < deadline - 15:
        try:
            result = minimize(_objective, x, method='SLSQP', bounds=bounds,
                              args=(5000.0,),
                              options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
            if result.fun < _objective(x, 5000.0) - 1e-12:
                x = result.x
        except Exception:
            pass

    # Stage 3: SLSQP with very tight tolerance
    if time.time() < deadline - 10:
        try:
            result = minimize(_objective, x, method='SLSQP', bounds=bounds,
                              args=(10000.0,),
                              options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-12})
            if result.fun < _objective(x, 10000.0) - 1e-12:
                x = result.x
        except Exception:
            pass

    # Stage 4: SLSQP with hard constraints
    if time.time() < deadline - 8:
        pos_slq, rad_slq = _slsqp_hard_constraints(
            x[:2 * N].reshape(N, 2), np.maximum(x[2 * N:], 1e-5), deadline)
        if np.sum(rad_slq) > np.sum(np.maximum(x[2 * N:], 1e-5)) - 1e-10:
            x = np.concatenate([pos_slq.flatten(), rad_slq])

    # Stage 5: High-penalty L-BFGS-B polish
    if time.time() < deadline - 6:
        try:
            result = minimize(_objective, x, method='L-BFGS-B', bounds=bounds,
                              args=(20000.0,),
                              options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12})
            if result.fun < _objective(x, 20000.0) - 1e-12:
                x = result.x
        except Exception:
            pass

    pos_out = x[:2 * N].reshape(N, 2)
    radii_out = np.maximum(x[2 * N:], 1e-5)
    pos_out, radii_out = _post_process(pos_out, radii_out)

    # Stage 6: Radius expansion
    if time.time() < deadline - 5:
        pos_out, radii_out = _try_radius_expansion(pos_out, radii_out)

    # Stage 7: Perturbation local search
    if time.time() < deadline - 3:
        pos_out, radii_out = _perturbation_local_search(
            pos_out, radii_out,
            n_perturbations=300, max_step=0.015, seed=123,
            deadline=deadline
        )

    # Stage 8: Final radius expansion
    if time.time() < deadline - 1:
        pos_out, radii_out = _try_radius_expansion(pos_out, radii_out, max_iterations=200)

    # Stage 9: One more round of local search + expansion
    if time.time() < deadline - 0.5:
        pos_out, radii_out = _perturbation_local_search(
            pos_out, radii_out,
            n_perturbations=100, max_step=0.005, seed=456,
            deadline=deadline
        )
        pos_out, radii_out = _try_radius_expansion(pos_out, radii_out, max_iterations=100)

    return pos_out, radii_out

def _polish(pos, radii, deadline):
    """High-precision polishing with multi-stage local search."""
    x = np.concatenate([pos.flatten(), radii])

    bounds = []
    for i in range(N):
        bounds.append((0.005, 0.995))
        bounds.append((0.005, 0.995))
    for i in range(N):
        bounds.append((0.005, 0.5))

    # Stage 1: SLSQP with high penalty
    try:
        result = minimize(_objective, x, method='SLSQP', bounds=bounds,
                          args=(5000.0,),
                          options={'maxiter': 800, 'ftol': 1e-15, 'gtol': 1e-12})
        if result.fun < _objective(x, 5000.0) - 1e-13:
            x = result.x
    except Exception:
        pass

    # Stage 2: SLSQP with hard constraints
    if time.time() < deadline - 12:
        pos_slq, rad_slq = _slsqp_hard_constraints(
            x[:2 * N].reshape(N, 2), np.maximum(x[2 * N:], 1e-5), deadline)
        if np.sum(rad_slq) > np.sum(np.maximum(x[2 * N:], 1e-5)) - 1e-10:
            x = np.concatenate([pos_slq.flatten(), rad_slq])

    # Stage 3: L-BFGS-B with even higher penalty
    if time.time() < deadline - 8:
        try:
            result = minimize(_objective, x, method='L-BFGS-B', bounds=bounds,
                              args=(10000.0,),
                              options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12})
            if result.fun < _objective(x, 10000.0) - 1e-13:
                x = result.x
        except Exception:
            pass

    pos_out = x[:2 * N].reshape(N, 2)
    radii_out = np.maximum(x[2 * N:], 1e-5)
    pos_out, radii_out = _post_process(pos_out, radii_out)

    # Stage 4: Radius expansion
    if time.time() < deadline - 6:
        pos_out, radii_out = _try_radius_expansion(pos_out, radii_out)

    # Stage 5: Perturbation local search
    if time.time() < deadline - 4:
        pos_out, radii_out = _perturbation_local_search(
            pos_out, radii_out,
            n_perturbations=400, max_step=0.02, seed=42,
            deadline=deadline
        )

    # Stage 6: Final radius expansion
    if time.time() < deadline - 2:
        pos_out, radii_out = _try_radius_expansion(pos_out, radii_out, max_iterations=150)

    # Stage 7: Additional local search with smaller steps
    if time.time() < deadline - 1:
        pos_out, radii_out = _perturbation_local_search(
            pos_out, radii_out,
            n_perturbations=200, max_step=0.005, seed=789,
            deadline=deadline
        )
        pos_out, radii_out = _try_radius_expansion(pos_out, radii_out, max_iterations=100)

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

        # Stage 1: Standard objective
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

        # Stage 2: Smooth objective
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

        # Stage 3: Simulated annealing
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

    # Final refinement with multi-stage SLSQP
    if time.time() < deadline - 5:
        try:
            pos_ref, rad_ref = _final_refinement(best_pos, best_radii, deadline)
            pos_pp, radii_pp = _post_process(pos_ref, rad_ref)
            score = np.sum(radii_pp)
            if _is_valid(pos_pp, radii_pp) and score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
        except Exception:
            pass

    # Final polish
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
