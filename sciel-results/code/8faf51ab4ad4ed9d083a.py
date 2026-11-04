"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
import numpy as np
import time
import random
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _elapsed(start_time):
    return time.time() - start_time

def _remaining(start_time, limit=148.0):
    return max(0.0, limit - _elapsed(start_time))

def _max_radius_for_positions(pos):
    """
    Vectorized: given positions (N,2), compute the maximum radius each circle
    can have such that all circles are disjoint and inside the unit square.
    r_i = min(boundary[i], min_{j != i} dist(i,j)/2)
    """
    dists = cdist(pos, pos)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists, axis=1) / 2.0
    boundary = np.minimum(np.minimum(pos[:, 0], 1.0 - pos[:, 0]),
                          np.minimum(pos[:, 1], 1.0 - pos[:, 1]))
    radii = np.minimum(boundary, min_pair)
    return np.maximum(radii, 0.0)


def _objective(pos):
    """Sum of max radii for given positions."""
    return float(np.sum(_max_radius_for_positions(pos)))


def _is_valid(pos, eps=1e-9):
    """Check pairwise non-overlap and boundary containment with computed radii."""
    radii = _max_radius_for_positions(pos)
    dists = cdist(pos, pos)
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            if dists[i, j] < radii[i] + radii[j] - eps:
                return False
        if (pos[i, 0] < radii[i] - eps or pos[i, 0] > 1.0 - radii[i] + eps or
                pos[i, 1] < radii[i] - eps or pos[i, 1] > 1.0 - radii[i] + eps):
            return False
    return True


def _project_feasible(pos, max_iter=500, tol=1e-10):
    """
    Project positions so that computed radii are feasible (no overlaps, in bounds).
    Iteratively reduces radii and adjusts positions to resolve conflicts.
    Returns (pos, radii) that are guaranteed feasible.
    """
    n = len(pos)
    pos = pos.copy()
    radii = _max_radius_for_positions(pos)
    
    for _ in range(max_iter):
        # Check if already feasible
        feasible = True
        dists = cdist(pos, pos)
        for i in range(n):
            for j in range(i + 1, n):
                if dists[i, j] < radii[i] + radii[j] - tol:
                    feasible = False
                    # Reduce the larger radius slightly
                    if radii[i] >= radii[j]:
                        radii[i] = max(0.0, dists[i, j] / 2.0 - 1e-12)
                    else:
                        radii[j] = max(0.0, dists[i, j] / 2.0 - 1e-12)
            # Check bounds
            boundary = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
            if radii[i] > boundary + tol:
                feasible = False
                radii[i] = max(0.0, boundary - 1e-12)
        
        if feasible:
            break
    
    # Final recomputation to ensure consistency
    radii = _max_radius_for_positions(pos)
    return pos, radii


def _score(pos):
    """Compute the score (sum of radii) for a position configuration."""
    return float(np.sum(_max_radius_for_positions(pos)))


def _hex_packing_init(n=N):
    """
    Initialize with a hexagonal packing pattern.
    This provides a good starting point with circles already well-separated.
    """
    # Hexagonal packing: rows offset by half spacing
    # For n circles, use roughly sqrt(n) rows
    n_rows = int(np.ceil(np.sqrt(n * 2.0 / np.sqrt(3))))
    spacing = 1.0 / (n_rows + 0.5)
    
    pos = []
    count = 0
    for row in range(n_rows + 1):
        row_offset = (row % 2) * spacing / 2.0
        n_cols = int(np.floor(1.0 / spacing)) + 1
        for col in range(n_cols):
            if count >= n:
                break
            x = row_offset + col * spacing
            y = row * spacing * np.sqrt(3) / 2.0
            if 0 <= x <= 1.0 and 0 <= y <= 1.0:
                pos.append([x, y])
                count += 1
        if count >= n:
            break
    
    # If we don't have enough, add random positions
    rng = np.random.RandomState(42)
    while len(pos) < n:
        x, y = rng.random(), rng.random()
        if all(np.hypot(x - p[0], y - p[1]) > 0.05 for p in pos):
            pos.append([x, y])
        else:
            pos.append([x, y])
    
    return np.array(pos[:n])


def _random_init(seed=0):
    rng = np.random.RandomState(seed)
    pos = []
    attempts = 0
    while len(pos) < N and attempts < 5000:
        attempts += 1
        x, y = rng.random(), rng.random()
        if all(np.hypot(x - px, y - py) > 0.06 for px, py in pos):
            pos.append([x, y])
    while len(pos) < N:
        pos.append([rng.random(), rng.random()])
    return np.array(pos)


def _grid_init():
    side = int(np.ceil(np.sqrt(N)))
    pos = []
    for i in range(side):
        for j in range(side):
            if len(pos) >= N:
                break
            pos.append([(i + 0.5) / side, (j + 0.5) / side])
    return np.array(pos[:N])


def _soft_objective(v, n, beta=50.0):
    """
    Vectorized smooth objective for gradient-based optimization.
    v = [x_0..x_{n-1}, y_0..y_{n-1}, r_0..r_{n-1}]
    Minimize: -sum(r) + penalty * (smooth wall violations + smooth pairwise violations)
    """
    x = v[0:n]
    y = v[n:2 * n]
    r = v[2 * n:3 * n]

    # Wall violations
    wall_min = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    wall_viol = r - wall_min
    wall_pen = np.sum(np.logaddexp(0.0, beta * wall_viol)) / beta

    # Pairwise violations - vectorized
    # Compute all pairwise distances
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    dists = np.sqrt(dx * dx + dy * dy + 1e-30)
    np.fill_diagonal(dists, np.inf)
    
    # Pairwise violations: r_i + r_j - dist_ij
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    viol = r_sum - dists
    # Only consider upper triangle to avoid double counting
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    viol_upper = viol[mask]
    pair_pen = np.sum(np.logaddexp(0.0, beta * viol_upper)) / beta

    return -np.sum(r) + 50.0 * (wall_pen + pair_pen)


def _soft_grad(v, n, beta=50.0):
    """
    Vectorized analytic gradient of _soft_objective.
    """
    x = v[0:n].copy()
    y = v[n:2 * n].copy()
    r = v[2 * n:3 * n].copy()

    grad = np.zeros(3 * n)

    # Wall terms
    wall_min = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    wall_viol = r - wall_min
    sig_wall = 1.0 / (1.0 + np.exp(-beta * wall_viol))

    for i in range(n):
        s = sig_wall[i]
        if s < 1e-12:
            continue
        dists = np.array([x[i], 1.0 - x[i], y[i], 1.0 - y[i]])
        min_idx = int(np.argmin(dists))
        factor = 50.0 * s
        grad[2 * n + i] += factor
        if min_idx == 0:
            grad[i] -= factor
        elif min_idx == 1:
            grad[i] += factor
        elif min_idx == 2:
            grad[n + i] -= factor
        else:
            grad[n + i] += factor

    # Pairwise terms - vectorized
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    dists_sq = dx * dx + dy * dy + 1e-30
    dists = np.sqrt(dists_sq)
    np.fill_diagonal(dists, 1.0)  # Avoid division by zero
    
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    viol = r_sum - dists
    
    # Softmax-like weights for violated pairs
    sig = 1.0 / (1.0 + np.exp(-beta * viol))
    np.fill_diagonal(sig, 0.0)
    
    # Gradient contribution from pairwise terms
    # d/dx_i sum_{j>i} log(1+exp(beta*(r_i+r_j-dist_ij)))
    # = sum_{j>i} beta * sig_ij * (dx_ij / dist_ij)
    # where dx_ij = x_i - x_j
    
    # For each i, sum over j != i
    factor_matrix = 50.0 * sig
    
    # Gradient for x
    grad_x = np.sum(factor_matrix * dx / dists, axis=1)
    grad_y = np.sum(factor_matrix * dy / dists, axis=1)
    grad_r = np.sum(factor_matrix, axis=1)
    
    grad[0:n] += grad_x
    grad[n:2 * n] += grad_y
    grad[2 * n:3 * n] += grad_r

    # Objective: -sum(r)
    grad[2 * n:3 * n] -= 1.0

    return grad


def _gradient_polish(pos, deadline, beta=50.0):
    """
    Final polish: use L-BFGS-B to optimize [x, y, r] jointly with smooth objective.
    Returns (pos, radii, score) if improved, else (pos, radii, score).
    """
    n = len(pos)
    r_init = _max_radius_for_positions(pos)
    v0 = np.concatenate([pos[:, 0], pos[:, 1], r_init])

    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))  # x
    for i in range(n):
        bounds.append((0.0, 1.0))  # y
    for i in range(n):
        bounds.append((0.0, 0.5))  # r

    try:
        res = minimize(
            _soft_objective, v0, args=(n, beta),
            jac=_soft_grad, method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-12}
        )
    except Exception:
        return pos, r_init, float(np.sum(r_init))

    if time.time() > deadline:
        return pos, r_init, float(np.sum(r_init))

    v_opt = res.x
    x_opt = v_opt[0:n]
    y_opt = v_opt[n:2 * n]
    r_opt = v_opt[2 * n:3 * n]

    pos_opt = np.column_stack([x_opt, y_opt])
    # Project to feasible region
    pos_opt, r_exact = _project_feasible(pos_opt)
    obj_opt = float(np.sum(r_exact))

    return pos_opt, r_exact, obj_opt


def _bubble_relaxation(pos, n_steps=2000, dt=0.002, pressure_coeff=10.0, 
                        repulsion_coeff=50.0, damping=0.95, deadline=None, seed=42):
    """
    Physics-based bubble relaxation.
    Circles are modeled as pressurized gas bubbles with internal pressure 
    proportional to r^2. They expand against repulsive contact forces and 
    boundary constraints, naturally maximizing sum of radii at equilibrium.
    
    Force model:
    - Internal pressure: F_i = pressure_coeff * r_i^2 * outward_normal
    - Repulsion between circles: F_ij = repulsion_coeff * max(0, r_i + r_j - dist_ij) * direction
    - Boundary repulsion: F_wall = wall_coeff * max(0, r_i - dist_to_wall) * inward_normal
    
    The system evolves via explicit Euler integration with damping.
    """
    rng = np.random.RandomState(seed)
    n = len(pos)
    current_pos = pos.copy()
    current_r = _max_radius_for_positions(current_pos)
    
    # Initialize velocities to zero
    vel = np.zeros((n, 2))
    
    best_pos = current_pos.copy()
    best_r = current_r.copy()
    best_score = float(np.sum(best_r))
    
    wall_coeff = 100.0
    
    for step in range(n_steps):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute forces
        forces = np.zeros((n, 2))
        
        # 1. Internal pressure (expansion force)
        # Each circle wants to expand; force is outward from center
        # We model this as a tendency to increase radius, which we handle by
        # allowing radii to grow when there's space
        for i in range(n):
            # Compute available space
            boundary = np.array([
                current_pos[i, 0], 1.0 - current_pos[i, 0],
                current_pos[i, 1], 1.0 - current_pos[i, 1]
            ])
            min_boundary = np.min(boundary)
            
            # Check pairwise constraints
            max_pair_r = min_boundary
            for j in range(n):
                if i == j:
                    continue
                dist = np.hypot(current_pos[i, 0] - current_pos[j, 0],
                               current_pos[i, 1] - current_pos[j, 1])
                max_pair_r = min(max_pair_r, dist / 2.0)
            
            # Pressure force: proportional to r^2, pushes outward
            # This is modeled as a force that increases radius when possible
            # For position dynamics, we use repulsion from other circles and walls
            pass
        
        # 2. Pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                dx = current_pos[i, 0] - current_pos[j, 0]
                dy = current_pos[i, 1] - current_pos[j, 1]
                dist = np.hypot(dx, dy)
                
                if dist < 1e-10:
                    # Jitter to break symmetry
                    dx = rng.uniform(-0.01, 0.01)
                    dy = rng.uniform(-0.01, 0.01)
                    dist = np.hypot(dx, dy)
                    if dist < 1e-10:
                        continue
                
                overlap = current_r[i] + current_r[j] - dist
                if overlap > 0:
                    # Repulsive force proportional to overlap
                    f_mag = repulsion_coeff * overlap
                    fx = f_mag * dx / dist
                    fy = f_mag * dy / dist
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
        
        # 3. Wall repulsion
        for i in range(n):
            # Left wall
            viol = current_r[i] - current_pos[i, 0]
            if viol > 0:
                forces[i, 0] += wall_coeff * viol
            # Right wall
            viol = current_r[i] - (1.0 - current_pos[i, 0])
            if viol > 0:
                forces[i, 0] -= wall_coeff * viol
            # Bottom wall
            viol = current_r[i] - current_pos[i, 1]
            if viol > 0:
                forces[i, 1] += wall_coeff * viol
            # Top wall
            viol = current_r[i] - (1.0 - current_pos[i, 1])
            if viol > 0:
                forces[i, 1] -= wall_coeff * viol
        
        # 4. Pressure-driven expansion: nudge radii up slightly
        # This simulates the internal pressure pushing circles to be larger
        # We implement this by slightly increasing radii and letting repulsion
        # push positions apart
        current_r = np.minimum(current_r + pressure_coeff * dt * current_r**2 * 0.01,
                               _max_radius_for_positions(current_pos))
        
        # Update velocities with damping
        vel = damping * vel + dt * forces
        vel = np.clip(vel, -0.05, 0.05)  # Limit velocity
        
        # Update positions
        current_pos = current_pos + dt * vel
        current_pos = np.clip(current_pos, 0.0, 1.0)
        
        # Recompute radii
        current_r = _max_radius_for_positions(current_pos)
        current_score = float(np.sum(current_r))
        
        if current_score > best_score:
            best_score = current_score
            best_pos = current_pos.copy()
            best_r = current_r.copy()
    
    return best_pos, best_r, best_score


def _local_search(pos, n_iter=4000, step_size=0.008, seed=42, deadline=None):
    """
    Local search: iteratively nudge each circle to improve the objective.
    """
    rng = np.random.RandomState(seed)
    current_pos = pos.copy()
    current_obj = _objective(current_pos)

    n = len(current_pos)
    improved = True
    it = 0
    while improved and it < n_iter:
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        it += 1
        order = rng.permutation(n)
        for i in order:
            for _ in range(15):
                dx = rng.uniform(-step_size, step_size)
                dy = rng.uniform(-step_size, step_size)
                new_pos = current_pos.copy()
                new_pos[i, 0] = np.clip(current_pos[i, 0] + dx, 0.0, 1.0)
                new_pos[i, 1] = np.clip(current_pos[i, 1] + dy, 0.0, 1.0)

                new_obj = _objective(new_pos)
                if new_obj > current_obj + 1e-13:
                    current_pos = new_pos
                    current_obj = new_obj
                    improved = True
                    break

    return current_pos, current_obj


def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill

    # Diverse initializations
    initializations = [_hex_packing_init()]
    initializations.append(_grid_init())
    for s in range(5):
        initializations.append(_random_init(seed=s * 997 + 13))

    best_pos = None
    best_obj = -np.inf

    # Phase 1: Bubble relaxation from each init
    for init_idx, init_pos in enumerate(initializations):
        rem = _remaining(start_time)
        if rem < 15:
            break

        # Use a portion of remaining time for bubble relaxation
        bubble_deadline = start_time + rem * 0.6
        b_pos, b_r, b_obj = _bubble_relaxation(
            init_pos, n_steps=3000, dt=0.0015,
            pressure_coeff=8.0, repulsion_coeff=40.0,
            damping=0.92, deadline=bubble_deadline,
            seed=init_idx * 7 + 1
        )

        if b_obj > best_obj:
            best_pos = b_pos
            best_obj = b_obj

    if best_pos is None:
        pos = _hex_packing_init()
        r = _max_radius_for_positions(pos)
        return np.column_stack([pos, r])

    # Phase 2: Local search refinement
    rem = _remaining(start_time)
    if rem > 10:
        ls_deadline = start_time + rem * 0.5
        ls_pos, ls_obj = _local_search(
            best_pos, n_iter=5000, step_size=0.004,
            seed=999, deadline=ls_deadline
        )
        if ls_obj > best_obj:
            best_pos = ls_pos
            best_obj = ls_obj

    # Phase 3: Another bubble relaxation pass on best
    rem = _remaining(start_time)
    if rem > 8:
        bubble_deadline = start_time + rem * 0.5
        b2_pos, b2_r, b2_obj = _bubble_relaxation(
            best_pos, n_steps=2000, dt=0.001,
            pressure_coeff=6.0, repulsion_coeff=35.0,
            damping=0.90, deadline=bubble_deadline,
            seed=1234
        )
        if b2_obj > best_obj:
            best_pos = b2_pos
            best_obj = b2_obj

    # Phase 4: Gradient-based polish (smooth surrogate)
    rem = _remaining(start_time)
    if rem > 5:
        grad_deadline = start_time + rem * 0.8
        p_pol, r_pol, obj_pol = _gradient_polish(best_pos, grad_deadline, beta=50.0)
        if obj_pol > best_obj:
            best_pos = p_pol
            best_obj = obj_pol

    # Final: project to ensure feasibility
    best_pos, best_r = _project_feasible(best_pos)
    best_obj = float(np.sum(best_r))

    # Build result
    r_final = _max_radius_for_positions(best_pos)
    result = np.column_stack([best_pos, r_final])
    return result

# EVOLVE-BLOCK-END
