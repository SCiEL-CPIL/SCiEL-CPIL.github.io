"""
50 equal circles in a unit square – rigid-body contact-graph assembly approach.

Strategy:
  1. Build a contact graph from a hexagonal initial placement.
  2. Use a force-directed / constraint-satisfaction method where each contact
     edge enforces exact distance = 2r, and walls enforce center-to-wall = r.
  3. Iterate: solve for positions given r (linearized), then adjust r to close
     the residual.
  4. Use simulated annealing with the contact-graph energy to escape local minima.
  5. Final polish with L-BFGS on a smooth surrogate.
"""
import numpy as np
import time
import scipy.optimize as opt
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

N = 50
DEADLINE = 145.0  # Safety margin before 150s kill


def _compute_radius(centers):
    """Compute the true maximum radius for a given configuration."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 2) or not np.all(np.isfinite(centers)):
        return 0.0
    
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    min_wall = np.min(np.minimum(wall_dists_x, wall_dists_y))
    
    if min_wall < 0:
        return 0.0
    
    tree = cKDTree(centers)
    dists, _ = tree.query(centers, k=2)
    min_pair = np.min(dists[:, 1]) / 2.0
    
    return min(min_pair, min_wall)


def _build_contact_graph(centers, r, margin=0.01):
    """
    Build a contact graph: edges between circles whose distance is close to 2r,
    and wall contacts where center-to-wall distance is close to r.
    Returns: (edge_list, wall_list)
      edge_list: list of (i, j) for inter-circle contacts
      wall_list: list of (i, axis, sign) for wall contacts
    """
    edges = []
    walls = []
    
    # Inter-circle contacts
    tree = cKDTree(centers)
    pairs = tree.query_pairs(2 * r + margin)
    for i, j in pairs:
        d = np.linalg.norm(centers[i] - centers[j])
        if abs(d - 2 * r) < margin:
            edges.append((i, j))
    
    # Wall contacts
    for i in range(N):
        x, y = centers[i]
        if abs(0.5 - x - r) < margin:
            walls.append((i, 0, 1))   # right wall
        if abs(-0.5 + x + r) < margin:
            walls.append((i, 0, -1))  # left wall
        if abs(0.5 - y - r) < margin:
            walls.append((i, 1, 1))   # top wall
        if abs(-0.5 + y + r) < margin:
            walls.append((i, 1, -1))  # bottom wall
    
    return edges, walls


def _solve_rigid_assembly(centers, r, edges, walls, max_iter=50, tol=1e-10):
    """
    Solve for positions that satisfy:
      - For each edge (i,j): |c_i - c_j| = 2r
      - For each wall (i, axis, sign): sign * c_i[axis] + 0.5 = r
    Using iterative linearized updates (Gauss-Seidel style on the contact graph).
    """
    pos = centers.copy().astype(np.float64)
    
    for iteration in range(max_iter):
        max_change = 0.0
        
        # Enforce wall constraints
        for (i, axis, sign) in walls:
            target = sign * r - 0.5 * sign  # if sign=1: x = r - 0.5? No.
            # Wall at x=0.5: center x = 0.5 - r => sign=1 means right wall
            # Wall at x=-0.5: center x = -0.5 + r => sign=-1 means left wall
            if axis == 0:
                if sign == 1:  # right wall
                    target_x = 0.5 - r
                else:  # left wall
                    target_x = -0.5 + r
                change = target_x - pos[i, 0]
                pos[i, 0] += change
                max_change = max(max_change, abs(change))
            else:
                if sign == 1:  # top wall
                    target_y = 0.5 - r
                else:  # bottom wall
                    target_y = -0.5 + r
                change = target_y - pos[i, 1]
                pos[i, 1] += change
                max_change = max(max_change, abs(change))
        
        # Enforce distance constraints via iterative relaxation
        for (i, j) in edges:
            diff = pos[i] - pos[j]
            d = np.linalg.norm(diff)
            if d < 1e-15:
                continue
            # Desired distance is 2r
            # Move each point halfway toward/away
            error = (d - 2 * r) / d
            correction = 0.5 * error * diff
            pos[i] -= correction
            pos[j] += correction
            max_change = max(max_change, 0.5 * abs(error) * d)
        
        if max_change < tol:
            break
    
    return pos


def _contact_energy(centers, r, edges, walls):
    """
    Compute the total constraint violation energy.
    """
    energy = 0.0
    
    for (i, j) in edges:
        d = np.linalg.norm(centers[i] - centers[j])
        energy += 0.5 * (d - 2 * r) ** 2
    
    for (i, axis, sign) in walls:
        if axis == 0:
            if sign == 1:
                err = (0.5 - centers[i, 0]) - r
            else:
                err = (centers[i, 0] + 0.5) - r
        else:
            if sign == 1:
                err = (0.5 - centers[i, 1]) - r
            else:
                err = (centers[i, 1] + 0.5) - r
        energy += 0.5 * err ** 2
    
    return energy


def _anneal_with_contacts(centers, r, rng, time_budget):
    """
    Simulated annealing on the contact-graph energy, periodically rebuilding
    the contact graph and adjusting r.
    """
    start_time = time.time()
    pos = centers.copy().astype(np.float64)
    best = pos.copy()
    best_r = _compute_radius(pos)
    
    T = 0.005
    T_min = 0.0001
    n_steps = 20000
    rebuild_interval = 200
    
    for step in range(n_steps):
        if time.time() - start_time > time_budget:
            break
        
        T = T_min + (T - T_min) * (1 - step / n_steps)
        
        # Rebuild contact graph periodically
        if step % rebuild_interval == 0:
            edges, walls = _build_contact_graph(pos, r, margin=0.02)
        
        # Current energy
        E_curr = _contact_energy(pos, r, edges, walls)
        
        # Propose a random move
        move_size = 0.02 * T / T_min * 0.1 + 0.005
        idx = rng.integers(0, N)
        axis = rng.integers(0, 2)
        delta = rng.normal(0, move_size)
        
        pos_new = pos.copy()
        pos_new[idx, axis] += delta
        pos_new = np.clip(pos_new, -0.5, 0.5)
        
        E_new = _contact_energy(pos_new, r, edges, walls)
        
        dE = E_new - E_curr
        if dE < 0 or rng.random() < np.exp(-dE / T):
            pos = pos_new
        
        # Periodically adjust r to close the gap
        if step % rebuild_interval == 0:
            # Solve for positions at current r
            pos = _solve_rigid_assembly(pos, r, edges, walls, max_iter=30)
            pos = np.clip(pos, -0.5, 0.5)
            
            # Estimate best r from current configuration
            r_est = _compute_radius(pos)
            if r_est > r * 0.95:
                r = min(r_est, r * 1.02)
            
            # Track best
            r_curr = _compute_radius(pos)
            if r_curr > best_r:
                best_r = r_curr
                best = pos.copy()
    
    # Final solve
    edges, walls = _build_contact_graph(pos, r, margin=0.02)
    pos = _solve_rigid_assembly(pos, r, edges, walls, max_iter=100)
    pos = np.clip(pos, -0.5, 0.5)
    r_curr = _compute_radius(pos)
    if r_curr > best_r:
        best_r = r_curr
        best = pos.copy()
    
    return best, best_r


def _soft_min_objective(x, tau=0.005):
    """
    Objective: minimize -soft_min(minimum clearance).
    """
    centers = x.reshape(-1, 2)
    
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wall_dists_x, wall_dists_y)
    
    tree = cKDTree(centers)
    dists, _ = tree.query(centers, k=2)
    pair_dists = dists[:, 1] / 2.0
    
    all_dists = np.concatenate([walls, pair_dists])
    
    min_d = np.min(all_dists)
    shifted = all_dists - min_d
    log_sum = np.log(np.sum(np.exp(-shifted / tau)))
    soft_min = min_d - tau * log_sum
    
    return -soft_min


def _grad_soft_min(x, tau=0.005):
    """Analytic gradient of -soft_min."""
    centers = x.reshape(-1, 2)
    
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wall_dists_x, wall_dists_y)
    
    iu, ju = np.triu_indices(N, k=1)
    diff = centers[iu] - centers[ju]
    dists = np.sqrt(np.sum(diff**2, axis=1))
    pair_dists = dists / 2.0
    
    all_dists = np.concatenate([walls, pair_dists])
    
    min_d = np.min(all_dists)
    shifted = all_dists - min_d
    exp_vals = np.exp(-shifted / tau)
    weights = exp_vals / np.sum(exp_vals)
    
    grad = np.zeros_like(x)
    
    # Wall gradients
    for i in range(N):
        w = weights[i]
        if wall_dists_x[i] <= wall_dists_y[i]:
            grad[2*i] += w * np.sign(centers[i, 0])
        if wall_dists_y[i] <= wall_dists_x[i]:
            grad[2*i+1] += w * np.sign(centers[i, 1])
    
    # Pair gradients
    for k in range(len(iu)):
        w = weights[N + k]
        i, j = iu[k], ju[k]
        d = dists[k]
        if d > 1e-12:
            factor = 0.5 * w / d
            diff_ij = centers[i] - centers[j]
            grad[2*i:2*i+2] -= factor * diff_ij
            grad[2*j:2*j+2] += factor * diff_ij
    
    return grad


def _lbfgs_polish(x0, tau, time_budget):
    """L-BFGS-B polish on soft-min objective."""
    start_time = time.time()
    
    def obj_and_grad(x):
        if time.time() - start_time > time_budget:
            return 0.0, np.zeros_like(x)
        return _soft_min_objective(x, tau), _grad_soft_min(x, tau)
    
    try:
        result = opt.minimize(
            obj_and_grad, x0, jac=True, method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': 300, 'ftol': 1e-16, 'gtol': 1e-13,
                     'maxfun': 1000, 'disp': False}
        )
        return np.clip(result.x.reshape(-1, 2), -0.5, 0.5)
    except Exception:
        return x0.reshape(-1, 2)


def _initial_hex_grid():
    """Hexagonal grid initial placement."""
    rows = 8
    cols = 7  # 8*7 = 56, we'll take 50
    
    dx = 1.0 / (cols + 0.5)
    dy = dx * np.sqrt(3) / 2
    
    points = []
    for row in range(rows):
        y = -0.5 + dy * (row + 0.5)
        if y > 0.5:
            break
        offset = dx / 2 if row % 2 == 1 else 0
        for col in range(cols):
            x = -0.5 + dx * (col + 0.5) + offset
            if -0.5 <= x <= 0.5:
                points.append([x, y])
                if len(points) >= N:
                    break
        if len(points) >= N:
            break
    
    if len(points) < N:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        for y in ticks:
            for x in ticks:
                if len(points) >= N:
                    break
                if not any(np.allclose([x, y], p) for p in points):
                    points.append([x, y])
            if len(points) >= N:
                break
    
    grid = np.array(points[:N])
    rng = np.random.default_rng(42)
    grid = grid + rng.normal(0, 0.002, grid.shape)
    grid = np.clip(grid, -0.49, 0.49)
    return grid


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_centers = _initial_hex_grid()
    best_radius = _compute_radius(best_centers)
    
    rng = np.random.default_rng(12345)
    
    # Phase 1: Contact-graph annealing
    if time.time() < deadline - 50:
        sa_budget = min(40.0, deadline - time.time() - 50)
        if sa_budget > 5:
            best_centers, best_radius = _anneal_with_contacts(
                best_centers, best_radius, rng, sa_budget
            )
    
    # Phase 2: Coarse-to-fine L-BFGS
    if time.time() < deadline - 30:
        x0 = best_centers.flatten()
        for tau in [0.02, 0.01, 0.005, 0.003, 0.002, 0.001]:
            if time.time() > deadline - 15:
                break
            tb = min(8.0, deadline - time.time() - 12)
            if tb < 2:
                break
            trial = _lbfgs_polish(x0, tau, tb)
            trial_r = _compute_radius(trial)
            if trial_r > best_radius:
                best_radius = trial_r
                best_centers = trial
            x0 = best_centers.flatten()
    
    # Phase 3: Another round of contact-graph refinement
    if time.time() < deadline - 20:
        sa_budget = min(15.0, deadline - time.time() - 20)
        if sa_budget > 3:
            best_centers, best_radius = _anneal_with_contacts(
                best_centers, best_radius, rng, sa_budget
            )
    
    # Phase 4: Final high-precision polish
    if time.time() < deadline - 5:
        tb = min(10.0, deadline - time.time() - 4)
        if tb > 2:
            trial = _lbfgs_polish(best_centers.flatten(), 0.0005, tb)
            trial_r = _compute_radius(trial)
            if trial_r > best_radius:
                best_radius = trial_r
                best_centers = trial
    
    # Final micro-polish
    if time.time() < deadline - 2:
        tb = min(5.0, deadline - time.time() - 1)
        if tb > 1:
            try:
                result = opt.minimize(
                    lambda v: _soft_min_objective(v, tau=0.0003),
                    best_centers.flatten(), method='L-BFGS-B',
                    bounds=[(-0.5, 0.5)] * (2 * N),
                    options={'maxiter': 100, 'ftol': 1e-18, 'gtol': 1e-15,
                             'maxfun': 300, 'disp': False}
                )
                polished = np.clip(result.x.reshape(-1, 2), -0.5, 0.5)
                r_polished = _compute_radius(polished)
                if r_polished > best_radius:
                    best_radius = r_polished
                    best_centers = polished
            except Exception:
                pass
    
    if not np.all(np.isfinite(best_centers)) or best_centers.shape != (N, 2):
        return _initial_hex_grid()
    
    return best_centers
