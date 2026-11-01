"""Model-and-optimize approach for 50 equal circles in a unit square."""
import numpy as np
import time
import scipy.optimize as opt
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
    
    # Use cKDTree for efficient nearest-neighbor distance
    tree = cKDTree(centers)
    # Query each point's nearest neighbor (excluding itself)
    dists, idx = tree.query(centers, k=2)
    min_pair = np.min(dists[:, 1]) / 2.0
    
    return min(min_pair, min_wall)

def _soft_min_objective(x, tau=0.008):
    """
    Objective: minimize -soft_min(minimum distance to constraint boundary).
    x: flattened array of 2*N coordinates
    """
    centers = x.reshape(-1, 2)
    
    # Wall distances
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wall_dists_x, wall_dists_y)
    
    # Pairwise distances using cKDTree
    tree = cKDTree(centers)
    dists, _ = tree.query(centers, k=2)
    pair_dists = dists[:, 1] / 2.0
    
    # Combine all constraint distances
    all_dists = np.concatenate([walls, pair_dists])
    
    # Soft-min using log-sum-exp
    min_d = np.min(all_dists)
    shifted = all_dists - min_d
    log_sum = np.log(np.sum(np.exp(-shifted / tau)))
    soft_min = min_d - tau * log_sum
    
    return -soft_min

def _grad_soft_min(x, tau=0.008):
    """
    Compute gradient of -soft_min objective analytically.
    """
    centers = x.reshape(-1, 2)
    
    # Wall distances
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wall_dists_x, wall_dists_y)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_dists = dists / 2.0
    
    # Upper triangle pairwise distances
    iu, ju = np.triu_indices(N, k=1)
    pair_upper = pair_dists[iu, ju]
    
    all_dists = np.concatenate([walls, pair_upper])
    
    # Soft-min weights: w_k = exp(-d_k/tau) / sum(exp(-d_j/tau))
    min_d = np.min(all_dists)
    shifted = all_dists - min_d
    exp_vals = np.exp(-shifted / tau)
    weights = exp_vals / np.sum(exp_vals)
    
    grad = np.zeros_like(x)
    
    # Wall gradient contributions
    for i in range(N):
        w_wall = weights[i]
        if walls[i] == wall_dists_x[i]:
            grad[2*i] += w_wall * np.sign(centers[i, 0])
        if walls[i] == wall_dists_y[i]:
            grad[2*i+1] += w_wall * np.sign(centers[i, 1])
    
    # Pairwise gradient contributions
    for k in range(len(iu)):
        w_pair = weights[N + k]
        i, j = iu[k], ju[k]
        d = dists[i, j]
        if d > 1e-12:
            factor = 0.5 * w_pair / d
            diff_ij = centers[i] - centers[j]
            grad[2*i:2*i+2] -= factor * diff_ij
            grad[2*j:2*j+2] += factor * diff_ij
    
    return grad

def _initial_grid():
    """Create initial hexagonal grid layout with slight jitter."""
    # Hexagonal packing: offset every other row
    rows = int(np.ceil(np.sqrt(N * 2 / np.sqrt(3))))
    cols = int(np.ceil(N / rows))
    
    # Hexagonal spacing
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
            if x < -0.5 or x > 0.5:
                continue
            points.append([x, y])
            if len(points) >= N:
                break
        if len(points) >= N:
            break
    
    # If we don't have enough, fill in with square grid
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
    
    rng = np.random.default_rng(12345)
    grid = grid + rng.normal(0, 0.003, grid.shape)
    grid = np.clip(grid, -0.49, 0.49)
    
    return grid

def _simulated_annealing(centers, rng, time_budget):
    """
    Simulated annealing using soft-min objective to escape local minima.
    """
    start_time = time.time()
    current = centers.copy()
    best = current.copy()
    best_r = _compute_radius(current)
    
    tau_start = 0.02
    tau_end = 0.003
    n_steps = 30000
    step_size_start = 0.05
    step_size_end = 0.002
    
    current_obj = _soft_min_objective(current.flatten(), tau_start)
    
    for step in range(n_steps):
        if time.time() - start_time > time_budget:
            break
        
        progress = step / n_steps
        tau = tau_start + (tau_end - tau_start) * progress
        step_size = step_size_start + (step_size_end - step_size_start) * progress
        
        # Propose move: random perturbation
        proposal = current + rng.normal(0, step_size, current.shape)
        proposal = np.clip(proposal, -0.5, 0.5)
        
        proposal_obj = _soft_min_objective(proposal.flatten(), tau)
        
        # Accept if better, or with probability exp(-delta/tau)
        delta = proposal_obj - current_obj
        if delta < 0 or rng.random() < np.exp(-delta / max(tau, 1e-10)):
            current = proposal
            current_obj = proposal_obj
            
            r_curr = _compute_radius(current)
            if r_curr > best_r:
                best_r = r_curr
                best = current.copy()
    
    return best, best_r

def _contact_polish(centers, time_budget):
    """
    SLSQP-based polish that identifies contact pairs and optimizes
    with explicit distance constraints, exploiting rigid-body structure.
    """
    start_time = time.time()
    current = centers.copy()
    best = current.copy()
    best_r = _compute_radius(current)
    
    # Identify likely contact pairs (within 1.1x of current min distance)
    tree = cKDTree(current)
    min_d = _compute_radius(current) * 2.0
    threshold = min_d * 1.15
    
    # Get all pairs within threshold
    pairs = tree.query_pairs(threshold)
    
    # Also identify wall contacts
    wall_x = 0.5 - np.abs(current[:, 0])
    wall_y = 0.5 - np.abs(current[:, 1])
    wall_min = np.minimum(wall_x, wall_y)
    
    # Build constraint list
    constraints = []
    for i, j in pairs:
        constraints.append({
            'type': 'eq',
            'fun': lambda x, i=i, j=j: np.sqrt(np.sum((x[2*i:2*i+2] - x[2*j:2*j+2])**2)) - min_d,
            'jac': lambda x, i=i, j=j: _contact_jac(x, i, j)
        })
    
    # Wall constraints for circles touching walls
    for i in range(N):
        if wall_min[i] < min_d * 0.95:
            if wall_x[i] <= wall_y[i]:
                # Touching left or right wall
                side = 1 if current[i, 0] > 0 else -1
                constraints.append({
                    'type': 'eq',
                    'fun': lambda x, i=i, side=side: side * x[2*i] + 0.5 - min_d,
                    'jac': lambda x, i=i, side=side: _wall_jac_x(x, i, side)
                })
            else:
                # Touching top or bottom wall
                side = 1 if current[i, 1] > 0 else -1
                constraints.append({
                    'type': 'eq',
                    'fun': lambda x, i=i, side=side: side * x[2*i+1] + 0.5 - min_d,
                    'jac': lambda x, i=i, side=side: _wall_jac_y(x, i, side)
                })
    
    if not constraints:
        return best, best_r
    
    # Objective: maximize minimum distance (soft-min surrogate)
    tau = 0.002
    
    def obj(x):
        return _soft_min_objective(x, tau)
    
    def obj_grad(x):
        return _grad_soft_min(x, tau)
    
    try:
        result = opt.minimize(
            obj, current.flatten(), jac=obj_grad, method='SLSQP',
            bounds=[(-0.5, 0.5)] * (2 * N),
            constraints=constraints,
            options={'maxiter': 300, 'ftol': 1e-14, 'disp': False}
        )
        polished = np.clip(result.x.reshape(-1, 2), -0.5, 0.5)
        r_polished = _compute_radius(polished)
        if r_polished > best_r:
            best_r = r_polished
            best = polished
    except Exception:
        pass
    
    return best, best_r

def _contact_jac(x, i, j):
    """Jacobian of distance constraint for pair (i,j)."""
    grad = np.zeros(2 * N)
    diff = x[2*i:2*i+2] - x[2*j:2*j+2]
    d = np.sqrt(np.sum(diff**2))
    if d > 1e-12:
        grad[2*i:2*i+2] = diff / d
        grad[2*j:2*j+2] = -diff / d
    return grad

def _wall_jac_x(x, i, side):
    """Jacobian of wall constraint (x-direction)."""
    grad = np.zeros(2 * N)
    grad[2*i] = side
    return grad

def _wall_jac_y(x, i, side):
    """Jacobian of wall constraint (y-direction)."""
    grad = np.zeros(2 * N)
    grad[2*i+1] = side
    return grad

def _lbfgs_optimize(x0, tau=0.008, time_budget=20.0):
    """Run L-BFGS-B optimization on the soft-min objective."""
    start_time = time.time()
    
    def obj_and_grad(x):
        if time.time() - start_time > time_budget:
            return 0.0, np.zeros_like(x)
        f = _soft_min_objective(x, tau)
        g = _grad_soft_min(x, tau)
        return f, g
    
    try:
        result = opt.minimize(
            obj_and_grad, x0, jac=True, method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12,
                     'maxfun': 2000, 'disp': False}
        )
        polished = result.x.reshape(-1, 2)
        polished = np.clip(polished, -0.5, 0.5)
        return polished
    except Exception:
        return x0.reshape(-1, 2)

def _high_precision_polish(centers, time_budget):
    """Final high-precision polish using very tight tolerances."""
    start_time = time.time()
    current = centers.copy()
    best = current.copy()
    best_r = _compute_radius(current)
    
    tau = 0.001
    
    def obj_and_grad(x):
        if time.time() - start_time > time_budget:
            return 0.0, np.zeros_like(x)
        f = _soft_min_objective(x, tau)
        g = _grad_soft_min(x, tau)
        return f, g
    
    try:
        result = opt.minimize(
            obj_and_grad, current.flatten(), jac=True, method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': 1000, 'ftol': 1e-18, 'gtol': 1e-15,
                     'maxfun': 5000, 'disp': False}
        )
        polished = result.x.reshape(-1, 2)
        polished = np.clip(polished, -0.5, 0.5)
        
        r_polished = _compute_radius(polished)
        if r_polished > best_r:
            best_r = r_polished
            best = polished
    except Exception:
        pass
    
    return best, best_r

def _binary_search_radius(centers, lo, hi, time_budget):
    """Binary search for maximum feasible radius with gradient polishing."""
    start_time = time.time()
    best = centers.copy()
    best_r = lo
    
    for _ in range(15):
        if time.time() - start_time > time_budget:
            break
        
        mid = (lo + hi) / 2
        trial = best.copy()
        
        x = trial.flatten().copy()
        lr = 0.001
        for _ in range(200):
            if time.time() - start_time > time_budget:
                break
            c = x.reshape(-1, 2)
            
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            pair_viol = np.sum(np.maximum(0, mid - dists / 2.0))
            
            wall_x = 0.5 - np.abs(c[:, 0])
            wall_y = 0.5 - np.abs(c[:, 1])
            wall_viol = np.sum(np.maximum(0, mid - np.minimum(wall_x, wall_y)))
            
            v = pair_viol + wall_viol
            if v < 1e-14:
                break
            
            grad = np.zeros_like(x)
            for i in range(N):
                for j in range(i+1, N):
                    d = dists[i, j]
                    if d < 2 * mid and d > 1e-12:
                        factor = 0.5 / d
                        diff_ij = c[i] - c[j]
                        grad[2*i:2*i+2] -= factor * diff_ij
                        grad[2*j:2*j+2] += factor * diff_ij
                if wall_x[i] < mid:
                    grad[2*i] += np.sign(c[i, 0])
                if wall_y[i] < mid:
                    grad[2*i+1] += np.sign(c[i, 1])
            
            x -= lr * grad
            x = np.clip(x, -0.5, 0.5)
        
        trial = x.reshape(-1, 2)
        trial_r = _compute_radius(trial)
        
        if trial_r >= mid - 1e-10:
            lo = mid
            best = trial
            best_r = trial_r
        else:
            hi = mid
    
    return best, best_r

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    best_centers = _initial_grid()
    best_radius = _compute_radius(best_centers)
    
    rng = np.random.default_rng(777)
    
    # Phase 1: Simulated Annealing to escape local minima
    if time.time() < deadline - 60:
        sa_budget = min(40.0, deadline - time.time() - 60)
        if sa_budget > 5:
            best_centers, best_radius = _simulated_annealing(best_centers, rng, sa_budget)
    
    # Phase 2: L-BFGS with multiple tau values
    if time.time() < deadline - 45:
        x0 = best_centers.flatten()
        for tau in [0.015, 0.01, 0.008, 0.005, 0.003]:
            if time.time() > deadline - 30:
                break
            trial = _lbfgs_optimize(x0, tau=tau, time_budget=min(10, deadline - time.time() - 25))
            trial_r = _compute_radius(trial)
            if trial_r > best_radius:
                best_radius = trial_r
                best_centers = trial
            x0 = best_centers.flatten()
    
    # Phase 3: Contact-based SLSQP polish
    if time.time() < deadline - 25:
        best_centers, best_radius = _contact_polish(
            best_centers, time_budget=min(20, deadline - time.time() - 20)
        )
    
    # Phase 4: Binary search
    if time.time() < deadline - 12:
        lo = best_radius
        hi = min(0.08, best_radius * 1.02)
        best_centers, best_radius = _binary_search_radius(
            best_centers, lo, hi, time_budget=min(10, deadline - time.time() - 8)
        )
    
    # Phase 5: High-precision polish
    if time.time() < deadline - 5:
        best_centers, best_radius = _high_precision_polish(
            best_centers, time_budget=min(8, deadline - time.time() - 3)
        )
    
    # Phase 6: Final contact polish if time remains
    if time.time() < deadline - 3:
        best_centers, best_radius = _contact_polish(
            best_centers, time_budget=min(5, deadline - time.time() - 2)
        )
    
    if not np.all(np.isfinite(best_centers)) or best_centers.shape != (N, 2):
        return _initial_grid()
    
    return best_centers
