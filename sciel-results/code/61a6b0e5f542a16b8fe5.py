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
    Compute gradient of -soft_min objective analytically using cKDTree for speed.
    Consistent with _soft_min_objective (uses nearest neighbors only).
    """
    centers = x.reshape(-1, 2)
    
    # Wall distances
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wall_dists_x, wall_dists_y)
    
    # Pairwise distances using cKDTree (nearest neighbor only, consistent with objective)
    tree = cKDTree(centers)
    dists, _ = tree.query(centers, k=2)
    pair_dists = dists[:, 1] / 2.0
    
    all_dists = np.concatenate([walls, pair_dists])
    
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
    
    # Pairwise gradient contributions (only nearest neighbor pairs)
    for i in range(N):
        w_pair = weights[N + i]
        if w_pair < 1e-15:
            continue
        _, nn_idx = tree.query(centers[i], k=2)
        j = nn_idx[1]
        d = dists[i, 1]
        if d > 1e-12:
            factor = 0.5 * w_pair / d
            diff_ij = centers[i] - centers[j]
            grad[2*i:2*i+2] -= factor * diff_ij
            grad[2*j:2*j+2] += factor * diff_ij
    
    return grad

def _initial_grid():
    """Create initial hexagonal packing layout with slight jitter."""
    # Hexagonal arrangement: alternate rows offset by half-spacing
    # For 50 circles, use 7 rows with varying counts
    # Row pattern for hex packing: 8,7,8,7,8,7,5 = 50
    row_counts = [8, 7, 8, 7, 8, 7, 5]
    
    points = []
    total_rows = len(row_counts)
    # Vertical spacing for hex packing
    vert_spacing = 0.9 / (total_rows - 1) if total_rows > 1 else 0
    
    for row_idx, count in enumerate(row_counts):
        y = -0.45 + row_idx * vert_spacing
        # Horizontal spacing
        horiz_spacing = 0.9 / count if count > 1 else 0
        x_start = -0.45 + (0.9 - horiz_spacing * (count - 1)) / 2
        # Offset alternating rows
        if row_idx % 2 == 1:
            x_offset = horiz_spacing / 2
        else:
            x_offset = 0
        
        for i in range(count):
            x = x_start + i * horiz_spacing + x_offset
            points.append([x, y])
    
    points = np.array(points[:N])
    
    rng = np.random.default_rng(12345)
    points = points + rng.normal(0, 0.008, points.shape)
    points = np.clip(points, -0.49, 0.49)
    
    return points

def _simulated_annealing(centers, rng, time_budget):
    """
    Simulated annealing to optimize the minimum radius directly.
    Uses cKDTree for fast distance queries and biased moves toward constrained circles.
    """
    start_time = time.time()
    current = centers.copy()
    best = current.copy()
    best_r = _compute_radius(current)
    
    # Temperature schedule - slower cooling for more thorough exploration
    T_start = 0.035
    T_end = 1e-7
    n_steps = 100000
    cooling = (T_end / T_start) ** (1.0 / max(n_steps, 1))
    T = T_start
    
    # Pre-allocate
    trial = current.copy()
    
    for step in range(n_steps):
        if time.time() - start_time > time_budget:
            break
        
        # Compute current min radius components using cKDTree
        tree = cKDTree(current)
        dists, _ = tree.query(current, k=2)
        min_pair = np.min(dists[:, 1]) / 2.0
        wall_x = 0.5 - np.abs(current[:, 0])
        wall_y = 0.5 - np.abs(current[:, 1])
        min_wall = np.min(np.minimum(wall_x, wall_y))
        curr_r = min(min_pair, min_wall)
        
        # Build constraint list for biasing
        all_dists = np.concatenate([np.minimum(wall_x, wall_y), dists[:, 1] / 2.0])
        min_d = np.min(all_dists)
        weights = np.exp(-(all_dists - min_d) / 0.004)
        weights /= np.sum(weights)
        
        # Pick which constraint to improve (biased toward tight constraints)
        idx = rng.choice(len(all_dists), p=weights)
        
        move_size = 0.004 + 0.025 * T / T_start
        
        if idx < N:
            # Wall constraint for circle idx - push away from wall
            i = idx
            direction = rng.integers(0, 4)
            delta = np.zeros(2)
            if direction == 0:
                delta[0] = move_size * rng.uniform(0.5, 1.0)
            elif direction == 1:
                delta[0] = -move_size * rng.uniform(0.5, 1.0)
            elif direction == 2:
                delta[1] = move_size * rng.uniform(0.5, 1.0)
            else:
                delta[1] = -move_size * rng.uniform(0.5, 1.0)
            
            trial = current.copy()
            trial[i] += delta
            trial = np.clip(trial, -0.5, 0.5)
        else:
            # Pair constraint - move nearest-neighbor pair apart
            i = idx - N
            _, nn_idx = tree.query(current[i], k=2)
            j = nn_idx[1]
            
            diff = current[i] - current[j]
            dist = np.linalg.norm(diff)
            if dist > 1e-9:
                unit = diff / dist
            else:
                unit = rng.normal(size=2)
                unit /= np.linalg.norm(unit)
            
            # Move both apart with some randomness
            trial = current.copy()
            frac = rng.uniform(0.3, 0.7)
            trial[i] += unit * move_size * frac
            trial[j] -= unit * move_size * (1.0 - frac)
            trial = np.clip(trial, -0.5, 0.5)
        
        # Evaluate
        trial_r = _compute_radius(trial)
        delta_r = trial_r - curr_r
        
        # Accept if improvement, or with probability exp(delta_r / T)
        if delta_r >= 0:
            current = trial
            if trial_r > best_r:
                best_r = trial_r
                best = trial.copy()
        else:
            if rng.random() < np.exp(delta_r / max(T, 1e-10)):
                current = trial
        
        T *= cooling
    
    return best, best_r

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
            options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10,
                     'maxfun': 1000, 'disp': False}
        )
        polished = result.x.reshape(-1, 2)
        polished = np.clip(polished, -0.5, 0.5)
        return polished
    except Exception:
        return x0.reshape(-1, 2)

def _contact_polish(centers, time_budget):
    """
    Contact-enforcement polish: iteratively identify touching pairs/walls
    and push them apart to maximize the minimum distance.
    This directly enforces the rigid-body contact constraints.
    """
    start_time = time.time()
    best = centers.copy()
    best_r = _compute_radius(best)
    
    # Iteratively find the tightest constraint and push apart
    for iteration in range(500):
        if time.time() - start_time > time_budget:
            break
        
        c = best.copy()
        tree = cKDTree(c)
        dists, _ = tree.query(c, k=2)
        min_pair_d = np.min(dists[:, 1])
        wall_x = 0.5 - np.abs(c[:, 0])
        wall_y = 0.5 - np.abs(c[:, 1])
        min_wall_d = np.min(np.minimum(wall_x, wall_y))
        curr_r = min(min_pair_d / 2.0, min_wall_d)
        
        # Find the most violated constraint
        if min_pair_d / 2.0 < min_wall_d:
            # Pair constraint is tightest
            min_idx = np.argmin(dists[:, 1])
            i = min_idx
            _, nn_idx = tree.query(c[i], k=2)
            j = nn_idx[1]
            diff = c[i] - c[j]
            d = np.linalg.norm(diff)
            if d > 1e-12:
                unit = diff / d
            else:
                unit = np.array([1.0, 0.0])
            
            # Push apart by a small amount
            push = 0.001
            c[i] += unit * push * 0.5
            c[j] -= unit * push * 0.5
        else:
            # Wall constraint is tightest
            walls = np.minimum(wall_x, wall_y)
            min_idx = np.argmin(walls)
            i = min_idx
            # Push away from nearest wall
            if wall_x[i] <= wall_y[i]:
                c[i, 0] += 0.001 * np.sign(c[i, 0]) if c[i, 0] != 0 else 0.001
            else:
                c[i, 1] += 0.001 * np.sign(c[i, 1]) if c[i, 1] != 0 else 0.001
        
        c = np.clip(c, -0.5, 0.5)
        r = _compute_radius(c)
        if r > best_r:
            best_r = r
            best = c
        elif best_r - r > 0.0001:
            # If we've degraded significantly, stop
            break
    
    return best, best_r

def _high_precision_polish(centers, time_budget):
    """
    High-precision polish using L-BFGS with very tight tolerances
    on a smaller soft-min tau to refine the solution, followed by contact polish.
    """
    start_time = time.time()
    best = centers.copy()
    best_r = _compute_radius(best)
    
    # Use progressively smaller tau for tighter optimization
    for tau in [0.003, 0.002, 0.001]:
        if time.time() - start_time > time_budget * 0.6:
            break
        
        x0 = best.flatten().copy()
        
        def obj_and_grad(x):
            if time.time() - start_time > time_budget * 0.8:
                return 0.0, np.zeros_like(x)
            f = _soft_min_objective(x, tau)
            g = _grad_soft_min(x, tau)
            return f, g
        
        try:
            result = opt.minimize(
                obj_and_grad, x0, jac=True, method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-12,
                         'maxfun': 2000, 'disp': False}
            )
            polished = result.x.reshape(-1, 2)
            polished = np.clip(polished, -0.5, 0.5)
            r_polished = _compute_radius(polished)
            if r_polished > best_r:
                best_r = r_polished
                best = polished
        except Exception:
            pass
    
    # Final contact polish
    if time.time() - start_time < time_budget * 0.8:
        best, best_r = _contact_polish(
            best, time_budget=min(time_budget * 0.2, 10.0)
        )
    
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
        
        # Use gradient descent to minimize violation for radius mid
        x = trial.flatten().copy()
        lr = 0.001
        for _ in range(80):
            if time.time() - start_time > time_budget:
                break
            c = x.reshape(-1, 2)
            
            # Compute violation
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            pair_viol = np.sum(np.maximum(0, mid - dists / 2.0))
            
            wall_x = 0.5 - np.abs(c[:, 0])
            wall_y = 0.5 - np.abs(c[:, 1])
            wall_viol = np.sum(np.maximum(0, mid - np.minimum(wall_x, wall_y)))
            
            v = pair_viol + wall_viol
            if v < 1e-12:
                break
            
            # Gradient
            grad = np.zeros_like(x)
            iu, ju = np.triu_indices(N, k=1)
            for k in range(len(iu)):
                i, j = iu[k], ju[k]
                d = dists[i, j]
                if d < 2 * mid and d > 1e-12:
                    factor = 0.5 / d
                    diff_ij = c[i] - c[j]
                    grad[2*i:2*i+2] -= factor * diff_ij
                    grad[2*j:2*j+2] += factor * diff_ij
            for i in range(N):
                if wall_x[i] < mid:
                    grad[2*i] += np.sign(c[i, 0])
                if wall_y[i] < mid:
                    grad[2*i+1] += np.sign(c[i, 1])
            
            x -= lr * grad
            x = np.clip(x, -0.5, 0.5)
        
        trial = x.reshape(-1, 2)
        trial_r = _compute_radius(trial)
        
        if trial_r >= mid - 1e-9:
            lo = mid
            best = trial
            best_r = trial_r
        else:
            hi = mid
    
    return best, best_r

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Initial configuration: hexagonal packing
    best_centers = _initial_grid()
    best_radius = _compute_radius(best_centers)
    
    rng = np.random.default_rng(777)
    
    # Phase 1: Simulated Annealing to escape local minima
    # Give more time to SA since it's the main exploration phase
    sa_budget = min(60.0, (deadline - time.time()) * 0.45)
    if sa_budget > 5:
        best_centers, best_radius = _simulated_annealing(best_centers, rng, sa_budget)
    
    # Phase 2: L-BFGS optimization with soft-min
    if time.time() < deadline - 45:
        x0 = best_centers.flatten()
        # Try with different tau values
        for tau in [0.01, 0.008, 0.005]:
            if time.time() > deadline - 30:
                break
            trial = _lbfgs_optimize(x0, tau=tau, time_budget=min(10, deadline - time.time() - 20))
            trial_r = _compute_radius(trial)
            if trial_r > best_radius:
                best_radius = trial_r
                best_centers = trial
            x0 = best_centers.flatten()
    
    # Phase 3: High-precision polish (includes contact polish)
    if time.time() < deadline - 30:
        best_centers, best_radius = _high_precision_polish(
            best_centers, time_budget=min(25, deadline - time.time() - 15)
        )
    
    # Phase 4: Binary search with gradient polish
    if time.time() < deadline - 15:
        lo = best_radius
        hi = min(0.08, best_radius * 1.03)
        best_centers, best_radius = _binary_search_radius(
            best_centers, lo, hi, time_budget=min(12, deadline - time.time() - 8)
        )
    
    # Final validation
    if not np.all(np.isfinite(best_centers)) or best_centers.shape != (N, 2):
        return _initial_grid()
    
    return best_centers
