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

def _soft_min_objective(x, tau=0.008, k_neighbors=5):
    """
    Objective: minimize -soft_min(minimum distance to constraint boundary).
    Uses top-k nearest neighbors for a better approximation of the true min distance.
    x: flattened array of 2*N coordinates
    """
    centers = x.reshape(-1, 2)
    
    # Wall distances
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wall_dists_x, wall_dists_y)
    
    # Pairwise distances using cKDTree - use top-k neighbors
    tree = cKDTree(centers)
    k = min(k_neighbors + 1, N)
    dists, idxs = tree.query(centers, k=k)
    # dists[:, 0] is 0 (self), dists[:, 1:] are distances to neighbors
    # Take the minimum neighbor distance for each point
    pair_dists = dists[:, 1] / 2.0  # nearest neighbor distance / 2
    
    # Also collect second nearest to get a better picture
    if k >= 3:
        pair_dists_2nd = dists[:, 2] / 2.0
        # Combine: use both nearest and second-nearest
        all_pair_dists = np.concatenate([pair_dists, pair_dists_2nd])
    else:
        all_pair_dists = pair_dists
    
    # Combine all constraint distances
    all_dists = np.concatenate([walls, all_pair_dists])
    
    # Soft-min using log-sum-exp
    min_d = np.min(all_dists)
    shifted = all_dists - min_d
    log_sum = np.log(np.sum(np.exp(-shifted / tau)))
    soft_min = min_d - tau * log_sum
    
    return -soft_min

def _grad_soft_min(x, tau=0.008, k_neighbors=5):
    """
    Compute gradient of -soft_min objective analytically using cKDTree for speed.
    Uses top-k nearest neighbors consistent with objective.
    """
    centers = x.reshape(-1, 2)
    
    # Wall distances
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wall_dists_x, wall_dists_y)
    
    # Pairwise distances using cKDTree
    tree = cKDTree(centers)
    k = min(k_neighbors + 1, N)
    dists, idxs = tree.query(centers, k=k)
    
    # Build list of (dist, i, j) for all neighbor pairs
    pair_dists_list = []
    pair_indices_list = []
    for i in range(N):
        for j_idx in range(1, k):
            j = idxs[i, j_idx]
            d = dists[i, j_idx]
            if d > 1e-12:
                pair_dists_list.append(d / 2.0)
                pair_indices_list.append((i, j))
    
    all_pair_dists = np.array(pair_dists_list) if pair_dists_list else np.array([0.0])
    
    # Combine all constraint distances
    all_dists = np.concatenate([walls, all_pair_dists])
    
    # Soft-min weights: w_k = exp(-d_k/tau) / sum(exp(-d_j/tau))
    min_d = np.min(all_dists)
    shifted = all_dists - min_d
    exp_vals = np.exp(-shifted / tau)
    weights = exp_vals / np.sum(exp_vals)
    
    grad = np.zeros_like(x)
    
    # Wall gradient contributions
    for i in range(N):
        w_wall = weights[i]
        if w_wall < 1e-15:
            continue
        if wall_dists_x[i] <= wall_dists_y[i]:
            grad[2*i] += w_wall * np.sign(centers[i, 0])
        else:
            grad[2*i+1] += w_wall * np.sign(centers[i, 1])
    
    # Pairwise gradient contributions
    for idx, (i, j) in enumerate(pair_indices_list):
        w_pair = weights[N + idx]
        if w_pair < 1e-15:
            continue
        d = all_pair_dists[idx] * 2.0  # actual distance
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
    Optimized to avoid redundant tree rebuilds.
    """
    start_time = time.time()
    current = centers.copy()
    best = current.copy()
    best_r = _compute_radius(current)
    
    # Temperature schedule
    T_start = 0.03
    T_end = 1e-7
    n_steps = 80000
    cooling = (T_end / T_start) ** (1.0 / max(n_steps, 1))
    T = T_start
    
    # Pre-allocate
    trial = current.copy()
    
    # Cache tree and distances, rebuild only when current changes significantly
    tree = cKDTree(current)
    cached_dists = None
    cached_wall = None
    last_tree_build = 0
    
    for step in range(n_steps):
        if time.time() - start_time > time_budget:
            break
        
        # Rebuild tree every 50 steps or if we've made a significant move
        if step % 50 == 0 or step == 0:
            tree = cKDTree(current)
            last_tree_build = step
        
        # Compute current min radius components
        dists, _ = tree.query(current, k=3)
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
        
        move_size = 0.003 + 0.02 * T / T_start
        
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
    Uses adaptive step sizes and simultaneous constraint resolution.
    """
    start_time = time.time()
    best = centers.copy()
    best_r = _compute_radius(best)
    
    # Adaptive step size
    step = 0.002
    no_improve_count = 0
    
    for iteration in range(300):
        if time.time() - start_time > time_budget:
            break
        
        c = best.copy()
        tree = cKDTree(c)
        dists, idxs = tree.query(c, k=4)
        min_pair_d = np.min(dists[:, 1])
        wall_x = 0.5 - np.abs(c[:, 0])
        wall_y = 0.5 - np.abs(c[:, 1])
        min_wall_d = np.min(np.minimum(wall_x, wall_y))
        curr_r = min(min_pair_d / 2.0, min_wall_d)
        
        # Find all near-min constraints and push them simultaneously
        threshold = curr_r * 1.001
        
        # Collect pair constraints
        pair_moves = []
        for i in range(N):
            for j_idx in range(1, 4):
                j = idxs[i, j_idx]
                if j < i:  # avoid double-counting
                    continue
                d = dists[i, j_idx]
                if d / 2.0 < threshold:
                    diff = c[i] - c[j]
                    dist = np.linalg.norm(diff)
                    if dist > 1e-12:
                        unit = diff / dist
                        pair_moves.append((i, j, unit, (d/2.0 - curr_r) + 0.001))
        
        # Collect wall constraints
        wall_moves = []
        walls = np.minimum(wall_x, wall_y)
        for i in range(N):
            if walls[i] < threshold:
                if wall_x[i] <= wall_y[i]:
                    wall_moves.append((i, 0, np.sign(c[i, 0]) if c[i, 0] != 0 else 1.0, (walls[i] - curr_r) + 0.001))
                else:
                    wall_moves.append((i, 1, np.sign(c[i, 1]) if c[i, 1] != 0 else 1.0, (walls[i] - curr_r) + 0.001))
        
        # Apply moves with adaptive step
        if not pair_moves and not wall_moves:
            break
        
        total_weight = sum(m[3] for m in pair_moves) + sum(m[3] for m in wall_moves)
        if total_weight < 1e-10:
            break
        
        for i, j, unit, weight in pair_moves:
            frac = 0.5 * step * weight / total_weight
            c[i] += unit * frac
            c[j] -= unit * frac
        
        for i, axis, sign, weight in wall_moves:
            move = step * weight / total_weight
            if axis == 0:
                c[i, 0] += sign * move
            else:
                c[i, 1] += sign * move
        
        c = np.clip(c, -0.5, 0.5)
        r = _compute_radius(c)
        
        if r > best_r:
            best_r = r
            best = c
            no_improve_count = 0
            step = min(step * 1.1, 0.005)
        else:
            no_improve_count += 1
            step = max(step * 0.8, 0.0005)
            if no_improve_count > 30:
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
    for tau in [0.004, 0.003, 0.002, 0.001]:
        if time.time() - start_time > time_budget * 0.55:
            break
        
        x0 = best.flatten().copy()
        
        def obj_and_grad(x):
            if time.time() - start_time > time_budget * 0.75:
                return 0.0, np.zeros_like(x)
            f = _soft_min_objective(x, tau, k_neighbors=5)
            g = _grad_soft_min(x, tau, k_neighbors=5)
            return f, g
        
        try:
            result = opt.minimize(
                obj_and_grad, x0, jac=True, method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 400, 'ftol': 1e-15, 'gtol': 1e-12,
                         'maxfun': 1500, 'disp': False}
            )
            polished = result.x.reshape(-1, 2)
            polished = np.clip(polished, -0.5, 0.5)
            r_polished = _compute_radius(polished)
            if r_polished > best_r:
                best_r = r_polished
                best = polished
        except Exception:
            pass
    
    # Final contact polish with remaining time
    remaining = time_budget - (time.time() - start_time)
    if remaining > 2:
        best, best_r = _contact_polish(
            best, time_budget=min(remaining * 0.5, 8.0)
        )
    
    return best, best_r

def _binary_search_radius(centers, lo, hi, time_budget):
    """Binary search for maximum feasible radius with gradient polishing."""
    start_time = time.time()
    best = centers.copy()
    best_r = lo
    
    for _ in range(12):
        if time.time() - start_time > time_budget:
            break
        
        mid = (lo + hi) / 2
        trial = best.copy()
        
        # Use gradient descent to minimize violation for radius mid
        x = trial.flatten().copy()
        lr = 0.0008
        for _ in range(60):
            if time.time() - start_time > time_budget:
                break
            c = x.reshape(-1, 2)
            
            # Compute violation using cKDTree for efficiency
            tree = cKDTree(c)
            dists, _ = tree.query(c, k=3)
            pair_viol = np.sum(np.maximum(0, mid - dists[:, 1] / 2.0))
            
            wall_x = 0.5 - np.abs(c[:, 0])
            wall_y = 0.5 - np.abs(c[:, 1])
            walls = np.minimum(wall_x, wall_y)
            wall_viol = np.sum(np.maximum(0, mid - walls))
            
            v = pair_viol + wall_viol
            if v < 1e-12:
                break
            
            # Gradient using cKDTree
            grad = np.zeros_like(x)
            dists_full, idxs_full = tree.query(c, k=3)
            for i in range(N):
                for j_idx in range(1, 3):
                    j = idxs_full[i, j_idx]
                    if j < i:
                        continue
                    d = dists_full[i, j_idx]
                    if d / 2.0 < mid and d > 1e-12:
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
    sa_budget = min(55.0, (deadline - time.time()) * 0.4)
    if sa_budget > 5:
        best_centers, best_radius = _simulated_annealing(best_centers, rng, sa_budget)
    
    # Phase 2: L-BFGS optimization with soft-min
    if time.time() < deadline - 40:
        x0 = best_centers.flatten()
        for tau in [0.01, 0.008, 0.005]:
            if time.time() > deadline - 25:
                break
            trial = _lbfgs_optimize(x0, tau=tau, time_budget=min(8, deadline - time.time() - 15))
            trial_r = _compute_radius(trial)
            if trial_r > best_radius:
                best_radius = trial_r
                best_centers = trial
            x0 = best_centers.flatten()
    
    # Phase 3: High-precision polish (includes contact polish)
    if time.time() < deadline - 20:
        best_centers, best_radius = _high_precision_polish(
            best_centers, time_budget=min(20, deadline - time.time() - 10)
        )
    
    # Phase 4: Binary search with gradient polish
    if time.time() < deadline - 10:
        lo = best_radius
        hi = min(0.08, best_radius * 1.03)
        best_centers, best_radius = _binary_search_radius(
            best_centers, lo, hi, time_budget=min(10, deadline - time.time() - 5)
        )
    
    # Final validation
    if not np.all(np.isfinite(best_centers)) or best_centers.shape != (N, 2):
        return _initial_grid()
    
    return best_centers
