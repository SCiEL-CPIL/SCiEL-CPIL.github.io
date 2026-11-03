"""Optimized packing of 100 equal spheres in a unit cube."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize


def _compute_radius(pts):
    """Compute the maximum valid radius for given points in [-1/2, 1/2]^3."""
    if len(pts) < 2:
        return 0.5
    d = pdist(pts)
    min_pair_dist = d.min() / 2.0
    max_coord = np.abs(pts).max(axis=1)
    wall_dist = (0.5 - max_coord).min()
    return min(min_pair_dist, wall_dist)


def _compute_active_sets(pts, r_target, tol=1e-6):
    """
    Identify active (tight) constraints: pairs with dist/2 close to r_target
    and walls with wall_dist close to r_target.
    Returns (active_pair_indices, active_wall_indices) where each is a list of (i,j) or i.
    """
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, 1e-10)
    pair_dists = np.sqrt(dist_sq) / 2.0
    np.fill_diagonal(pair_dists, np.inf)
    
    abs_pts = np.abs(pts)
    wall_dists = 0.5 - np.max(abs_pts, axis=1)
    
    # Active pairs: dist/2 within tol of r_target (and not far above)
    active_pairs = []
    iu = np.triu_indices(n, k=1)
    for idx in range(len(iu[0])):
        d_val = pair_dists[iu[0][idx], iu[1][idx]]
        if d_val <= r_target + tol:
            active_pairs.append((iu[0][idx], iu[1][idx]))
    
    # Active walls
    active_walls = []
    for i in range(n):
        if wall_dists[i] <= r_target + tol:
            active_walls.append(i)
    
    return active_pairs, active_walls, pair_dists, wall_dists


def _grid_init(n):
    """Generate a grid initialization with various grid shapes."""
    best_pts = None
    best_r = 0
    
    # Try different grid configurations
    for m1 in range(3, 9):
        for m2 in range(3, 9):
            m3 = max(1, int(round(n / (m1 * m2))))
            if m1 * m2 * m3 < n:
                continue
            
            ticks1 = (np.arange(m1) + 0.5) / m1 - 0.5
            ticks2 = (np.arange(m2) + 0.5) / m2 - 0.5
            ticks3 = (np.arange(m3) + 0.5) / m3 - 0.5
            
            grid = np.array([(x, y, z) for z in ticks3 for y in ticks2 for x in ticks1])
            if len(grid) >= n:
                subset = grid[:n]
                r = _compute_radius(subset)
                if r > best_r:
                    best_r = r
                    best_pts = subset
    
    if best_pts is None:
        m = int(np.ceil(n ** (1.0 / 3.0)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_pts = grid[:n]
        best_r = _compute_radius(best_pts)
    
    return best_pts, best_r


def _bcc_init(n):
    """Generate BCC lattice initialization."""
    best_pts = None
    best_r = 0
    
    for a in np.linspace(0.22, 0.38, 25):
        n_cells = int(1.0 / a)
        pts = []
        for i in range(n_cells + 1):
            for j in range(n_cells + 1):
                for k in range(n_cells + 1):
                    x = -0.5 + i * a
                    y = -0.5 + j * a
                    z = -0.5 + k * a
                    if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5 and -0.5 <= z <= 0.5:
                        pts.append([x, y, z])
                    x2 = -0.5 + (i + 0.5) * a
                    y2 = -0.5 + (j + 0.5) * a
                    z2 = -0.5 + (k + 0.5) * a
                    if -0.5 <= x2 <= 0.5 and -0.5 <= y2 <= 0.5 and -0.5 <= z2 <= 0.5:
                        pts.append([x2, y2, z2])
        
        pts = np.array(pts)
        if len(pts) < n:
            continue
        
        # Select n points with best spacing using greedy approach
        np.random.seed(42)
        for trial in range(20):
            if trial == 0:
                subset = pts[:n]
            else:
                idx = np.random.choice(len(pts), n, replace=False)
                subset = pts[idx]
            
            r = _compute_radius(subset)
            if r > best_r:
                best_r = r
                best_pts = subset
    
    if best_pts is None:
        pts, r = _grid_init(n)
        return pts, r
    
    return best_pts, best_r


def _random_init(n, seed=None):
    """Generate random initialization."""
    if seed is not None:
        np.random.seed(seed)
    pts = np.random.uniform(-0.45, 0.45, size=(n, 3))
    r = _compute_radius(pts)
    return pts, r


def _soft_min_energy(pts, target_r, beta=20.0):
    """
    Soft-min epigraph energy: measures how much the minimum radius falls short of target_r.
    Uses log-sum-exp to create a smooth approximation of max(0, target_r - min_constraint).
    
    Constraints: for all pairs (i,j), dist(i,j)/2 >= r, and for all points, wall_dist >= r.
    We want to maximize r, so we minimize the violation: sum of exp(-beta*(constraint - target_r))
    for constraints that are violated (constraint < target_r).
    
    Returns (energy, gradient) where energy is the soft violation and gradient w.r.t. pts.
    """
    pts = np.array(pts, dtype=np.float64)
    n = len(pts)
    
    # Compute all pair distances / 2
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, 1e-10)
    pair_dists = np.sqrt(dist_sq) / 2.0
    np.fill_diagonal(pair_dists, np.inf)
    
    # Wall distances
    abs_pts = np.abs(pts)
    wall_dists = 0.5 - np.max(abs_pts, axis=1)
    
    # Collect all constraint values
    # Pair constraints: upper triangle only
    iu = np.triu_indices(n, k=1)
    pair_vals = pair_dists[iu]
    
    # Wall constraints
    wall_vals = wall_dists
    
    # All constraints
    all_vals = np.concatenate([pair_vals, wall_vals])
    
    # Violation: target_r - val (positive when violated)
    violations = target_r - all_vals
    
    # Soft-min: sum of exp(-beta * violation) for violated constraints
    # Only consider constraints that are actually violated (violation > 0)
    # Use softplus-like: for each constraint, contribution is exp(-beta * max(0, target_r - val))
    # This is smooth and focuses on the most violated constraints
    
    # For numerical stability, clip violations
    clipped_viol = np.clip(violations, 0, 10.0)
    
    # Energy: sum of exp(-beta * clipped_violation)
    # This is 1 for satisfied constraints, < 1 for violated ones
    # We want to minimize the sum, so violated constraints contribute more
    energy_terms = np.exp(-beta * clipped_viol)
    energy = np.sum(energy_terms)
    
    # Gradient computation
    grad = np.zeros_like(pts)
    
    # Pair gradients
    # d/d(pts_i) of exp(-beta * (target_r - dist(i,j)/2))
    # = exp(-beta * v_ij) * beta * (1/2) * d(dist(i,j))/d(pts_i)
    # = exp(-beta * v_ij) * beta/2 * (pts_i - pts_j) / dist(i,j)
    # But only when violation > 0 (i.e., clipped_viol > 0)
    
    exp_terms = energy_terms[:len(pair_vals)]  # pair part
    mask = clipped_viol[:len(pair_vals)] > 0  # only violated pairs
    
    for idx in range(len(pair_vals)):
        if not mask[idx]:
            continue
        i, j = iu[0][idx], iu[1][idx]
        d = pts[i] - pts[j]
        dist_ij = np.linalg.norm(d)
        if dist_ij < 1e-10:
            continue
        weight = exp_terms[idx] * beta / 2.0 / dist_ij
        grad[i] += weight * d
        grad[j] -= weight * d
    
    # Wall gradients
    # d/d(pts_i) of exp(-beta * (target_r - (0.5 - max(|pts_i|))))
    # The max is non-smooth, use subgradient: direction toward the coordinate with max abs value
    wall_exp = energy_terms[len(pair_vals):]
    wall_mask = clipped_viol[len(pair_vals):] > 0
    
    for idx in range(len(wall_vals)):
        if not wall_mask[idx]:
            continue
        i = idx
        max_idx = np.argmax(abs_pts[i])
        sign = np.sign(pts[i, max_idx])
        if sign == 0:
            sign = 1.0
        weight = wall_exp[idx] * beta * sign
        grad[i, max_idx] -= weight
    
    return energy, grad


def _active_set_energy(pts, target_r, beta=20.0, n_active=20):
    """
    Active-set based smooth energy: only considers the tightest constraints.
    This creates a better-conditioned optimization problem by focusing on
    the most critical constraints, mimicking SOCP active-set methods.
    
    Returns (energy, gradient).
    """
    pts = np.array(pts, dtype=np.float64)
    n = len(pts)
    
    # Compute all pair distances / 2
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, 1e-10)
    pair_dists = np.sqrt(dist_sq) / 2.0
    np.fill_diagonal(pair_dists, np.inf)
    
    # Wall distances
    abs_pts = np.abs(pts)
    wall_dists = 0.5 - np.max(abs_pts, axis=1)
    
    # Find the n_active tightest pair constraints
    iu = np.triu_indices(n, k=1)
    pair_vals = pair_dists[iu]
    
    # Combine pair and wall constraints with their types
    # We'll select the tightest among all constraints
    all_vals = np.concatenate([pair_vals, wall_dists])
    
    # Select the n_active tightest (smallest) constraint values
    if len(all_vals) > n_active:
        # Get indices of the n_active smallest
        active_indices = np.argpartition(all_vals, n_active)[:n_active]
    else:
        active_indices = np.arange(len(all_vals))
    
    # Build energy from active constraints only
    active_vals = all_vals[active_indices]
    violations = target_r - active_vals
    clipped_viol = np.clip(violations, 0, 10.0)
    energy_terms = np.exp(-beta * clipped_viol)
    energy = np.sum(energy_terms)
    
    # Gradient
    grad = np.zeros_like(pts)
    
    # Map active indices back to pairs or walls
    n_pairs = len(pair_vals)
    for k in range(len(active_indices)):
        idx = active_indices[k]
        viol = clipped_viol[k]
        if viol <= 0:
            continue
        exp_term = energy_terms[k]
        
        if idx < n_pairs:
            # Pair constraint
            i, j = iu[0][idx], iu[1][idx]
            d = pts[i] - pts[j]
            dist_ij = np.linalg.norm(d)
            if dist_ij < 1e-10:
                continue
            weight = exp_term * beta / 2.0 / dist_ij
            grad[i] += weight * d
            grad[j] -= weight * d
        else:
            # Wall constraint
            i = idx - n_pairs
            max_idx = np.argmax(abs_pts[i])
            sign = np.sign(pts[i, max_idx])
            if sign == 0:
                sign = 1.0
            weight = exp_term * beta * sign
            grad[i, max_idx] -= weight
    
    return energy, grad


def _optimize_soft_min(pts, target_r, deadline, beta=20.0, max_iter=500, use_active_set=False):
    """
    Minimize soft-min violation energy using L-BFGS-B.
    Optionally uses active-set based energy for better conditioning.
    """
    pts = np.array(pts, dtype=np.float64).copy()
    n = len(pts)
    x0 = pts.flatten()
    
    if use_active_set:
        def objective(x):
            p = x.reshape(-1, 3)
            p = np.clip(p, -0.5, 0.5)
            e, _ = _active_set_energy(p, target_r, beta)
            return e
        
        def gradient(x):
            p = x.reshape(-1, 3)
            p = np.clip(p, -0.5, 0.5)
            _, g = _active_set_energy(p, target_r, beta)
            return g.flatten()
    else:
        def objective(x):
            p = x.reshape(-1, 3)
            p = np.clip(p, -0.5, 0.5)
            e, _ = _soft_min_energy(p, target_r, beta)
            return e
        
        def gradient(x):
            p = x.reshape(-1, 3)
            p = np.clip(p, -0.5, 0.5)
            _, g = _soft_min_energy(p, target_r, beta)
            return g.flatten()
    
    bounds = [(-0.5, 0.5)] * (n * 3)
    
    best_r = _compute_radius(pts)
    best_pts = pts.copy()
    
    res = minimize(
        objective, x0, method='L-BFGS-B', jac=gradient,
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
    )
    
    new_pts = res.x.reshape(-1, 3)
    new_pts = np.clip(new_pts, -0.5, 0.5)
    new_r = _compute_radius(new_pts)
    
    if new_r > best_r:
        best_r = new_r
        best_pts = new_pts
    
    return best_pts, best_r


def _iterative_soft_min(pts, deadline):
    """
    Iteratively increase target radius and minimize soft-min violation.
    Uses active-set based optimization for better conditioning, with progressive tightening.
    """
    pts = np.array(pts, dtype=np.float64).copy()
    current_r = _compute_radius(pts)
    best_pts = pts.copy()
    best_r = current_r
    
    # Start slightly above current radius
    target_r = current_r * 1.005
    max_target = 0.112  # Upper bound based on known best
    beta = 15.0
    
    # Adaptive beta: increase as we get closer to target
    while target_r < max_target and time.time() < deadline - 15:
        # Try to achieve target_r using active-set method
        opt_pts, opt_r = _optimize_soft_min(pts, target_r, deadline, beta=beta, max_iter=400, use_active_set=True)
        
        if opt_r > best_r:
            best_r = opt_r
            best_pts = opt_pts.copy()
            pts = opt_pts.copy()
            # Increase target: move toward the achieved radius
            target_r = min(opt_r * 1.003, max_target)
            # Increase beta for sharper focus
            beta = min(beta * 1.1, 50.0)
        else:
            # Try with full soft-min as fallback
            opt_pts2, opt_r2 = _optimize_soft_min(pts, target_r, deadline, beta=beta, max_iter=300, use_active_set=False)
            if opt_r2 > best_r:
                best_r = opt_r2
                best_pts = opt_pts2.copy()
                pts = opt_pts2.copy()
                target_r = min(opt_r2 * 1.003, max_target)
                beta = min(beta * 1.1, 50.0)
            else:
                # Didn't improve, try smaller step
                target_r = min(target_r * 1.001, max_target)
                # If stuck, break
                if target_r - current_r < 1e-7:
                    break
    
    return best_pts, best_r


def _local_polish(pts, deadline, max_iter=1000):
    """
    High-precision local polish using exact constraint-based gradient descent.
    Focuses on the tightest constraints with adaptive step sizes.
    Uses a multi-constraint approach: moves points away from all near-tight constraints simultaneously.
    """
    pts = np.array(pts, dtype=np.float64).copy()
    n = len(pts)
    best_r = _compute_radius(pts)
    best_pts = pts.copy()
    
    lr = 0.0005
    stall_count = 0
    tol = 1e-5  # Consider constraints within this of the minimum as "active"
    
    for it in range(max_iter):
        if time.time() > deadline - 2:
            break
        
        # Compute all pair distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(dist_sq, 1e-10)
        dist = np.sqrt(dist_sq)
        np.fill_diagonal(dist, np.inf)
        
        # Compute wall distances
        abs_pts = np.abs(pts)
        wall_dists = 0.5 - np.max(abs_pts, axis=1)
        
        # Find the minimum radius
        min_pair_dist = dist.min() / 2.0
        min_wall_dist = wall_dists.min()
        current_r = min(min_pair_dist, min_wall_dist)
        
        # Build gradient from all near-active constraints
        grad = np.zeros_like(pts)
        
        # Pair constraints that are near the minimum
        threshold = current_r + tol
        for i in range(n):
            for j in range(i + 1, n):
                d_val = dist[i, j] / 2.0
                if d_val <= threshold:
                    # Weight by how tight this constraint is
                    tightness = 1.0 - (d_val - current_r) / tol
                    d_vec = pts[i] - pts[j]
                    norm = np.linalg.norm(d_vec)
                    if norm > 1e-10:
                        unit = d_vec / norm
                        grad[i] += tightness * unit
                        grad[j] -= tightness * unit
        
        # Wall constraints that are near the minimum
        for i in range(n):
            if wall_dists[i] <= threshold:
                tightness = 1.0 - (wall_dists[i] - current_r) / tol
                max_idx = np.argmax(abs_pts[i])
                sign = np.sign(pts[i, max_idx])
                if sign == 0:
                    sign = 1.0
                grad[i, max_idx] -= tightness * sign
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-10:
            grad /= grad_norm
        
        # Apply update
        pts += lr * grad
        pts = np.clip(pts, -0.5, 0.5)
        
        r = _compute_radius(pts)
        if r > best_r + 1e-15:
            best_r = r
            best_pts = pts.copy()
            stall_count = 0
            lr = min(lr * 1.02, 0.003)
        else:
            pts = best_pts.copy()
            lr *= 0.85
            stall_count += 1
            if stall_count > 80:
                break
    
    return best_pts, best_r


def _anneal_and_refine(pts, deadline):
    """
    Simulated annealing with periodic local refinement.
    Uses larger moves and periodic re-optimization.
    """
    pts = np.array(pts, dtype=np.float64).copy()
    n = len(pts)
    best_r = _compute_radius(pts)
    best_pts = pts.copy()
    
    T = 0.015
    T_min = 0.0005
    cooling = 0.99
    step_size = 0.008
    max_steps = 5000
    
    step_count = 0
    while time.time() < deadline - 10 and T > T_min and step_count < max_steps:
        step_count += 1
        new_pts = pts.copy()
        idx = np.random.randint(n)
        delta = np.random.randn(3) * step_size
        new_pts[idx] += delta
        new_pts = np.clip(new_pts, -0.5, 0.5)
        
        new_r = _compute_radius(new_pts)
        delta_r = new_r - best_r
        
        if delta_r > 0 or np.random.rand() < np.exp(delta_r / T):
            pts = new_pts
            if new_r > best_r:
                best_r = new_r
                best_pts = new_pts.copy()
        
        T *= cooling
        
        # Periodic refinement
        if step_count % 200 == 0 and time.time() < deadline - 20:
            refined_pts, refined_r = _local_polish(pts, deadline)
            if refined_r > best_r:
                best_r = refined_r
                best_pts = refined_pts.copy()
                pts = refined_pts.copy()
    
    return best_pts, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    # Multiple initializations
    inits = []
    
    pts_grid, r_grid = _grid_init(N)
    inits.append((pts_grid, r_grid))
    
    pts_bcc, r_bcc = _bcc_init(N)
    inits.append((pts_bcc, r_bcc))
    
    for seed in [42, 123, 456, 789, 1000, 2000]:
        pts_rand, r_rand = _random_init(N, seed=seed)
        inits.append((pts_rand, r_rand))
    
    inits.sort(key=lambda x: x[1], reverse=True)
    
    best_pts = inits[0][0]
    best_r = inits[0][1]
    
    # Phase 1: Active-set soft-min optimization from best initializations
    for init_idx, (init_pts, init_r) in enumerate(inits[:4]):  # Top 4 inits
        if time.time() > deadline - 50:
            break
        
        if init_r < best_r * 0.80 and init_idx > 0:
            continue
        
        try:
            opt_pts, opt_r = _iterative_soft_min(init_pts, deadline)
            if opt_r > best_r:
                best_r = opt_r
                best_pts = opt_pts.copy()
        except Exception:
            continue
    
    # Phase 2: Additional passes with perturbed conditions
    if time.time() < deadline - 35:
        try:
            np.random.seed(12345)
            perturbed = best_pts + np.random.randn(N, 3) * 0.004
            perturbed = np.clip(perturbed, -0.5, 0.5)
            opt_pts, opt_r = _iterative_soft_min(perturbed, deadline)
            if opt_r > best_r:
                best_r = opt_r
                best_pts = opt_pts.copy()
        except Exception:
            pass
    
    # Phase 3: Simulated annealing for global exploration
    if time.time() < deadline - 25:
        try:
            anneal_pts, anneal_r = _anneal_and_refine(best_pts, deadline)
            if anneal_r > best_r:
                best_r = anneal_r
                best_pts = anneal_pts.copy()
        except Exception:
            pass
    
    # Phase 4: Final high-precision polish (multiple passes)
    if time.time() < deadline - 3:
        polished_pts, polished_r = _local_polish(best_pts, deadline, max_iter=2000)
        if polished_r > best_r:
            best_r = polished_r
            best_pts = polished_pts.copy()
    
    # Phase 5: One more polish pass if time allows
    if time.time() < deadline - 1:
        polished_pts, polished_r = _local_polish(best_pts, deadline, max_iter=500)
        if polished_r > best_r:
            best_r = polished_r
            best_pts = polished_pts.copy()
    
    # Final verification and clipping
    best_pts = np.clip(best_pts, -0.5, 0.5)
    
    return best_pts
# EVOLVE-BLOCK-END
