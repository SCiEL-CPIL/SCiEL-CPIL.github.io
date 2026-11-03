"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, linprog
import time

def _compute_score(centers):
    """Compute the radius score for given centers."""
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    boundary = 0.5 - np.abs(centers)
    min_boundary = np.min(boundary)
    return min(min_pair, min_boundary)

def _active_constrained_objective(x, active_pairs, active_bounds, sigma):
    """Objective and gradient using only active constraints (closest pairs + boundary)."""
    centers = x.reshape(N, 3)
    
    # Compute distances for active pairs only
    pair_vals = np.zeros(len(active_pairs))
    pair_grads = np.zeros((len(active_pairs), N, 3))
    
    for k, (i, j) in enumerate(active_pairs):
        d = centers[i] - centers[j]
        dist = np.sqrt(np.sum(d**2))
        if dist < 1e-12:
            pair_vals[k] = 1e-12
            continue
        pair_vals[k] = dist / 2.0
        unit = d / (2.0 * dist)
        pair_grads[k, i] = unit
        pair_grads[k, j] = -unit
    
    # Boundary constraints
    bound_vals = np.zeros(len(active_bounds))
    bound_grads = np.zeros((len(active_bounds), N, 3))
    for k, (idx, axis) in enumerate(active_bounds):
        bound_vals[k] = 0.5 - abs(centers[idx, axis])
        bound_grads[k, idx, axis] = -np.sign(centers[idx, axis])
    
    # Combine all constraints
    all_vals = np.concatenate([pair_vals, bound_vals])
    all_grads = np.concatenate([pair_grads, bound_grads], axis=0)
    
    # Softmin
    scaled = -all_vals / sigma
    max_scaled = np.max(scaled)
    exp_vals = np.exp(scaled - max_scaled)
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp
    
    softmin_val = -sigma * (max_scaled + np.log(sum_exp))
    objective = -softmin_val
    
    # Gradient: sum of weighted gradients
    grad = np.zeros_like(x)
    for k in range(len(all_vals)):
        grad += weights[k] * all_grads[k].ravel()
    
    return objective, grad

def _find_active_constraints(centers, n_pairs=100, n_bounds=20):
    """Find the n_pairs closest pairs and n_bounds closest-to-boundary indices."""
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Get n_pairs closest pairs
    flat_idx = np.argsort(dists.ravel())[:n_pairs * 2]
    pairs = []
    for idx in flat_idx:
        i, j = np.unravel_index(idx, (N, N))
        if i < j:
            pairs.append((i, j))
        elif j < i:
            pairs.append((j, i))
        if len(pairs) >= n_pairs:
            break
    
    # Remove duplicates
    seen = set()
    unique_pairs = []
    for p in pairs:
        key = (min(p), max(p))
        if key not in seen:
            seen.add(key)
            unique_pairs.append(key)
    unique_pairs = unique_pairs[:n_pairs]
    
    # Find closest to boundary - return (index, axis) pairs
    boundary_dists = 0.5 - np.abs(centers)  # shape (N, 3)
    flat_bd = boundary_dists.ravel()
    bd_sorted = np.argsort(flat_bd)[:n_bounds]
    active_bounds = []
    for idx in bd_sorted:
        i, k = np.unravel_index(idx, (N, 3))
        active_bounds.append((i, k))
    
    return unique_pairs, active_bounds

def _direct_minimax_polish(x, max_iter=200, tol=1e-12, deadline=145.0):
    """Direct minimax polish: maximize r such that all constraints >= r."""
    centers = x.reshape(N, 3).copy()
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        # Find current minimum constraint value
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair_val = np.min(dists) / 2.0
        boundary_val = np.min(0.5 - np.abs(centers))
        r = min(min_pair_val, boundary_val)
        
        # Find all constraints that are close to the minimum (within 10%)
        threshold = r * 1.1
        
        # Active pairs
        active_pairs = []
        for i in range(N):
            for j in range(i+1, N):
                d = centers[i] - centers[j]
                dist = np.sqrt(np.sum(d**2))
                if dist / 2.0 <= threshold:
                    active_pairs.append((i, j))
        
        # Active bounds
        active_bounds = []
        for i in range(N):
            for k in range(3):
                if 0.5 - abs(centers[i, k]) <= threshold:
                    active_bounds.append((i, k))
        
        if len(active_pairs) == 0 and len(active_bounds) == 0:
            break
        
        # Build constraint Jacobian and values
        m = len(active_pairs) + len(active_bounds)
        if m == 0:
            break
        
        A = np.zeros((m, N * 3))
        b = np.zeros(m)
        
        row = 0
        for i, j in active_pairs:
            d = centers[i] - centers[j]
            dist = np.sqrt(np.sum(d**2))
            if dist < 1e-12:
                continue
            unit = d / (2.0 * dist)
            A[row, i*3:(i+1)*3] = unit
            A[row, j*3:(j+1)*3] = -unit
            b[row] = dist / 2.0
            row += 1
        
        for i, k in active_bounds:
            A[row, i*3 + k] = -np.sign(centers[i, k])
            b[row] = 0.5 - abs(centers[i, k])
            row += 1
        
        m = row
        if m == 0:
            break
        
        # Solve: maximize r subject to A @ x + r * ones >= b, |x| <= 0.5
        # Equivalent: minimize -r s.t. -A @ x + r >= -b, -0.5 <= x <= 0.5
        # Use scipy linprog
        c = np.zeros(N * 3 + 1)
        c[-1] = -1  # maximize r => minimize -r
        
        # Inequality constraints: -A @ x + r >= -b  =>  A @ x - r <= b
        A_ub = np.hstack([A, -np.ones((m, 1))])
        b_ub = b
        
        # Bounds
        bounds = [(-0.5, 0.5)] * (N * 3) + [(-0.1, 0.5)]
        
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                new_x = res.x[:N*3]
                new_r = res.x[-1]
                if new_r > r - 1e-10:
                    centers = new_x.reshape(N, 3)
                    centers = np.clip(centers, -0.5, 0.5)
                else:
                    break
            else:
                break
        except Exception:
            break
        
        # Check convergence
        new_dists = np.sqrt(np.sum((centers[:, np.newaxis, :] - centers[np.newaxis, :, :])**2, axis=2))
        np.fill_diagonal(new_dists, np.inf)
        new_r = min(np.min(new_dists) / 2.0, np.min(0.5 - np.abs(centers)))
        if abs(new_r - r) < tol:
            break
    
    return centers.reshape(-1)

def _make_bcc_initial():
    """Generate BCC lattice initialization for 50 spheres."""
    points = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                points.append((i/3 - 0.5, j/3 - 0.5, k/3 - 0.5))
                points.append((i/3 - 0.5 + 1/6, j/3 - 0.5 + 1/6, k/3 - 0.5 + 1/6))
    points = np.array(points)
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(points))
    return points[perm[:N]].copy()

def _make_fcc_initial():
    """Generate FCC lattice initialization for 50 spheres."""
    points = []
    a = 1.0 / 2.0  # cube side is 1, FCC lattice constant
    # FCC points in unit cell: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)
    fcc_offsets = [(0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for dx, dy, dz in fcc_offsets:
                    x = (i + dx) / 3.0 - 0.5
                    y = (j + dy) / 3.0 - 0.5
                    z = (k + dz) / 3.0 - 0.5
                    if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5 and -0.5 <= z <= 0.5:
                        points.append((x, y, z))
    points = np.array(points)
    rng = np.random.default_rng(7)
    perm = rng.permutation(len(points))
    return points[perm[:N]].copy()

def _make_hcp_initial():
    """Generate HCP-like initialization for 50 spheres."""
    points = []
    # Create layers of hexagonal packing
    layer_y = [0, 1/3, 2/3]
    for layer_idx, y_base in enumerate(layer_y):
        for i in range(4):
            for j in range(4):
                x = (i + 0.5) / 4.0 - 0.5
                z = (j + 0.5) / 4.0 - 0.5
                if layer_idx % 2 == 1:
                    x += 0.125
                y = y_base - 0.5 + 0.5 / 3.0
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5 and -0.5 <= z <= 0.5:
                    points.append((x, y, z))
    points = np.array(points)
    rng = np.random.default_rng(13)
    perm = rng.permutation(len(points))
    return points[perm[:N]].copy()

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # seconds, leaving 5s margin for safety
    
    best_score = -1
    best_centers = None
    
    rng = np.random.default_rng(42)
    
    # Generate initial configurations
    configs = []
    
    # Config 1: BCC lattice
    configs.append(_make_bcc_initial())
    
    # Config 2: FCC lattice
    configs.append(_make_fcc_initial())
    
    # Config 3: HCP lattice
    configs.append(_make_hcp_initial())
    
    # Config 4: 4x4x4 grid subset
    ticks = (np.arange(4) + 0.5) / 4.0 - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    perm = rng.permutation(64)
    kept = perm[:50]
    configs.append(grid[kept].copy())
    
    # Config 5: Random
    configs.append(rng.uniform(-0.42, 0.42, (N, 3)))
    
    # Config 6: Perturbed BCC
    bcc = _make_bcc_initial()
    configs.append(np.clip(bcc + rng.normal(0, 0.03, (N, 3)), -0.48, 0.48))
    
    # Config 7: Perturbed FCC
    fcc = _make_fcc_initial()
    configs.append(np.clip(fcc + rng.normal(0, 0.03, (N, 3)), -0.48, 0.48))
    
    # Config 8: Random with different seed
    rng2 = np.random.default_rng(123)
    configs.append(rng2.uniform(-0.42, 0.42, (N, 3)))
    
    # Config 9: Perturbed grid
    ticks6 = (np.arange(4) + 0.5) / 4.0 - 0.5
    grid6 = np.array([(x, y, z) for z in ticks6 for y in ticks6 for x in ticks6])
    rng6 = np.random.default_rng(999)
    perm6 = rng6.permutation(64)
    kept6 = perm6[:50]
    configs.append(np.clip(grid6[kept6] + rng6.normal(0, 0.02, (N, 3)), -0.48, 0.48))
    
    # Config 10: Perturbed HCP
    hcp = _make_hcp_initial()
    configs.append(np.clip(hcp + rng.normal(0, 0.03, (N, 3)), -0.48, 0.48))
    
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    for config in configs:
        if time.time() - start_time > deadline - 12:
            break
        
        x = config.reshape(-1).copy()
        
        # Phase 1: Multi-phase annealing with decreasing sigma using active constraints
        sigmas = [0.08, 0.04, 0.02, 0.01, 0.005, 0.002, 0.001]
        
        for sigma in sigmas:
            if time.time() - start_time > deadline - 6:
                break
            
            centers = x.reshape(N, 3)
            active_pairs, active_bounds = _find_active_constraints(centers, n_pairs=80, n_bounds=15)
            
            if len(active_pairs) == 0 and len(active_bounds) == 0:
                continue
            
            try:
                result = minimize(
                    lambda x, s=sigma, ap=active_pairs, ab=active_bounds: _active_constrained_objective(x, ap, ab, s)[0],
                    x,
                    jac=lambda x, s=sigma, ap=active_pairs, ab=active_bounds: _active_constrained_objective(x, ap, ab, s)[1],
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 40}
                )
                x = result.x
            except Exception:
                pass
        
        # Phase 2: Direct minimax polish
        if time.time() - start_time < deadline - 4:
            try:
                x = _direct_minimax_polish(x, max_iter=150, deadline=deadline)
            except Exception:
                pass
        
        centers = x.reshape(N, 3)
        centers = np.clip(centers, -0.5, 0.5)
        score = _compute_score(centers)
        
        if score > best_score:
            best_score = score
            best_centers = centers.copy()
    
    # Fallback
    if best_centers is None or best_score < 0.12:
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N]
    
    return best_centers

# EVOLVE-BLOCK-END
