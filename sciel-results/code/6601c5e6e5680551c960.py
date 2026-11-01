import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 100
_DEADLINE = None

def _get_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 145.0
    return _DEADLINE

def _time_left():
    return _get_deadline() - time.time()

def _compute_radius(centers):
    """
    Compute the largest common radius for the given centers.
    Centers are in [-0.5, 0.5]^3.
    """
    # Distance to walls
    dists_to_walls = 0.5 - np.abs(centers)
    min_wall_dist = np.min(dists_to_walls)
    
    # Pairwise distances using cKDTree for speed
    tree = cKDTree(centers)
    # We need the minimum pairwise distance.
    # Querying pairs with a large radius is expensive. 
    # Instead, we can find the nearest neighbor for each point and take the min.
    # However, query_pairs with a small radius might miss the global min if it's larger than the radius.
    # A safer bet for N=100 is to just compute all pairwise distances if N is small, 
    # or use query_pairs with a radius that is guaranteed to cover the min distance.
    # Since we don't know the min distance, let's use a heuristic:
    # The max possible distance is sqrt(3)/2 ~ 0.866.
    # But for N=100, O(N^2) is 10000 ops, which is fast.
    # Let's stick to cKDTree query_pairs with a generous radius to be safe, 
    # or just compute cdist if N is small enough.
    # For N=100, cdist is ~10k operations, very fast.
    
    # Let's use cKDTree.query_pairs with a radius that is likely to contain the min pair.
    # If we set r=0.5, we might miss pairs if the min dist > 0.5, which is unlikely for packing.
    # Actually, the min pair distance is what we want.
    # If we query_pairs(r=0.1), we might miss the min if it's 0.2.
    # So, let's just compute the full distance matrix for N=100. It's fast enough.
    # But wait, _compute_radius is called frequently in polish/SA.
    # cKDTree query_pairs(r=0.5) is better. If no pairs found, min_dist is inf (impossible for packing).
    # Let's assume the min pair dist is < 0.5.
    
    pairs = tree.query_pairs(r=0.5)
    if len(pairs) == 0:
        min_pair_dist = np.inf
    else:
        # Extract distances for these pairs
        # This is a bit slow in python loop.
        # Alternative: use tree.query to get nearest neighbor for each point.
        # The min of nearest neighbors is the min pair distance.
        # This is O(N log N) and guaranteed to find the global min pair dist.
        dists, _ = tree.query(centers, k=2)
        # dists[:, 0] is 0, dists[:, 1] is dist to nearest neighbor
        min_pair_dist = np.min(dists[:, 1])
    
    return min(min_wall_dist, min_pair_dist / 2.0)

def _soft_min_objective(centers_flat, beta):
    """
    Objective function: soft-min of wall distances and pairwise distances/2.
    """
    centers = centers_flat.reshape(-1, 3)
    
    # 1. Wall distances
    dists_to_walls = 0.5 - np.abs(centers)
    wall_vals = dists_to_walls.flatten()
    
    # 2. Pairwise distances / 2
    # For the objective, we need all pairwise distances.
    # N=100, so 100x100 matrix is small.
    # Using cdist is efficient.
    from scipy.spatial.distance import cdist
    dists = cdist(centers, centers, metric='euclidean')
    # We only need upper triangle
    iu = np.triu_indices(N, k=1)
    pair_vals = dists[iu] / 2.0
    
    all_vals = np.concatenate([wall_vals, pair_vals])
    
    # Soft min: -1/beta * log(sum(exp(-beta * x)))
    # Numerical stability: subtract max
    neg_beta_x = -beta * all_vals
    max_val = np.max(neg_beta_x)
    sum_exp = np.sum(np.exp(neg_beta_x - max_val))
    obj = -(max_val + np.log(sum_exp)) / beta
    
    return obj

def _build_bcc_init():
    """
    Build a BCC (body-centered cubic) lattice initialization.
    """
    # 5x5x4 grid of BCC cells? 5*5*4 = 100 cells. Each cell has 2 points? No, BCC has 2 points per cell.
    # 100 points total.
    # Let's use a 5x5x2 grid of cells, each with 2 points (corner and center)?
    # Or just a regular grid with noise.
    # The seed used 5x5x2 cells, 2 points each -> 100 points.
    
    nx, ny, nz = 5, 5, 2
    points = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                cx = -0.5 + (i + 0.5) * (1.0 / nx)
                cy = -0.5 + (j + 0.5) * (1.0 / ny)
                cz = -0.5 + (k + 0.5) * (1.0 / nz)
                # Point 1: Center of the sub-cell
                points.append([cx, cy, cz])
                # Point 2: Offset
                # In BCC, the second point is at the body center.
                # Here, let's just use a small offset to break symmetry.
                points.append([cx + 0.25/nx, cy + 0.25/ny, cz + 0.25/nz])
    
    centers = np.array(points[:N])
    
    # Add slight noise
    rng = np.random.default_rng(42)
    centers += rng.normal(0, 0.005, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    
    return centers

def _build_halton_init(seed=42):
    """
    Generate N points using a Halton sequence (low discrepancy).
    """
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r

    bases = [2, 3, 5]
    points = []
    for i in range(1, N + 1):
        pt = [halton(i, b) for b in bases]
        pt_mapped = [p - 0.5 for p in pt]
        points.append(pt_mapped)
        
    centers = np.array(points)
    
    # Add slight noise to break symmetry
    rng = np.random.default_rng(seed)
    centers += rng.normal(0, 0.01, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    
    return centers

def _build_grid_init(nx, ny, nz, noise_scale=0.0, seed=42):
    """Build a grid initialization."""
    if nx == 1: xs = np.array([0.0])
    else: xs = np.linspace(-0.5, 0.5, nx)
    
    if ny == 1: ys = np.array([0.0])
    else: ys = np.linspace(-0.5, 0.5, ny)
    
    if nz == 1: zs = np.array([0.0])
    else: zs = np.linspace(-0.5, 0.5, nz)
    
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    centers = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    
    if len(centers) != N:
        if len(centers) < N:
            rng = np.random.default_rng(seed)
            extra = rng.uniform(-0.45, 0.45, (N - len(centers), 3))
            centers = np.vstack([centers, extra])
        else:
            centers = centers[:N]

    if noise_scale > 0:
        rng = np.random.default_rng(seed)
        centers += rng.normal(0, noise_scale, centers.shape)
    
    centers = np.clip(centers, -0.499, 0.499)
    return centers

def _polish_repulsion(centers, deadline, iu):
    """
    Local search using repulsion forces with KD-tree acceleration.
    """
    centers = centers.copy()
    r_current = _compute_radius(centers)
    best_r = r_current
    best_centers = centers.copy()
    
    step_size = 0.003
    decay = 0.97
    min_step = 1e-7
    margin = 1.05
    
    no_improve_count = 0
    max_no_improve = 100
    
    while _time_left() > 1.0 and step_size > min_step and no_improve_count < max_no_improve:
        r = _compute_radius(centers)
        if r > best_r + 1e-12:
            best_r = r
            best_centers = centers.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # Compute forces
        forces = np.zeros_like(centers)
        
        # Wall forces
        for k in range(3):
            dists = 0.5 - np.abs(centers[:, k])
            limit = max(r_current * margin, 0.01)
            mask = dists < limit
            if np.any(mask):
                dirs = -np.sign(centers[mask, k])
                dirs[dirs == 0] = 1.0
                strengths = (limit - dists[mask]) / limit
                forces[mask, k] += dirs * strengths
                
        # Pairwise forces using KD-tree for efficiency
        tree = cKDTree(centers)
        search_radius = 2 * max(r_current * margin, 0.01)
        
        # Query neighbors
        neighbor_pairs = tree.query_pairs(r=search_radius)
        
        for i, j in neighbor_pairs:
            if i == j:
                continue
            diff = centers[i] - centers[j]
            dist = np.linalg.norm(diff)
            if dist > 1e-9 and dist < search_radius:
                strength = (search_radius - dist) / search_radius
                unit_diff = diff / dist
                forces[i] += strength * unit_diff
                forces[j] -= strength * unit_diff
        
        # Normalize forces
        norms = np.linalg.norm(forces, axis=1, keepdims=True)
        norms[norms == 0] = 1
        forces = forces / norms
        
        centers += step_size * forces
        centers = np.clip(centers, -0.4999, 0.4999)
        
        step_size *= decay
        
    return best_centers, best_r

def _optimize_from_initial(centers, deadline, time_budget=20.0):
    """
    Run optimization from initial centers.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    start_t = time.time()
    remaining = time_budget
    
    # Phase 1: Soft-min optimization with adaptive beta
    beta_start = 30.0
    beta_end = 3000.0
    n_beta_steps = 6
    
    if remaining > 8.0:
        for step in range(n_beta_steps):
            if _time_left() < 2.0 or (time.time() - start_t) > remaining - 2.0:
                break
                
            beta = beta_start * (beta_end / beta_start) ** (step / (n_beta_steps - 1))
            
            x0 = best_centers.flatten()
            
            try:
                max_iter = 150 if step < 3 else 80
                res = minimize(
                    _soft_min_objective,
                    x0,
                    args=(beta,),
                    method='L-BFGS-B',
                    bounds=[(-0.4999, 0.4999)] * (N * 3),
                    options={'maxiter': max_iter, 'ftol': 1e-13, 'gtol': 1e-9, 'maxls': 20}
                )
                
                new_centers = res.x.reshape(-1, 3)
                new_centers = np.clip(new_centers, -0.4999, 0.4999)
                
                r = _compute_radius(new_centers)
                if r > best_radius:
                    best_radius = r
                    best_centers = new_centers
                    
            except Exception:
                pass
    
    # Phase 2: Simulated Annealing
    if _time_left() > 5.0 and (time.time() - start_t) < remaining - 5.0:
        sa_centers, sa_radius = _simulated_annealing(best_centers, best_radius, deadline, budget=remaining - 5.0)
        if sa_radius > best_radius:
            best_radius = sa_radius
            best_centers = sa_centers
            
    # Phase 3: Polish with repulsion
    if _time_left() > 2.0 and (time.time() - start_t) < remaining:
        polish_deadline = min(deadline, start_t + remaining)
        polish_centers, polish_radius = _polish_repulsion(best_centers, polish_deadline, None)
        if polish_radius > best_radius:
            best_radius = polish_radius
            best_centers = polish_centers
            
    return best_centers, best_radius

def _simulated_annealing(centers, current_r, deadline, budget=10.0):
    """
    Simulated annealing to escape local minima.
    """
    centers = centers.copy()
    best_centers = centers.copy()
    best_r = current_r
    
    T = 0.05
    T_min = 1e-6
    cooling_rate = 0.99
    max_iter = 10000
    
    rng = np.random.default_rng(123)
    
    start_t = time.time()
    
    for i in range(max_iter):
        if _time_left() < 1.0 or (time.time() - start_t) > budget:
            break
            
        # Generate a random perturbation
        # Move a random subset of points
        n_move = rng.integers(1, 5)
        indices = rng.choice(N, n_move, replace=False)
        
        # Random displacement
        displacement = rng.normal(0, 0.01, (n_move, 3))
        
        new_centers = centers.copy()
        new_centers[indices] += displacement
        new_centers = np.clip(new_centers, -0.4999, 0.4999)
        
        new_r = _compute_radius(new_centers)
        
        # Acceptance criterion
        if new_r > current_r:
            current_r = new_r
            centers = new_centers
            if new_r > best_r:
                best_r = new_r
                best_centers = new_centers.copy()
        else:
            # Metropolis criterion
            delta = current_r - new_r
            if delta < 0:
                accept = True
            else:
                accept = rng.random() < np.exp(-delta / T)
                
            if accept:
                current_r = new_r
                centers = new_centers
                
        T *= cooling_rate
        
    return best_centers, best_r

def construct():
    global _DEADLINE
    _DEADLINE = time.time() + 145.0
    
    best_centers = None
    best_radius = 0.0
    
    def try_init(centers, time_budget=18.0):
        nonlocal best_centers, best_radius
        r_init = _compute_radius(centers)
        if r_init > best_radius:
            best_radius = r_init
            best_centers = centers.copy()
        
        if _time_left() > time_budget + 2.0:
            opt_centers, opt_radius = _optimize_from_initial(centers, _DEADLINE, time_budget)
            if opt_radius > best_radius:
                best_radius = opt_radius
                best_centers = opt_centers
        else:
            if _time_left() > 3.0:
                polish_centers, polish_radius = _polish_repulsion(centers, _DEADLINE, None)
                if polish_radius > best_radius:
                    best_radius = polish_radius
                    best_centers = polish_centers
    
    # Initializations
    if _time_left() > 25.0:
        centers = _build_bcc_init()
        try_init(centers, time_budget=25.0)
    
    grid_configs = [
        (5, 5, 4),
        (4, 5, 5),
        (5, 4, 5),
    ]
    
    for nx, ny, nz in grid_configs:
        if _time_left() < 25.0:
            break
        for noise in [0.0, 0.015]:
            if _time_left() < 20.0:
                break
            centers = _build_grid_init(nx, ny, nz, noise, seed=nx*100+ny*10+nz)
            try_init(centers, time_budget=18.0)
    
    if _time_left() > 20.0:
        centers = _build_halton_init()
        try_init(centers, time_budget=18.0)
    
    rng = np.random.default_rng(123)
    for seed in [123, 456, 789]:
        if _time_left() < 20.0:
            break
        centers = rng.uniform(-0.45, 0.45, (N, 3))
        try_init(centers, time_budget=15.0)
    
    # Final high-precision polish
    if best_centers is not None and _time_left() > 3.0:
        # Use a very high beta for the final polish
        x0 = best_centers.flatten()
        beta = 200000.0
        try:
            res = minimize(
                _soft_min_objective,
                x0,
                args=(beta,),
                method='L-BFGS-B',
                bounds=[(-0.4999, 0.4999)] * (N * 3),
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
            )
            new_centers = res.x.reshape(-1, 3)
            new_centers = np.clip(new_centers, -0.4999, 0.4999)
            r = _compute_radius(new_centers)
            if r > best_radius:
                best_radius = r
                best_centers = new_centers
        except Exception:
            pass
        
        # Also try repulsion polish
        if _time_left() > 2.0:
            polish_centers, polish_radius = _polish_repulsion(best_centers, _DEADLINE, None)
            if polish_radius > best_radius:
                best_radius = polish_radius
                best_centers = polish_centers
    
    if best_centers is None:
        best_centers = _build_grid_init(5, 5, 4, 0.0)
        
    return best_centers
