import numpy as np
import time
from scipy.optimize import minimize

N = 100

def _compute_radius(centers):
    """
    Compute the largest common radius for the given centers.
    Centers are in [-0.5, 0.5]^3.
    """
    # Distance to walls
    dists_to_walls = 0.5 - np.abs(centers)
    min_wall_dist = np.min(dists_to_walls)
    
    # Pairwise distances
    from scipy.spatial.distance import cdist
    dists = cdist(centers, centers, metric='euclidean')
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    return min(min_wall_dist, min_pair_dist / 2.0)

def _soft_min_objective(centers_flat, beta, iu):
    """
    Objective function for L-BFGS-B.
    Minimizes the soft-min approximation of the radius.
    """
    centers = centers_flat.reshape(-1, 3)
    
    # 1. Wall distances
    dists_to_walls = 0.5 - np.abs(centers)
    wall_vals = dists_to_walls.flatten()
    
    # 2. Pairwise distances / 2
    from scipy.spatial.distance import cdist
    dists = cdist(centers, centers, metric='euclidean')
    pair_vals = dists / 2.0
    pair_vals_unique = pair_vals[iu]
    
    all_vals = np.concatenate([wall_vals, pair_vals_unique])
    
    # Calculate LSE: 1/beta * ln(sum(exp(-beta * x)))
    neg_beta_x = -beta * all_vals
    max_val = np.max(neg_beta_x)
    sum_exp = np.sum(np.exp(neg_beta_x - max_val))
    obj = (max_val + np.log(sum_exp)) / beta
    
    return obj

def _build_grid_init(nx, ny, nz, noise_scale=0.0):
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
        return None
    
    if noise_scale > 0:
        rng = np.random.default_rng(int(np.sum(np.round(centers*1000))) % 10000)
        centers += rng.normal(0, noise_scale, centers.shape)
    
    centers = np.clip(centers, -0.499, 0.499)
    return centers

def _build_bcc_init():
    """
    Build a BCC lattice initialization.
    """
    a = (2.0 / N) ** (1/3.0)
    n_cells = int(np.ceil(1.0 / a)) + 2
    
    points = []
    for i in range(-1, n_cells):
        for j in range(-1, n_cells):
            for k in range(-1, n_cells):
                points.append([i*a, j*a, k*a])
                points.append([i*a + a/2, j*a + a/2, k*a + a/2])
    
    points = np.array(points)
    mask = np.all(np.abs(points) <= 0.5, axis=1)
    points = points[mask]
    
    if len(points) > N:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(points), N, replace=False)
        points = points[indices]
    elif len(points) < N:
        return _build_grid_init(5, 5, 4, 0.01)
        
    return points

def _build_sobol_init():
    """
    Build a Sobol sequence initialization for better space-filling.
    """
    from scipy.stats import qmc
    sampler = qmc.Sobol(d=3, scramble=True)
    # Get enough points
    m = 10
    points = sampler.random(2**m)  # 1024 points
    # Scale to [-0.5, 0.5]
    points = points * 1.0 - 0.5
    # Select N points with maximum minimum distance (greedy)
    selected = [points[0]]
    remaining = list(range(1, len(points)))
    
    while len(selected) < N and remaining:
        # For each remaining point, compute min distance to selected
        best_idx = -1
        best_min_dist = -1
        for idx in remaining:
            d = np.min(np.linalg.norm(points[idx] - np.array(selected), axis=1))
            if d > best_min_dist:
                best_min_dist = d
                best_idx = idx
        if best_idx >= 0:
            selected.append(points[best_idx])
            remaining.remove(best_idx)
    
    centers = np.array(selected)
    centers = np.clip(centers, -0.499, 0.499)
    return centers

def _build_halton_init():
    """
    Build a Halton sequence initialization.
    """
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    points = []
    for i in range(1, 1000):
        x = halton(i, 2)
        y = halton(i, 3)
        z = halton(i, 5)
        points.append([x - 0.5, y - 0.5, z - 0.5])
    
    points = np.array(points)
    
    # Select N points with maximum minimum distance (greedy)
    selected = [points[0]]
    remaining = list(range(1, len(points)))
    
    while len(selected) < N and remaining:
        best_idx = -1
        best_min_dist = -1
        for idx in remaining:
            d = np.min(np.linalg.norm(points[idx] - np.array(selected), axis=1))
            if d > best_min_dist:
                best_min_dist = d
                best_idx = idx
        if best_idx >= 0:
            selected.append(points[best_idx])
            remaining.remove(best_idx)
    
    centers = np.array(selected)
    centers = np.clip(centers, -0.499, 0.499)
    return centers

def _simulated_annealing(centers, deadline, iu, max_iter=2000):
    """
    Simulated annealing to escape local minima.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    current = centers.copy()
    
    T_start = 0.05
    T_end = 0.001
    n_steps = max_iter
    
    for i in range(n_steps):
        if time.time() > deadline - 2.0:
            break
        
        T = T_start * (T_end / T_start) ** (i / max(1, n_steps - 1))
        
        rng = np.random.default_rng(42 + i)
        perturbation = rng.normal(0, T, current.shape)
        candidate = np.clip(current + perturbation, -0.5, 0.5)
        
        r_candidate = _compute_radius(candidate)
        r_current = _compute_radius(current)
        
        if r_candidate >= r_current:
            current = candidate
        else:
            delta = r_current - r_candidate
            if delta > 0 and rng.random() < np.exp(-delta / max(T, 1e-10)):
                current = candidate
        
        if r_candidate > best_radius:
            best_radius = r_candidate
            best_centers = candidate.copy()
    
    return best_centers, best_radius

def _polish_repulsion(centers, deadline, iu):
    """
    Repulsion-based polish: push apart the closest pairs.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    for iteration in range(100):
        if time.time() > deadline - 3.0:
            break
        
        # Find the closest pair
        from scipy.spatial.distance import cdist
        dists = cdist(best_centers, best_centers, metric='euclidean')
        np.fill_diagonal(dists, np.inf)
        min_idx = np.unravel_index(np.argmin(dists), dists.shape)
        i, j = min_idx[0], min_idx[1]
        
        # Also check wall constraints
        wall_dists = 0.5 - np.abs(best_centers)
        min_wall_idx = np.unravel_index(np.argmin(wall_dists), wall_dists.shape)
        wall_i = min_wall_idx[0]
        
        # Push the closest pair apart
        diff = best_centers[i] - best_centers[j]
        norm = np.linalg.norm(diff)
        if norm > 1e-10:
            direction = diff / norm
            push = 0.005
            best_centers[i] += direction * push
            best_centers[j] -= direction * push
        
        # Push wall-violating spheres inward
        if wall_dists[wall_i, min_wall_idx[1]] < best_radius:
            axis = min_wall_idx[1]
            sign = 1 if best_centers[wall_i, axis] > 0 else -1
            best_centers[wall_i, axis] -= sign * 0.005
        
        best_centers = np.clip(best_centers, -0.5, 0.5)
        
        r = _compute_radius(best_centers)
        if r > best_radius:
            best_radius = r
    
    return best_centers, best_radius

def _polish_bottleneck(centers, deadline, iu):
    """
    Focus on the bottleneck constraints: optimize only the spheres involved in the minimum distance.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    for iteration in range(50):
        if time.time() > deadline - 3.0:
            break
        
        # Find bottleneck: closest pair and closest wall
        from scipy.spatial.distance import cdist
        dists = cdist(best_centers, best_centers, metric='euclidean')
        np.fill_diagonal(dists, np.inf)
        min_idx = np.unravel_index(np.argmin(dists), dists.shape)
        i, j = min_idx[0], min_idx[1]
        
        wall_dists = 0.5 - np.abs(best_centers)
        min_wall_idx = np.unravel_index(np.argmin(wall_dists), wall_dists.shape)
        wi = min_wall_idx[0]
        
        # Identify bottleneck spheres
        bottleneck = set([i, j, wi])
        bottleneck = list(bottleneck)
        
        if len(bottleneck) < 2:
            break
        
        # Optimize only these spheres
        def bottleneck_obj(x_flat):
            x = x_flat.reshape(-1, 3)
            full = best_centers.copy()
            for k, idx in enumerate(bottleneck):
                full[idx] = x[k]
            
            wall = 0.5 - np.abs(full)
            min_wall = np.min(wall)
            
            d = cdist(full, full, metric='euclidean')
            np.fill_diagonal(d, np.inf)
            min_pair = np.min(d) / 2.0
            
            return -min(min_wall, min_pair)
        
        x0 = np.array([best_centers[idx] for idx in bottleneck]).flatten()
        bounds = [(-0.5, 0.5)] * (len(bottleneck) * 3)
        
        try:
            res = minimize(bottleneck_obj, x0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 100, 'ftol': 1e-12})
            
            new_x = res.x.reshape(-1, 3)
            for k, idx in enumerate(bottleneck):
                best_centers[idx] = new_x[k]
            
            best_centers = np.clip(best_centers, -0.5, 0.5)
            r = _compute_radius(best_centers)
            if r > best_radius:
                best_radius = r
        except:
            pass
    
    return best_centers, best_radius

def _polish_high_beta(centers, deadline, iu):
    """
    High-precision polish pass with very high beta and tight tolerances.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    beta = 10000.0
    
    if time.time() > deadline - 3.0:
        return best_centers, best_radius
    
    try:
        res = minimize(
            _soft_min_objective,
            best_centers.flatten(),
            args=(beta, iu),
            method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (N * 3),
            options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-10, 'maxls': 30}
        )
        
        new_centers = res.x.reshape(-1, 3)
        new_centers = np.clip(new_centers, -0.5, 0.5)
        
        r = _compute_radius(new_centers)
        if r > best_radius:
            best_radius = r
            best_centers = new_centers
            
    except Exception:
        pass
    
    return best_centers, best_radius

def _polish_very_high_beta(centers, deadline, iu):
    """
    Very high beta polish for final refinement.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    for beta in [20000.0, 50000.0]:
        if time.time() > deadline - 2.0:
            break
        
        try:
            res = minimize(
                _soft_min_objective,
                best_centers.flatten(),
                args=(beta, iu),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 300, 'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 20}
            )
            
            new_centers = res.x.reshape(-1, 3)
            new_centers = np.clip(new_centers, -0.5, 0.5)
            
            r = _compute_radius(new_centers)
            if r > best_radius:
                best_radius = r
                best_centers = new_centers
                
        except Exception:
            pass
    
    return best_centers, best_radius

def _coordinate_descent_minmax(centers, deadline, iu):
    """
    Coordinate descent: optimize one coordinate at a time to maximize the minimum constraint.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    for iteration in range(200):
        if time.time() > deadline - 3.0:
            break
        
        improved = False
        for i in range(N):
            for axis in range(3):
                # Try small perturbations in each direction
                for delta in [0.001, 0.002, 0.005, -0.001, -0.002, -0.005]:
                    candidate = best_centers.copy()
                    candidate[i, axis] += delta
                    candidate[i, axis] = np.clip(candidate[i, axis], -0.5, 0.5)
                    
                    r = _compute_radius(candidate)
                    if r > best_radius + 1e-12:
                        best_radius = r
                        best_centers = candidate
                        improved = True
        
        if not improved:
            break
    
    return best_centers, best_radius

def _optimize_from_initial(centers, deadline, iu):
    """
    Run optimization from initial centers.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    # Beta schedule
    beta_start = 100.0
    beta_end = 5000.0
    n_beta_steps = 10
    
    if time.time() > deadline - 5.0:
        return best_centers, best_radius

    for step in range(n_beta_steps):
        if time.time() > deadline - 1.0:
            break
            
        if n_beta_steps == 1:
            beta = beta_start
        else:
            beta = beta_start * (beta_end / beta_start) ** (step / (n_beta_steps - 1))
        
        x0 = best_centers.flatten()
        
        try:
            res = minimize(
                _soft_min_objective,
                x0,
                args=(beta, iu),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-7, 'maxls': 20}
            )
            
            new_centers = res.x.reshape(-1, 3)
            new_centers = np.clip(new_centers, -0.5, 0.5)
            
            r = _compute_radius(new_centers)
            if r > best_radius:
                best_radius = r
                best_centers = new_centers
                
        except Exception:
            pass
            
    # Simulated annealing phase to escape local minima
    if time.time() < deadline - 10.0:
        sa_centers, sa_radius = _simulated_annealing(best_centers, deadline, iu, max_iter=1000)
        if sa_radius > best_radius:
            best_radius = sa_radius
            best_centers = sa_centers
    
    # Repulsion polish
    if time.time() < deadline - 8.0:
        rep_centers, rep_radius = _polish_repulsion(best_centers, deadline, iu)
        if rep_radius > best_radius:
            best_radius = rep_radius
            best_centers = rep_centers
    
    # Bottleneck polish
    if time.time() < deadline - 6.0:
        bot_centers, bot_radius = _polish_bottleneck(best_centers, deadline, iu)
        if bot_radius > best_radius:
            best_radius = bot_radius
            best_centers = bot_centers
    
    # High-precision polish
    if time.time() < deadline - 5.0:
        polish_centers, polish_radius = _polish_high_beta(best_centers, deadline, iu)
        if polish_radius > best_radius:
            best_radius = polish_radius
            best_centers = polish_centers
    
    # Coordinate descent
    if time.time() < deadline - 4.0:
        cd_centers, cd_radius = _coordinate_descent_minmax(best_centers, deadline, iu)
        if cd_radius > best_radius:
            best_radius = cd_radius
            best_centers = cd_centers
    
    # Very high beta final polish
    if time.time() < deadline - 3.0:
        vhb_centers, vhb_radius = _polish_very_high_beta(best_centers, deadline, iu)
        if vhb_radius > best_radius:
            best_radius = vhb_radius
            best_centers = vhb_centers
    
    return best_centers, best_radius

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    iu = np.triu_indices(N, k=1)
    
    best_centers = None
    best_radius = 0.0
    
    # 1. Grid Initializations
    grid_configs = [
        (4, 5, 5),
        (5, 4, 5),
        (5, 5, 4),
        (10, 10, 1),
        (10, 5, 2),
    ]
    
    noise_scales = [0.0, 0.01, 0.02]
    
    for nx, ny, nz in grid_configs:
        if time.time() > deadline - 10.0:
            break
        
        for noise in noise_scales:
            if time.time() > deadline - 5.0:
                break
            
            centers = _build_grid_init(nx, ny, nz, noise)
            if centers is None:
                continue
                
            r_init = _compute_radius(centers)
            
            if r_init > best_radius:
                best_radius = r_init
                best_centers = centers.copy()
            
            opt_centers, opt_radius = _optimize_from_initial(centers, deadline, iu)
            
            if opt_radius > best_radius:
                best_radius = opt_radius
                best_centers = opt_centers
    
    # 2. BCC Initialization
    if time.time() < deadline - 15.0:
        centers = _build_bcc_init()
        if centers is not None and len(centers) == N:
            r_init = _compute_radius(centers)
            if r_init > best_radius:
                best_radius = r_init
                best_centers = centers.copy()
            
            opt_centers, opt_radius = _optimize_from_initial(centers, deadline, iu)
            if opt_radius > best_radius:
                best_radius = opt_radius
                best_centers = opt_centers
    
    # 3. Sobol Initialization
    if time.time() < deadline - 15.0:
        centers = _build_sobol_init()
        if centers is not None and len(centers) == N:
            r_init = _compute_radius(centers)
            if r_init > best_radius:
                best_radius = r_init
                best_centers = centers.copy()
            
            opt_centers, opt_radius = _optimize_from_initial(centers, deadline, iu)
            if opt_radius > best_radius:
                best_radius = opt_radius
                best_centers = opt_centers
    
    # 4. Halton Initialization
    if time.time() < deadline - 15.0:
        centers = _build_halton_init()
        if centers is not None and len(centers) == N:
            r_init = _compute_radius(centers)
            if r_init > best_radius:
                best_radius = r_init
                best_centers = centers.copy()
            
            opt_centers, opt_radius = _optimize_from_initial(centers, deadline, iu)
            if opt_radius > best_radius:
                best_radius = opt_radius
                best_centers = opt_centers

    # 5. Random Initializations
    rng = np.random.default_rng(123)
    n_random_starts = 5
    
    if time.time() < deadline - 10.0:
        for i in range(n_random_starts):
            if time.time() > deadline - 5.0:
                break
            
            centers = rng.uniform(-0.45, 0.45, (N, 3))
            r_init = _compute_radius(centers)
            
            if r_init > best_radius:
                best_radius = r_init
                best_centers = centers.copy()
            
            opt_centers, opt_radius = _optimize_from_initial(centers, deadline, iu)
            
            if opt_radius > best_radius:
                best_radius = opt_radius
                best_centers = opt_centers

    # Final polish on the best solution
    if best_centers is not None and time.time() < deadline - 3.0:
        final_centers, final_radius = _polish_high_beta(best_centers, deadline, iu)
        if final_radius > best_radius:
            best_radius = final_radius
            best_centers = final_centers
        
        # Very high beta final
        if time.time() < deadline - 2.0:
            vhb_centers, vhb_radius = _polish_very_high_beta(best_centers, deadline, iu)
            if vhb_radius > best_radius:
                best_radius = vhb_radius
                best_centers = vhb_centers

    if best_centers is None:
        best_centers = _build_grid_init(5, 5, 4, 0.0)
    
    return best_centers
