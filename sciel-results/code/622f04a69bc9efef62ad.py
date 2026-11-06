"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt


def _radius(centers):
    """Compute the maximum common radius for given centers (vectorized)."""
    n = len(centers)
    if n < 2:
        wall = 0.5 - np.max(np.abs(centers), axis=1)
        return np.min(wall)
    # Pairwise distances (upper triangle)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(d2, np.inf)
    min_pair_dist = np.sqrt(np.min(d2))
    # Distance to walls
    wall = 0.5 - np.max(np.abs(centers), axis=1)
    min_wall = np.min(wall)
    return min(min_pair_dist / 2.0, min_wall)

def _soft_radius(centers, beta):
    """
    Compute a smooth approximation of the radius using log-sum-exp.
    We want to maximize min(pairwise_dist/2, wall_dist).
    soft_min(d) = -(1/beta) * log(sum(exp(-beta * d_i)))
    We want to maximize soft_min(d).
    So we minimize -soft_min(d).
    """
    n = len(centers)
    # Pairwise distances / 2
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(d2, np.inf)
    pair_dists = np.sqrt(d2) / 2.0
    
    # Wall distances
    wall_dists = 0.5 - np.abs(centers)
    
    # Combine all distances into one array
    # pair_dists is (n, n), wall_dists is (n, 3)
    # We need to flatten and concatenate
    pair_flat = pair_dists[np.triu_indices(n, k=1)]
    wall_flat = wall_dists.flatten()
    
    all_dists = np.concatenate([pair_flat, wall_flat])
    
    # Soft min: -(1/beta) * logsumexp(-beta * all_dists)
    # logsumexp(x) = max(x) + log(sum(exp(x - max(x))))
    x = -beta * all_dists
    x_max = np.max(x)
    # Handle case where x_max is very large (beta very large, distances very small)
    if x_max > 700:
        # Avoid overflow
        log_sum = x_max + np.log(np.sum(np.exp(x - x_max)))
    else:
        log_sum = x_max + np.log(np.sum(np.exp(x - x_max)))
    
    soft_min_val = -log_sum / beta
    return soft_min_val

def _objective_smooth(x, beta):
    """Objective function for smooth optimization."""
    centers = x.reshape(N, 3)
    # Penalty for going out of bounds
    penalty = 0.0
    for i in range(N):
        for j in range(3):
            if centers[i, j] > 0.5:
                penalty += (centers[i, j] - 0.5) ** 2
            elif centers[i, j] < -0.5:
                penalty += (centers[i, j] + 0.5) ** 2
    return -_soft_radius(centers, beta) + 100.0 * penalty

def _bcc_init(n_points):
    """Generate initial points on a distorted BCC lattice."""
    points = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                ox = -0.5 + i * (1.0/3.0)
                oy = -0.5 + j * (1.0/3.0)
                oz = -0.5 + k * (1.0/3.0)
                
                points.append([ox, oy, oz])
                points.append([ox + 1.0/6.0, oy + 1.0/6.0, oz + 1.0/6.0])
                
    points = np.array(points)
    return points[:n_points].copy()

def _fcc_init(n_points):
    """Generate initial points on an FCC lattice."""
    # FCC lattice points in a unit cell: corners and face centers
    points = []
    # 3x3x3 grid of cells
    for i in range(3):
        for j in range(3):
            for k in range(3):
                ox = -0.5 + i * (1.0/3.0)
                oy = -0.5 + j * (1.0/3.0)
                oz = -0.5 + k * (1.0/3.0)
                
                # Corner
                points.append([ox, oy, oz])
                # Face centers
                points.append([ox + 1.0/6.0, oy + 1.0/6.0, oz])
                points.append([ox + 1.0/6.0, oy, oz + 1.0/6.0])
                points.append([ox, oy + 1.0/6.0, oz + 1.0/6.0])
                
    points = np.array(points)
    # We have 3*3*3*4 = 108 points, need 50
    # Take first 50
    return points[:n_points].copy()

def _random_init(n_points, seed=None):
    """Generate random initial points."""
    if seed is not None:
        np.random.seed(seed)
    return np.random.uniform(-0.45, 0.45, (n_points, 3))

def _sa_optimize(centers, time_budget):
    """Simulated Annealing optimization."""
    start = time.time()
    centers = centers.copy()
    n = len(centers)
    current_r = _radius(centers)
    best_r = current_r
    best_centers = centers.copy()
    
    T = 0.005
    T_min = 1e-5
    step_size = 0.01
    decay = 0.9995
    
    while time.time() - start < time_budget:
        # Pick a random point
        i = np.random.randint(n)
        
        # Perturb
        delta = np.random.normal(0, step_size, 3)
        new_center = centers[i].copy()
        new_center += delta
        new_center = np.clip(new_center, -0.5, 0.5)
        
        # Calculate new radius
        new_centers = centers.copy()
        new_centers[i] = new_center
        new_r = _radius(new_centers)
        
        # Acceptance criterion
        if new_r >= current_r:
            centers = new_centers
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_centers = centers.copy()
        else:
            if np.random.rand() < np.exp((new_r - current_r) / T):
                centers = new_centers
                current_r = new_r
                if current_r > best_r:
                    best_r = current_r
                    best_centers = centers.copy()
        
        T *= decay
        step_size *= decay
        if T < T_min:
            T = T_min
            step_size = max(step_size, 0.001)
            
    return best_centers, best_r

def _local_optimize_smooth(centers, time_budget):
    """Local optimization using smooth objective with adaptive beta."""
    start = time.time()
    centers = centers.copy()
    x0 = centers.flatten()
    
    best_centers = centers.copy()
    best_r = _radius(best_centers)
    
    # Adaptive beta: start with small beta (smooth), increase over time
    beta_start = 50.0
    beta_end = 500.0
    n_iter = 50
    
    for i in range(n_iter):
        if time.time() - start > time_budget:
            break
        
        # Interpolate beta
        progress = i / n_iter
        beta = beta_start + (beta_end - beta_start) * progress
        
        # Use Nelder-Mead for a few iterations
        result = opt.minimize(_objective_smooth, x0, args=(beta,), 
                             method='Nelder-Mead', 
                             options={'maxiter': 20, 'xatol': 1e-8, 'fatol': 1e-8})
        
        new_centers = result.x.reshape(N, 3)
        # Clip to ensure validity
        new_centers = np.clip(new_centers, -0.5, 0.5)
        
        new_r = _radius(new_centers)
        if new_r > best_r:
            best_r = new_r
            best_centers = new_centers.copy()
            x0 = new_centers.flatten()
    
    return best_centers, best_r

def _coordinate_descent(centers, time_budget):
    """Coordinate Descent polish."""
    start = time.time()
    centers = centers.copy()
    n = len(centers)
    current_r = _radius(centers)
    best_r = current_r
    best_centers = centers.copy()
    
    while time.time() - start < time_budget:
        improved = False
        for i in range(n):
            for d in range(3):
                # Optimize coordinate d of point i
                val = centers[i, d]
                
                candidates = np.linspace(-0.5, 0.5, 30)
                best_val = val
                best_r_local = current_r
                
                for c_val in candidates:
                    new_centers = centers.copy()
                    new_centers[i, d] = c_val
                    r = _radius(new_centers)
                    if r > best_r_local:
                        best_r_local = r
                        best_val = c_val
                
                if best_val != val:
                    centers[i, d] = best_val
                    current_r = best_r_local
                    improved = True
                    if current_r > best_r:
                        best_r = current_r
                        best_centers = centers.copy()
        
        if not improved:
            break
            
    return best_centers, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    best_centers = None
    best_r = -1.0

    # Try multiple initializations
    inits = [
        ("bcc", _bcc_init(N)),
        ("fcc", _fcc_init(N)),
        ("rand1", _random_init(N, seed=42)),
        ("rand2", _random_init(N, seed=123)),
    ]
    
    for name, init_centers in inits:
        if time.time() - start_time > deadline - 30:
            break
        
        current_best = init_centers.copy()
        current_r = _radius(current_best)
        
        # Phase 1: Simulated Annealing
        if time.time() - start_time < deadline - 25:
            budget = min(20.0, deadline - 25 - (time.time() - start_time))
            sa_centers, sa_r = _sa_optimize(current_best, budget)
            if sa_r > current_r:
                current_r = sa_r
                current_best = sa_centers.copy()
        
        # Phase 2: Smooth Local Optimization
        if time.time() - start_time < deadline - 15:
            budget = min(15.0, deadline - 15 - (time.time() - start_time))
            lo_centers, lo_r = _local_optimize_smooth(current_best, budget)
            if lo_r > current_r:
                current_r = lo_r
                current_best = lo_centers.copy()
        
        # Phase 3: Coordinate Descent
        if time.time() - start_time < deadline - 5:
            budget = min(10.0, deadline - 5 - (time.time() - start_time))
            cd_centers, cd_r = _coordinate_descent(current_best, budget)
            if cd_r > current_r:
                current_r = cd_r
                current_best = cd_centers.copy()
        
        if current_r > best_r:
            best_r = current_r
            best_centers = current_best.copy()

    # Final validity check
    if best_centers is None or not np.all(np.isfinite(best_centers)) or np.any(np.abs(best_centers) > 0.5):
        # Fallback to grid
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers = grid[:N].copy()

    return best_centers
# EVOLVE-BLOCK-END
