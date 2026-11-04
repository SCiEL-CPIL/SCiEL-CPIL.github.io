import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 100

def _compute_radius(centers):
    """
    Computes the maximum common radius for a given set of centers.
    Centers: (N, 3) array.
    Cube: [-0.5, 0.5]^3.
    """
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
        
    # Wall constraints: distance from center to nearest wall
    max_abs = np.max(np.abs(centers), axis=1)
    wall_dists = 0.5 - max_abs
    min_wall_dist = np.min(wall_dists)
    
    if min_wall_dist < 0:
        return 0.0

    # Pairwise constraints
    dists = cdist(centers, centers)
    # Exclude self-distance (diagonal)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return r

def _init_fcc(n_spheres=100):
    """
    Generate FCC lattice initialization.
    """
    # FCC: 4 atoms per cell. 100 spheres -> 25 cells.
    # 25 is 5x5x1 or 1x5x5 or 1x1x25 (too long).
    # Let's try 5x5x1.
    nx, ny, nz = 5, 5, 1
    a = 1.0 / max(nx, ny, nz)
    
    centers_list = []
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                base = np.array([ix, iy, iz]) * a
                for dx, dy, dz in [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]:
                    pos = base + np.array([dx, dy, dz]) * a
                    centers_list.append(pos)
    
    centers = np.array(centers_list)
    # Center
    centers = centers - np.mean(centers, axis=0)
    # Scale to fit
    max_extent = np.max(np.abs(centers))
    if max_extent > 0:
        centers = centers / max_extent * 0.49
    
    if len(centers) > n_spheres:
        # Randomly select if too many
        rng = np.random.RandomState(42)
        idx = rng.choice(len(centers), size=n_spheres, replace=False)
        centers = centers[idx]
    elif len(centers) < n_spheres:
        rng = np.random.RandomState(42)
        extra = rng.uniform(-0.45, 0.45, size=(n_spheres - len(centers), 3))
        centers = np.vstack([centers, extra])
        
    return centers

def _init_grid(n_spheres=100):
    """
    Simple grid initialization.
    """
    # 10x10x1 grid
    xs = np.linspace(-0.45, 0.45, 10)
    ys = np.linspace(-0.45, 0.45, 10)
    centers = np.array([[x, y, 0] for x in xs for y in ys])
    return centers[:n_spheres]

def _local_optimize(centers, max_iters=50, deadline=None):
    """
    Local optimization using active-set gradient descent.
    Minimizes the soft-min of the active constraints (closest pairs and walls).
    """
    centers = centers.copy()
    n = N
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    best_centers = centers.copy()
    best_r = _compute_radius(centers)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute distances
        dists = cdist(centers, centers)
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        max_abs = np.max(np.abs(centers), axis=1)
        wall_dists = 0.5 - max_abs
        min_wall_dist = np.min(wall_dists)
        
        r_curr = min(min_pair_dist / 2.0, min_wall_dist)
        
        # Determine active set
        # Pairs within r_curr + margin
        margin = 0.005
        active_pair_mask = dists[i_idx, j_idx] < (r_curr * 2.0) + margin
        active_pairs = [(i_idx[k], j_idx[k]) for k in range(len(i_idx)) if active_pair_mask[k]]
        
        # Walls within r_curr + margin
        active_walls = []
        for i in range(n):
            if wall_dists[i] < r_curr + margin:
                max_val = max_abs[i]
                for k in range(3):
                    if np.isclose(abs(centers[i, k]), max_val, atol=1e-10):
                        active_walls.append((i, k))
                        break
        
        if not active_pairs and not active_walls:
            break
            
        # Compute gradient of soft-min of active constraints
        # We use a small temperature for the soft-min to approximate the min
        temp = 0.001
        
        # Collect active distances
        act_dists = []
        act_grads_info = []
        
        for i, j in active_pairs:
            d = np.linalg.norm(centers[i] - centers[j]) / 2.0
            act_dists.append(d)
            act_grads_info.append(('pair', i, j, d))
            
        for i, k in active_walls:
            d = 0.5 - abs(centers[i, k])
            act_dists.append(d)
            act_grads_info.append(('wall', i, k, d))
            
        act_dists = np.array(act_dists)
        
        # Soft min weights
        scaled = -act_dists / temp
        max_scaled = np.max(scaled)
        exp_scaled = np.exp(scaled - max_scaled)
        weights = exp_scaled / np.sum(exp_scaled)
        
        grad = np.zeros_like(centers)
        
        for w, info in zip(weights, act_grads_info):
            if w < 1e-6:
                continue
            if info[0] == 'pair':
                _, i, j, d = info
                vec = centers[i] - centers[j]
                dist = np.linalg.norm(vec)
                if dist > 1e-10:
                    factor = -w / (2 * dist)
                    grad[i] += factor * vec
                    grad[j] -= factor * vec
            else:
                _, i, k, d = info
                grad[i, k] += w * np.sign(centers[i, k])
                
        # Step
        step_size = 0.01
        centers = centers - step_size * grad
        centers = np.clip(centers, -0.49999, 0.49999)
        
        r_new = _compute_radius(centers)
        if r_new > best_r:
            best_r = r_new
            best_centers = centers.copy()
            
    return best_centers

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_radius = 0.0
    best_centers = None
    
    # Initializations
    inits = [
        _init_fcc(N),
        _init_grid(N)
    ]
    
    # Add a random init
    rng = np.random.RandomState(42)
    inits.append(rng.uniform(-0.45, 0.45, size=(N, 3)))
    
    # Basin Hopping
    # We will run basin hopping from each init
    for init in inits:
        if time.time() > deadline - 10.0:
            break
            
        current_centers = init.copy()
        current_radius = _compute_radius(current_centers)
        
        # Initial local optimization
        current_centers = _local_optimize(current_centers, max_iters=100, deadline=deadline - 5.0)
        current_radius = _compute_radius(current_centers)
        
        if current_radius > best_radius:
            best_radius = current_radius
            best_centers = current_centers.copy()
            
        # Basin Hopping Loop
        T = 0.01
        kick_scale = 0.05
        
        while time.time() < deadline - 5.0:
            # Kick
            kick = rng.normal(0, kick_scale, size=(N, 3))
            kicked_centers = current_centers + kick
            kicked_centers = np.clip(kicked_centers, -0.49999, 0.49999)
            
            # Local Optimize
            new_centers = _local_optimize(kicked_centers, max_iters=50, deadline=deadline - 2.0)
            new_radius = _compute_radius(new_centers)
            
            # Accept/Reject
            # Energy = -radius
            delta_E = -new_radius - (-current_radius) # new_radius - current_radius
            if new_radius > current_radius or np.random.rand() < np.exp(-delta_E / T):
                current_centers = new_centers
                current_radius = new_radius
                
                if current_radius > best_radius:
                    best_radius = current_radius
                    best_centers = current_centers.copy()
            
            # Anneal
            T *= 0.9
            kick_scale *= 0.95
            
            if T < 1e-5:
                break
                
    return best_centers
