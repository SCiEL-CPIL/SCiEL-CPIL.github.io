import numpy as np
import time
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START
def _compute_radius(c):
    """Compute the common radius for centres c (N,3) in cube [-0.5,0.5]^3."""
    if len(c) == 0:
        return 0.0
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    wall_dists = 0.5 - np.abs(c)
    min_wall = np.min(wall_dists)
    return min(min_pair / 2.0, min_wall)

def _neg_radius(c):
    """Objective to minimize: negative of the common radius."""
    return -_compute_radius(c)

def _get_bcc_init():
    """Body-centered cubic initialization."""
    # BCC unit cell has 2 atoms: corners (8 shared) + center (1)
    # For 50 spheres, we can use a 4x4x3 grid of BCC cells
    # Each BCC cell has 2 spheres, so 4*4*3*2 = 96 is too many.
    # Let's use a different approach: place spheres in a BCC-like pattern.
    
    # Actually, let's use a simpler structured initialization.
    # 5x5x2 grid is the natural fit for 50 spheres.
    # But let's also try FCC-like and other patterns.
    
    # For now, return the 5x5x2 grid as a baseline
    m = 5
    ticks_xy = (np.arange(m) + 0.5) / m - 0.5
    ticks_z = np.array([-0.25, 0.25])
    
    grid = []
    for z in ticks_z:
        for y in ticks_xy:
            for x in ticks_xy:
                grid.append([x, y, z])
    return np.array(grid)

def _get_fcc_init():
    """FCC-like initialization."""
    # FCC has 4 atoms per unit cell
    # Let's create a pattern that approximates FCC packing
    # This is complex, so let's use a perturbed grid instead
    
    m = 5
    ticks_xy = (np.arange(m) + 0.5) / m - 0.5
    ticks_z = np.array([-0.25, 0.25])
    
    grid = []
    for z in ticks_z:
        for y in ticks_xy:
            for x in ticks_xy:
                # Add small offset to break symmetry
                offset = 0.02 * np.sin(x * 10) * np.cos(y * 10)
                grid.append([x + offset, y - offset, z])
    return np.array(grid)

def _get_body_centered_init():
    """Body-centered initialization with alternating layers."""
    m = 5
    ticks_xy = (np.arange(m) + 0.5) / m - 0.5
    ticks_z = np.array([-0.25, 0.25])
    
    grid = []
    for z_idx, z in enumerate(ticks_z):
        for y in ticks_xy:
            for x in ticks_xy:
                # Alternate layers slightly offset
                if z_idx == 0:
                    grid.append([x, y, z])
                else:
                    grid.append([x + 0.1, y + 0.1, z])
    return np.array(grid)

def _local_refine_smooth(c, time_limit, seed=0):
    """Refine using smooth surrogate (soft-min) with L-BFGS-B."""
    c = c.copy()
    start_time = time.time()
    
    # Soft-min objective
    def soft_min_obj(x):
        c_local = x.reshape(-1, 3)
        n = c_local.shape[0]
        
        # Pairwise distances
        diff = c_local[:, np.newaxis, :] - c_local[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=-1))
        np.fill_diagonal(dists, np.inf)
        i_idx, j_idx = np.triu_indices(n, k=1)
        pair_dists = dists[i_idx, j_idx]
        
        # Wall distances
        wall_dists = (0.5 - np.abs(c_local)).flatten()
        
        all_dists = np.concatenate([pair_dists, wall_dists])
        
        beta = 100.0
        d_max = np.max(all_dists)
        exp_terms = np.exp(-beta * (all_dists - d_max))
        S = np.sum(exp_terms)
        if S == 0:
            S = 1e-300
        
        obj = (1.0 / beta) * (np.log(S) - beta * d_max)
        return obj
    
    try:
        res = minimize(soft_min_obj, c.flatten(), method='L-BFGS-B',
                       options={'maxiter': 500, 'maxfun': 500, 'disp': False})
        if res.success:
            c = res.x.reshape(-1, 3)
    except Exception:
        pass
    
    return c

def _local_refine_direct(c, time_limit, seed=0):
    """Refine using direct radius maximization with Nelder-Mead."""
    c = c.copy()
    start_time = time.time()
    
    try:
        res = minimize(_neg_radius, c.flatten(), method='Nelder-Mead',
                       options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-8, 'disp': False})
        if res.success:
            c = res.x.reshape(-1, 3)
    except Exception:
        pass
    
    return c

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    # Multiple initialization strategies
    inits = {
        'grid': _get_bcc_init(),
        'fcc': _get_fcc_init(),
        'bcc_alt': _get_body_centered_init()
    }
    
    best_c = inits['grid'].copy()
    best_r = _compute_radius(best_c)
    
    rng = np.random.RandomState(42)
    seed_idx = 0
    
    while time.time() - start_time < deadline - 10.0:
        # Choose an initialization strategy
        init_name = list(inits.keys())[seed_idx % len(inits)]
        c_init = inits[init_name].copy()
        
        # Add perturbation
        noise_scale = 0.03 + 0.02 * (seed_idx % 3)
        c_init = c_init + rng.normal(0, noise_scale, c_init.shape)
        c_init = np.clip(c_init, -0.49, 0.49)
        
        # Alternate between smooth and direct refinement
        if seed_idx % 2 == 0:
            c_refined = _local_refine_smooth(c_init, 30.0, seed=seed_idx)
        else:
            c_refined = _local_refine_direct(c_init, 30.0, seed=seed_idx)
        
        # Evaluate
        r_refined = _compute_radius(c_refined)
        
        if r_refined > best_r:
            best_r = r_refined
            best_c = c_refined.copy()
        
        seed_idx += 1
    
    # Final check
    if not np.all(np.isfinite(best_c)):
        return inits['grid']
        
    # Ensure strictly inside
    best_c = np.clip(best_c, -0.49999, 0.49999)
    
    return best_c
# EVOLVE-BLOCK-END
