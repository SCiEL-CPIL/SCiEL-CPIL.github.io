"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, cdist

def _bcc_lattice(n_points):
    """Generate n_points centers on a BCC lattice scaled to fit [-0.5, 0.5]^3."""
    # BCC unit cell has 2 atoms: corners (8 shared) + body center (1)
    # For n points, use a cubic grid of BCC cells
    # Each BCC cell contributes 2 points
    # We need ceil(n/2) cells in a 3D arrangement
    
    n_cells = int(np.ceil(n_points / 2))
    m = int(np.ceil(n_cells ** (1.0/3.0) - 1e-9))
    
    # Generate all BCC lattice points in an m x m x m supercell
    points = []
    for ix in range(m):
        for iy in range(m):
            for iz in range(m):
                # Corner of this cell
                corner = np.array([ix, iy, iz])
                # Body center of this cell
                body_center = corner + 0.5
                points.append(corner)
                points.append(body_center)
    
    points = np.array(points)
    
    # Scale to fit in [-0.5, 0.5]^3
    # The lattice spans [0, m-1+0.5] in each dimension
    span = m - 1 + 0.5
    if span > 0:
        points = points / span - 0.5
    else:
        points = np.zeros_like(points)
    
    # Take first n_points
    return points[:n_points].copy()

def _compute_radius(c):
    """Compute the maximum common radius for given centres."""
    if not np.all(np.isfinite(c)):
        return 0.0
    if c.shape != (N, 3):
        return 0.0
    # Check all centres are within the cube
    if np.any(np.abs(c) > 0.5 + 1e-10):
        return 0.0
    # Minimum distance to walls
    wall_min = 0.5 - np.max(np.abs(c), axis=1)
    if np.any(wall_min < -1e-10):
        return 0.0
    r_wall = np.min(wall_min)
    # Minimum pairwise distance
    d = pdist(c)
    if len(d) == 0:
        r_pair = np.inf
    else:
        r_pair = np.min(d) / 2.0
    return min(r_wall, r_pair)

def _smooth_energy(c, beta=100.0):
    """Smooth surrogate: soft-min of pairwise distances/2 and wall distances."""
    if not np.all(np.isfinite(c)):
        return 1.0
    if np.any(np.abs(c) > 0.5 + 1e-8):
        return 1.0
    
    # Wall distances
    wall_dists = 0.5 - np.max(np.abs(c), axis=1)
    
    # Pairwise distances / 2
    d = pdist(c) / 2.0
    
    # Combine all constraints into a single array
    all_dists = np.concatenate([wall_dists, d])
    
    # Soft min using log-sum-exp trick
    min_val = np.min(all_dists)
    if min_val < -1e-6:
        return 1.0
    
    # Soft-min: -log(sum(exp(-beta*x))) / beta
    # For numerical stability
    shifted = -beta * (all_dists - min_val)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min = min_val - log_sum / beta
    
    return -soft_min  # Negative to minimize

def _direct_energy(c):
    """Direct negative radius."""
    r = _compute_radius(c)
    if r <= 0:
        return 1.0
    return -r

def _local_optimize(c, deadline, use_smooth=True, max_iters=200):
    """Local optimization using L-BFGS-B."""
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    if use_smooth:
        obj = lambda x: _smooth_energy(x.reshape(N, 3))
        obj2 = None
    else:
        obj = lambda x: _direct_energy(x.reshape(N, 3))
        obj2 = None
    
    # Use L-BFGS-B with the smooth surrogate first, then direct
    try:
        res = minimize(
            obj, c.ravel(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iters, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 500}
        )
        c_opt = res.x.reshape(N, 3)
        
        # Polish with direct objective
        if time.time() < deadline - 5:
            res2 = minimize(
                _direct_energy, c_opt.ravel(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 200}
            )
            c_opt = res2.x.reshape(N, 3)
    except Exception:
        c_opt = c.copy()
    
    return c_opt

def construct():
    t0 = time.time()
    deadline = t0 + 145.0  # safety margin before 150s limit
    
    # Initialize with BCC lattice
    centres = _bcc_lattice(N)
    best_centres = centres.copy()
    best_r = _compute_radius(best_centres)
    
    rng = np.random.RandomState(42)
    
    # Multi-start local optimization
    n_restarts = 8
    
    for restart in range(n_restarts):
        if time.time() > deadline - 20.0:
            break
        
        # Generate starting point
        if restart == 0:
            c = best_centres.copy()
        else:
            # Mix of BCC perturbation and random
            if restart % 2 == 1:
                # Perturb best
                c = best_centres + rng.normal(0, 0.02, (N, 3))
            else:
                # New BCC with random rotation
                c = _bcc_lattice(N)
                # Apply random rotation
                from scipy.spatial.transform import Rotation
                rot = Rotation.random(random_state=rng)
                c = rot.apply(c)
            c = np.clip(c, -0.49, 0.49)
        
        # Local optimization
        c_opt = _local_optimize(c, deadline, use_smooth=True, max_iters=150)
        
        # Validate
        r_opt = _compute_radius(c_opt)
        if r_opt > best_r + 1e-10:
            best_r = r_opt
            best_centres = c_opt.copy()
    
    # Final polish on best configuration
    if time.time() < deadline - 10:
        best_centres = _local_optimize(best_centres, deadline, use_smooth=False, max_iters=300)
        final_r = _compute_radius(best_centres)
        if final_r > best_r:
            best_r = final_r
    
    # Final validation
    if not np.all(np.isfinite(best_centres)):
        best_centres = _bcc_lattice(N)
    if best_centres.shape != (N, 3):
        best_centres = _bcc_lattice(N)
    if np.any(np.abs(best_centres) > 0.5 + 1e-10):
        best_centres = np.clip(best_centres, -0.4999, 0.4999)
    
    return best_centres
# EVOLVE-BLOCK-END
