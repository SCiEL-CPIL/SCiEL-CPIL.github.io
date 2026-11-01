"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_radius(c):
    """Compute the common radius for centres c (N,3) in cube [-0.5,0.5]^3."""
    # Pairwise distances using broadcasting
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    # Wall distances
    wall_dists = 0.5 - np.abs(c)
    min_wall = np.min(wall_dists)
    return min(min_pair / 2.0, min_wall)

def _negative_radius(x):
    """Objective function: negative of the radius (to minimize)."""
    c = x.reshape(N, 3)
    # Penalize if any point is outside the cube
    if np.any(c < -0.5 + 1e-10) or np.any(c > 0.5 - 1e-10):
        return 1e10
    return -_compute_radius(c)

def _fcc_initial():
    """Generate an FCC-like initial configuration for 50 spheres."""
    # FCC structure has spheres at corners and face centers of cubes
    # We'll create a 3D grid with FCC offsets
    # For 50 spheres, try a 4x4x4 grid with some FCC-like perturbations
    
    # Start with a 4x4x4 grid (64 positions), then select 50
    # But first, let's try a better arrangement
    
    # Create FCC lattice points
    # FCC basis: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)
    # Scale to fit in [-0.5, 0.5]^3
    
    # Let's use a 3x3x3 grid of FCC unit cells
    # Each cell has 4 points, so 27*4 = 108 points, we need 50
    # Scale factor: we want points in [-0.5, 0.5]^3
    
    points = []
    # FCC unit cell basis vectors
    a = 1.0  # lattice constant
    
    # Generate FCC points in a 3x3x3 supercell
    for i in range(3):
        for j in range(3):
            for k in range(3):
                # Corner
                points.append([i*a, j*a, k*a])
                # Face centers
                points.append([(i+0.5)*a, (j+0.5)*a, k*a])
                points.append([(i+0.5)*a, j*a, (k+0.5)*a])
                points.append([i*a, (j+0.5)*a, (k+0.5)*a])
    
    points = np.array(points)
    
    # Center and scale to fit in [-0.5, 0.5]^3
    points = points - points.mean(axis=0)
    # Scale so max extent is 1.0
    max_extent = np.max(np.abs(points))
    if max_extent > 0:
        points = points / max_extent * 0.45  # Leave some margin
    
    # Select 50 points that are most spread out
    # Use a simple approach: sort by distance from center, take alternating
    # Or use a greedy farthest-point sampling
    
    # Greedy farthest point sampling
    selected = [0]  # Start with first point
    remaining = list(range(1, len(points)))
    
    while len(selected) < N and remaining:
        # Find point in remaining that maximizes minimum distance to selected
        best_idx = -1
        best_min_dist = -1
        for idx in remaining:
            dists_to_selected = np.linalg.norm(points[idx] - points[selected], axis=1)
            min_dist = np.min(dists_to_selected)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = idx
        selected.append(best_idx)
        remaining.remove(best_idx)
    
    return points[selected]

def _grid_initial():
    """Generate a grid-based initial configuration."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N]

def _random_initial():
    """Generate a random initial configuration."""
    return np.random.uniform(-0.45, 0.45, size=(N, 3))

def construct():
    """Main function: optimize sphere placement to maximize radius."""
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin
    
    # Try multiple initializations and keep the best
    best_c = None
    best_r = -1.0
    
    # Initial configurations to try
    inits = [
        ('fcc', _fcc_initial()),
        ('grid', _grid_initial()),
        ('random', _random_initial()),
    ]
    
    for name, c0 in inits:
        if time.time() > deadline:
            break
        
        c0 = c0.copy()
        # Ensure within bounds
        c0 = np.clip(c0, -0.49, 0.49)
        
        x0 = c0.flatten()
        
        # Use L-BFGS-B for optimization
        try:
            result = minimize(
                _negative_radius,
                x0,
                method='L-BFGS-B',
                bounds=[(-0.49, 0.49)] * (N * 3),
                options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            c_opt = result.x.reshape(N, 3)
            c_opt = np.clip(c_opt, -0.49, 0.49)
            r_opt = _compute_radius(c_opt)
            
            if r_opt > best_r:
                best_r = r_opt
                best_c = c_opt.copy()
                
        except Exception:
            pass
        
        if time.time() > deadline:
            break
    
    # If we have time, try local refinement from the best configuration
    if best_c is not None and time.time() < deadline:
        x0 = best_c.flatten()
        try:
            result = minimize(
                _negative_radius,
                x0,
                method='L-BFGS-B',
                bounds=[(-0.49, 0.49)] * (N * 3),
                options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            c_opt = result.x.reshape(N, 3)
            c_opt = np.clip(c_opt, -0.49, 0.49)
            r_opt = _compute_radius(c_opt)
            
            if r_opt > best_r:
                best_r = r_opt
                best_c = c_opt.copy()
        except Exception:
            pass
    
    # Final safety: ensure valid output
    if best_c is None:
        best_c = _grid_initial()
    
    # Clip to ensure within cube
    best_c = np.clip(best_c, -0.499999, 0.499999)
    
    return best_c

# EVOLVE-BLOCK-END
