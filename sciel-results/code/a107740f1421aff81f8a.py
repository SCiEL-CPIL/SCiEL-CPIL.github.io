import numpy as np
from scipy.optimize import minimize
import time

N = 16
D = 2

def pairwise_distances(pts):
    """Compute all pairwise distances."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    # Exclude self-distances
    iu = np.triu_indices(N, k=1)
    return dists[iu]

def pairwise_distances_sq(pts):
    """Compute all pairwise squared distances (faster, no sqrt)."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    iu = np.triu_indices(N, k=1)
    return dist_sq[iu]

def objective(x):
    """(max_dist / min_dist)^2"""
    pts = x.reshape(N, D)
    d_sq = pairwise_distances_sq(pts)
    # Handle degenerate cases
    if np.any(d_sq < 1e-24):
        return 1e10
    d = np.sqrt(d_sq)
    ratio = d.max() / d.min()
    return ratio**2

def objective_smooth(x, beta=200.0):
    """Smooth approximation using log-sum-exp on squared distances."""
    pts = x.reshape(N, D)
    d_sq = pairwise_distances_sq(pts)
    if np.any(d_sq < 1e-24):
        return 1e10
    d = np.sqrt(d_sq)
    # log-sum-exp for max
    max_d = np.log(np.sum(np.exp(beta * d))) / beta
    # For min, use negative log-sum-exp: min = -logsumexp(-beta*d)/beta
    min_d = -np.log(np.sum(np.exp(-beta * d))) / beta
    if min_d < 1e-12:
        return 1e10
    ratio = max_d / min_d
    return ratio**2

def make_hexagonal_cluster(params):
    """
    Generate 16 points as a hexagonal lattice cluster with 3 parameters:
    params[0] = aspect ratio (y-scale relative to x)
    params[1] = shear angle (radians)
    params[2] = radial scaling factor for outer ring
    
    Uses a 4-3-4-3-2 arrangement typical of hexagonal packing.
    """
    aspect = params[0]
    shear = params[1]
    outer_scale = params[2]
    
    # Base hexagonal lattice positions (unit spacing)
    # Row 0: 4 points
    # Row 1: 3 points (offset)
    # Row 2: 4 points
    # Row 3: 3 points (offset)
    # Row 4: 2 points (centered)
    
    base_points = []
    # Row 0 (top): 4 points
    for i in range(4):
        x = (i - 1.5) * 1.0
        y = 2.0
        base_points.append([x, y])
    # Row 1: 3 points, offset by 0.5
    for i in range(3):
        x = (i - 1.0) * 1.0 + 0.5
        y = 2.0 - np.sqrt(3)/2
        base_points.append([x, y])
    # Row 2: 4 points
    for i in range(4):
        x = (i - 1.5) * 1.0
        y = 2.0 - np.sqrt(3)
        base_points.append([x, y])
    # Row 3: 3 points, offset by 0.5
    for i in range(3):
        x = (i - 1.0) * 1.0 + 0.5
        y = 2.0 - 1.5*np.sqrt(3)
        base_points.append([x, y])
    # Row 4 (bottom): 2 points
    for i in range(2):
        x = (i - 0.5) * 1.0
        y = 2.0 - 2.0*np.sqrt(3)
        base_points.append([x, y])
    
    base_points = np.array(base_points, dtype=float)
    
    # Center the points
    centroid = base_points.mean(axis=0)
    base_points -= centroid
    
    # Apply aspect ratio (scale y)
    base_points[:, 1] *= aspect
    
    # Apply shear transformation
    shear_matrix = np.array([[1, np.tan(shear)], [0, 1]])
    base_points = base_points @ shear_matrix.T
    
    # Identify outer ring points (those farther from center)
    dists_from_center = np.linalg.norm(base_points, axis=1)
    median_dist = np.median(dists_from_center)
    outer_mask = dists_from_center > median_dist
    
    # Apply radial scaling to outer ring only
    if np.any(outer_mask):
        outer_points = base_points[outer_mask]
        # Scale outer points radially
        outer_dists = np.linalg.norm(outer_points, axis=1)
        outer_dists[outer_dists < 1e-12] = 1e-12
        scale_factors = outer_scale / outer_dists
        base_points[outer_mask] *= scale_factors[:, np.newaxis]
    
    return base_points

def make_seeds():
    """Generate multiple diverse initial configurations."""
    seeds = []
    
    # 1. 4x4 grid
    grid = np.array([(i, j) for i in range(4) for j in range(4)], dtype=float)
    seeds.append(grid.flatten())
    
    # 2. Hexagonal lattice subset
    # Hex lattice: rows offset
    hex_pts = []
    for row in range(4):
        x_offset = (row % 2) * 0.5
        for col in range(4):
            hex_pts.append([col + x_offset, row * np.sqrt(3)/2])
    hex_pts = np.array(hex_pts[:16])
    seeds.append(hex_pts.flatten())
    
    # 3. Concentric circles: 1 center + 5 + 10
    pts = [np.array([0.0, 0.0])]
    # Inner ring: 5 points
    for i in range(5):
        angle = 2 * np.pi * i / 5
        pts.append(np.array([1.0 * np.cos(angle), 1.0 * np.sin(angle)]))
    # Outer ring: 10 points
    for i in range(10):
        angle = 2 * np.pi * i / 10 + np.pi/10
        pts.append(np.array([2.2 * np.cos(angle), 2.2 * np.sin(angle)]))
    seeds.append(np.array(pts).flatten())
    
    # 4. Two concentric octagons
    pts = []
    for i in range(8):
        angle = 2 * np.pi * i / 8
        pts.append(np.array([1.0 * np.cos(angle), 1.0 * np.sin(angle)]))
    for i in range(8):
        angle = 2 * np.pi * i / 8 + np.pi/8
        pts.append(np.array([2.0 * np.cos(angle), 2.0 * np.sin(angle)]))
    seeds.append(np.array(pts).flatten())
    
    # 5. 1 center + 6 + 9 (hexagon + outer)
    pts = [np.array([0.0, 0.0])]
    for i in range(6):
        angle = 2 * np.pi * i / 6
        pts.append(np.array([1.0 * np.cos(angle), 1.0 * np.sin(angle)]))
    for i in range(9):
        angle = 2 * np.pi * i / 9 + np.pi/9
        pts.append(np.array([2.0 * np.cos(angle), 2.0 * np.sin(angle)]))
    seeds.append(np.array(pts).flatten())
    
    # 6. 2 center + 6 + 8
    pts = [np.array([-0.5, 0.0]), np.array([0.5, 0.0])]
    for i in range(6):
        angle = 2 * np.pi * i / 6
        pts.append(np.array([1.5 * np.cos(angle), 1.5 * np.sin(angle)]))
    for i in range(8):
        angle = 2 * np.pi * i / 8 + np.pi/8
        pts.append(np.array([2.8 * np.cos(angle), 2.8 * np.sin(angle)]))
    seeds.append(np.array(pts).flatten())
    
    # 7. Regular 16-gon (degenerate but a seed)
    pts = []
    for i in range(16):
        angle = 2 * np.pi * i / 16
        pts.append(np.array([np.cos(angle), np.sin(angle)]))
    seeds.append(np.array(pts).flatten())
    
    # 8. 4x4 grid with slight perturbation
    grid2 = np.array([(i + 0.1*(i%2), j + 0.1*(j%2)) for i in range(4) for j in range(4)], dtype=float)
    seeds.append(grid2.flatten())
    
    # 9. Hexagonal cluster with various parameter settings
    hex_params_list = [
        [1.0, 0.0, 1.0],      # perfect hexagon
        [1.1, 0.1, 1.05],     # slightly stretched
        [0.9, -0.1, 0.95],    # slightly compressed
        [1.0, 0.2, 1.1],      # sheared and expanded
        [1.2, 0.0, 0.9],      # tall and contracted outer
    ]
    for params in hex_params_list:
        pts = make_hexagonal_cluster(np.array(params))
        seeds.append(pts.flatten())
    
    return seeds

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    seeds = make_seeds()
    best_x = None
    best_val = np.inf
    
    # Phase 1: Optimize hexagonal cluster parameters (3 DOF)
    hex_param_bounds = [(0.5, 2.0), (-0.5, 0.5), (0.5, 2.0)]
    hex_param_seeds = [
        [1.0, 0.0, 1.0],
        [1.1, 0.1, 1.05],
        [0.9, -0.1, 0.95],
        [1.2, 0.2, 1.1],
        [0.8, 0.0, 1.2],
    ]
    
    for hex_seed in hex_param_seeds:
        if time.time() > deadline:
            break
        
        try:
            # First optimize parameters with smooth objective
            def hex_objective(params):
                pts = make_hexagonal_cluster(params)
                return objective_smooth(pts.flatten(), beta=200.0)
            
            res = minimize(
                hex_objective,
                np.array(hex_seed),
                method='L-BFGS-B',
                bounds=hex_param_bounds,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if res.success or res.fun < 1e8:
                # Convert to full point configuration
                pts = make_hexagonal_cluster(res.x)
                val = objective(pts.flatten())
                if val < best_val:
                    best_val = val
                    best_x = pts.flatten().copy()
        except Exception:
            pass
    
    # Phase 2: Full 32-DOF optimization from best hex result and other seeds
    all_starts = list(seeds)
    if best_x is not None:
        all_starts.insert(0, best_x)
    
    for seed_idx, seed in enumerate(all_starts):
        if time.time() > deadline:
            break
        
        # Skip if we've already tried this as hex start
        if seed_idx == 0 and best_x is not None:
            continue
        
        # Try L-BFGS-B with smooth objective
        try:
            res = minimize(
                objective_smooth,
                seed,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            val = objective(res.x)
            if val < best_val:
                best_val = val
                best_x = res.x.copy()
        except Exception:
            pass
        
        # Try Nelder-Mead with exact objective
        if time.time() > deadline:
            break
        try:
            res = minimize(
                objective,
                seed,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
            )
            val = objective(res.x)
            if val < best_val:
                best_val = val
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: High-precision polish if time remains
    if best_x is not None and time.time() < deadline - 10.0:
        try:
            res = minimize(
                objective_smooth,
                best_x,
                method='L-BFGS-B',
                options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            val = objective(res.x)
            if val < best_val:
                best_val = val
                best_x = res.x.copy()
        except Exception:
            pass
        
        # Final Nelder-Mead polish
        if time.time() < deadline - 5.0:
            try:
                res = minimize(
                    objective,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
                )
                val = objective(res.x)
                if val < best_val:
                    best_val = val
                    best_x = res.x.copy()
            except Exception:
                pass
    
    if best_x is None:
        # Fallback to grid
        grid = np.array([(i, j) for i in range(4) for j in range(4)], dtype=float)
        return grid
    
    return best_x.reshape(N, D)
