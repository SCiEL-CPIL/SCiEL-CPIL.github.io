import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _hex_lattice_init(n_points, radius=1.0, spacing_factor=1.0, jitter=0.0, rotation=0.0):
    """
    Generate initial points using a hexagonal lattice.
    We generate a large grid and pick the n_points closest to the center.
    """
    # Estimate spacing
    # Area of unit circle is pi. Area per circle is approx pi/n.
    # Hex packing density is pi/sqrt(12) ~ 0.9069.
    # So effective area per circle in hex lattice is (pi/n) / 0.9069.
    # Area of hex cell is (sqrt(3)/2) * s^2.
    # s^2 = (pi/n) / (0.9069 * sqrt(3)/2)
    area_per_circle = np.pi / n_points
    hex_area_factor = np.sqrt(3) / 2
    density = np.pi / np.sqrt(12)
    s = np.sqrt(area_per_circle / (density * hex_area_factor))
    s *= spacing_factor
    
    # Generate points
    k = int(2.0 / s) + 3
    points = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * np.sqrt(3) / 2
            points.append((x, y))
    points = np.array(points)
    
    # Rotate
    if rotation != 0:
        theta = np.deg2rad(rotation)
        c, s_ = np.cos(theta), np.sin(theta)
        rot_mat = np.array([[c, -s_], [s_, c]])
        points = points @ rot_mat.T

    # Filter points inside the circle (with some margin for radius)
    # We want points that can potentially hold a circle of radius r.
    # For initialization, let's just take points inside radius 1.
    dists = np.linalg.norm(points, axis=1)
    valid = dists < 1.0
    
    points = points[valid]
    dists = dists[valid]
    
    # Sort by distance to center, pick closest n_points
    if len(points) >= n_points:
        indices = np.argsort(dists)[:n_points]
        points = points[indices]
    else:
        # If not enough points, pad with random points
        # This shouldn't happen with reasonable s
        pass
        
    # Add jitter
    if jitter > 0:
        points += np.random.normal(0, jitter, points.shape)
        
    return points

def _violation(pos, r):
    """
    Calculate the sum of squared violations of the constraints.
    pos: (N, 2) array of centers.
    r: radius.
    """
    # Boundary violations: |c_i| + r <= 1  => |c_i| <= 1 - r
    # Violation is max(0, |c_i| - (1 - r))^2
    dists_center = np.linalg.norm(pos, axis=1)
    boundary_viol = np.maximum(0, dists_center - (1.0 - r))
    boundary_viol = np.sum(boundary_viol ** 2)
    
    # Pairwise violations: |c_i - c_j| >= 2r
    # Violation is max(0, 2r - |c_i - c_j|)^2
    # Compute pairwise distances
    # Using broadcasting
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Only consider i < j
    iu = np.triu_indices(N, 1)
    pair_dists = dists[iu]
    pair_viol = np.maximum(0, 2.0 * r - pair_dists)
    pair_viol = np.sum(pair_viol ** 2)
    
    return boundary_viol + pair_viol

def _optimize_for_radius(pos_init, r_target):
    """
    Minimize the violation for a fixed radius r_target.
    Returns the optimized positions and the minimum violation.
    """
    def objective(x):
        pos = x.reshape(N, 2)
        return _violation(pos, r_target)
        
    x0 = pos_init.flatten()
    
    # Use L-BFGS-B
    res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-12})
    
    pos_opt = res.x.reshape(N, 2)
    viol = res.fun
    
    return pos_opt, viol

def _find_max_radius(pos_init, r_low=0.05, r_high=0.15, tol=1e-6, max_iters=50):
    """
    Binary search for the maximum feasible radius.
    """
    pos_current = pos_init.copy()
    
    for _ in range(max_iters):
        if r_high - r_low < tol:
            break
            
        r_mid = (r_low + r_high) / 2.0
        
        # Try to fit circles of radius r_mid
        pos_opt, viol = _optimize_for_radius(pos_current, r_mid)
        
        # Check if feasible
        # We consider it feasible if violation is very small
        if viol < 1e-8:
            r_low = r_mid
            pos_current = pos_opt
        else:
            r_high = r_mid
            # Do not update pos_current to a failed state? 
            # Or update to the best attempt? 
            # Updating to the best attempt might help if it's close.
            # But if it's far, it might mislead.
            # Let's keep the last known feasible position.
            
    # Final polish at r_low
    pos_final, viol_final = _optimize_for_radius(pos_current, r_low)
    
    # Verify final radius
    # Calculate the actual max radius for this configuration
    dists_center = np.linalg.norm(pos_final, axis=1)
    r_boundary = np.min(1.0 - dists_center)
    
    diff = pos_final[:, np.newaxis, :] - pos_final[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(N, 1)
    r_pair = np.min(dists[iu]) / 2.0
    
    r_actual = min(r_boundary, r_pair)
    
    return r_actual, pos_final

def construct():
    start_time = time.time()
    time_limit = 140.0  # Safety margin
    
    best_r = 0.0
    best_pos = None
    
    # Generate diverse initial configurations
    inits = []
    
    # 1. Standard Hex Lattice
    inits.append(_hex_lattice_init(N, jitter=0.01))
    
    # 2. Rotated Hex Lattice
    inits.append(_hex_lattice_init(N, jitter=0.01, rotation=15))
    
    # 3. Rotated Hex Lattice 30 deg
    inits.append(_hex_lattice_init(N, jitter=0.01, rotation=30))
    
    # 4. Slightly different spacing
    inits.append(_hex_lattice_init(N, jitter=0.02, spacing_factor=0.95))
    inits.append(_hex_lattice_init(N, jitter=0.02, spacing_factor=1.05))
    
    # 5. Random
    # Generate random points in unit disk
    for _ in range(2):
        angles = np.random.uniform(0, 2 * np.pi, N)
        radii = np.sqrt(np.random.uniform(0, 1, N))
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        inits.append(np.column_stack((x, y)))
        
    # Iterate through inits
    for i, init_pos in enumerate(inits):
        if time.time() - start_time > time_limit:
            break
            
        # Run optimization
        try:
            r, pos = _find_max_radius(init_pos)
            
            if r > best_r:
                best_r = r
                best_pos = pos
        except Exception as e:
            pass
            
    if best_pos is None:
        # Fallback to a simple lattice if everything failed
        best_pos = _hex_lattice_init(N, jitter=0.0)
        
    return best_pos

# EVOLVE-BLOCK-END
