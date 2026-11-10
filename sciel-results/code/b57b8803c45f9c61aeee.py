# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the (x,y) coordinates.

    """
    n = 16
    d = 2
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 590  # 10s margin
    
    def get_score(pts):
        if pts.shape != (n, 2):
            return 0.0
        dists = pdist(pts)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        ratio = d_min / d_max
        return ratio ** 2

    def soft_min_max_objective(flat_params):
        """
        Parameters: [r1, r2, r3, r4, theta1, theta2, theta3, theta4]
        4 rings with 1, 5, 5, 5 points (or other splits)
        Actually, let's use a more flexible approach:
        Parameterize by radial distortion of a 4x4 grid.
        
        New approach: use a parameterized warp of a 4x4 grid.
        params: [a, b, c, d, e] where:
        - a: radial scaling factor
        - b: quartic radial distortion
        - c: angular modulation amplitude
        - d: angular frequency (fixed at 4 for symmetry)
        - e: additional radial power
        
        Actually, let me use a simpler, more effective parameterization:
        Place points on 4 concentric rings with adjustable radii and angular offsets.
        Split: 1 + 4 + 5 + 6 = 16 points
        """
        # Parse parameters
        # [r0, r1, r2, r3, off0, off1, off2, off3]
        r0 = flat_params[0]  # radius of ring 0 (1 point)
        r1 = flat_params[1]  # radius of ring 1 (4 points)
        r2 = flat_params[2]  # radius of ring 2 (5 points)
        r3 = flat_params[3]  # radius of ring 3 (6 points)
        off0 = flat_params[4]  # angular offset ring 0
        off1 = flat_params[5]  # angular offset ring 1
        off2 = flat_params[6]  # angular offset ring 2
        off3 = flat_params[7]  # angular offset ring 3
        
        pts = []
        
        # Ring 0: 1 point at center
        pts.append([r0 * np.cos(off0), r0 * np.sin(off0)])
        
        # Ring 1: 4 points
        angles1 = np.linspace(0, 2*np.pi, 4, endpoint=False) + off1
        pts.extend(np.column_stack((r1 * np.cos(angles1), r1 * np.sin(angles1))))
        
        # Ring 2: 5 points
        angles2 = np.linspace(0, 2*np.pi, 5, endpoint=False) + off2
        pts.extend(np.column_stack((r2 * np.cos(angles2), r2 * np.sin(angles2))))
        
        # Ring 3: 6 points
        angles3 = np.linspace(0, 2*np.pi, 6, endpoint=False) + off3
        pts.extend(np.column_stack((r3 * np.cos(angles3), r3 * np.sin(angles3))))
        
        pts = np.array(pts)
        
        # Compute distances
        dists = pdist(pts)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        ratio = d_min / d_max
        return -(ratio ** 2)
    
    def smooth_objective(flat_params):
        """
        Smooth version using log-sum-exp to approximate min and max.
        This provides gradients and avoids the kink at the min/max.
        """
        # Parse parameters same as above
        r0 = flat_params[0]
        r1 = flat_params[1]
        r2 = flat_params[2]
        r3 = flat_params[3]
        off0 = flat_params[4]
        off1 = flat_params[5]
        off2 = flat_params[6]
        off3 = flat_params[7]
        
        pts = []
        
        # Ring 0: 1 point at center
        pts.append([r0 * np.cos(off0), r0 * np.sin(off0)])
        
        # Ring 1: 4 points
        angles1 = np.linspace(0, 2*np.pi, 4, endpoint=False) + off1
        pts.extend(np.column_stack((r1 * np.cos(angles1), r1 * np.sin(angles1))))
        
        # Ring 2: 5 points
        angles2 = np.linspace(0, 2*np.pi, 5, endpoint=False) + off2
        pts.extend(np.column_stack((r2 * np.cos(angles2), r2 * np.sin(angles2))))
        
        # Ring 3: 6 points
        angles3 = np.linspace(0, 2*np.pi, 6, endpoint=False) + off3
        pts.extend(np.column_stack((r3 * np.cos(angles3), r3 * np.sin(angles3))))
        
        pts = np.array(pts)
        
        # Compute distances
        dists = pdist(pts)
        if len(dists) == 0:
            return 0.0
        
        # Use soft min and max with temperature
        temp = 0.1
        log_dists = np.log(dists + 1e-12)
        
        # Soft min: -logsumexp(-dists/temp) * temp
        soft_min = -temp * np.log(np.sum(np.exp(-dists / temp)))
        # Soft max: logsumexp(dists/temp) * temp
        soft_max = temp * np.log(np.sum(np.exp(dists / temp)))
        
        if soft_max == 0 or np.isnan(soft_min) or np.isnan(soft_max):
            return 0.0
        
        ratio = soft_min / soft_max
        return -(ratio ** 2)

    best_points = None
    best_score = -np.inf
    
    # Strategy 1: Optimize ring-based parameterization
    # Initial guesses for ring radii and offsets
    ring_configs = []
    
    # Config A: evenly spaced rings
    ring_configs.append([0.0, 0.5, 1.0, 1.5, 0.0, 0.0, 0.0, 0.0])
    
    # Config B: slightly different spacing
    ring_configs.append([0.0, 0.6, 1.1, 1.6, 0.0, np.pi/4, 0.0, np.pi/6])
    
    # Config C: more compressed inner
    ring_configs.append([0.0, 0.4, 0.9, 1.4, 0.0, np.pi/4, np.pi/5, np.pi/6])
    
    # Config D: expanded outer
    ring_configs.append([0.0, 0.7, 1.2, 1.8, 0.0, np.pi/4, 0.0, np.pi/6])
    
    # Config E: based on optimal packing intuition
    ring_configs.append([0.0, 0.55, 1.05, 1.55, 0.0, np.pi/4, np.pi/5, np.pi/6])
    
    for init_params in ring_configs:
        if time.time() > deadline:
            break
        
        try:
            # Use smooth objective for gradient-based optimization
            res = minimize(smooth_objective, init_params, method='L-BFGS-B',
                          options={'maxiter': 500, 'ftol': 1e-12})
            
            # Reconstruct points from optimal parameters
            opt_params = res.x
            r0, r1, r2, r3 = opt_params[0], opt_params[1], opt_params[2], opt_params[3]
            off0, off1, off2, off3 = opt_params[4], opt_params[5], opt_params[6], opt_params[7]
            
            pts = []
            pts.append([r0 * np.cos(off0), r0 * np.sin(off0)])
            angles1 = np.linspace(0, 2*np.pi, 4, endpoint=False) + off1
            pts.extend(np.column_stack((r1 * np.cos(angles1), r1 * np.sin(angles1))))
            angles2 = np.linspace(0, 2*np.pi, 5, endpoint=False) + off2
            pts.extend(np.column_stack((r2 * np.cos(angles2), r2 * np.sin(angles2))))
            angles3 = np.linspace(0, 2*np.pi, 6, endpoint=False) + off3
            pts.extend(np.column_stack((r3 * np.cos(angles3), r3 * np.sin(angles3))))
            
            opt_pts = np.array(pts)
            score = get_score(opt_pts)
            
            if score > best_score:
                best_score = score
                best_points = opt_pts.copy()
                
        except Exception:
            continue
    
    # Strategy 2: Direct optimization from good initial configs with Nelder-Mead
    # 4x4 Grid with radial distortion
    x_grid = np.linspace(-1, 1, 4)
    y_grid = np.linspace(-1, 1, 4)
    X, Y = np.meshgrid(x_grid, y_grid)
    grid_pts = np.column_stack((X.ravel(), Y.ravel()))
    
    # Apply radial warp: r' = r * (1 + a*r^2)
    for a in [0.0, 0.1, 0.2, -0.1]:
        if time.time() > deadline:
            break
        
        r = np.linalg.norm(grid_pts, axis=1)
        theta = np.arctan2(grid_pts[:, 1], grid_pts[:, 0])
        r_new = r * (1 + a * r**2)
        warped_pts = np.column_stack((r_new * np.cos(theta), r_new * np.sin(theta)))
        
        try:
            res = minimize(soft_min_max_objective, warped_pts.flatten(), 
                          method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
            opt_pts = res.x.reshape(n, 2)
            score = get_score(opt_pts)
            if score > best_score:
                best_score = score
                best_points = opt_pts.copy()
        except Exception:
            continue
    
    # Strategy 3: Hexagonal lattice refinement
    hex_pts_list = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            x = 1.5 * q
            y = np.sqrt(3) * (r + 0.5 * q)
            hex_pts_list.append([x, y])
    hex_pts = np.array(hex_pts_list)
    hex_pts -= hex_pts.mean(axis=0)
    dists_to_center = np.linalg.norm(hex_pts, axis=1)
    idx = np.argsort(dists_to_center)[:n]
    hex_subset = hex_pts[idx]
    
    if time.time() < deadline:
        try:
            res = minimize(soft_min_max_objective, hex_subset.flatten(),
                          method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
            opt_pts = res.x.reshape(n, 2)
            score = get_score(opt_pts)
            if score > best_score:
                best_score = score
                best_points = opt_pts.copy()
        except Exception:
            pass
    
    # Fallback
    if best_points is None:
        # Use the best ring config
        for init_params in ring_configs:
            r0, r1, r2, r3 = init_params[0], init_params[1], init_params[2], init_params[3]
            off0, off1, off2, off3 = init_params[4], init_params[5], init_params[6], init_params[7]
            
            pts = []
            pts.append([r0 * np.cos(off0), r0 * np.sin(off0)])
            angles1 = np.linspace(0, 2*np.pi, 4, endpoint=False) + off1
            pts.extend(np.column_stack((r1 * np.cos(angles1), r1 * np.sin(angles1))))
            angles2 = np.linspace(0, 2*np.pi, 5, endpoint=False) + off2
            pts.extend(np.column_stack((r2 * np.cos(angles2), r2 * np.sin(angles2))))
            angles3 = np.linspace(0, 2*np.pi, 6, endpoint=False) + off3
            pts.extend(np.column_stack((r3 * np.cos(angles3), r3 * np.sin(angles3))))
            
            candidate = np.array(pts)
            score = get_score(candidate)
            if score > best_score:
                best_score = score
                best_points = candidate.copy()
        
        if best_points is None:
            best_points = grid_pts

    return best_points

# EVOLVE-BLOCK-END
