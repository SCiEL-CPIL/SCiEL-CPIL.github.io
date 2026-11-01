"""
Packing 200 equal circles in a unit square using Force-Directed Relaxation 
and Bisection on Radius.

Strategy:
1. Initialize with a hexagonal lattice.
2. Use a bisection search on the target radius r.
3. For a given r, use a force-directed method to find a configuration where 
   all circles are non-overlapping and inside the square.
4. The force-directed method minimizes the sum of squared penetration depths.
"""
import numpy as np
import time
from scipy.spatial import cKDTree

N = 200
DEADLINE = 115.0  # Safety margin for 120s limit

def _check_validity(centers, r):
    """Check if centers form a valid packing with radius r."""
    if not np.all(np.isfinite(centers)):
        return False
    if centers.shape != (N, 2):
        return False
    
    # Wall constraints
    walls = np.minimum(
        np.minimum(0.5 - centers[:, 0], centers[:, 0] + 0.5),
        np.minimum(0.5 - centers[:, 1], centers[:, 1] + 0.5)
    )
    if np.min(walls) < r - 1e-9:
        return False
        
    # Pair constraints
    tree = cKDTree(centers)
    # Query 2 nearest neighbors to get min distance
    # Note: k=2 gives self and nearest. 
    dists, _ = tree.query(centers, k=2)
    min_pair_dist = np.min(dists[:, 1])
    
    if min_pair_dist < 2.0 * r - 1e-9:
        return False
        
    return True

def _compute_radius(centers):
    """Compute the exact largest common radius for given centers."""
    if not np.all(np.isfinite(centers)):
        return 0.0
    if centers.shape != (N, 2):
        return 0.0
        
    walls = np.minimum(
        np.minimum(0.5 - centers[:, 0], centers[:, 0] + 0.5),
        np.minimum(0.5 - centers[:, 1], centers[:, 1] + 0.5)
    )
    wall_min = np.min(walls)
    
    tree = cKDTree(centers)
    dists, _ = tree.query(centers, k=2)
    pair_min = np.min(dists[:, 1]) / 2.0
    
    return min(wall_min, pair_min)

def _hex_init(n):
    """Generate initial positions using a hexagonal lattice pattern."""
    best = None
    best_r = -1
    
    # Try a few scales and orientations
    for orientation in [0, np.pi/6, np.pi/3, np.pi/12]:
        m = int(np.ceil(np.sqrt(n)))
        
        for scale in np.linspace(0.9, 1.1, 10):
            s = scale / m
            dx = s
            dy = s * np.sqrt(3) / 2
            
            pts = []
            for i in range(m):
                for j in range(m):
                    x = (j - (m-1)/2) * dx
                    y = (i - (m-1)/2) * dy
                    if i % 2 == 1:
                        x += dx / 2
                    pts.append([x, y])
            
            pts = np.array(pts[:n])
            
            if orientation != 0:
                c, s_rot = np.cos(orientation), np.sin(orientation)
                R = np.array([[c, -s_rot], [s_rot, c]])
                pts = pts @ R.T
            
            # Scale to fit within bounds with some margin
            max_abs = np.max(np.abs(pts))
            if max_abs > 0:
                pts = pts * (0.48 / max_abs)
            
            r = _compute_radius(pts)
            if r > best_r:
                best_r = r
                best = pts.copy()
    
    if best is None:
        # Fallback: square grid
        m = int(np.ceil(np.sqrt(n)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best = grid[:n]
        
    return best

def _force_directed_relax(centers, r_target, max_iters=200, time_limit=10.0):
    """
    Relax the configuration to satisfy constraints for radius r_target.
    Uses a repulsive force model: if distance < 2*r, push apart.
    Also applies a force to push away from walls if distance < r.
    
    Returns: (success, new_centers)
    """
    start_time = time.time()
    pts = centers.copy()
    
    # Learning rate schedule
    step_size = 0.001
    decay = 0.995
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
            
        forces = np.zeros_like(pts)
        
        # 1. Pairwise repulsion
        tree = cKDTree(pts)
        # Query neighbors within a cutoff slightly larger than 2*r
        cutoff = 2.0 * r_target * 1.5
        pairs = tree.query_pairs(cutoff)
        
        for i, j in pairs:
            diff = pts[i] - pts[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-8:
                # Random nudge to avoid singularity
                diff = np.array([np.random.randn(), np.random.randn()])
                dist = np.linalg.norm(diff)
                
            if dist < 2.0 * r_target:
                # Repulsion strength proportional to penetration
                penetration = (2.0 * r_target - dist)
                # Normalize direction
                dir_vec = diff / dist
                force = penetration * dir_vec
                forces[i] += force
                forces[j] -= force
        
        # 2. Wall repulsion
        # Left wall
        d_left = pts[:, 0] + 0.5
        mask = d_left < r_target
        if np.any(mask):
            forces[mask, 0] += (r_target - d_left[mask])
            
        # Right wall
        d_right = 0.5 - pts[:, 0]
        mask = d_right < r_target
        if np.any(mask):
            forces[mask, 0] -= (r_target - d_right[mask])
            
        # Bottom wall
        d_bottom = pts[:, 1] + 0.5
        mask = d_bottom < r_target
        if np.any(mask):
            forces[mask, 1] += (r_target - d_bottom[mask])
            
        # Top wall
        d_top = 0.5 - pts[:, 1]
        mask = d_top < r_target
        if np.any(mask):
            forces[mask, 1] -= (r_target - d_top[mask])
        
        # Apply forces
        pts += step_size * forces
        
        # Clip to hard bounds to prevent divergence
        pts[:, 0] = np.clip(pts[:, 0], -0.5, 0.5)
        pts[:, 1] = np.clip(pts[:, 1], -0.5, 0.5)
        
        # Check convergence: max violation
        # Re-compute violations efficiently
        tree = cKDTree(pts)
        dists, _ = tree.query(pts, k=2)
        min_pair_dist = np.min(dists[:, 1])
        
        walls = np.minimum(
            np.minimum(0.5 - pts[:, 0], pts[:, 0] + 0.5),
            np.minimum(0.5 - pts[:, 1], pts[:, 1] + 0.5)
        )
        min_wall_dist = np.min(walls)
        
        current_r = min(min_wall_dist, min_pair_dist / 2.0)
        
        if current_r >= r_target - 1e-7:
            return True, pts
            
        # Decay step size
        step_size *= decay
        
    return False, pts

def _optimize_packing(centers, time_limit=60.0):
    """
    Main optimization loop: Bisection on radius using force-directed relaxation.
    """
    start_time = time.time()
    best_centers = centers.copy()
    best_r = _compute_radius(best_centers)
    
    # Initial lower bound is the current radius
    r_low = best_r
    # Upper bound: estimate based on density
    # Area per circle ~ 1/N. r ~ sqrt(1/(N*pi))
    r_est = np.sqrt(1.0 / (N * np.pi))
    r_high = r_est * 1.1
    
    # Ensure r_low < r_high
    if r_low >= r_high:
        r_high = r_low * 1.05
        
    # Bisection loop
    while time.time() - start_time < time_limit - 5.0:
        r_mid = (r_low + r_high) / 2.0
        
        # Try to find a valid configuration for r_mid
        # Use a short time limit for the relaxation to keep bisection fast
        relax_time = min(5.0, time_limit - (time.time() - start_time) - 2.0)
        if relax_time < 1.0:
            break
            
        success, new_centers = _force_directed_relax(best_centers, r_mid, max_iters=100, time_limit=relax_time)
        
        if success:
            r_low = r_mid
            best_centers = new_centers
            best_r = r_mid
        else:
            r_high = r_mid
            
        # Convergence check
        if r_high - r_low < 1e-7:
            break
            
    # Final polish: try to improve best_r with a longer relaxation on the best found centers
    if time.time() - start_time < time_limit - 2.0:
        # Try to relax at current best_r to ensure validity, or slightly higher
        # We just re-verify and maybe nudge
        success, polished_centers = _force_directed_relax(best_centers, best_r, max_iters=200, time_limit=time_limit - (time.time() - start_time) - 1.0)
        if success:
            final_r = _compute_radius(polished_centers)
            if final_r > best_r:
                best_r = final_r
                best_centers = polished_centers
            else:
                # Keep polished centers if they are valid for best_r
                best_centers = polished_centers
        else:
            # If relaxation failed, stick to the previous best_centers which were valid for best_r
            # (assuming best_r was achieved in a previous step)
            pass
            
    return best_centers, best_r

def construct():
    start_time = time.time()
    
    # 1. Initialization
    centers = _hex_init(N)
    
    # 2. Optimization
    centers, r = _optimize_packing(centers, time_limit=DEADLINE - 5.0)
    
    # 3. Final Validation and Cleanup
    # Ensure points are strictly inside or on boundary
    centers[:, 0] = np.clip(centers[:, 0], -0.5, 0.5)
    centers[:, 1] = np.clip(centers[:, 1], -0.5, 0.5)
    
    # Verify validity
    if not _check_validity(centers, r):
        # If invalid, reduce r slightly and re-check
        # This shouldn't happen often if _force_directed_relax is robust
        for scale in [0.99, 0.98, 0.95]:
            r_test = r * scale
            if _check_validity(centers, r_test):
                r = r_test
                break
        else:
            # Fallback to a very safe radius
            r = 0.01
            # Ensure validity
            if not _check_validity(centers, r):
                # Last resort: grid
                m = int(np.ceil(np.sqrt(N)))
                ticks = (np.arange(m) + 0.5) / m - 0.5
                grid = np.array([(x, y) for y in ticks for x in ticks])
                centers = grid[:N]
                r = _compute_radius(centers)
                
    return centers

if __name__ == "__main__":
    # Simple test
    t0 = time.time()
    c = construct()
    t1 = time.time()
    r = _compute_radius(c)
    print(f"Time: {t1-t0:.2f}s, Radius: {r:.8f}")
