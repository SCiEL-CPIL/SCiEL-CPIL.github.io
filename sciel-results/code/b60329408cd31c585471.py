import numpy as np
import time
from scipy.spatial import cKDTree

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    def compute_radius(c):
        """Compute the maximum common radius for given centers."""
        if not np.all(np.isfinite(c)):
            return 0.0
        if np.any(c > 0.5) or np.any(c < -0.5):
            return 0.0
        # Wall distance
        wall_dists = 0.5 - np.abs(c)
        r_wall = np.min(wall_dists)
        
        # Pairwise distance using cKDTree for efficiency
        tree = cKDTree(c)
        # Query each point to find its nearest neighbor (k=2 includes self)
        dists, _ = tree.query(c, k=2)
        r_pair = np.min(dists[:, 1]) / 2.0
        
        return min(r_wall, r_pair)

    def repulsion_optimize(centers, max_iters=5000, time_limit=None):
        """
        Force-directed local search to maximize the minimum radius.
        This directly resolves closest-pair and wall-contact constraints.
        """
        centers = centers.copy()
        r = compute_radius(centers)
        best_centers = centers.copy()
        best_r = r
        
        for it in range(max_iters):
            if time_limit is not None and time.time() > time_limit:
                break
            
            # Compute current radius and identify constraints
            wall_dists = 0.5 - np.abs(centers)
            r_wall = np.min(wall_dists)
            
            tree = cKDTree(centers)
            dists, _ = tree.query(centers, k=2)
            min_pair_dist = np.min(dists[:, 1])
            r_pair = min_pair_dist / 2.0
            
            current_r = min(r_wall, r_pair)
            
            # If we've stalled significantly, break
            if it > 100 and current_r <= best_r * 1.0001:
                break
            
            # Compute force/gradient
            grad = np.zeros_like(centers)
            
            # Wall constraints: push inward if close to wall
            # Target distance is slightly larger than current radius to create pressure
            target_r = current_r + 0.005
            for k in range(3):
                d_wall = 0.5 - np.abs(centers[:, k])
                mask = d_wall < target_r
                if np.any(mask):
                    # Force magnitude proportional to violation
                    f_mag = np.clip((target_r - d_wall[mask]) * 2.0, 0, None)
                    # Direction: toward center (negative of sign of coordinate)
                    dirs = -np.sign(centers[mask, k])
                    dirs[dirs == 0] = 1.0
                    grad[mask, k] += f_mag * dirs
            
            # Pair constraints: push apart if closer than target distance
            target_dist = 2 * target_r
            # Get all pairs within target distance
            pairs = tree.query_pairs(target_dist)
            
            for i, j in pairs:
                if i == j:
                    continue
                diff = centers[i] - centers[j]
                d = np.linalg.norm(diff)
                if d > 1e-10 and d < target_dist:
                    # Force proportional to overlap
                    f_mag = (target_dist - d) / d * 1.0
                    grad[i] += f_mag * diff
                    grad[j] -= f_mag * diff
            
            # Normalize gradient
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-10:
                grad = grad / grad_norm
            else:
                break # No forces acting
            
            # Adaptive step size
            base_step = 0.05
            decay = 1.0 / (1.0 + it / 100.0)
            step = base_step * decay
            
            # Try the step
            new_centers = centers + step * grad
            new_centers = np.clip(new_centers, -0.499, 0.499)
            
            new_r = compute_radius(new_centers)
            
            if new_r > r:
                r = new_r
                centers = new_centers
                if r > best_r:
                    best_r = r
                    best_centers = centers.copy()
            else:
                # Try smaller step
                step *= 0.5
                new_centers = centers + step * grad
                new_centers = np.clip(new_centers, -0.499, 0.499)
                new_r = compute_radius(new_centers)
                if new_r > r:
                    r = new_r
                    centers = new_centers
                    if r > best_r:
                        best_r = r
                        best_centers = centers.copy()
                else:
                    # Try even smaller step or perturb
                    step *= 0.5
                    if step < 1e-6:
                        # Add small random jitter to escape local minima
                        centers += np.random.normal(0, 0.01, centers.shape)
                        centers = np.clip(centers, -0.499, 0.499)
                        r = compute_radius(centers)
                        if r > best_r:
                            best_r = r
                            best_centers = centers.copy()
        
        return best_centers, best_r

    # 1. Initial Configurations
    np.random.seed(42)
    
    best_centers = None
    best_radius = 0.0
    
    initializations = []
    
    # Configuration 1: 5x5x4 Grid
    nx, ny, nz = 5, 5, 4
    xs = np.linspace(-0.4, 0.4, nx)
    ys = np.linspace(-0.4, 0.4, ny)
    zs = np.linspace(-0.3, 0.3, nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    base_centers = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    
    # Add jittered versions
    for jitter in [0.0, 0.01, 0.02, 0.05]:
        centers = base_centers + np.random.normal(0, jitter, base_centers.shape)
        centers = np.clip(centers, -0.49, 0.49)
        initializations.append(centers)
    
    # Configuration 2: Random Uniform
    for _ in range(3):
        centers = np.random.uniform(-0.45, 0.45, (N, 3))
        initializations.append(centers)
        
    # Configuration 3: FCC-like lattice (might be better for dense packing)
    # Generate a grid of FCC points
    a = 0.35 # lattice constant
    fcc_points = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            for k in range(-2, 3):
                fcc_points.append([i*a, j*a, k*a])
                fcc_points.append([i*a + a/2, j*a + a/2, k*a])
                fcc_points.append([i*a + a/2, j*a, k*a + a/2])
                fcc_points.append([i*a, j*a + a/2, k*a + a/2])
    
    fcc_array = np.array(fcc_points)
    mask = np.all(np.abs(fcc_array) <= 0.49, axis=1)
    fcc_array = fcc_array[mask]
    
    if len(fcc_array) >= N:
        # Randomly select 100 points
        idx = np.random.choice(len(fcc_array), N, replace=False)
        fcc_centers = fcc_array[idx]
        for jitter in [0.0, 0.02]:
            centers = fcc_centers + np.random.normal(0, jitter, fcc_centers.shape)
            centers = np.clip(centers, -0.49, 0.49)
            initializations.append(centers)
            
    # 2. Optimization Loop
    # We prioritize initializations that start with a higher radius
    # but we also want to explore different basins.
    # Sort by initial radius (descending) to spend time on promising ones first?
    # Or just iterate through them all with a time budget.
    
    # Let's allocate time based on remaining budget.
    # We have ~145s.
    # Let's say we try each initialization for a fixed amount of time, or until convergence.
    
    for idx, init_centers in enumerate(initializations):
        if time.time() > deadline - 5:
            break
        
        # Estimate time per optimization.
        # If we have 10 initializations, we have ~14s each.
        # But some might converge faster.
        # Let's just run repulsion_optimize with a time limit.
        
        opt_centers, r = repulsion_optimize(
            init_centers, 
            max_iters=20000, 
            time_limit=deadline - 10 - (len(initializations) - idx) * 1.0 # Rough budget
        )
        
        if r > best_radius:
            best_radius = r
            best_centers = opt_centers.copy()
            
    # 3. Final Polish
    # If we have time left, do a final polish on the best solution found.
    if best_centers is not None and time.time() < deadline - 5:
        opt_centers, r = repulsion_optimize(
            best_centers,
            max_iters=50000,
            time_limit=deadline - 1
        )
        if r > best_radius:
            best_radius = r
            best_centers = opt_centers.copy()

    return best_centers
