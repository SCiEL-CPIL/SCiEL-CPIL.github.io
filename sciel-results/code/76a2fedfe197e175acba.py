import numpy as np
import time
from scipy.spatial import cKDTree

N = 100

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin (5s before 150s limit)
    
    def get_radius(centers):
        """Compute the maximum common radius for given centers."""
        if not np.all(np.isfinite(centers)):
            return 0.0
        if np.any(centers > 0.5) or np.any(centers < -0.5):
            return 0.0
        # Wall distance
        wall_dists = 0.5 - np.abs(centers)
        r_wall = np.min(wall_dists)
        # Pairwise distance using KDTree for efficiency
        tree = cKDTree(centers)
        # Get all pairwise distances (upper triangle)
        dists, _ = tree.query(centers, k=2)  # k=2 includes self, so take second column
        r_pair = np.min(dists[:, 1]) / 2.0
        return min(r_wall, r_pair)

    def optimize_spheres(centers, max_iters=2000, time_limit=None):
        """Optimize sphere packing using gradient ascent on the minimum radius."""
        centers = centers.copy()
        r = get_radius(centers)
        best_centers = centers.copy()
        best_r = r
        
        # Track best radius
        for it in range(max_iters):
            if time_limit is not None and time.time() > time_limit:
                break
            
            # Compute current radius and identify bottleneck constraints
            wall_dists = 0.5 - np.abs(centers)
            r_wall = np.min(wall_dists)
            
            tree = cKDTree(centers)
            dists, _ = tree.query(centers, k=2)
            min_pair_dist = np.min(dists[:, 1])
            r_pair = min_pair_dist / 2.0
            
            current_r = min(r_wall, r_pair)
            
            # If no improvement possible, break
            if current_r <= best_r * 1.0001:
                if it > 100:
                    break
            
            # Compute gradient: push centers to increase minimum radius
            grad = np.zeros_like(centers)
            
            # Wall constraints: if wall_dist < r, push inward
            for k in range(3):
                d_wall = 0.5 - np.abs(centers[:, k])
                # Only apply force to centers close to wall
                mask = d_wall < current_r + 0.01
                if np.any(mask):
                    # Force magnitude proportional to how close to wall
                    f_mag = np.clip((current_r + 0.01 - d_wall[mask]) * 0.5, 0, None)
                    # Direction: toward center (negative of sign)
                    dirs = -np.sign(centers[mask, k])
                    dirs[dirs == 0] = 1
                    grad[mask, k] += f_mag * dirs
            
            # Pair constraints: if dist < 2*r, push apart
            # Get all pairs within target distance
            target_dist = 2 * (current_r + 0.01)
            pairs = tree.query_pairs(target_dist)
            
            for i, j in pairs:
                if i == j:
                    continue
                diff = centers[i] - centers[j]
                d = np.linalg.norm(diff)
                if d > 1e-10 and d < target_dist:
                    # Force proportional to overlap
                    f_mag = (target_dist - d) / d * 0.5
                    grad[i] += f_mag * diff
                    grad[j] -= f_mag * diff
            
            # Normalize gradient to avoid huge steps
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-10:
                grad = grad / grad_norm
            
            # Adaptive step size
            base_step = 0.05
            decay = 1.0 / (1.0 + it / 50.0)
            step = base_step * decay
            
            # Try the step
            new_centers = centers + step * grad
            new_centers = np.clip(new_centers, -0.499, 0.499)
            
            new_r = get_radius(new_centers)
            
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
                new_r = get_radius(new_centers)
                if new_r > r:
                    r = new_r
                    centers = new_centers
                    if r > best_r:
                        best_r = r
                        best_centers = centers.copy()
        
        return best_centers, best_r

    # Generate initial configurations
    np.random.seed(42)
    
    best_centers = None
    best_radius = 0.0
    
    # Configuration 1: Regular grid with jitter
    nx, ny, nz = 5, 5, 4
    xs = np.linspace(-0.42, 0.42, nx)
    ys = np.linspace(-0.42, 0.42, ny)
    zs = np.linspace(-0.42, 0.42, nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    base_centers = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    
    # Configuration 2: Face-centered cubic-like
    # Place spheres at FCC lattice points
    a = 0.7  # lattice constant
    fcc_points = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                # Corner points
                fcc_points.append([i*a, j*a, k*a])
                # Face centers
                fcc_points.append([i*a, j*a + a/2, k*a + a/2])
                fcc_points.append([i*a + a/2, j*a, k*a + a/2])
                fcc_points.append([i*a + a/2, j*a + a/2, k*a])
    
    fcc_array = np.array(fcc_points)
    # Filter to keep only points inside cube and select 100
    mask = np.all(np.abs(fcc_array) <= 0.45, axis=1)
    fcc_array = fcc_array[mask]
    if len(fcc_array) >= N:
        # Select 100 points
        idx = np.random.choice(len(fcc_array), N, replace=False)
        fcc_centers = fcc_array[idx]
    else:
        fcc_centers = base_centers.copy()
    
    # Try multiple initializations
    initializations = []
    
    # Grid with various jitter levels
    for jitter in [0.0, 0.02, 0.04, 0.06]:
        centers = base_centers + np.random.normal(0, jitter, base_centers.shape)
        centers = np.clip(centers, -0.49, 0.49)
        initializations.append(centers)
    
    # FCC with jitter
    for jitter in [0.0, 0.02, 0.04]:
        centers = fcc_centers + np.random.normal(0, jitter, fcc_centers.shape)
        centers = np.clip(centers, -0.49, 0.49)
        initializations.append(centers)
    
    # Random initialization
    for _ in range(3):
        centers = np.random.uniform(-0.45, 0.45, (N, 3))
        initializations.append(centers)
    
    # Optimize each initialization
    for idx, init_centers in enumerate(initializations):
        if time.time() > deadline - 5:
            break
        
        # Determine time budget for this optimization
        remaining = deadline - 5 - time.time()
        if remaining < 2:
            break
        
        # Allocate time based on initialization quality
        r_init = get_radius(init_centers)
        
        # More promising starts get more time
        if r_init > 0.08:
            max_iters = 3000
        elif r_init > 0.06:
            max_iters = 2000
        else:
            max_iters = 1000
        
        opt_centers, r = optimize_spheres(init_centers, max_iters=max_iters, time_limit=deadline - 3)
        
        if r > best_radius:
            best_radius = r
            best_centers = opt_centers
    
    # Final polish on the best solution if time permits
    if best_centers is not None and time.time() < deadline - 10:
        remaining = deadline - 10 - time.time()
        if remaining > 5:
            # Do additional local optimization
            opt_centers, r = optimize_spheres(best_centers, max_iters=2000, time_limit=deadline - 3)
            if r > best_radius:
                best_radius = r
                best_centers = opt_centers
    
    # Ensure valid output
    if best_centers is None or not np.all(np.isfinite(best_centers)):
        return base_centers
    
    # Final clip to ensure validity
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers
