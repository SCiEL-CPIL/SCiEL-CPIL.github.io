import numpy as np
import time
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START
def _compute_radius(centers):
    """Compute the actual radius for a configuration."""
    c = centers
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist2, np.inf)
    min_pair_dist = np.sqrt(np.min(dist2))
    
    abs_c = np.abs(c)
    max_coord = np.max(abs_c, axis=1)
    min_wall_dist = np.min(0.5 - max_coord)
    
    return min(min_pair_dist / 2.0, min_wall_dist)

def _softmin_radius(centers, alpha=50.0):
    """
    Smooth approximation of the minimum radius using log-sum-exp.
    Returns a smooth lower bound on the actual radius.
    """
    c = centers
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist2, np.inf)
    
    # Soft-min of pair distances
    # Use log-sum-exp for stability
    dist = np.sqrt(dist2)
    mask = ~np.eye(N, dtype=bool)
    dist_pairs = dist[mask]
    
    # Soft-min: -log(sum(exp(-alpha * d))) / alpha
    # Clip to avoid overflow
    min_d = np.min(dist_pairs)
    exp_terms = np.exp(-alpha * (dist_pairs - min_d))
    log_sum = np.log(np.sum(exp_terms))
    softmin_pair = min_d - log_sum / alpha
    
    # Wall distances
    abs_c = np.abs(c)
    max_coord = np.max(abs_c, axis=1)
    d_walls = 0.5 - max_coord
    
    # Soft-min of wall distances
    min_w = np.min(d_walls)
    exp_w = np.exp(-alpha * (d_walls - min_w))
    log_sum_w = np.log(np.sum(exp_w))
    softmin_wall = min_w - log_sum_w / alpha
    
    # Combine: soft-min of pair_dist/2 and wall_dist
    # Use another soft-min
    a = softmin_pair / 2.0
    b = softmin_wall
    min_ab = min(a, b)
    exp_ab = np.exp(-alpha * (np.array([a, b]) - min_ab))
    log_sum_ab = np.log(np.sum(exp_ab))
    result = min_ab - log_sum_ab / alpha
    
    return result

def _smooth_objective(x_flat, alpha=50.0):
    """
    Objective to maximize: soft-min radius.
    We minimize the negative of this.
    """
    c = x_flat.reshape(N, 3)
    r_soft = _softmin_radius(c, alpha)
    return -r_soft

def _lloyd_step(centers, alpha=50.0, n_iter=50):
    """
    Voronoi-Lloyd iteration: move each center to the centroid of its Voronoi cell.
    This equalizes the packing.
    """
    from scipy.spatial import Voronoi
    
    current = centers.copy()
    
    for _ in range(n_iter):
        try:
            vor = Voronoi(current)
            
            # For each point, find the centroid of its Voronoi cell
            # Clip to the cube [-0.5, 0.5]^3
            new_centers = np.zeros_like(current)
            
            for i in range(N):
                # Get the vertices of the Voronoi region for point i
                region_idx = vor.point_region[i]
                region = vor.regions[region_idx]
                
                if len(region) == 0:
                    # Unbounded region, keep the point
                    new_centers[i] = current[i]
                    continue
                
                vertices = vor.vertices[region]
                if len(vertices) == 0:
                    new_centers[i] = current[i]
                    continue
                
                # Centroid of the Voronoi cell
                centroid = np.mean(vertices, axis=0)
                
                # Clip to the cube
                centroid = np.clip(centroid, -0.5 + 0.01, 0.5 - 0.01)
                
                new_centers[i] = centroid
            
            # Check if converged
            if np.max(np.abs(new_centers - current)) < 1e-10:
                current = new_centers
                break
            
            current = new_centers
            
        except Exception:
            # Voronoi can fail for degenerate cases
            break
    
    return current

def _local_search(centers, step_size=0.01, n_iter=1000):
    """
    Simple local search: try small random perturbations and accept if radius improves.
    """
    current = centers.copy()
    current_r = _compute_radius(current)
    best = current.copy()
    best_r = current_r
    
    rng = np.random.RandomState(42)
    
    for i in range(n_iter):
        # Random perturbation
        delta = rng.randn(N, 3) * step_size
        candidate = current + delta
        candidate = np.clip(candidate, -0.5 + 1e-6, 0.5 - 1e-6)
        
        cand_r = _compute_radius(candidate)
        
        if cand_r > current_r:
            current = candidate
            current_r = cand_r
            if cand_r > best_r:
                best = candidate.copy()
                best_r = cand_r
        else:
            # Metropolis acceptance with small probability
            if rng.rand() < 0.01:
                current = candidate
                current_r = cand_r
        
        # Decrease step size
        step_size *= 0.999
    
    return best, best_r

def _seed_fcc():
    """Generate an FCC-like initial configuration."""
    # FCC lattice points in a cube
    a = 1.0 / 3.0  # Lattice spacing
    points = []
    
    # Corner points
    for i in range(4):
        for j in range(4):
            for k in range(4):
                x = (i - 1.5) * a
                y = (j - 1.5) * a
                z = (k - 1.5) * a
                if abs(x) <= 0.5 and abs(y) <= 0.5 and abs(z) <= 0.5:
                    points.append([x, y, z])
    
    # Face center points
    for i in range(4):
        for j in range(4):
            x = (i - 1.5) * a
            y = (j - 1.5) * a
            z = 0.0
            if abs(x) <= 0.5 and abs(y) <= 0.5:
                points.append([x, y, z])
            x = (i - 1.5) * a
            y = 0.0
            z = (j - 1.5) * a
            if abs(x) <= 0.5 and abs(z) <= 0.5:
                points.append([x, y, z])
            x = 0.0
            y = (i - 1.5) * a
            z = (j - 1.5) * a
            if abs(y) <= 0.5 and abs(z) <= 0.5:
                points.append([x, y, z])
    
    points = np.array(points)
    
    # Remove duplicates and filter to cube
    # Simple deduplication
    unique = []
    for p in points:
        is_dup = False
        for u in unique:
            if np.linalg.norm(p - u) < 1e-6:
                is_dup = True
                break
        if not is_dup:
            unique.append(p)
    
    points = np.array(unique)
    
    # Keep only points inside the cube
    mask = (np.abs(points) <= 0.5).all(axis=1)
    points = points[mask]
    
    # If we have more than N, take the first N
    if len(points) >= N:
        return points[:N].copy()
    
    # If we have fewer, pad with random points
    if len(points) < N:
        rng = np.random.RandomState(42)
        extra = rng.uniform(-0.4, 0.4, size=(N - len(points), 3))
        points = np.vstack([points, extra])
    
    return points

def _seed_grid():
    """Generate a grid initial configuration."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()

def construct():
    start_time = time.time()
    timeout = 115  # Safety margin
    
    # Try multiple seeding strategies
    seeds = [_seed_fcc(), _seed_grid()]
    
    best_centers = None
    best_r = 0.0
    
    for seed in seeds:
        # Lloyd iteration to equalize
        lloyd_centers = _lloyd_step(seed, n_iter=30)
        
        # Smooth optimization to maximize soft-min radius
        x0 = lloyd_centers.flatten()
        bounds = [(-0.5 + 1e-6, 0.5 - 1e-6)] * (3 * N)
        
        # Use L-BFGS-B to maximize soft-min radius (minimize negative)
        result = minimize(
            _smooth_objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        opt_centers = result.x.reshape(N, 3)
        
        # Local search polish
        ls_centers, ls_r = _local_search(opt_centers, step_size=0.005, n_iter=500)
        
        # Check time
        if time.time() - start_time > timeout:
            break
        
        if ls_r > best_r:
            best_r = ls_r
            best_centers = ls_centers
    
    # Final polish with tighter tolerance
    if best_centers is not None and time.time() - start_time < timeout - 10:
        final_centers, final_r = _local_search(best_centers, step_size=0.002, n_iter=1000)
        if final_r >= best_r:
            best_centers = final_centers
    
    # Ensure valid output
    best_centers = np.clip(best_centers, -0.5 + 1e-6, 0.5 - 1e-6)
    
    return best_centers
# EVOLVE-BLOCK-END
