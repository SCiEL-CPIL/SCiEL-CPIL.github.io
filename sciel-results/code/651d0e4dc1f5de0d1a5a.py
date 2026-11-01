"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import Voronoi
from scipy.optimize import minimize

def _compute_radius(centers):
    """Compute the maximum radius for given centers."""
    if len(centers) != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 0.0
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    min_pair_dist = np.min(dists[np.triu_indices(N, k=1)])
    
    wall_dists = 0.5 - np.abs(centers)
    min_wall_dist = np.min(wall_dists)
    
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return r

def _softmin_radius(centers, beta=20.0):
    """
    Soft-min approximation of the radius.
    Uses log-sum-exp trick for numerical stability.
    """
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    
    # Pair distances (upper triangle)
    iu = np.triu_indices(N, k=1)
    pair_dists_sq = dists_sq[iu]
    
    # Wall distances
    wall_dists = 0.5 - np.abs(centers)
    # Square them for consistency
    wall_dists_sq = wall_dists**2
    
    # Combine all squared distances
    all_dists_sq = np.concatenate([pair_dists_sq, wall_dists_sq])
    
    # Soft-min: -1/beta * log(sum(exp(-beta * x)))
    # This approximates min(x) as beta -> infinity
    scaled = -beta * all_dists_sq
    max_scaled = np.max(scaled)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    softmin_sq = -log_sum / beta
    
    # The soft-min of squared distances approximates (min distance)^2
    # So soft-min of distances is sqrt(softmin_sq)
    softmin_dist = np.sqrt(np.maximum(softmin_sq, 1e-15))
    
    # The radius is half the min pair distance, but we also have wall constraint
    # Actually, let's be more careful. The radius is:
    # r = min(min_pair_dist/2, min_wall_dist)
    # Let's use a combined soft-min
    
    # Pair contribution: min_pair_dist / 2
    pair_softmin_sq = -1.0 / beta * (max_scaled_pair + np.log(np.sum(np.exp(-beta * pair_dists_sq - max_scaled_pair))))
    pair_softmin = np.sqrt(np.maximum(pair_softmin_sq, 1e-15)) / 2.0
    
    wall_softmin_sq = -1.0 / beta * (max_scaled_wall + np.log(np.sum(np.exp(-beta * wall_dists_sq - max_scaled_wall))))
    wall_softmin = np.sqrt(np.maximum(wall_softmin_sq, 1e-15))
    
    # Soft-min of the two
    vals = np.array([pair_softmin, wall_softmin])
    scaled_vals = -beta * vals
    max_sv = np.max(scaled_vals)
    combined_softmin = -(max_sv + np.log(np.sum(np.exp(scaled_vals - max_sv)))) / beta
    
    return combined_softmin

def _lloyd_step(centers, max_iter=30, time_limit=10.0):
    """
    Lloyd's algorithm: move each center to the centroid of its Voronoi cell
    clipped by the cube boundary.
    """
    start_time = time.time()
    centers = centers.copy()
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Compute Voronoi diagram
        try:
            vor = Voronoi(centers)
        except Exception:
            break
        
        new_centers = centers.copy()
        for i in range(N):
            # Get vertices of Voronoi cell i
            region_idx = vor.point_region[i]
            region = vor.regions[region_idx]
            
            if len(region) == 0 or -1 in region:
                # Unbounded cell - keep center or move towards center
                continue
            
            vertices = vor.vertices[region]
            
            # Clip vertices to cube [-0.5, 0.5]^3
            vertices = np.clip(vertices, -0.5, 0.5)
            
            # Compute centroid
            if len(vertices) > 0:
                centroid = np.mean(vertices, axis=0)
                # Clip to cube
                centroid = np.clip(centroid, -0.5, 0.5)
                new_centers[i] = centroid
        
        # Check for improvement
        old_score = _compute_radius(centers)
        new_score = _compute_radius(new_centers)
        
        if new_score >= old_score - 1e-12:
            centers = new_centers
        else:
            break
    
    return centers

def _metropolis_mc(centers, time_limit=15.0, rng=None):
    """
    Metropolis Monte Carlo for sphere packing.
    Try random moves and accept if they improve or with probability exp(-delta/T).
    """
    if rng is None:
        rng = np.random.default_rng()
    
    start_time = time.time()
    centers = centers.copy()
    best_centers = centers.copy()
    best_score = _compute_radius(centers)
    
    T = 0.005  # Initial temperature
    T_min = 0.0001
    step_size = 0.05
    n_accepted = 0
    n_total = 0
    
    while time.time() - start_time < time_limit:
        # Pick a random center to move
        i = rng.integers(N)
        
        # Random displacement
        delta = rng.normal(0, step_size, 3)
        new_center = centers[i] + delta
        
        # Keep within bounds
        new_center = np.clip(new_center, -0.499, 0.499)
        
        # Compute score change
        old_score = _compute_radius(centers)
        centers[i] = new_center
        new_score = _compute_radius(centers)
        
        n_total += 1
        delta_score = new_score - old_score
        
        if delta_score > 0:
            n_accepted += 1
        else:
            # Accept with probability exp(delta_score / T)
            if rng.random() < np.exp(delta_score / T):
                n_accepted += 1
            else:
                centers[i] = new_center - delta  # Revert
        
        # Update best
        if new_score > best_score:
            best_score = new_score
            best_centers = centers.copy()
        
        # Cool down
        T = max(T * 0.9995, T_min)
        step_size = max(0.005, step_size * 0.9998)
    
    return best_centers

def _softmin_polish(centers, time_limit=10.0):
    """
    Polish using soft-min objective with decreasing beta.
    """
    start_time = time.time()
    centers = centers.copy()
    
    # Decreasing beta schedule
    betas = [5.0, 10.0, 20.0, 50.0, 100.0]
    
    for beta in betas:
        if time.time() - start_time > time_limit:
            break
        
        def obj(x):
            c = x.reshape(N, 3)
            return -_softmin_radius(c, beta=beta)
        
        def obj_grad(x):
            c = x.reshape(N, 3)
            # Numerical gradient
            eps = 1e-6
            grad = np.zeros_like(c)
            for i in range(N):
                for j in range(3):
                    c_plus = c.copy()
                    c_plus[i, j] += eps
                    f_plus = -_softmin_radius(c_plus, beta=beta)
                    
                    c_minus = c.copy()
                    c_minus[i, j] -= eps
                    f_minus = -_softmin_radius(c_minus, beta=beta)
                    
                    grad[i, j] = (f_plus - f_minus) / (2 * eps)
            return grad.flatten()
        
        x0 = centers.flatten()
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        state = {'stop': False}
        def callback(xk):
            if time.time() - start_time > time_limit:
                state['stop'] = True
                return True
            return False
        
        try:
            res = minimize(obj, x0, method='L-BFGS-B', jac=obj_grad, bounds=bounds,
                           options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6, 'callback': callback})
            new_centers = res.x.reshape(N, 3)
            new_centers = np.clip(new_centers, -0.5, 0.5)
            
            old_score = _compute_radius(centers)
            new_score = _compute_radius(new_centers)
            
            if new_score >= old_score - 1e-12:
                centers = new_centers
        except Exception:
            pass
    
    return centers

def _local_search(centers, time_limit=8.0, rng=None):
    """
    Local search with adaptive step size.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    start_time = time.time()
    centers = centers.copy()
    best_score = _compute_radius(centers)
    
    step_size = 0.01
    no_improve = 0
    
    while time.time() - start_time < time_limit:
        i = rng.integers(N)
        delta = rng.normal(0, step_size, 3)
        new_center = np.clip(centers[i] + delta, -0.499, 0.499)
        
        old_score = _compute_radius(centers)
        centers[i] = new_center
        new_score = _compute_radius(centers)
        
        if new_score > old_score:
            no_improve = 0
            step_size = min(0.02, step_size * 1.1)
            best_score = new_score
        else:
            centers[i] = new_center - delta  # Revert
            no_improve += 1
            if no_improve > 50:
                step_size *= 0.9
                no_improve = 0
                if step_size < 0.001:
                    break
    
    return centers

def _seed_fcc():
    """
    Generate FCC-like seed points.
    FCC has 4 atoms per unit cell. We'll use a subset.
    """
    # FCC basis in [0,1]^3
    fcc_basis = np.array([
        [0, 0, 0],
        [0.5, 0.5, 0],
        [0.5, 0, 0.5],
        [0, 0.5, 0.5]
    ])
    
    # Generate points in a 3x3x3 grid of FCC cells
    points = []
    for ix in range(3):
        for iy in range(3):
            for iz in range(3):
                for b in fcc_basis:
                    p = np.array([ix, iy, iz]) + b
                    if p[0] <= 1.0 and p[1] <= 1.0 and p[2] <= 1.0:
                        # Map to [-0.5, 0.5]^3
                        p_mapped = p - 0.5
                        points.append(p_mapped)
    
    points = np.array(points)
    
    # If we have more than N, randomly sample
    if len(points) > N:
        idx = np.random.default_rng(42).permutation(len(points))[:N]
        points = points[idx]
    elif len(points) < N:
        # Fill remaining with random points
        remaining = N - len(points)
        rng = np.random.default_rng(42)
        extra = rng.uniform(-0.45, 0.45, (remaining, 3))
        points = np.vstack([points, extra])
    
    return points

def _seed_grid():
    """Generate a jittered grid seed."""
    m = 5
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid_full = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(grid_full))[:N]
    centers = grid_full[idx].copy()
    centers += rng.normal(0, 0.02, centers.shape)
    centers = np.clip(centers, -0.49, 0.49)
    
    return centers

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_score = 0.0
    best_centers = None
    
    rng = np.random.default_rng(42)
    
    # Multiple initialization strategies
    seeds = []
    seeds.append(_seed_fcc())
    seeds.append(_seed_grid())
    
    # Random seed
    seeds.append(rng.uniform(-0.45, 0.45, (N, 3)))
    
    for seed_idx, centers in enumerate(seeds):
        if time.time() > deadline - 30:
            break
        
        centers = np.clip(centers, -0.49, 0.49)
        
        # Phase 1: Lloyd equalization
        time_left = deadline - time.time()
        lloyd_time = min(15.0, time_left * 0.25)
        if lloyd_time > 1.0:
            centers = _lloyd_step(centers, max_iter=30, time_limit=lloyd_time)
        
        # Phase 2: Metropolis MC for escape from local minima
        time_left = deadline - time.time()
        mc_time = min(20.0, time_left * 0.3)
        if mc_time > 1.0:
            centers = _metropolis_mc(centers, time_limit=mc_time, rng=rng)
        
        # Phase 3: Soft-min polish
        time_left = deadline - time.time()
        polish_time = min(15.0, time_left * 0.3)
        if polish_time > 1.0:
            centers = _softmin_polish(centers, time_limit=polish_time)
        
        # Phase 4: Local search fine-tuning
        time_left = deadline - time.time()
        ls_time = min(10.0, time_left * 0.2)
        if ls_time > 0.5:
            centers = _local_search(centers, time_limit=ls_time, rng=rng)
        
        score = _compute_radius(centers)
        
        if score > best_score:
            best_score = score
            best_centers = centers.copy()
    
    # Final polish on best candidate
    if best_centers is not None:
        time_left = deadline - time.time()
        if time_left > 5.0:
            final_time = time_left - 2.0
            # Additional local search
            refined = _local_search(best_centers, time_limit=final_time * 0.5, rng=rng)
            score = _compute_radius(refined)
            if score > best_score:
                best_score = score
                best_centers = refined.copy()
            
            # Final soft-min polish with high beta
            time_left = deadline - time.time()
            if time_left > 3.0:
                refined = _softmin_polish(best_centers, time_limit=time_left - 1.0)
                score = _compute_radius(refined)
                if score > best_score:
                    best_score = score
                    best_centers = refined.copy()
    
    if best_centers is None:
        # Fallback
        best_centers = _seed_grid()
    
    return best_centers
# EVOLVE-BLOCK-END
