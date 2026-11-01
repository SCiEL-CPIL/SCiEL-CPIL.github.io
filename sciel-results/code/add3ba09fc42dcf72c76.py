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

def _energy_and_grad(centers, r):
    """
    Compute the squared violation energy and its gradient for a target radius r.
    Energy = sum over pairs (max(0, 2r - d_ij))^2 + sum over walls (max(0, r - d_ik))^2
    """
    c = centers
    n = len(c)
    
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist2, np.inf)
    
    dist = np.sqrt(dist2)
    mask = ~np.eye(n, dtype=bool)
    
    # Only upper triangle for pairs
    iu = np.triu_indices(n, k=1)
    pair_dists = dist[iu]
    pair_diffs = diff[iu] # shape (num_pairs, 3)
    
    # Violations for pairs: v_ij = max(0, 2r - d_ij)
    v_pairs = np.maximum(0, 2*r - pair_dists)
    pair_energy = np.sum(v_pairs ** 2)
    
    # Gradient for pairs
    # d/dx_i (max(0, 2r - d_ij)^2) = 2 * (2r - d_ij) * (-1) * (x_i - x_j) / d_ij  if 2r > d_ij
    # = -2 * (2r - d_ij) * (x_i - x_j) / d_ij
    # Let u_ij = x_i - x_j. d_ij = ||u_ij||
    # grad_i = -2 * (2r - d_ij) * u_ij / d_ij
    # grad_j = -2 * (2r - d_ij) * (-u_ij) / d_ij = 2 * (2r - d_ij) * u_ij / d_ij
    
    grad = np.zeros_like(c)
    active_pairs = v_pairs > 0
    if np.any(active_pairs):
        active_dists = pair_dists[active_pairs]
        active_diffs = pair_diffs[active_pairs]
        active_v = v_pairs[active_pairs]
        
        # factor = 2 * (2r - d) / d
        factors = 2.0 * active_v / active_dists
        # contribution to grad_i is -factors * active_diffs
        # contribution to grad_j is +factors * active_diffs
        
        # We need to scatter these back to the correct indices
        i_idx, j_idx = iu[0][active_pairs], iu[1][active_pairs]
        
        # Use np.add.at for accumulation
        np.add.at(grad, i_idx, -factors[:, np.newaxis] * active_diffs)
        np.add.at(grad, j_idx, factors[:, np.newaxis] * active_diffs)
    
    # Wall distances
    # d_ik = 0.5 - |c_ik|
    # Violation v_ik = max(0, r - d_ik) = max(0, r - (0.5 - |c_ik|)) = max(0, r - 0.5 + |c_ik|)
    # Let's stick to the definition: d_wall = 0.5 - |c|
    abs_c = np.abs(c)
    wall_dists = 0.5 - abs_c
    v_walls = np.maximum(0, r - wall_dists)
    wall_energy = np.sum(v_walls ** 2)
    
    # Gradient for walls
    # d/dx_ik (max(0, r - (0.5 - |x_ik|))^2)
    # = 2 * (r - (0.5 - |x_ik|)) * (d/dx_ik (0.5 - |x_ik|))
    # = 2 * v_ik * (-sign(x_ik))
    # Note: if x_ik = 0, sign is 0, but the function is not differentiable. 
    # However, in practice, centers rarely land exactly on 0. 
    # We can use subgradient 0 or just sign. Let's use sign.
    active_walls = v_walls > 0
    if np.any(active_walls):
        active_v = v_walls[active_walls]
        active_signs = np.sign(c[active_walls])
        # grad = 2 * v * (-sign(x))
        np.add.at(grad, active_walls, -2.0 * active_v[:, np.newaxis] * active_signs)
    
    total_energy = pair_energy + wall_energy
    return total_energy, grad

def _minimize_for_radius(r, initial_centers, time_limit=10.0):
    """
    Minimize the violation energy for a fixed radius r.
    Returns the resulting centers.
    """
    start_time = time.time()
    x0 = initial_centers.flatten()
    bounds = [(-0.5, 0.5)] * (3 * N)
    
    def obj(x):
        c = x.reshape(N, 3)
        e, _ = _energy_and_grad(c, r)
        return e
    
    def grad(x):
        c = x.reshape(N, 3)
        _, g = _energy_and_grad(c, r)
        return g.flatten()
    
    state = {'stop': False}
    def callback(xk):
        if time.time() - start_time > time_limit:
            state['stop'] = True
            return True
        return False
    
    try:
        res = minimize(obj, x0, jac=grad, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'callback': callback, 'disp': False})
        return res.x.reshape(N, 3)
    except Exception:
        return initial_centers

def _lloyd_step(centers, max_iter=20, time_limit=10.0):
    """
    Lloyd's algorithm: move each center to the centroid of its Voronoi cell
    clipped by the cube boundary.
    """
    start_time = time.time()
    centers = centers.copy()
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        try:
            vor = Voronoi(centers)
        except Exception:
            break
        
        new_centers = centers.copy()
        for i in range(N):
            region_idx = vor.point_region[i]
            region = vor.regions[region_idx]
            
            if len(region) == 0 or -1 in region:
                continue
            
            vertices = vor.vertices[region]
            vertices = np.clip(vertices, -0.5, 0.5)
            
            if len(vertices) > 0:
                centroid = np.mean(vertices, axis=0)
                centroid = np.clip(centroid, -0.5, 0.5)
                new_centers[i] = centroid
        
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
    """
    if rng is None:
        rng = np.random.default_rng()
    
    start_time = time.time()
    centers = centers.copy()
    best_centers = centers.copy()
    best_score = _compute_radius(centers)
    
    T = 0.005
    T_min = 0.0001
    step_size = 0.05
    
    while time.time() - start_time < time_limit:
        i = rng.integers(N)
        delta = rng.normal(0, step_size, 3)
        new_center = centers[i] + delta
        new_center = np.clip(new_center, -0.499, 0.499)
        
        old_score = _compute_radius(centers)
        centers[i] = new_center
        new_score = _compute_radius(centers)
        
        delta_score = new_score - old_score
        
        if delta_score > 0:
            pass
        else:
            if rng.random() < np.exp(delta_score / T):
                pass
            else:
                centers[i] = new_center - delta
        
        if new_score > best_score:
            best_score = new_score
            best_centers = centers.copy()
        
        T = max(T * 0.9995, T_min)
        step_size = max(0.005, step_size * 0.9998)
    
    return best_centers

def _seed_fcc():
    """Generate FCC-like seed points."""
    fcc_basis = np.array([
        [0, 0, 0],
        [0.5, 0.5, 0],
        [0.5, 0, 0.5],
        [0, 0.5, 0.5]
    ])
    
    points = []
    for ix in range(3):
        for iy in range(3):
            for iz in range(3):
                for b in fcc_basis:
                    p = np.array([ix, iy, iz]) + b
                    if p[0] <= 1.0 and p[1] <= 1.0 and p[2] <= 1.0:
                        p_mapped = p - 0.5
                        points.append(p_mapped)
    
    points = np.array(points)
    
    if len(points) > N:
        idx = np.random.default_rng(42).permutation(len(points))[:N]
        points = points[idx]
    elif len(points) < N:
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

def _seed_bcc():
    """Generate BCC-like seed points."""
    bcc_basis = np.array([
        [0, 0, 0],
        [0.5, 0.5, 0.5]
    ])
    
    points = []
    for ix in range(3):
        for iy in range(3):
            for iz in range(3):
                for b in bcc_basis:
                    p = np.array([ix, iy, iz]) + b
                    if p[0] <= 1.0 and p[1] <= 1.0 and p[2] <= 1.0:
                        p_mapped = p - 0.5
                        points.append(p_mapped)
    
    points = np.array(points)
    
    if len(points) > N:
        idx = np.random.default_rng(42).permutation(len(points))[:N]
        points = points[idx]
    elif len(points) < N:
        remaining = N - len(points)
        rng = np.random.default_rng(42)
        extra = rng.uniform(-0.45, 0.45, (remaining, 3))
        points = np.vstack([points, extra])
    
    return points

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
    seeds.append(_seed_bcc())
    
    for seed_idx, centers in enumerate(seeds):
        if time.time() > deadline - 40:
            break
        
        centers = np.clip(centers, -0.49, 0.49)
        
        # Phase 1: Lloyd equalization
        time_left = deadline - time.time()
        lloyd_time = min(10.0, time_left * 0.2)
        if lloyd_time > 1.0:
            centers = _lloyd_step(centers, max_iter=20, time_limit=lloyd_time)
        
        # Phase 2: Metropolis MC for escape from local minima
        time_left = deadline - time.time()
        mc_time = min(15.0, time_left * 0.3)
        if mc_time > 1.0:
            centers = _metropolis_mc(centers, time_limit=mc_time, rng=rng)
        
        # Phase 3: Binary Search on Radius with Gradient Descent
        # This is the core improvement from the inspirations
        time_left = deadline - time.time()
        if time_left > 15.0:
            current = centers.copy()
            current_score = _compute_radius(current)
            
            # Binary search range
            r_low = current_score
            r_high = 0.140  # Known best is ~0.136, so 0.14 is a safe upper bound
            
            # We will try to find a configuration for r_mid
            # If successful (energy ~ 0), update r_low
            # If not, update r_high
            # We use the result of the minimization as the new starting point for the next iteration
            # to maintain continuity.
            
            max_bs_iters = 10
            for bs_iter in range(max_bs_iters):
                if time.time() > deadline - 10:
                    break
                
                r_mid = (r_low + r_high) / 2.0
                
                # Minimize energy for r_mid
                # Time limit per minimization step
                step_time = min(10.0, (deadline - time.time()) / (max_bs_iters - bs_iter))
                if step_time < 1.0:
                    break
                
                new_centers = _minimize_for_radius(r_mid, current, time_limit=step_time)
                new_centers = np.clip(new_centers, -0.5, 0.5)
                
                # Check if we found a valid configuration
                # Energy should be close to 0
                e, _ = _energy_and_grad(new_centers, r_mid)
                
                # If energy is small enough, we can claim r_mid is feasible
                # Threshold: 1e-6 is usually sufficient for double precision
                if e < 1e-6:
                    r_low = r_mid
                    current = new_centers
                    current_score = _compute_radius(current)
                else:
                    r_high = r_mid
                    # Even if not feasible, the new_centers might be better than current
                    # for the actual score, or at least a better starting point
                    # Let's update current to new_centers to help the next iteration
                    # but only if it's not wildly worse?
                    # Actually, keeping the minimizer's output is usually best for continuity.
                    current = new_centers
            
            # After binary search, we have 'current' which is the best configuration found
            # for the highest feasible radius (r_low) or a configuration near it.
            # Let's check its actual score.
            final_score = _compute_radius(current)
            
            if final_score > best_score:
                best_score = final_score
                best_centers = current.copy()
        
        # Phase 4: Local search fine-tuning on the best candidate found so far
        time_left = deadline - time.time()
        if time_left > 5.0:
            # Use the best_centers found in this seed loop (or previous)
            if best_centers is not None:
                refined = _metropolis_mc(best_centers, time_limit=min(5.0, time_left * 0.5), rng=rng)
                score = _compute_radius(refined)
                if score > best_score:
                    best_score = score
                    best_centers = refined.copy()
    
    # Final polish on best candidate
    if best_centers is not None:
        time_left = deadline - time.time()
        if time_left > 5.0:
            # One last attempt to push the radius up with binary search on the best solution
            current = best_centers.copy()
            current_score = _compute_radius(current)
            r_low = current_score
            r_high = current_score + 0.005 # Small increment
            
            for _ in range(5):
                if time.time() > deadline - 2:
                    break
                r_mid = (r_low + r_high) / 2.0
                new_centers = _minimize_for_radius(r_mid, current, time_limit=5.0)
                new_centers = np.clip(new_centers, -0.5, 0.5)
                e, _ = _energy_and_grad(new_centers, r_mid)
                if e < 1e-6:
                    r_low = r_mid
                    current = new_centers
                    current_score = _compute_radius(current)
                else:
                    r_high = r_mid
                    current = new_centers
            
            final_score = _compute_radius(current)
            if final_score > best_score:
                best_score = final_score
                best_centers = current.copy()
    
    if best_centers is None:
        # Fallback
        best_centers = _seed_grid()
    
    return best_centers
# EVOLVE-BLOCK-END
