"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import Voronoi, Delaunay
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

def _softmin_radius(centers, beta=50.0):
    """
    Soft-min approximation of the radius using log-sum-exp.
    Returns a smooth approximation of the minimum radius.
    """
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    
    # Pair distances (upper triangle)
    iu = np.triu_indices(N, k=1)
    pair_dists_sq = dists_sq[iu]
    
    # Wall distances squared
    wall_dists = 0.5 - np.abs(centers)
    wall_dists_sq = wall_dists**2
    
    # Soft-min of pair distances / 2
    # We want to minimize the radius, so we maximize -radius
    # radius = min(min_pair_dist/2, min_wall_dist)
    # -radius = max(-min_pair_dist/2, -min_wall_dist)
    # Let's use soft-max for the negative
    
    # For pair distances: we want soft-min of dist/2
    # soft_min(x) = -1/beta * log(sum(exp(-beta*x)))
    # But we have squared distances. Let's work with squared distances.
    # min_dist^2 = min(pair_dists_sq)
    # min_dist = sqrt(min(pair_dists_sq))
    # radius_pair = min_dist / 2 = sqrt(min(pair_dists_sq)) / 2
    
    # Let's compute soft-min of squared distances, then take sqrt
    # soft_min_sq = -1/beta * log(sum(exp(-beta * x_i)))
    
    # Pair part
    if len(pair_dists_sq) > 0:
        scaled_pair = -beta * pair_dists_sq
        max_pair = np.max(scaled_pair)
        log_sum_pair = max_pair + np.log(np.sum(np.exp(scaled_pair - max_pair)))
        softmin_pair_sq = -log_sum_pair / beta
        softmin_pair_dist = np.sqrt(np.maximum(softmin_pair_sq, 1e-15)) / 2.0
    else:
        softmin_pair_dist = 0.5
    
    # Wall part
    if len(wall_dists_sq) > 0:
        scaled_wall = -beta * wall_dists_sq
        max_wall = np.max(scaled_wall)
        log_sum_wall = max_wall + np.log(np.sum(np.exp(scaled_wall - max_wall)))
        softmin_wall_sq = -log_sum_wall / beta
        softmin_wall_dist = np.sqrt(np.maximum(softmin_wall_sq, 1e-15))
    else:
        softmin_wall_dist = 0.5
    
    # Combine: soft-max of the two (since we want the min of the two radii)
    # min(a, b) = -max(-a, -b)
    # soft_max(a, b) = 1/beta * log(exp(beta*a) + exp(beta*b))
    # soft_min(a, b) = -soft_max(-a, -b)
    
    vals = np.array([softmin_pair_dist, softmin_wall_dist])
    scaled_vals = -beta * vals
    max_sv = np.max(scaled_vals)
    log_sum_sv = max_sv + np.log(np.sum(np.exp(scaled_vals - max_sv)))
    combined_softmin = -log_sum_sv / beta
    
    return combined_softmin

def _softmin_grad(centers, beta=50.0):
    """
    Analytical gradient of -softmin_radius with respect to centers.
    """
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    
    iu = np.triu_indices(N, k=1)
    pair_dists_sq = dists_sq[iu]
    
    wall_dists = 0.5 - np.abs(centers)
    wall_dists_sq = wall_dists**2
    
    grad = np.zeros_like(centers)
    
    # Pair part
    if len(pair_dists_sq) > 0:
        scaled_pair = -beta * pair_dists_sq
        max_pair = np.max(scaled_pair)
        exp_pair = np.exp(scaled_pair - max_pair)
        sum_exp_pair = np.sum(exp_pair)
        weights_pair = exp_pair / sum_exp_pair  # softmax weights
        
        # d(softmin_pair_sq)/d(centers)
        # softmin_pair_sq = -1/beta * log(sum(exp(-beta * dists_sq)))
        # d(softmin_pair_sq)/d(dists_sq_i) = -1/beta * (-beta * exp(-beta*dists_sq_i)) / sum(...)
        # = exp_pair_i / sum_exp_pair = weights_pair[i]
        # Wait: d/dx [-1/beta * log(sum(exp(-beta*x_i)))] = -1/beta * 1/sum(...) * sum(-beta * exp(-beta*x_i) * dx_i/dx)
        # For a specific pair i,j: d(softmin_sq)/d(centers) involves the gradient of dists_sq[i,j]
        # dists_sq[i,j] = |c_i - c_j|^2
        # d/d(c_i) = 2*(c_i - c_j)
        # d/d(c_j) = 2*(c_j - c_i)
        
        # So d(softmin_pair_sq)/d(c_i) = sum over j of weights_pair[i,j] * 2*(c_i - c_j)
        # But weights are indexed by the flattened upper triangle.
        
        # Let's compute the gradient contribution
        for idx, (i, j) in enumerate(zip(iu[0], iu[1])):
            w = weights_pair[idx]
            # d(softmin_sq)/d(c_i) += w * 2 * (c_i - c_j)
            # d(softmin_sq)/d(c_j) += w * 2 * (c_j - c_i)
            grad[i] += w * 2.0 * (centers[i] - centers[j])
            grad[j] += w * 2.0 * (centers[j] - centers[i])
        
        # Now we have d(softmin_pair_sq)/d(centers)
        # softmin_pair_dist = sqrt(softmin_pair_sq) / 2
        # d(softmin_pair_dist)/d(centers) = (1/(2*sqrt(softmin_pair_sq))) * 1/2 * d(softmin_pair_sq)/d(centers)
        # = 1/(4*sqrt(softmin_pair_sq)) * d(softmin_pair_sq)/d(centers)
        
        softmin_pair_sq_val = -1.0/beta * (max_pair + np.log(sum_exp_pair))
        if softmin_pair_sq_val > 1e-15:
            factor = 1.0 / (4.0 * np.sqrt(softmin_pair_sq_val))
            grad_pair_dist = factor * grad
        else:
            grad_pair_dist = np.zeros_like(centers)
        
        # Store for combination
        pair_dist_val = np.sqrt(max(softmin_pair_sq_val, 1e-15)) / 2.0
    else:
        grad_pair_dist = np.zeros_like(centers)
        pair_dist_val = 0.5
    
    # Wall part
    if len(wall_dists_sq) > 0:
        scaled_wall = -beta * wall_dists_sq
        max_wall = np.max(scaled_wall)
        exp_wall = np.exp(scaled_wall - max_wall)
        sum_exp_wall = np.sum(exp_wall)
        weights_wall = exp_wall / sum_exp_wall
        
        # d(softmin_wall_sq)/d(wall_dists_sq_i) = weights_wall[i]
        # wall_dists_sq_i = (0.5 - |c_i,k|)^2 for some k
        # Actually wall_dists is a flat array of N*3 values
        # wall_dists[i*3 + k] = 0.5 - |centers[i, k]|
        # d(wall_dists[i*3+k])/d(centers[i,k]) = -sign(centers[i,k])
        # d(wall_dists_sq[i*3+k])/d(centers[i,k]) = 2*(0.5 - |centers[i,k]|) * (-sign(centers[i,k]))
        
        grad_wall_sq = np.zeros_like(centers)
        for idx in range(N * 3):
            i = idx // 3
            k = idx % 3
            w = weights_wall[idx]
            wd = wall_dists[i, k]
            c_val = centers[i, k]
            if abs(c_val) > 1e-10:
                sign_c = 1.0 if c_val > 0 else -1.0
            else:
                sign_c = 0.0
            grad_wall_sq[i, k] = w * 2.0 * wd * (-sign_c)
        
        softmin_wall_sq_val = -1.0/beta * (max_wall + np.log(sum_exp_wall))
        if softmin_wall_sq_val > 1e-15:
            factor = 1.0 / (2.0 * np.sqrt(softmin_wall_sq_val))
            grad_wall_dist = factor * grad_wall_sq
        else:
            grad_wall_dist = np.zeros_like(centers)
        
        wall_dist_val = np.sqrt(max(softmin_wall_sq_val, 1e-15))
    else:
        grad_wall_dist = np.zeros_like(centers)
        wall_dist_val = 0.5
    
    # Combine: combined = soft_min(pair_dist, wall_dist)
    # = -soft_max(-pair_dist, -wall_dist)
    # soft_max(a, b) = 1/beta * log(exp(beta*a) + exp(beta*b))
    # Let a = -pair_dist, b = -wall_dist
    # d(soft_max)/da = exp(beta*a)/(exp(beta*a)+exp(beta*b))
    # d(soft_max)/db = exp(beta*b)/(exp(beta*a)+exp(beta*b))
    
    exp_a = np.exp(beta * (-pair_dist_val))
    exp_b = np.exp(beta * (-wall_dist_val))
    sum_exp = exp_a + exp_b
    
    w_a = exp_a / sum_exp
    w_b = exp_b / sum_exp
    
    # d(combined)/d(centers) = -d(soft_max)/d(centers)
    # = -[w_a * d(-pair_dist)/d(centers) + w_b * d(-wall_dist)/d(centers)]
    # = -[w_a * (-grad_pair_dist) + w_b * (-grad_wall_dist)]
    # = w_a * grad_pair_dist + w_b * grad_wall_dist
    
    grad_combined = w_a * grad_pair_dist + w_b * grad_wall_dist
    
    # We want gradient of -softmin_radius, which is -grad_combined
    return -grad_combined

def _lloyd_step(centers, max_iter=20, time_limit=8.0):
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

def _gradient_descent(centers, time_limit=15.0, beta_start=10.0, beta_end=200.0):
    """
    Gradient descent with adaptive beta for the soft-min objective.
    """
    start_time = time.time()
    centers = centers.copy()
    best_score = _compute_radius(centers)
    best_centers = centers.copy()
    
    n_betas = 10
    betas = np.geomspace(beta_start, beta_end, n_betas)
    
    for beta in betas:
        if time.time() - start_time > time_limit:
            break
        
        # Use L-BFGS-B with analytical gradient
        def obj(x):
            c = x.reshape(N, 3)
            return -_softmin_radius(c, beta=beta)
        
        def obj_grad(x):
            c = x.reshape(N, 3)
            g = _softmin_grad(c, beta=beta)
            return g.flatten()
        
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
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'callback': callback,
                                    'maxls': 50})
            new_centers = res.x.reshape(N, 3)
            new_centers = np.clip(new_centers, -0.5, 0.5)
            
            new_score = _compute_radius(new_centers)
            if new_score > best_score:
                best_score = new_score
                best_centers = new_centers.copy()
            centers = new_centers
        except Exception:
            pass
    
    return best_centers

def _metropolis_mc(centers, time_limit=20.0, rng=None):
    """
    Metropolis Monte Carlo for sphere packing.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    start_time = time.time()
    centers = centers.copy()
    best_centers = centers.copy()
    best_score = _compute_radius(centers)
    
    T = 0.003
    T_min = 0.00005
    step_size = 0.03
    
    while time.time() - start_time < time_limit:
        i = rng.integers(N)
        delta = rng.normal(0, step_size, 3)
        new_center = centers[i] + delta
        new_center = np.clip(new_center, -0.499, 0.499)
        
        old_score = _compute_radius(centers)
        centers[i] = new_center
        new_score = _compute_radius(centers)
        
        delta_score = new_score - old_score
        
        if delta_score > 0 or rng.random() < np.exp(delta_score / T):
            pass
        else:
            centers[i] = new_center - delta
        
        if new_score > best_score:
            best_score = new_score
            best_centers = centers.copy()
        
        T = max(T * 0.999, T_min)
        step_size = max(0.003, step_size * 0.9995)
    
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

def _seed_random():
    """Random seed."""
    rng = np.random.default_rng(42)
    return rng.uniform(-0.45, 0.45, (N, 3))

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s
    
    best_score = 0.0
    best_centers = None
    
    rng = np.random.default_rng(42)
    
    # Multiple initialization strategies
    seeds = []
    seeds.append(_seed_fcc())
    seeds.append(_seed_grid())
    seeds.append(_seed_random())
    
    for seed_idx, centers in enumerate(seeds):
        if time.time() > deadline - 40:
            break
        
        centers = np.clip(centers, -0.49, 0.49)
        
        # Phase 1: Lloyd equalization
        time_left = deadline - time.time()
        lloyd_time = min(10.0, time_left * 0.2)
        if lloyd_time > 1.0:
            centers = _lloyd_step(centers, max_iter=20, time_limit=lloyd_time)
        
        # Phase 2: Gradient descent with soft-min
        time_left = deadline - time.time()
        gd_time = min(30.0, time_left * 0.3)
        if gd_time > 2.0:
            centers = _gradient_descent(centers, time_limit=gd_time, beta_start=10.0, beta_end=200.0)
        
        # Phase 3: Metropolis MC for escape from local minima
        time_left = deadline - time.time()
        mc_time = min(20.0, time_left * 0.25)
        if mc_time > 2.0:
            centers = _metropolis_mc(centers, time_limit=mc_time, rng=rng)
        
        # Phase 4: Final gradient polish
        time_left = deadline - time.time()
        polish_time = min(15.0, time_left * 0.2)
        if polish_time > 1.0:
            centers = _gradient_descent(centers, time_limit=polish_time, beta_start=50.0, beta_end=500.0)
        
        score = _compute_radius(centers)
        
        if score > best_score:
            best_score = score
            best_centers = centers.copy()
    
    # Final polish on best candidate
    if best_centers is not None:
        time_left = deadline - time.time()
        if time_left > 5.0:
            final_time = time_left - 2.0
            refined = _gradient_descent(best_centers, time_limit=final_time, beta_start=100.0, beta_end=1000.0)
            score = _compute_radius(refined)
            if score > best_score:
                best_score = score
                best_centers = refined.copy()
    
    if best_centers is None:
        best_centers = _seed_grid()
    
    return best_centers
# EVOLVE-BLOCK-END
