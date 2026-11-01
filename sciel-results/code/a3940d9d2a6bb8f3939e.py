"""Rewritten program for packing 200 equal circles in a unit square."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist


def _compute_radius(c):
    """Compute the largest common radius for given centres."""
    n = c.shape[0]
    boundary_dists = 0.5 - np.abs(c)
    min_boundary = np.min(boundary_dists)
    
    if n < 2:
        return min_boundary
    
    dists = pdist(c, metric='euclidean')
    min_pair_dist = np.min(dists)
    
    return min(min_pair_dist / 2.0, min_boundary)


def _soft_min_objective(params, temperature=0.001):
    """
    Smooth surrogate: minimize the soft-min of all constraints.
    Uses log-sum-exp to approximate min of (pair_dists/2, boundary_dists).
    Lower is better (we minimize negative of soft-max, i.e. maximize soft-min).
    """
    c = np.clip(params.reshape(-1, 2), -0.5 + 1e-10, 0.5 - 1e-10)
    n = c.shape[0]
    
    # Boundary constraints: distance to each wall
    bx = 0.5 - np.abs(c[:, 0])
    by = 0.5 - np.abs(c[:, 1])
    boundary_vals = np.concatenate([bx, by])
    
    # Pairwise distances / 2
    dists = pdist(c, metric='euclidean') / 2.0
    
    # All constraint values
    all_vals = np.concatenate([dists, boundary_vals])
    
    # Soft max (we want to maximize the minimum, so maximize soft-max of constraints)
    # Objective = -soft_max(all_vals)
    # soft_max(x) = T * log(sum(exp(x_i / T)))
    # As T -> 0, soft_max -> max(x)
    # We want to MAXIMIZE min, which is equivalent to minimizing -min
    # Use: -T * log(sum(exp(-x_i / T)))  which is -soft_min
    # Actually: soft_min(x) = -T * log(sum(exp(-x_i / T)))
    # We want to maximize soft_min, so minimize -soft_min = T * log(sum(exp(-x_i / T)))
    
    neg_vals = -all_vals / temperature
    # Numerical stability
    max_neg = np.max(neg_vals)
    if max_neg > 700:
        max_neg = 700
    shifted = neg_vals - max_neg
    log_sum = np.log(np.sum(np.exp(shifted))) + max_neg
    obj = -temperature * log_sum  # = -soft_min, we minimize this
    
    return obj


def _initialize_hexagonal():
    """Initialize with a hexagonal pattern optimized for the square."""
    best_c = None
    best_r = 0
    
    for d in np.linspace(0.06, 0.08, 15):
        points = []
        row_spacing = d * np.sqrt(3) / 2
        x_offset = d / 2
        
        y = -0.5 + d / 2
        row = 0
        while y <= 0.5 - d / 2:
            x = -0.5 + d / 2
            if row % 2 == 1:
                x += x_offset
            while x <= 0.5 - d / 2:
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    points.append([x, y])
                x += d
            y += row_spacing
            row += 1
        
        points = np.array(points)
        if len(points) >= N:
            c = points[:N].copy()
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best_c = c.copy()
    
    if best_c is not None:
        return best_c
    
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def _force_directed_init(c0, n_iters=2000, max_time=8.0):
    """
    Physical relaxation: apply repulsive forces between circles and
    attractive forces toward center, with boundary repulsion.
    This produces a well-spread initial configuration.
    """
    start = time.time()
    c = c0.copy().astype(np.float64)
    n = c.shape[0]
    
    # Estimate typical spacing
    spacing = 1.0 / np.sqrt(n)
    
    # Force parameters
    repel_strength = 1.0
    attract_strength = 0.01
    boundary_strength = 5.0
    damping = 0.95
    
    for it in range(n_iters):
        if time.time() - start > max_time:
            break
        
        # Compute forces
        forces = np.zeros_like(c)
        
        # Pairwise repulsion (O(n^2) but n=200 is manageable)
        for i in range(n):
            for j in range(i + 1, n):
                diff = c[i] - c[j]
                dist_sq = np.dot(diff, diff)
                dist = np.sqrt(dist_sq + 1e-12)
                if dist < spacing * 2.5:
                    # Repulsive force inversely proportional to distance
                    f = repel_strength / (dist * dist + 1e-8)
                    forces[i] += f * diff / dist
                    forces[j] -= f * diff / dist
        
        # Attraction toward center
        for i in range(n):
            forces[i] -= attract_strength * c[i]
        
        # Boundary repulsion
        margin = 0.05
        for i in range(n):
            for axis in range(2):
                if c[i, axis] > 0.5 - margin:
                    forces[i, axis] -= boundary_strength * (c[i, axis] - (0.5 - margin))
                elif c[i, axis] < -0.5 + margin:
                    forces[i, axis] += boundary_strength * ((-0.5 + margin) - c[i, axis])
        
        # Update positions with damping
        c += forces * 0.01
        c *= damping
        c = np.clip(c, -0.49, 0.49)
        
        # Periodic re-centering
        if it % 100 == 0:
            c -= np.mean(c, axis=0)
            c = np.clip(c, -0.49, 0.49)
    
    return c


def _local_optimize_smooth(c0, max_time=25.0):
    """
    Local optimization using smooth log-sum-exp surrogate with L-BFGS-B.
    Multiple temperature levels for progressive refinement.
    """
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    # Progressive temperature annealing
    temperatures = [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]
    
    for temp in temperatures:
        if time.time() - start_time > max_time - 2:
            break
        
        try:
            result = minimize(
                _soft_min_objective,
                best_c.flatten(),
                args=(temp,),
                method='L-BFGS-B',
                bounds=[(-0.5 + 1e-10, 0.5 - 1e-10)] * (2 * N),
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-10, 0.5 - 1e-10)
            r_new = _compute_radius(c_new)
            if r_new > best_r:
                best_c = c_new
                best_r = r_new
        except Exception:
            pass
    
    # Final SLSQP with hard constraints for precision
    if time.time() - start_time < max_time - 3:
        try:
            def _slsqp_obj(params):
                c = np.clip(params.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
                r = _compute_radius(c)
                return -r
            
            # Hard constraint: all points inside [−0.5+eps, 0.5−eps]
            bounds = [(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * N)
            
            result = minimize(
                _slsqp_obj,
                best_c.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-14}
            )
            
            c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
            r_new = _compute_radius(c_new)
            if r_new > best_r:
                best_c = c_new
                best_r = r_new
        except Exception:
            pass
    
    return best_c, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    best_radius = 0.0
    best_c = None
    
    # Phase 1: Hexagonal initialization
    c_hex = _initialize_hexagonal()
    r_hex = _compute_radius(c_hex)
    if r_hex > best_radius:
        best_radius = r_hex
        best_c = c_hex.copy()
    
    # Phase 2: Force-directed relaxation to spread out
    if time.time() < deadline - 20:
        c_fd = _force_directed_init(c_hex, n_iters=1500, max_time=min(8.0, deadline - time.time() - 15))
        r_fd = _compute_radius(c_fd)
        if r_fd > best_radius:
            best_radius = r_fd
            best_c = c_fd.copy()
    
    # Phase 3: Smooth optimization from best config
    if time.time() < deadline - 15:
        c_opt, r_opt = _local_optimize_smooth(best_c, max_time=min(35.0, deadline - time.time() - 5))
        if r_opt > best_radius:
            best_radius = r_opt
            best_c = c_opt.copy()
    
    # Phase 4: Additional restarts from perturbed configs
    np.random.seed(42)
    for trial in range(3):
        if time.time() > deadline - 15:
            break
        
        # Perturb the best solution
        c_pert = best_c + np.random.normal(0, 0.02, size=(N, 2))
        c_pert = np.clip(c_pert, -0.49, 0.49)
        r_pert = _compute_radius(c_pert)
        
        if r_pert > best_radius:
            best_radius = r_pert
            best_c = c_pert.copy()
        
        if time.time() < deadline - 10:
            c_opt, r_opt = _local_optimize_smooth(c_pert, max_time=min(15.0, deadline - time.time() - 3))
            if r_opt > best_radius:
                best_radius = r_opt
                best_c = c_opt.copy()
    
    if best_c is None:
        best_c = _initialize_hexagonal()
    
    best_c = np.clip(best_c, -0.5, 0.5)
    
    return best_c
# EVOLVE-BLOCK-END
