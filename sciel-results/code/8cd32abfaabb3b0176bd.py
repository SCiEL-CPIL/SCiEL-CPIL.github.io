import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 50
CUBE_HALF = 0.5

def _compute_radius(p):
    """Compute the valid radius for a configuration."""
    if p.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(p)):
        return 0.0
    # Check bounds
    if np.any(np.abs(p) > CUBE_HALF + 1e-9):
        return 0.0
    
    # Boundary constraint: distance to closest wall
    # dist_i = 0.5 - max(|x_i|, |y_i|, |z_i|)
    b = CUBE_HALF - np.max(np.abs(p), axis=1)
    min_b = np.min(b)
    
    # Pairwise constraint
    diffs = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_d = np.min(dists) / 2.0
    
    return min(min_b, min_d)

def _softmin_radius_and_grad(p, beta=100.0):
    """
    Smooth approximation of min radius using softmin, with analytic gradient.
    Returns (softmin_value, gradient_wrt_p).
    """
    # Boundary distances
    abs_p = np.abs(p)
    max_abs = np.max(abs_p, axis=1)
    b = CUBE_HALF - max_abs
    
    # Pairwise distances
    dists = cdist(p, p, metric='euclidean')
    np.fill_diagonal(dists, np.inf)
    pd = dists / 2.0
    
    # Collect constraints
    iu = np.triu_indices(N, k=1)
    pd_upper = pd[iu]
    
    all_constraints = np.concatenate([b, pd_upper])
    
    # Softmin: f = min_val - (1/beta) * log(sum(exp(-beta * (x - min_val))))
    min_val = np.min(all_constraints)
    shifted = all_constraints - min_val
    exp_vals = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_vals)
    log_sum = np.log(sum_exp)
    softmin_val = min_val - log_sum / beta
    
    # Gradient
    weights = exp_vals / sum_exp
    grad = np.zeros_like(p)
    
    # Boundary part
    boundary_weights = weights[:N]
    max_axes = np.argmax(abs_p, axis=1)
    for j in range(N):
        w = boundary_weights[j]
        if w > 1e-15:
            k = max_axes[j]
            grad[j, k] += w * (-np.sign(p[j, k]))
            
    # Pairwise part
    pair_weights = weights[N:]
    if np.any(pair_weights > 1e-15):
        i_idx = iu[0]
        j_idx = iu[1]
        diffs = p[i_idx] - p[j_idx]
        dists_pair = np.linalg.norm(diffs, axis=1)
        safe_dists = np.where(dists_pair > 1e-12, dists_pair, 1.0)
        unit_vecs = diffs / safe_dists[:, np.newaxis]
        
        w_scaled = pair_weights * 0.5
        np.add.at(grad, i_idx, w_scaled[:, np.newaxis] * unit_vecs)
        np.add.at(grad, j_idx, -w_scaled[:, np.newaxis] * unit_vecs)
        
    return softmin_val, grad

def _smooth_obj_and_grad(x, beta=100.0):
    p = x.reshape(N, 3)
    val, grad = _softmin_radius_and_grad(p, beta)
    return -val, -grad

def _generate_fcc_seed(rng, n_points=200):
    """Generate a random subset of FCC lattice points, centered and scaled."""
    # Generate a large chunk of FCC lattice
    # FCC basis: (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
    basis = np.array([[0,0,0], [0.5,0.5,0], [0.5,0,0.5], [0,0.5,0.5]])
    lattice_points = []
    # Range -5 to 5 is enough to get > 200 points
    for i in range(-5, 6):
        for j in range(-5, 6):
            for k in range(-5, 6):
                for b in basis:
                    lattice_points.append([i + b[0], j + b[1], k + b[2]])
    
    lattice_points = np.array(lattice_points)
    
    # Randomly select n_points
    idx = rng.choice(len(lattice_points), n_points, replace=False)
    points = lattice_points[idx]
    
    # Center
    points -= np.mean(points, axis=0)
    
    # Scale to fit in cube with some margin (0.45)
    max_coord = np.max(np.abs(points))
    if max_coord > 0:
        scale = 0.45 / max_coord
        points *= scale
        
    return points

def _repulsion_anneal(p0, target_r, n_steps=500, time_limit=None, start_time=None):
    """Vectorized repulsion annealing."""
    p = p0.copy()
    limits = CUBE_HALF - target_r
    
    for step in range(n_steps):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        forces = np.zeros_like(p)
        
        # Boundary forces
        for axis in range(3):
            over_pos = p[:, axis] > limits
            over_neg = p[:, axis] < -limits
            forces[over_pos, axis] -= 1.0
            forces[over_neg, axis] += 1.0
        
        # Pairwise repulsion (Vectorized)
        diffs = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        
        mask = dists < 2 * target_r
        np.fill_diagonal(mask, False)
        
        if np.any(mask):
            safe_dists = np.where(dists > 1e-10, dists, 1.0)
            unit_vecs = diffs / safe_dists[:, :, np.newaxis]
            unit_vecs = unit_vecs * mask[:, :, np.newaxis]
            pair_forces = np.sum(unit_vecs, axis=1)
            forces += pair_forces
        
        # Adaptive step size
        step_size = 0.02 * (1.0 - 0.5 * step / n_steps)
        p += step_size * forces
        p = np.clip(p, -CUBE_HALF, CUBE_HALF)
        
    return p

def _metropolis_mc(p0, target_r, n_steps=20000, time_limit=None, start_time=None):
    """Metropolis Monte Carlo to escape local minima."""
    p = p0.copy()
    current_r = _compute_radius(p)
    best_r = current_r
    best_p = p.copy()
    
    step_size = 0.02
    T = 0.01  # Temperature
    
    for step in range(n_steps):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        # Pick a random point and axis
        i = np.random.randint(N)
        axis = np.random.randint(3)
        delta = np.random.normal(0, step_size)
        
        p_trial = p.copy()
        p_trial[i, axis] += delta
        p_trial = np.clip(p_trial, -CUBE_HALF, CUBE_HALF)
        
        r_trial = _compute_radius(p_trial)
        
        # Accept if better, or with probability exp(-(r_trial - current_r) / T)
        if r_trial > current_r:
            accept = True
        else:
            # Penalty for worse radius
            diff = current_r - r_trial
            if diff > 0.05: # Hard cutoff for very bad moves
                accept = False
            else:
                accept = np.random.random() < np.exp(-diff / T)
                
        if accept:
            p = p_trial
            current_r = r_trial
            if current_r > best_r:
                best_r = current_r
                best_p = p.copy()
        
        # Anneal temperature
        if step % 1000 == 0 and step > 0:
            T *= 0.8
            step_size *= 0.95
            
    return best_p, best_r

def _active_set_polish(p, time_limit, start_time):
    """
    Polish by optimizing only points involved in critical constraints.
    """
    p = p.copy()
    current_r = _compute_radius(p)
    best_r = current_r
    best_p = p.copy()
    
    # Identify active points
    def get_active_points(p, r):
        active = set()
        # Boundary
        b = CUBE_HALF - np.max(np.abs(p), axis=1)
        active.update(np.where(b < r + 1e-8)[0])
        
        # Pairs
        dists = cdist(p, p)
        np.fill_diagonal(dists, np.inf)
        # Pairs with dist < 2r + tol
        tol = 1e-8
        active_pairs = np.argwhere(dists < 2*r + tol)
        for i, j in active_pairs:
            active.add(i)
            active.add(j)
        return sorted(active)
    
    active = get_active_points(p, current_r)
    if not active:
        active = list(range(N))
        
    # Optimize only active points using L-BFGS on soft-min
    # We fix inactive points to their current positions
    
    # Create a wrapper for optimization that only updates active points
    # Variables: x of length len(active)*3
    n_active = len(active)
    indices = np.array(active)
    
    def obj_grad(x):
        p_trial = p.copy()
        p_trial[indices] = x.reshape(n_active, 3)
        val, grad = _softmin_radius_and_grad(p_trial, beta=200.0)
        # Map gradient back to x
        grad_x = grad[indices].flatten()
        return -val, -grad_x
        
    x0 = p[indices].flatten()
    bounds = [(-CUBE_HALF, CUBE_HALF)] * (n_active * 3)
    
    try:
        result = minimize(obj_grad, x0, method='L-BFGS-B', jac=True, bounds=bounds,
                          options={'maxiter': 100, 'gtol': 1e-8})
        p_new = p.copy()
        p_new[indices] = result.x.reshape(n_active, 3)
        r_new = _compute_radius(p_new)
        
        if r_new > best_r + 1e-12:
            best_r = r_new
            best_p = p_new
    except Exception:
        pass
        
    return best_p, best_r

def construct():
    start_time = time.time()
    total_time_limit = 145.0 # Safety margin
    
    # 1. Generate diverse seeds
    rng = np.random.RandomState(42)
    seeds = []
    # 5 random FCC seeds
    for i in range(5):
        p_seed = _generate_fcc_seed(rng)
        seeds.append(p_seed)
        
    # 1 Random uniform seed
    p_rand = rng.uniform(-0.45, 0.45, size=(N, 3))
    seeds.append(p_rand)
    
    best_p = None
    best_r = -1.0
    
    # Phase 1: Quick annealing on all seeds to get a decent start
    # Allocate 10% of time
    t1_limit = start_time + 0.1 * (total_time_limit - (time.time() - start_time))
    
    for p_seed in seeds:
        if time.time() > t1_limit:
            break
        p_annealed = _repulsion_anneal(p_seed, 0.1, n_steps=100, 
                                        time_limit=t1_limit, start_time=start_time)
        r_annealed = _compute_radius(p_annealed)
        if r_annealed > best_r:
            best_r = r_annealed
            best_p = p_annealed
            
    if best_p is None:
        best_p = seeds[0]
        best_r = _compute_radius(best_p)
        
    # Phase 2: Metropolis MC to explore
    # Allocate 40% of time
    t2_limit = start_time + 0.5 * total_time_limit
    p_mc, r_mc = _metropolis_mc(best_p, best_r, n_steps=20000, 
                                time_limit=t2_limit, start_time=start_time)
    if r_mc > best_r:
        best_r = r_mc
        best_p = p_mc
        
    # Phase 3: Gradient Polish + Active Set
    # Allocate remaining time
    t3_limit = start_time + total_time_limit
    
    # Global soft-min polish
    def global_obj_grad(x):
        return _smooth_obj_and_grad(x, beta=100.0)
        
    p_flat = best_p.flatten()
    bounds = [(-CUBE_HALF, CUBE_HALF)] * (N * 3)
    
    try:
        result = minimize(global_obj_grad, p_flat, method='L-BFGS-B', jac=True, bounds=bounds,
                          options={'maxiter': 200, 'gtol': 1e-8})
        p_polished = result.x.reshape(N, 3)
        r_polished = _compute_radius(p_polished)
        if r_polished > best_r:
            best_r = r_polished
            best_p = p_polished
    except Exception:
        pass
        
    # Active set polish
    if time.time() < t3_limit - 5:
        p_as, r_as = _active_set_polish(best_p, t3_limit, start_time)
        if r_as > best_r:
            best_r = r_as
            best_p = p_as
            
    # Final validation
    final_r = _compute_radius(best_p)
    if final_r > best_r:
        best_r = final_r
        
    return best_p
