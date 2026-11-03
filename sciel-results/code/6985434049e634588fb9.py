"""Combined program: BCC init + SA exploration + soft-min L-BFGS-B refinement."""
import numpy as np
import time
from scipy.spatial import cKDTree
from scipy.optimize import minimize

N = 100
HALF = 0.5
START_TIME = time.time()
TIME_LIMIT = 120.0
SAFETY_MARGIN = 8.0

def _check_time():
    return (time.time() - START_TIME) > (TIME_LIMIT - SAFETY_MARGIN)

def _compute_radius(C):
    """Compute the maximum possible radius for a set of centers C."""
    if len(C) < 2:
        return HALF - np.max(np.abs(C))
    dist_to_bound = HALF - np.abs(C).max(axis=1)
    r_bound = np.min(dist_to_bound)
    tree = cKDTree(C)
    distances, _ = tree.query(C, k=2)
    min_dist = np.min(distances[:, 1])
    r_pair = min_dist / 2.0
    return min(r_bound, r_pair)

def _generate_bcc_points(n_points=N):
    """Generate initial points based on a Body-Centered Cubic (BCC) lattice."""
    size = 4
    raw_points = []
    for i in range(size):
        for j in range(size):
            for k in range(size):
                raw_points.append([i, j, k])
                raw_points.append([i + 0.5, j + 0.5, k + 0.5])
    raw_points = np.array(raw_points)
    center = np.array([size/2 - 0.25, size/2 - 0.25, size/2 - 0.25])
    raw_points = raw_points - center
    max_coord = np.max(np.abs(raw_points))
    scale = 0.42 / max_coord
    raw_points *= scale
    idx = np.random.default_rng(42).permutation(len(raw_points))
    raw_points = raw_points[idx]
    return raw_points[:n_points].copy()

def _soft_min_objective_and_grad(flat_pos, beta=80.0):
    """
    Soft-min objective: maximize the soft-minimum of all effective distances.
    Returns (objective, gradient) for minimization of -soft_min.
    """
    pos = flat_pos.reshape(N, 3)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    d_walls = HALF - np.abs(pos)
    wall_dists = d_walls.flatten()
    all_eff_dists = np.concatenate([pair_dists / 2.0, wall_dists])
    scaled = -beta * all_eff_dists
    max_scaled = np.max(scaled)
    log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min_val = -log_sum_exp / beta
    weights = np.exp(scaled - max_scaled)
    weights /= np.sum(weights)
    grad_pos = np.zeros_like(pos)
    for idx in range(len(iu[0])):
        i, j = iu[0][idx], iu[1][idx]
        w = weights[idx]
        d = pair_dists[idx]
        if d > 1e-14:
            vec = pos[i] - pos[j]
            coeff = w / (2.0 * d)
            grad_pos[i] += coeff * vec
            grad_pos[j] -= coeff * vec
    for idx in range(N * 3):
        i = idx // 3
        k = idx % 3
        w = weights[len(pair_dists) + idx]
        s = np.sign(pos[i, k])
        if s == 0:
            s = 1.0
        grad_pos[i, k] -= w * s
    return -soft_min_val, -grad_pos.flatten()

def _soft_min_optimize(pos, beta=80.0, max_iter=300, ftol=1e-14, gtol=1e-12):
    """Optimize using soft-min objective with L-BFGS-B."""
    bounds = [(-HALF, HALF)] * (N * 3)
    try:
        res = minimize(
            lambda x: _soft_min_objective_and_grad(x, beta)[0],
            pos.flatten(),
            jac=lambda x: _soft_min_objective_and_grad(x, beta)[1],
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol}
        )
        if np.isfinite(res.fun):
            new_pos = res.x.reshape(N, 3)
            if np.all(np.isfinite(new_pos)):
                return new_pos
    except Exception:
        pass
    return pos

def _sa_step(current_centers, current_radius, T, rng):
    """One simulated annealing step. Returns (new_centers, new_radius, accepted)."""
    i = rng.integers(N)
    step_size = T * 1.5
    move = rng.normal(0, step_size, 3)
    new_centers = current_centers.copy()
    new_centers[i] += move
    if np.any(np.abs(new_centers[i]) > HALF - 1e-9):
        return current_centers, current_radius, False
    new_radius = _compute_radius(new_centers)
    delta = new_radius - current_radius
    if delta > 0:
        return new_centers, new_radius, True
    if rng.random() < np.exp(delta / T):
        return new_centers, new_radius, True
    return current_centers, current_radius, False

def construct():
    start_time = time.time()
    deadline = start_time + 112.0
    
    # Phase 1: Generate BCC initial points and do initial soft-min optimization
    centers = _generate_bcc_points(N)
    centers = np.clip(centers, -HALF + 1e-9, HALF - 1e-9)
    
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    # Initial soft-min optimization to get a good starting point
    if not _check_time():
        optimized = _soft_min_optimize(centers, beta=80.0, max_iter=200)
        opt_radius = _compute_radius(optimized)
        if opt_radius > best_radius:
            best_centers = optimized
            best_radius = opt_radius
    
    current_centers = best_centers.copy()
    current_radius = best_radius
    
    # Phase 2: Simulated Annealing with periodic soft-min refinement
    T_start = 0.04
    T_end = 0.002
    alpha = 0.9997
    T = T_start
    rng = np.random.default_rng(123)
    
    sa_iterations = 0
    last_refine_time = start_time
    refine_interval = 5.0  # seconds between refinements
    
    while T > T_end and time.time() < deadline:
        # SA step
        current_centers, current_radius, accepted = _sa_step(
            current_centers, current_radius, T, rng
        )
        sa_iterations += 1
        
        if current_radius > best_radius:
            best_radius = current_radius
            best_centers = current_centers.copy()
        
        T *= alpha
        
        # Periodic soft-min refinement to escape local minima
        now = time.time()
        if (now - last_refine_time > refine_interval and 
            not _check_time() and
            sa_iterations % 50 == 0):
            last_refine_time = now
            refined = _soft_min_optimize(current_centers, beta=100.0, max_iter=150)
            refined_radius = _compute_radius(refined)
            if refined_radius > current_radius:
                current_centers = refined
                current_radius = refined_radius
                if refined_radius > best_radius:
                    best_radius = refined_radius
                    best_centers = refined.copy()
        
        # Every 500 SA steps, do a more aggressive restart from best
        if sa_iterations % 500 == 0 and not _check_time():
            # Perturb best solution and re-optimize
            perturbed = best_centers + rng.normal(0, 0.02, (N, 3))
            perturbed = np.clip(perturbed, -HALF + 1e-9, HALF - 1e-9)
            refined = _soft_min_optimize(perturbed, beta=60.0, max_iter=100)
            refined_radius = _compute_radius(refined)
            if refined_radius > best_radius:
                best_radius = refined_radius
                best_centers = refined
                current_centers = refined
                current_radius = refined_radius
    
    # Phase 3: Final high-precision soft-min refinement on best solution
    if not _check_time():
        final_refined = _soft_min_optimize(best_centers, beta=200.0, max_iter=500, ftol=1e-15, gtol=1e-14)
        final_radius = _compute_radius(final_refined)
        if final_radius > best_radius:
            best_radius = final_radius
            best_centers = final_refined
    
    # Phase 4: Final local polish with small steps if time permits
    if time.time() < deadline - 3.0:
        step = 0.002
        for _ in range(50):
            improved = False
            for i in range(N):
                for d in range(3):
                    for sign in [-1, 1]:
                        new_c = best_centers.copy()
                        new_c[i, d] += sign * step
                        if np.any(np.abs(new_c[i]) > HALF - 1e-9):
                            continue
                        new_r = _compute_radius(new_c)
                        if new_r > best_radius + 1e-12:
                            best_centers = new_c
                            best_radius = new_r
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
    
    # Ensure all points are strictly inside the cube
    best_centers = np.clip(best_centers, -HALF + 1e-10, HALF - 1e-10)
    
    return best_centers
