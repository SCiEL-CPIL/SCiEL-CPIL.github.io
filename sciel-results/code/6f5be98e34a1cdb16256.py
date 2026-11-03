import numpy as np
import time
from scipy.optimize import minimize

N = 100
HALF = 0.5

def _compute_radius(centers):
    """
    Computes the maximum common radius for a given set of centers.
    Centers: (N, 3) array.
    Cube: [-0.5, 0.5]^3.
    """
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
        
    # Wall constraints: distance from center to nearest wall
    max_abs = np.max(np.abs(centers), axis=1)
    wall_dists = 0.5 - max_abs
    min_wall_dist = np.min(wall_dists)
    
    if min_wall_dist < 0:
        return 0.0

    # Pairwise constraints
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Set diagonal to infinity to exclude self-distance
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # The radius is half the minimum pairwise distance, limited by wall distance
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return r

def _soft_min_objective_and_grad(x, beta=50.0):
    """
    Smooth surrogate: maximize the soft-min of all constraint distances.
    Objective = -soft_min(2*r_pairwise, wall_dists)
    where soft_min uses log-sum-exp for smoothness.
    """
    centers = x.reshape(-1, 3)
    n = len(centers)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    dists = np.sqrt(dists_sq + 1e-15)
    np.fill_diagonal(dists, np.inf)
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_dists = dists[i_idx, j_idx]
    half_pair_dists = pair_dists / 2.0
    
    # Wall distances
    max_abs = np.max(np.abs(centers), axis=1)
    wall_dists = 0.5 - max_abs
    
    # Combine all constraints
    all_dists = np.concatenate([half_pair_dists, wall_dists])
    
    # Soft-min via log-sum-exp (numerically stable)
    min_d = np.min(all_dists)
    shifted = all_dists - min_d
    log_sum = np.log(np.sum(np.exp(-beta * shifted))) / beta
    soft_min_val = min_d - log_sum
    
    # Gradient of soft-min
    weights = np.exp(-beta * (all_dists - min_d))
    weights /= np.sum(weights)
    
    grad = np.zeros_like(centers)
    
    # Pairwise gradients (vectorized)
    vecs = centers[i_idx] - centers[j_idx] # (P, 3)
    d_safe = np.where(pair_dists > 1e-15, pair_dists, 1.0)
    factors = weights / (2.0 * d_safe) # (P,)
    mask_pairs = (weights >= 1e-10) & (pair_dists > 1e-15)
    factors = np.where(mask_pairs, factors, 0.0)
    forces = factors[:, np.newaxis] * vecs # (P, 3)
    np.add.at(grad, i_idx, forces)
    np.add.at(grad, j_idx, -forces)
    
    # Wall gradients
    wall_weights = weights[n*(n-1)//2:]
    mask_wall = wall_weights >= 1e-10
    if np.any(mask_wall):
        idxs = np.where(mask_wall)[0]
        for idx in idxs:
            max_k = np.argmax(np.abs(centers[idx]))
            sign = np.sign(centers[idx, max_k])
            grad[idx, max_k] -= wall_weights[idx] * sign
    
    # We want to maximize soft_min, so minimize -soft_min
    objective = -soft_min_val
    return objective, grad.flatten()

def _hard_energy_and_grad(x, R):
    """
    Hard squared-violation energy for feasibility checking.
    """
    centers = x.reshape(-1, 3)
    n = len(centers)
    
    energy = 0.0
    grad = np.zeros_like(centers)
    
    # 1. Pairwise terms
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-15)
    np.fill_diagonal(dists, np.inf)
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    d_ij = dists[i_idx, j_idx]
    mask_pairs = d_ij < 2.0 * R
    
    if np.any(mask_pairs):
        v_ij = 2.0 * R - d_ij[mask_pairs]
        d_valid = d_ij[mask_pairs]
        i_valid = i_idx[mask_pairs]
        j_valid = j_idx[mask_pairs]
        
        energy += np.sum(v_ij**2)
        
        factors = 2.0 * v_ij / d_valid
        vecs = centers[i_valid] - centers[j_valid]
        forces = factors[:, np.newaxis] * vecs
        
        np.add.at(grad, i_valid, -forces)
        np.add.at(grad, j_valid, forces)

    # 2. Wall terms
    abs_centers = np.abs(centers)
    max_abs = np.max(abs_centers, axis=1)
    wall_dists = 0.5 - max_abs
    
    mask_wall = wall_dists < R
    if np.any(mask_wall):
        violations_wall = R - wall_dists[mask_wall]
        i_valid_wall = np.where(mask_wall)[0]
        
        energy += np.sum(violations_wall**2)
        
        for idx in i_valid_wall:
            v = violations_wall[np.where(i_valid_wall == idx)[0][0]]
            max_k = np.argmax(abs_centers[idx])
            
            grad[idx, max_k] += 2.0 * v * np.sign(centers[idx, max_k])
            
    return energy, grad.flatten()

def _optimize_soft_min_multi_phase(init_centers, max_iter=500, ftol=1e-14, gtol=1e-12):
    """
    Optimize using soft-min surrogate with multiple beta phases.
    """
    bounds = [(-0.5, 0.5)] * (N * 3)
    current_pos = init_centers.flatten()
    
    # Phase 1: Low beta for exploration
    beta1 = 10.0
    def obj_grad1(x):
        return _soft_min_objective_and_grad(x, beta=beta1)
    try:
        res1 = minimize(obj_grad1, current_pos, jac=True, method='L-BFGS-B', bounds=bounds, options={'maxiter': max_iter//2, 'ftol': ftol, 'gtol': gtol, 'disp': False})
        current_pos = res1.x
    except Exception:
        pass
        
    # Phase 2: Medium beta for refinement
    beta2 = 50.0
    def obj_grad2(x):
        return _soft_min_objective_and_grad(x, beta=beta2)
    try:
        res2 = minimize(obj_grad2, current_pos, jac=True, method='L-BFGS-B', bounds=bounds, options={'maxiter': max_iter//2, 'ftol': ftol, 'gtol': gtol, 'disp': False})
        current_pos = res2.x
    except Exception:
        pass
        
    # Phase 3: High beta for sharpness
    beta3 = 200.0
    def obj_grad3(x):
        return _soft_min_objective_and_grad(x, beta=beta3)
    try:
        res3 = minimize(obj_grad3, current_pos, jac=True, method='L-BFGS-B', bounds=bounds, options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol, 'disp': False})
        current_pos = res3.x
    except Exception:
        pass
        
    new_centers = current_pos.reshape(N, 3)
    return new_centers

def _optimize_hard(init_centers, target_R, max_iter=500, ftol=1e-14, gtol=1e-12):
    """
    Optimize using hard energy for feasibility.
    """
    def obj_grad(x):
        val, g = _hard_energy_and_grad(x, target_R)
        return val, g

    bounds = [(-0.5, 0.5)] * (N * 3)
    
    try:
        res = minimize(
            obj_grad,
            init_centers.flatten(),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol, 'disp': False}
        )
        new_centers = res.x.reshape(N, 3)
        final_energy, _ = _hard_energy_and_grad(new_centers, target_R)
        return new_centers, final_energy, res.success
    except Exception:
        return init_centers, np.inf, False

def _generate_diverse_starts(rng, n_starts=8):
    """
    Generate diverse initial configurations.
    """
    starts = []
    
    # 1. Grid-based 5x5x4 = 100 points
    nx, ny, nz = 5, 5, 4
    xs = np.linspace(-0.5 + 0.5/nx, 0.5 - 0.5/nx, nx)
    ys = np.linspace(-0.5 + 0.5/ny, 0.5 - 0.5/ny, ny)
    zs = np.linspace(-0.5 + 0.5/nz, 0.5 - 0.5/nz, nz)
    grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    starts.append(grid[:N].copy() + rng.normal(0, 0.01, size=(N, 3)))
    
    # 2. Random uniform
    starts.append(rng.uniform(-0.45, 0.45, size=(N, 3)))
    
    # 3. BCC-like lattice (4x4x4 cells, 2 pts/cell = 128, take 100)
    lattice_pts = []
    for ix in range(4):
        for iy in range(4):
            for iz in range(4):
                x = -0.5 + (ix + 0.5) * 0.5
                y = -0.5 + (iy + 0.5) * 0.5
                z = -0.5 + (iz + 0.5) * 0.5
                lattice_pts.append([x, y, z])
                lattice_pts.append([x + 0.125, y + 0.125, z + 0.125])
    lattice = np.array(lattice_pts[:N])
    starts.append(lattice + rng.normal(0, 0.005, size=(N, 3)))
    
    # 4. FCC-like lattice (shifted BCC)
    fcc_pts = []
    for ix in range(4):
        for iy in range(4):
            for iz in range(4):
                x = -0.5 + (ix + 0.5) * 0.5
                y = -0.5 + (iy + 0.5) * 0.5
                z = -0.5 + (iz + 0.5) * 0.5
                fcc_pts.append([x, y, z])
                fcc_pts.append([x + 0.25, y + 0.25, z])
    fcc = np.array(fcc_pts[:N])
    starts.append(fcc + rng.normal(0, 0.005, size=(N, 3)))
    
    # 5-8. More random with different scales
    for _ in range(n_starts - 4):
        starts.append(rng.uniform(-0.45, 0.45, size=(N, 3)))
    
    return [np.clip(s, -0.49, 0.49) for s in starts]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    rng = np.random.default_rng(42)
    
    # 1. Multi-start initialization with multi-phase soft-min
    starts = _generate_diverse_starts(rng, n_starts=8)
    
    best_pos = None
    best_r = 0.0
    
    for s in starts:
        if time.time() > deadline - 40.0:
            break
        # Use multi-phase soft-min optimization
        pos = _optimize_soft_min_multi_phase(s, max_iter=500)
        r = _compute_radius(pos)
        if r > best_r:
            best_r = r
            best_pos = pos
    
    if best_pos is None or best_r <= 0:
        best_pos = rng.uniform(-0.45, 0.45, size=(N, 3))
        best_r = _compute_radius(best_pos)
    
    # 2. Iterative refinement with hard optimization
    current_r = best_r
    step_size = 0.0004
    consecutive_failures = 0
    max_failures = 3 # Reduce max failures to try more steps
    rng_search = np.random.default_rng(123)
    
    while time.time() < deadline - 10.0:
        target_r = current_r + step_size
        
        if consecutive_failures >= max_failures:
            step_size *= 0.8
            consecutive_failures = 0
            if step_size < 1e-8:
                break
        
        # Generate candidates: current best + perturbations
        candidates = [best_pos.copy()]
        for sigma in [0.002, 0.005, 0.01]:
            candidates.append(best_pos + rng_search.normal(0, sigma, size=(N, 3)))
        
        success = False
        best_candidate = None
        best_candidate_r = 0.0
        
        for start_pos in candidates:
            if time.time() > deadline - 15.0:
                break
            start_pos = np.clip(start_pos, -0.5, 0.5)
            
            # Try hard optimization at target radius
            hard_pos, energy, _ = _optimize_hard(start_pos, target_r, max_iter=400)
            hard_r = _compute_radius(hard_pos)
            
            # Also try soft-min polish (more iterations due to speedup)
            soft_pos = _optimize_soft_min_multi_phase(start_pos, max_iter=300)
            soft_r = _compute_radius(soft_pos)
            
            # Pick the best result
            if soft_r > hard_r:
                cand_r, cand_pos = soft_r, soft_pos
            else:
                cand_r, cand_pos = hard_r, hard_pos
            
            if cand_r > best_candidate_r:
                best_candidate_r = cand_r
                best_candidate = cand_pos
            
            # Check for success at target radius
            if energy < 1e-10 and hard_r >= target_r - 1e-7:
                best_pos = hard_pos
                best_r = hard_r
                current_r = hard_r
                success = True
                consecutive_failures = 0
                break
        
        if not success:
            # Accept improvement even if target not fully met
            if best_candidate is not None and best_candidate_r > current_r:
                best_pos = best_candidate
                best_r = best_candidate_r
                current_r = best_candidate_r
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                
                # Polish current best
                if time.time() < deadline - 20.0 and consecutive_failures % 2 == 0:
                    polished = _optimize_soft_min_multi_phase(best_pos, max_iter=400)
                    polished_r = _compute_radius(polished)
                    if polished_r > best_r:
                        best_pos = polished
                        best_r = polished_r
                        current_r = best_r

    # 3. Final high-precision polish
    if time.time() < deadline - 5.0:
        # Push slightly beyond
        final_target = best_r + 1e-6
        polished_pos, final_energy, _ = _optimize_hard(best_pos, final_target, max_iter=1000, ftol=1e-15, gtol=1e-14)
        final_r = _compute_radius(polished_pos)
        
        if final_r > best_r:
            best_pos = polished_pos
            best_r = final_r
        
        # Soft-min polish for fine-tuning
        if time.time() < deadline - 2.0:
            soft_polished = _optimize_soft_min_multi_phase(best_pos, max_iter=500)
            soft_r = _compute_radius(soft_polished)
            if soft_r > best_r:
                best_pos = soft_polished
                best_r = soft_r
        
        # Final hard polish at best_r
        if time.time() < deadline - 1.0:
            final_pos, _, _ = _optimize_hard(best_pos, best_r, max_iter=500, ftol=1e-15, gtol=1e-14)
            final_r = _compute_radius(final_pos)
            if final_r > best_r:
                best_pos = final_pos
                best_r = final_r

    return best_pos
