import numpy as np
import time
from scipy.optimize import minimize
import sys

N = 50
DEADLINE = 130.0  # seconds, leaving 20s margin before the 150s kill

def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    centers = np.asarray(centers)
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Wall constraint: min_i min_k (0.5 - |c_ik|)
    wall = np.min(0.5 - np.abs(centers))
    if wall <= 0:
        return 0.0
    # Pairwise constraint: min_{i<j} |c_i - c_j| / 2
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    i_idx, j_idx = np.triu_indices(N, k=1)
    min_pair = np.min(dist[i_idx, j_idx]) / 2.0
    return min(wall, min_pair)

def _energy_and_grad(x, R):
    """
    Calculate energy (sum of squared overlaps) and gradient for a fixed radius R.
    x is a flat array of shape (N*3,).
    """
    centers = x.reshape(N, 3)
    energy = 0.0
    grad_centers = np.zeros_like(centers)
    
    # Pairwise overlaps
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    i_idx, j_idx = np.triu_indices(N, k=1)
    d_ij = dist[i_idx, j_idx]
    
    target = 2.0 * R
    viol = np.maximum(0.0, target - d_ij)
    energy += np.sum(viol ** 2)
    
    mask = viol > 0
    if np.any(mask):
        d_masked = d_ij[mask]
        viol_masked = viol[mask]
        i_masked = i_idx[mask]
        j_masked = j_idx[mask]
        
        vecs = centers[i_masked] - centers[j_masked]
        coeff = -2.0 * viol_masked / d_masked
        
        np.add.at(grad_centers, i_masked, coeff[:, np.newaxis] * vecs)
        np.add.at(grad_centers, j_masked, -coeff[:, np.newaxis] * vecs)
    
    # Wall overlaps
    L = 0.5 - R
    abs_c = np.abs(centers)
    wall_viol = np.maximum(0.0, abs_c - L)
    energy += np.sum(wall_viol ** 2)
    grad_centers += 2.0 * wall_viol * np.sign(centers)
    
    grad = grad_centers.ravel()
    return energy, grad

def _soft_min_obj_and_grad(x, beta):
    """
    Objective: negative of the soft-minimum distance (to be minimized).
    """
    centers = x.reshape(N, 3)
    
    wall_dists = 0.5 - np.abs(centers)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2) + 1e-15)
    i_idx, j_idx = np.triu_indices(N, k=1)
    d_ij = dist[i_idx, j_idx] / 2.0
    
    all_dists = np.concatenate([wall_dists.ravel(), d_ij])
    
    min_d = np.min(all_dists)
    shifted = all_dists - min_d
    exp_terms = np.exp(-shifted / beta)
    sum_exp = np.sum(exp_terms)
    
    value = -min_d + beta * np.log(sum_exp)
    
    weights = exp_terms / sum_exp
    grad_dists = -weights
    
    grad_centers = np.zeros_like(centers)
    
    wall_weights = grad_dists[:N*3].reshape(N, 3)
    grad_centers -= wall_weights * np.sign(centers)
    
    pair_weights = grad_dists[N*3:]
    i_masked = i_idx
    j_masked = j_idx
    
    vecs = centers[i_masked] - centers[j_masked]
    d_masked = dist[i_masked, j_masked]
    
    coeff = pair_weights / (2.0 * d_masked)
    np.add.at(grad_centers, i_masked, coeff[:, np.newaxis] * vecs)
    np.add.at(grad_centers, j_masked, -coeff[:, np.newaxis] * vecs)
    
    return value, grad_centers.ravel()

def _generate_initial_guess(seed=0):
    """Generate diverse initial guesses."""
    rng = np.random.default_rng(seed)
    centers = np.zeros((N, 3))
    strategy = seed % 6
    
    if strategy == 0:
        # 5x5x2 grid with offset
        pts = []
        for i in range(5):
            for j in range(5):
                for k in range(2):
                    x = -0.4 + i * 0.2
                    y = -0.4 + j * 0.2
                    z = -0.2 + k * 0.4
                    pts.append([x, y, z])
        centers[:50] = np.array(pts[:50])
        centers += rng.normal(0, 0.02, (N, 3))
        
    elif strategy == 1:
        # BCC-like: 3x3x3 corners + body centers + extras
        pts = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    x = (i - 1) * 0.4
                    y = (j - 1) * 0.4
                    z = (k - 1) * 0.4
                    pts.append([x, y, z])
                    if i < 2 and j < 2 and k < 2:
                        pts.append([x + 0.2, y + 0.2, z + 0.2])
        centers[:35] = np.array(pts[:35])
        centers[35:] = rng.uniform(-0.35, 0.35, (15, 3))
        
    elif strategy == 2:
        # Random uniform
        centers = rng.uniform(-0.4, 0.4, (N, 3))
        
    elif strategy == 3:
        # Two-layer 5x5 with stagger
        bottom = []
        top = []
        for i in range(5):
            for j in range(5):
                x = -0.4 + i * 0.2
                y = -0.4 + j * 0.2
                bottom.append([x, y, -0.15])
                tx = -0.3 + i * 0.2
                ty = -0.3 + j * 0.2
                top.append([tx, ty, 0.15])
        centers[:25] = np.array(bottom)
        centers[25:] = np.array(top)
        centers += rng.normal(0, 0.015, (N, 3))
        
    elif strategy == 4:
        # FCC-like: face-centered cubic
        pts = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    x = (i - 1) * 0.35
                    y = (j - 1) * 0.35
                    z = (k - 1) * 0.35
                    pts.append([x, y, z])
                    if i < 2:
                        pts.append([x + 0.175, y, z])
                    if j < 2:
                        pts.append([x, y + 0.175, z])
                    if k < 2:
                        pts.append([x, y, z + 0.175])
        pts = np.array(pts[:50])
        centers = pts + rng.normal(0, 0.02, (N, 3))
        
    else:
        # Hexagonal layers
        pts = []
        layer_z = [-0.3, -0.1, 0.1, 0.3]
        for li, z in enumerate(layer_z):
            n_per_layer = 13 if li % 2 == 0 else 12
            for i in range(n_per_layer):
                row = i // 4
                col = i % 4
                x = -0.36 + col * 0.24 + (0.12 if row % 2 else 0)
                y = -0.3 + row * 0.2
                pts.append([x, y, z])
        centers[:50] = np.array(pts[:50])
        centers += rng.normal(0, 0.02, (N, 3))
    
    centers = np.clip(centers, -0.45, 0.45)
    return centers

def _optimize_soft_min(x0, time_budget):
    """Run L-BFGS-B with beta annealing."""
    start_time = time.time()
    current_x = x0.copy()
    bounds = [(-0.5, 0.5)] * (N * 3)
    beta_schedule = [0.05, 0.03, 0.015, 0.008, 0.004, 0.002]
    
    for beta in beta_schedule:
        if time.time() - start_time > time_budget:
            break
        try:
            res = minimize(
                _soft_min_obj_and_grad,
                current_x,
                args=(beta,),
                jac=True,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50, 'iprint': -1}
            )
            current_x = res.x
        except Exception:
            pass
    
    return current_x

def _refine_with_energy(centers, r_est, time_budget):
    """
    Use energy minimization to try to achieve radius r_est.
    Binary search on radius to find the maximum achievable.
    """
    start_time = time.time()
    best_centers = centers.copy()
    best_r = _compute_radius(centers)
    
    # Try to increase radius slightly using energy minimization
    # Binary search: find max R such that energy can be driven to ~0
    lo = best_r
    hi = min(0.145, best_r + 0.005)
    
    for _ in range(8):  # ~8 bisection steps
        if time.time() - start_time > time_budget:
            break
        mid = (lo + hi) / 2.0
        if mid <= lo:
            break
        
        x0 = best_centers.ravel()
        bounds = [(-0.5 + mid, 0.5 - mid)] * (N * 3)
        
        try:
            res = minimize(
                _energy_and_grad,
                x0,
                args=(mid,),
                jac=True,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 30, 'iprint': -1}
            )
            new_centers = res.x.reshape(N, 3)
            new_r = _compute_radius(new_centers)
            
            if res.fun < 1e-6 and new_r >= mid - 1e-8:
                # Achieved radius mid
                lo = mid
                best_centers = new_centers
                best_r = new_r
            else:
                hi = mid
        except Exception:
            hi = mid
    
    return best_centers, best_r

def construct():
    """Construct optimal sphere packing."""
    start_time = time.time()
    
    best_radius = 0.0
    best_centers = None
    
    num_restarts = 20
    
    for restart_idx in range(num_restarts):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 20:
            break
        
        centers = _generate_initial_guess(seed=restart_idx)
        x0 = centers.ravel()
        
        remaining = DEADLINE - 20 - elapsed
        time_per_restart = remaining / max(1, num_restarts - restart_idx)
        time_per_restart = min(time_per_restart, 20.0)
        
        try:
            # Phase 1: Soft-min optimization
            t1 = time.time()
            new_x = _optimize_soft_min(x0, time_per_restart * 0.6)
            new_centers = new_x.reshape(N, 3)
            r_soft = _compute_radius(new_centers)
            
            # Phase 2: Energy-based refinement
            t2 = time.time()
            refine_budget = max(1.0, time_per_restart - (t2 - t1))
            refined_centers, r_refined = _refine_with_energy(new_centers, r_soft, refine_budget)
            
            r = max(r_soft, r_refined)
            if r_refined > r_soft:
                new_centers = refined_centers
            
            if r > best_radius:
                best_radius = r
                best_centers = new_centers.copy()
                
        except Exception:
            pass
    
    # Final polish on the best result
    if best_centers is not None and time.time() - start_time < DEADLINE - 5:
        x_best = best_centers.ravel()
        bounds = [(-0.5, 0.5)] * (N * 3)
        try:
            res = minimize(
                _soft_min_obj_and_grad,
                x_best,
                args=(0.002,),
                jac=True,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50, 'iprint': -1}
            )
            polished = res.x.reshape(N, 3)
            r_polished = _compute_radius(polished)
            
            # Try energy refinement on polished
            final_budget = max(1.0, DEADLINE - 3 - (time.time() - start_time))
            final_centers, r_final = _refine_with_energy(polished, r_polished, final_budget)
            
            if r_final > best_radius:
                best_radius = r_final
                best_centers = final_centers
            elif r_polished > best_radius:
                best_radius = r_polished
                best_centers = polished
        except Exception:
            pass
    
    if best_centers is None:
        centers = _generate_initial_guess(seed=99)
        return centers
        
    return best_centers
