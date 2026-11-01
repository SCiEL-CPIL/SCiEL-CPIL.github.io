"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_score = -1.0
    best_points = None
    
    # Precompute upper triangle indices for pairwise distances
    iu = np.triu_indices(N, k=1)
    num_pairs = len(iu[0])
    
    def calculate_score(points):
        if points is None:
            return 0.0
        pts = points.reshape(N, 3)
        if np.any(pts < -0.5) or np.any(pts > 0.5):
            return 0.0
        if not np.all(np.isfinite(pts)):
            return 0.0
            
        # Pairwise distances using precomputed indices
        diff = pts[iu[0]] - pts[iu[1]]
        dists = np.sqrt(np.sum(diff * diff, axis=1))
        min_pair = np.min(dists)
        
        boundary_dists = 0.5 - np.abs(pts)
        min_boundary = np.min(boundary_dists)
        
        r = min(min_pair / 2.0, min_boundary)
        return r
    
    def penalty_objective_and_grad(x, r_target, mu):
        """
        Penalty method: minimize sum of violations of constraint d_ij >= 2*r_target
        and boundary constraint b_i >= r_target.
        Objective: mu * sum(max(0, 2*r_target - d_ij)^2 + max(0, r_target - b_i)^2)
        """
        pts = x.reshape(N, 3)
        
        # Pairwise part
        diff = pts[iu[0]] - pts[iu[1]]
        dists_sq = np.sum(diff * diff, axis=1)
        dists = np.sqrt(dists_sq + 1e-20)  # Avoid division by zero
        
        # Violations: max(0, 2*r_target - dists)
        violations_pair = np.maximum(0.0, 2.0 * r_target - dists)
        obj_pair = mu * np.sum(violations_pair ** 2)
        
        # Gradient for pairwise part
        # d/dx_i sum_j max(0, 2r - d_ij)^2
        # For active constraints (d_ij < 2r):
        # d/dx_i (2r - d_ij)^2 = 2*(2r - d_ij)*(-d(d_ij)/dx_i)
        # d(d_ij)/dx_i = (x_i - x_j)/d_ij for i, and -(x_i - x_j)/d_ij for j
        # So d/dx_i = 2*(2r - d_ij)*(x_i - x_j)/d_ij
        # And d/dx_j = -2*(2r - d_ij)*(x_i - x_j)/d_ij
        
        grad = np.zeros_like(pts)
        active = violations_pair > 0
        if np.any(active):
            active_idx = np.where(active)[0]
            active_dists = dists[active_idx]
            active_viol = violations_pair[active_idx]
            active_diff = diff[active_idx]
            
            # Factor: 2 * (2r - d) / d
            factors = 2.0 * active_viol / active_dists
            
            # Accumulate gradients
            # For each active pair (i, j) with index k:
            # grad[i] += factors[k] * (pts[i] - pts[j])
            # grad[j] -= factors[k] * (pts[i] - pts[j])
            
            for k in active_idx:
                i, j = iu[0][k], iu[1][k]
                f = factors[np.where(active_idx == k)[0][0]]
                grad[i] += f * active_diff[np.where(active_idx == k)[0][0]]
                grad[j] -= f * active_diff[np.where(active_idx == k)[0][0]]
        
        # Boundary part
        b_dists = 0.5 - np.abs(pts)
        violations_bound = np.maximum(0.0, r_target - b_dists)
        obj_bound = mu * np.sum(violations_bound ** 2)
        
        # Gradient for boundary part
        # For each point i and each coordinate axis:
        # b_i = min(0.5 - |x_i|, 0.5 - |y_i|, 0.5 - |z_i|)
        # If active (b_i < r_target), we need the subgradient
        
        active_bound = violations_bound > 0
        if np.any(active_bound):
            active_bound_idx = np.where(active_bound)[0]
            for i in active_bound_idx:
                # Find which face is closest
                min_coord = np.argmin(b_dists[i])
                coord_val = pts[i, min_coord]
                if coord_val > 0:
                    sign = 1.0
                elif coord_val < 0:
                    sign = -1.0
                else:
                    sign = 0.0
                
                if sign != 0:
                    # d/dx_i (r_target - b_i)^2 = 2*(r_target - b_i)*(-db_i/dx_i)
                    # db_i/dx_i = -sign * e_min_coord
                    # So d/dx_i = 2*(r_target - b_i)*sign*e_min_coord
                    grad[i, min_coord] += 2.0 * violations_bound[i] * sign
        
        total_obj = obj_pair + obj_bound
        total_grad = grad.flatten()
        
        return total_obj, total_grad
    
    def init_grid():
        m = 5
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N].flatten()

    def init_random():
        return np.random.uniform(-0.4, 0.4, size=(N, 3)).flatten()

    def init_fcc():
        # Body-centered cubic style
        pts = []
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    pts.append([i*0.2-0.5, j*0.2-0.5, k*0.2-0.5])
                    if i < 4 and j < 4 and k < 4:
                        pts.append([i*0.2-0.5+0.1, j*0.2-0.5+0.1, k*0.2-0.5+0.1])
        pts = np.array(pts[:N])
        return pts.flatten()

    starts = [
        init_grid(),
        init_grid() + np.random.normal(0, 0.015, size=(N*3,)),
        init_grid() + np.random.normal(0, 0.03, size=(N*3,)),
        init_random(),
        init_random() + np.random.normal(0, 0.05, size=(N*3,)),
        init_fcc()
    ]
    
    # Clip starts to valid range
    for i in range(len(starts)):
        starts[i] = np.clip(starts[i], -0.49, 0.49)

    bounds = [(-0.5, 0.5)] * (N * 3)
    
    # Penalty method: iteratively increase r_target and mu
    # Start with a small r and increase it
    r_targets = [0.05, 0.08, 0.10, 0.11, 0.111, 0.1113, 0.11138, 0.111382]
    
    for x0 in starts:
        if time.time() > deadline:
            break
        
        try:
            current_pts = x0.reshape(N, 3).copy()
            current_score = calculate_score(current_pts)
            
            for r_target in r_targets:
                if time.time() > deadline - 5:
                    break
                
                # Use penalty method with increasing mu
                mu = 10.0
                for _ in range(5):  # Inner iterations for fixed r_target
                    if time.time() > deadline - 3:
                        break
                    
                    res = minimize(penalty_objective_and_grad, current_pts.flatten(), 
                                   method='L-BFGS-B', bounds=bounds,
                                   args=(r_target, mu),
                                   jac=True,
                                   options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
                    
                    if res.fun < 1e6:
                        new_pts = res.x.reshape(N, 3)
                        new_pts = np.clip(new_pts, -0.5, 0.5)
                        new_score = calculate_score(new_pts)
                        
                        if new_score >= current_score - 1e-6:
                            current_pts = new_pts
                            current_score = new_score
                        
                        # Check if we can increase r_target
                        if new_score >= r_target - 1e-4:
                            break
                    else:
                        break
                    
                    mu *= 2.0
                
                # Update best
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_pts.copy()
                    
        except Exception:
            continue
    
    # Final refinement with high-precision penalty method
    if best_points is not None and time.time() < deadline - 15:
        try:
            # Use a very high r_target close to the known best
            r_target = 0.1113823475125
            mu = 100.0
            
            for _ in range(3):
                if time.time() > deadline - 5:
                    break
                
                res = minimize(penalty_objective_and_grad, best_points.flatten(),
                               method='L-BFGS-B', bounds=bounds,
                               args=(r_target, mu),
                               jac=True,
                               options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12})
                
                if res.fun < 1e4:
                    new_pts = res.x.reshape(N, 3)
                    new_pts = np.clip(new_pts, -0.5, 0.5)
                    new_score = calculate_score(new_pts)
                    
                    if new_score > best_score:
                        best_score = new_score
                        best_points = new_pts.copy()
                
                mu *= 5.0
        except Exception:
            pass
    
    if best_points is None:
        best_points = init_grid().reshape(N, 3)
        
    return best_points
# EVOLVE-BLOCK-END
