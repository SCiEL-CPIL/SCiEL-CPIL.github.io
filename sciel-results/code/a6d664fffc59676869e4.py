import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill

    def compute_log_score_exact(params):
        """Compute log of normalized product of pairwise distances (Exact Max)."""
        x = params[:7]
        y = params[7:14]
        x7 = -np.sum(x)
        y7 = -np.sum(y)
        xs = np.append(x, x7)
        ys = np.append(y, y7)
        points = np.column_stack((xs, ys))
        
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        iu = np.triu_indices(8, k=1)
        pair_dists_sq = dists_sq[iu]
        
        min_d = np.min(pair_dists_sq)
        if min_d < 1e-16:
            return -1e10
        
        D_sq = np.max(pair_dists_sq)
        if D_sq < 1e-16:
            return -1e10
        
        # log_prod = 0.5 * sum(log(pair_dists_sq))
        # - 28 * log(D) = -14 * log(D_sq)
        log_prod = 0.5 * np.sum(np.log(pair_dists_sq))
        log_diam_term = 14.0 * np.log(D_sq)
        
        return log_prod - log_diam_term

    def compute_log_score_smooth(params, tau=10.0):
        """Compute log of normalized product of pairwise distances (Smooth Max)."""
        x = params[:7]
        y = params[7:14]
        x7 = -np.sum(x)
        y7 = -np.sum(y)
        xs = np.append(x, x7)
        ys = np.append(y, y7)
        points = np.column_stack((xs, ys))
        
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        iu = np.triu_indices(8, k=1)
        pair_dists_sq = dists_sq[iu]
        
        min_d = np.min(pair_dists_sq)
        if min_d < 1e-16:
            return -1e10
        
        # Smooth approximation for log(D_sq)
        # log(D_sq) = log(max(pair_dists_sq))
        # approx: (1/tau) * log(sum(exp(tau * log(pair_dists_sq))))
        # = (1/tau) * log(sum(pair_dists_sq^tau))
        
        # To avoid overflow with large tau or large values, we can scale
        # But pair_dists_sq are usually O(1) to O(10).
        # Let's use log-sum-exp trick for stability if needed, but for tau=10, direct power is likely fine.
        
        try:
            # Using logsumexp for stability
            log_d_sq = np.log(pair_dists_sq)
            # max_log = np.max(log_d_sq)
            # sum_exp = np.sum(np.exp(log_d_sq * tau - max_log * tau))
            # smooth_log_d_sq = (max_log * tau + np.log(sum_exp)) / tau
            
            # Simpler version:
            # Since tau is small (10), direct computation:
            powered = pair_dists_sq ** tau
            sum_powered = np.sum(powered)
            if sum_powered <= 0:
                return -1e10
            smooth_log_d_sq = np.log(sum_powered) / tau
        except Exception:
            # Fallback to exact if numerical issues
            D_sq = np.max(pair_dists_sq)
            if D_sq < 1e-16:
                return -1e10
            smooth_log_d_sq = np.log(D_sq)

        log_prod = 0.5 * np.sum(np.log(pair_dists_sq))
        
        # Objective is log_prod - 14 * log(D_sq)
        # Note: log(D) = 0.5 log(D_sq). So -28 log(D) = -14 log(D_sq).
        # The term in compute_log_score_exact was 14.0 * np.log(D_sq).
        # So we subtract 14.0 * smooth_log_d_sq.
        
        return log_prod - 14.0 * smooth_log_d_sq

    def objective_smooth(params):
        return -compute_log_score_smooth(params)

    def objective_exact(params):
        return -compute_log_score_exact(params)

    def points_to_params(points):
        """Convert 8 points (centered) to 14 parameters."""
        com = np.mean(points, axis=0)
        points = points - com
        xs = points[:, 0]
        ys = points[:, 1]
        return np.append(xs[:7], ys[:7])

    def params_to_points(params):
        """Convert 14 parameters to 8 points."""
        x = params[:7]
        y = params[7:14]
        x7 = -np.sum(x)
        y7 = -np.sum(y)
        xs = np.append(x, x7)
        ys = np.append(y, y7)
        return np.column_stack((xs, ys))

    def normalize_diameter(points):
        """Scale points to have diameter exactly 2."""
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        D_sq = np.max(dists_sq)
        if D_sq > 0:
            return points * (2.0 / np.sqrt(D_sq))
        return points

    # Generate diverse initial configurations
    def get_octagon(scale=1.0):
        """Regular octagon."""
        angles = np.arange(8) * (np.pi / 4)
        return scale * np.column_stack((np.cos(angles), np.sin(angles)))

    def get_two_squares(r1, r2, angle_offset=0):
        """Two concentric squares."""
        angles1 = np.array([0, np.pi/2, np.pi, 3*np.pi/2]) + angle_offset
        angles2 = np.array([np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4]) + angle_offset
        p1 = r1 * np.column_stack((np.cos(angles1), np.sin(angles1)))
        p2 = r2 * np.column_stack((np.cos(angles2), np.sin(angles2)))
        return np.vstack((p1, p2))

    def get_stretched_octagon(sx, sy):
        """Stretched octagon."""
        angles = np.arange(8) * (np.pi / 4)
        xs = sx * np.cos(angles)
        ys = sy * np.sin(angles)
        return np.column_stack((xs, ys))

    def get_perturbed_octagon(perturbation=0.1, seed=0):
        """Octagon with random perturbations."""
        rng = np.random.default_rng(seed)
        oct = get_octagon()
        oct += rng.normal(0, perturbation, size=oct.shape)
        return oct

    # Create initial configurations
    inits = []
    
    # Regular octagon
    inits.append(get_octagon(1.0))
    
    # Stretched octagons
    inits.append(get_stretched_octagon(1.3, 0.8))
    inits.append(get_stretched_octagon(0.8, 1.3))
    inits.append(get_stretched_octagon(1.5, 0.7))
    inits.append(get_stretched_octagon(0.7, 1.5))
    
    # Two squares with different radii
    inits.append(get_two_squares(0.6, 1.0))
    inits.append(get_two_squares(0.7, 1.0))
    inits.append(get_two_squares(0.8, 1.0))
    inits.append(get_two_squares(0.5, 1.0))
    inits.append(get_two_squares(0.7, 1.2))
    inits.append(get_two_squares(0.8, 1.2))
    
    # Perturbed octagons
    for seed in range(10):
        inits.append(get_perturbed_octagon(0.15, seed))
    
    # Random configurations
    rng = np.random.default_rng(42)
    for _ in range(10):
        pts = rng.normal(0, 1, (8, 2))
        inits.append(pts)
    
    # Convert all to params
    init_params = [points_to_params(p) for p in inits]
    
    best_score = -np.inf
    best_params = None
    best_points = None
    
    # Optimization loop with time management
    n_inits = len(init_params)
    
    # Phase 1: Smooth optimization with L-BFGS-B
    for i, init in enumerate(init_params):
        elapsed = time.time() - start_time
        remaining = time_limit - elapsed
        
        if remaining < 10:
            break
        
        # Choose max iterations based on remaining time
        # L-BFGS-B is faster than Nelder-Mead per iteration?
        # Let's allow up to 2000 iterations, but check time.
        maxiter = min(2000, int(remaining * 20))
        
        if maxiter < 100:
            break
        
        try:
            res = minimize(objective_smooth, init, method='L-BFGS-B', 
                         options={'maxiter': maxiter, 'gtol': 1e-8, 'ftol': 1e-8})
            
            if res.fun < 1e9:  # Valid result
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_params = res.x.copy()
                    best_points = params_to_points(res.x)
                    
        except Exception:
            continue

    # Phase 2: Polish with Exact objective using Nelder-Mead
    # We only polish the best candidate found in Phase 1
    if best_params is not None:
        elapsed = time.time() - start_time
        remaining = time_limit - elapsed
        
        if remaining > 10:
            try:
                # Use Nelder-Mead for non-smooth exact objective
                # Start from the best params found
                maxiter_polish = min(5000, int(remaining * 50))
                
                res_polish = minimize(objective_exact, best_params, method='Nelder-Mead', 
                                     options={'maxiter': maxiter_polish, 'xatol': 1e-10, 'fatol': 1e-10,
                                              'adaptive': True})
                
                if res_polish.fun < 1e9:
                    score = -res_polish.fun
                    if score > best_score:
                        best_score = score
                        best_params = res_polish.x.copy()
                        best_points = params_to_points(res_polish.x)
                        
            except Exception:
                pass

    # If we found a good solution, use it
    if best_params is not None and best_points is not None:
        points = best_points
        points = normalize_diameter(points)
        
        # Final validation: ensure distinct points and valid coordinates
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        iu = np.triu_indices(8, k=1)
        if np.min(dists[iu]) < 1e-6:
            # Points are too close, fall back
            pass
        elif np.any(np.abs(points) >= 1e6):
            # Coordinates out of bounds, scale down
            points = points * (1e5 / np.max(np.abs(points)))
        else:
            return points.tolist()
    
    # Fallback: return regular octagon
    angles = np.arange(8) * (np.pi / 4)
    points = np.column_stack((np.cos(angles), np.sin(angles)))
    points = normalize_diameter(points)
    return points.tolist()
