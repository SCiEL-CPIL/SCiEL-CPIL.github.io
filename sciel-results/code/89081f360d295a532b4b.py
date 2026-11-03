import numpy as np
import time
import math
from scipy.optimize import least_squares

N = 300
DEADLINE = 145.0 # Safety margin from 150s limit

def compute_score(points):
    """
    Compute the exact packing radius for a set of points.
    Score = min(min_pair_dist / 2, min_boundary_dist)
    """
    rad = np.linalg.norm(points, axis=1)
    min_bound = 1.0 - np.max(rad)
    
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    score = min(min_pair / 2.0, min_bound)
    return score

def soft_min_value(points, k=100.0):
    """
    Compute the smooth minimum of pairwise distances and boundary distances.
    This is used as a surrogate objective for optimization.
    """
    diff = points[:, np.newaxis, :] - points[np.newaxis, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    rad = np.linalg.norm(points, axis=1)
    bound_dists = 1.0 - rad
    
    # We need the minimum of all pair distances and boundary distances.
    # To avoid memory issues with a huge vector, we can compute in chunks or just flatten.
    # N=300 -> 44850 pairs + 300 bounds = 45150 values. This is small.
    pair_dists = dists[np.triu_indices(N, k=1)]
    all_dists = np.concatenate([pair_dists, bound_dists])
    
    # Soft min: -1/k * log(sum(exp(-k * d)))
    # To prevent underflow/overflow, shift by min
    d_min = np.min(all_dists)
    exp_terms = np.exp(-k * (all_dists - d_min))
    sum_exp = np.sum(exp_terms)
    
    if sum_exp == 0:
        return d_min
    
    soft_min = d_min - (1.0 / k) * np.log(sum_exp)
    return soft_min

def polish_least_squares(points, time_budget, k=100.0, R_target=0.06):
    """
    Polish using scipy.optimize.least_squares.
    We minimize (R_target - soft_min_value(x))^2.
    """
    start_time = time.time()
    
    def residuals(x_flat):
        x = x_flat.reshape(N, 2)
        # Project inside container to ensure validity during optimization
        rad = np.linalg.norm(x, axis=1)
        mask = rad > 1.0
        if np.any(mask):
            x = x.copy()
            x[mask] /= rad[mask, np.newaxis]
        
        s_val = soft_min_value(x, k)
        # We want to maximize s_val, so minimize (R_target - s_val)^2
        # If s_val > R_target, this still pushes it down, but R_target is high.
        # Actually, if s_val can exceed R_target, we should just minimize -s_val?
        # No, least_squares minimizes sum of squares.
        # If we use residual = R_target - s_val, and s_val approaches R_target, residual -> 0.
        # If s_val > R_target, residual becomes negative, square is positive, gradient pushes back towards R_target?
        # Let's check: f(s) = (R - s)^2. f'(s) = -2(R-s).
        # If s < R, f' < 0, so increasing s decreases f. Good.
        # If s > R, f' > 0, so increasing s increases f. Bad, it pushes s down.
        # So we need R_target to be strictly greater than the maximum possible radius.
        # Max radius is ~0.053. Let's use 0.1.
        
        return np.array([R_target - s_val])

    x0 = points.flatten()
    
    # Use trf (trust-region reflective) which handles bounds well if we add them, 
    # but we are projecting inside anyway.
    # We don't have explicit bounds in least_squares if we project, but we can add them.
    # Bounds: -1 <= x <= 1 is too loose. 
    # Actually, we can just use the projection.
    
    try:
        # max_nfev is important. 
        # Each eval is ~1ms. 1000 evals = 1s.
        # Time budget is e.g. 10s. So 10000 evals.
        result = least_squares(
            residuals, 
            x0, 
            method='trf', 
            max_nfev=20000,
            xtol=1e-12, 
            ftol=1e-12
        )
        
        if time.time() - start_time > time_budget:
            # If it took too long, we might have hit the max_nfev or time limit inside?
            # least_squares doesn't have a time limit, only nfev.
            # We should wrap it in a way that respects time.
            # For now, rely on max_nfev.
            pass
            
        x_opt = result.x.reshape(N, 2)
        score_opt = compute_score(x_opt)
        
        return x_opt, score_opt
        
    except Exception as e:
        print(f"LSQ failed: {e}")
        return points, compute_score(points)

def construct():
    start_time = time.time()
    
    # 1. Initialization: Hexagonal Lattice
    # Optimal density for hex packing is pi / (2 * sqrt(3)) ~ 0.9069
    # Area of unit circle = pi.
    # Area of N circles = N * pi * r^2.
    # N * r^2 ~ Area * density = pi * 0.9069
    # r^2 ~ 0.9069 / N
    # r ~ sqrt(0.9069 / 300) ~ sqrt(0.00302) ~ 0.055
    # This is close to 0.053.
    
    s_init = np.sqrt(2 * np.pi / (np.sqrt(3) * N))
    # Scale down slightly to ensure they fit
    s_gen = s_init * 0.95
    y_step = s_gen * np.sqrt(3) / 2
    max_j = int(1.0 / y_step) + 2
    max_i = int(1.0 / s_gen) + 2
    
    points_list = []
    for j in range(-max_j, max_j + 1):
        y = j * y_step
        offset = 0 if j % 2 == 0 else s_gen / 2
        for i in range(-max_i, max_i + 1):
            x = i * s_gen + offset
            if np.hypot(x, y) < 0.99:
                points_list.append([x, y])
                
    points = np.array(points_list)
    
    if len(points) > N:
        dists_to_center = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists_to_center)[:N]
        points = points[idx]
    elif len(points) < N:
        # Fill remaining with random points near center
        while len(points) < N:
            p = np.random.uniform(-0.5, 0.5, 2)
            points = np.vstack([points, p])
        points = points[:N]
        
    best_points = points.copy()
    best_score = compute_score(points)
    
    # 2. Optimization Phase
    # We will run least_squares multiple times.
    # First run: on the initial hex grid.
    # Second run: on a slightly perturbed version to escape local minima.
    
    num_runs = 5
    time_per_run = (DEADLINE - (time.time() - start_time)) / num_runs
    
    for i in range(num_runs):
        if time.time() - start_time > DEADLINE - 5.0:
            break
            
        if i == 0:
            current_points = best_points.copy()
        else:
            # Perturb best points
            # Small perturbation to escape local minima
            noise = np.random.normal(0, 0.01, size=best_points.shape)
            current_points = best_points + noise
            # Project inside
            rad = np.linalg.norm(current_points, axis=1)
            mask = rad > 1.0
            if np.any(mask):
                current_points[mask] /= rad[mask, np.newaxis]
                
        # Run LSQ polish
        # Allocate time based on remaining budget
        run_time_budget = min(time_per_run, DEADLINE - (time.time() - start_time) - 2.0)
        
        if run_time_budget < 1.0:
            continue
            
        polished_points, polished_score = polish_least_squares(current_points, run_time_budget)
        
        if polished_score > best_score:
            best_score = polished_score
            best_points = polished_points
        else:
            # If LSQ didn't improve, maybe the initial perturbation was bad?
            # Or LSQ got stuck.
            pass
            
    # Final validation
    rad_final = np.linalg.norm(best_points, axis=1)
    if np.any(rad_final > 1.0) or not np.all(np.isfinite(best_points)):
        # Fallback to a simple valid configuration if something went wrong
        # (Unlikely, but safety first)
        return np.zeros((N, 2))
        
    return best_points
