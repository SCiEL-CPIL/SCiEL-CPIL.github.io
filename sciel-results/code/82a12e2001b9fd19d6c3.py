import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 50
DEADLINE = 145.0  # safety margin before 150s kill

def _pairwise_indices(n):
    """Return array of (i, j) pairs for i < j."""
    iu = np.triu_indices(n, k=1)
    return np.column_stack((iu[0], iu[1]))

# Precompute indices for speed
PAIR_IDX = _pairwise_indices(N)

def _score_vec(points):
    """Vectorized score computation: min pairwise distance / 2 and min boundary distance."""
    points = points.reshape(-1, 2)
    n = len(points)
    radii = 1.0 - np.linalg.norm(points, axis=1)
    r_bound = np.min(radii)
    
    ci = points[PAIR_IDX[:, 0]]
    cj = points[PAIR_IDX[:, 1]]
    dists = np.linalg.norm(ci - cj, axis=1)
    r_pair = np.min(dists) / 2.0
    
    return min(r_bound, r_pair)

def _soft_min_obj(x, beta, r_target_offset=0.0):
    """
    Objective for L-BFGS-B: minimize -soft_min(distances).
    We want to maximize the minimum distance, so we minimize the negative of a soft approximation of the min.
    soft_min(a) = -1/beta * log(sum(exp(-beta * a_i)))
    So we minimize: 1/beta * log(sum(exp(-beta * a_i)))
    This is smooth and differentiable.
    """
    c = x.reshape(N, 2)
    
    # Boundary distances
    b = 1.0 - np.hypot(c[:, 0], c[:, 1])
    
    # Pairwise distances
    ci = c[PAIR_IDX[:, 0]]
    cj = c[PAIR_IDX[:, 1]]
    d = np.linalg.norm(ci - cj, axis=1) / 2.0
    
    # Combine all constraints
    # We want max(min(b, d)).
    # Let z = [b, d]. We want max(min(z)).
    # Approximate min(z) with soft_min(z).
    # soft_min(z) = -1/beta * log(sum(exp(-beta * z)))
    # We want to maximize this, so minimize -soft_min(z) = 1/beta * log(sum(exp(-beta * z)))
    
    z = np.concatenate([b, d])
    
    # Numerical stability
    max_z = np.max(z)
    # We use -beta * z, so max(-beta * z) = -beta * min(z)
    # logsumexp(-beta * z) = log(sum(exp(-beta * z)))
    # To be stable, we can do:
    # sum(exp(-beta * (z - min_z))) * exp(-beta * min_z)
    # log(sum(exp(-beta * z))) = -beta * min_z + log(sum(exp(-beta * (z - min_z))))
    
    min_z = np.min(z)
    # If min_z is very small or negative, exp might overflow? 
    # z are distances, usually > 0. 
    # Let's just use the standard trick.
    # exp(-beta * (z - min_z)) is <= 1.
    exp_terms = np.exp(-beta * (z - min_z))
    log_sum = np.log(np.sum(exp_terms))
    log_sum_exp = -beta * min_z + log_sum
    
    obj = log_sum_exp / beta
    
    return obj

def _param_to_points(params):
    """
    Map 8 parameters to 50 points.
    Structure:
    - 1 center at (0,0)
    - 6 points in Ring 1: radius r1, angle offset theta1
    - 12 points in Ring 2: radius r2, angle offset theta2
    - 31 points in Ring 3: radius r3, angle offset theta3
    Total 1+6+12+31 = 50.
    """
    r1, th1, r2, th2, r3, th3 = params[:6]
    # The remaining 2 params can be small perturbations or ignored if we stick to 6.
    # Let's stick to 6 for simplicity in the global search, or allow 8 with global rotation/shift?
    # The prompt says "8D". Let's use:
    # r1, th1, r2, th2, r3, th3, cx, cy (center shift)
    # This allows the whole cluster to shift.
    cx, cy = params[6], params[7]
    
    points = np.zeros((50, 2))
    
    # Center
    points[0] = [cx, cy]
    
    # Ring 1 (6 points)
    angles1 = th1 + np.arange(6) * (2 * np.pi / 6)
    points[1:7, 0] = cx + r1 * np.cos(angles1)
    points[1:7, 1] = cy + r1 * np.sin(angles1)
    
    # Ring 2 (12 points)
    angles2 = th2 + np.arange(12) * (2 * np.pi / 12)
    points[7:19, 0] = cx + r2 * np.cos(angles2)
    points[7:19, 1] = cy + r2 * np.sin(angles2)
    
    # Ring 3 (31 points)
    # 31 points in a ring is not symmetric. We distribute them evenly.
    angles3 = th3 + np.arange(31) * (2 * np.pi / 31)
    points[19:50, 0] = cx + r3 * np.cos(angles3)
    points[19:50, 1] = cy + r3 * np.sin(angles3)
    
    return points

def _local_optimize(c, time_limit=10.0, beta_schedule=None):
    """Local optimization using L-BFGS-B with smooth-min objective."""
    if beta_schedule is None:
        beta_schedule = [100.0, 500.0, 1000.0, 2000.0]
    
    best_c = c.copy()
    best_r = _score_vec(best_c)
    
    t_start = time.time()
    
    for beta in beta_schedule:
        if time.time() - t_start >= time_limit - 2:
            break
        
        def objective(x):
            return _soft_min_obj(x, beta)
        
        def callback(x):
            if time.time() - t_start >= time_limit - 1:
                raise Exception("timeout")
        
        try:
            res = minimize(
                objective,
                best_c.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-14},
                callback=callback
            )
            if np.all(np.isfinite(res.x)):
                cand = res.x.reshape(N, 2)
                r = _score_vec(cand)
                if r > best_r:
                    best_r = r
                    best_c = cand.copy()
        except Exception:
            pass
    
    return best_c, best_r

def construct():
    t_start = time.time()
    
    # 1. Global Search on Parametric Model
    # Bounds for parameters:
    # r1: [0, 0.5], th1: [0, 2pi]
    # r2: [0.1, 0.8], th2: [0, 2pi]
    # r3: [0.5, 1.0], th3: [0, 2pi]
    # cx, cy: [-0.2, 0.2]
    
    bounds = [
        (0.0, 0.5),   # r1
        (0.0, 2*np.pi), # th1
        (0.1, 0.8),   # r2
        (0.0, 2*np.pi), # th2
        (0.5, 1.0),   # r3
        (0.0, 2*np.pi), # th3
        (-0.2, 0.2),  # cx
        (-0.2, 0.2)   # cy
    ]
    
    def param_objective(params):
        points = _param_to_points(params)
        # Use a soft min score for the global search to guide it
        # We want to maximize the min distance.
        # _soft_min_obj returns the negative log-sum-exp scaled.
        # Lower is better for _soft_min_obj.
        return _soft_min_obj(points.flatten(), beta=50.0)
    
    best_points = None
    best_score = 0.0
    
    # Run Differential Evolution
    # Time limit for DE: ~30s
    de_deadline = time.time() + 30.0
    
    try:
        # Using DE to find a good topology
        # popsize=15, tol=1e-8
        res_de = differential_evolution(
            param_objective,
            bounds,
            seed=42,
            maxiter=100,
            tol=1e-8,
            popsize=15,
            # Callback to check time? DE doesn't have a direct callback for early stop in all versions, 
            # but we can wrap it or just let it run and check time after.
            # To be safe, we'll run a limited number of iterations.
        )
        
        if res_de.success or np.isfinite(res_de.fun):
            best_params = res_de.x
            points_de = _param_to_points(best_params)
            best_points = points_de
            best_score = _score_vec(best_points)
            
    except Exception:
        pass
    
    # If DE failed or didn't run, fallback to hex init
    if best_points is None:
        # Hexagonal init
        points = []
        s = 0.2
        max_coord = 5
        for j in range(-max_coord, max_coord + 1):
            for i in range(-max_coord, max_coord + 1):
                x = (i + 0.5 * (j % 2)) * s
                y = j * s * np.sqrt(3) / 2
                if np.hypot(x, y) <= 1.0 - s / 2:
                    points.append([x, y])
        points = np.array(points)
        if len(points) >= N:
            dists = np.linalg.norm(points, axis=1)
            idx = np.argsort(dists)[:N]
            best_points = points[idx]
            best_score = _score_vec(best_points)
        else:
            # Random fallback
            best_points = np.random.randn(N, 2) * 0.5
            best_score = _score_vec(best_points)

    # 2. Local Optimization from Best Global Result
    # Use remaining time for local optimization
    time_left = DEADLINE - (time.time() - t_start)
    
    if time_left > 10:
        # Run local optimization with a generous time limit
        c_opt, r_opt = _local_optimize(best_points, time_limit=min(60.0, time_left - 10), beta_schedule=[50.0, 200.0, 1000.0, 2000.0])
        if r_opt > best_score:
            best_score = r_opt
            best_points = c_opt.copy()

    # 3. Final Polish with Highest Beta
    time_left = DEADLINE - (time.time() - t_start)
    if time_left > 5:
        c_final, r_final = _local_optimize(best_points, time_limit=min(10.0, time_left - 2), beta_schedule=[5000.0])
        if r_final > best_score:
            best_score = r_final
            best_points = c_final.copy()

    # Ensure validity
    c_final = best_points.reshape(N, 2)
    r2 = np.hypot(c_final[:, 0], c_final[:, 1])
    for i in range(N):
        if r2[i] > 1 - best_score + 1e-10:
            scale = (1 - best_score) / r2[i]
            c_final[i] *= scale
    
    # Final score check
    r_check = _score_vec(c_final)
    if r_check < best_score:
        best_score = r_check
        
    return c_final
