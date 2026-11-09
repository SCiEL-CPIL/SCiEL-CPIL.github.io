import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s margin for safety
    
    n = 8
    # Precompute indices for pairs
    i_idx, j_idx = np.triu_indices(n, k=1)
    num_pairs = len(i_idx)
    
    def compute_distances(points):
        # points: (n, 2)
        # Returns squared distances for all pairs i < j
        diffs = points[i_idx] - points[j_idx]
        return np.sum(diffs**2, axis=1)
    
    def objective(x):
        points = x.reshape(n, 2)
        sq_dists = compute_distances(points)
        # Avoid log(0)
        sq_dists = np.maximum(sq_dists, 1e-10)
        # Maximize sum of log(dist) = 0.5 * sum of log(sq_dist)
        # Minimize negative
        return -0.5 * np.sum(np.log(sq_dists))
    
    def constraints(x):
        points = x.reshape(n, 2)
        sq_dists = compute_distances(points)
        # Constraint: sq_dist <= 4 (diameter <= 2)
        # Inequality constraint: 4 - sq_dist >= 0
        return 4.0 - sq_dists

    # Initial guesses
    inits = []
    
    # 1. Regular Octagon
    angles = 2 * np.pi * np.arange(n) / n
    init_reg = np.column_stack([np.cos(angles), np.sin(angles)])
    inits.append(init_reg.flatten())
    
    # 2. Perturbed Regular Octagon
    np.random.seed(42)
    for _ in range(5):
        noise = np.random.normal(0, 0.1, (n, 2))
        init_pert = init_reg + noise
        inits.append(init_pert.flatten())
        
    # 3. Two clusters (left and right)
    # Try to split points into two groups
    left = np.column_stack([np.cos(2*np.pi*np.arange(4)/4) * 0.5 - 0.5, np.sin(2*np.pi*np.arange(4)/4) * 0.5])
    right = np.column_stack([np.cos(2*np.pi*np.arange(4)/4) * 0.5 + 0.5, np.sin(2*np.pi*np.arange(4)/4) * 0.5])
    init_cluster = np.vstack([left, right])
    inits.append(init_cluster.flatten())
    
    # 4. Random points within unit circle
    for _ in range(10):
        r = np.random.uniform(0.5, 1.0, n)
        theta = np.random.uniform(0, 2*np.pi, n)
        init_rand = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        inits.append(init_rand.flatten())

    best_score = 0.0
    best_points = None
    
    def get_score(points):
        # Scale to diameter 2
        dists = np.sqrt(compute_distances(points))
        d_max = np.max(dists)
        if d_max < 1e-10:
            return 0.0
        scaled_dists = dists * (2.0 / d_max)
        # Product of ordered distances
        # Delta = prod_{i != j} d_ij = (prod_{i < j} d_ij)^2
        # Score = Delta / n^n
        # log_score = 2 * sum(log(d_ij)) - n * log(n)
        d_safe = np.maximum(scaled_dists, 1e-10)
        log_prod_unordered = np.sum(np.log(d_safe))
        log_delta = 2.0 * log_prod_unordered
        log_score = log_delta - n * np.log(n)
        return np.exp(log_score)

    for i, x0 in enumerate(inits):
        if time.time() > deadline:
            break
            
        try:
            # Use SLSQP with constraints
            # bounds: coordinates should be reasonable, e.g., [-2, 2]
            bounds = [(-2, 2)] * (2 * n)
            
            res = minimize(
                objective, 
                x0, 
                method='SLSQP', 
                bounds=bounds,
                constraints=[{'type': 'ineq', 'fun': constraints}],
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            
            if res.success or res.fun < 1e10:
                points = res.x.reshape(n, 2)
                score = get_score(points)
                if score > best_score:
                    best_score = score
                    best_points = points.copy()
                    
        except Exception as e:
            pass

    # If SLSQP didn't find anything better, try Nelder-Mead on the objective with a penalty for diameter > 2
    # This is a fallback for cases where constraints are hard to satisfy
    if best_score <= 1.0:
        def objective_penalty(x):
            points = x.reshape(n, 2)
            sq_dists = compute_distances(points)
            d_max_sq = np.max(sq_dists)
            # Penalty if diameter > 2
            penalty = 0.0
            if d_max_sq > 4.0:
                penalty = 100.0 * (d_max_sq - 4.0)
            
            sq_dists_safe = np.maximum(sq_dists, 1e-10)
            obj_val = -0.5 * np.sum(np.log(sq_dists_safe))
            return obj_val + penalty

        for i in range(10):
            if time.time() > deadline:
                break
            x0 = np.random.uniform(-1, 1, 2 * n)
            try:
                res = minimize(
                    objective_penalty, 
                    x0, 
                    method='Nelder-Mead', 
                    options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8}
                )
                points = res.x.reshape(n, 2)
                # Check if diameter is valid (close to 2)
                dists = np.sqrt(compute_distances(points))
                d_max = np.max(dists)
                if d_max < 1e-10:
                    continue
                # Scale to diameter 2
                scaled_points = points * (2.0 / d_max)
                score = get_score(scaled_points)
                if score > best_score:
                    best_score = score
                    best_points = scaled_points.copy()
            except Exception as e:
                pass

    if best_points is None:
        # Return regular octagon
        angles = 2 * np.pi * np.arange(n) / n
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    else:
        # Ensure diameter is exactly 2 for final output consistency, though score function handles it
        dists = np.sqrt(compute_distances(best_points))
        d_max = np.max(dists)
        if d_max > 1e-10:
            best_points = best_points * (2.0 / d_max)

    return [[float(p[0]), float(p[1])] for p in best_points]
