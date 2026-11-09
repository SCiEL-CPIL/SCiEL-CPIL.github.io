import numpy as np
from scipy.optimize import differential_evolution, minimize
import math
import time

N = 100

def construct():
    # Fixed point at North Pole
    fixed_point = np.array([0.0, 0.0, 1.0])
    n_opt = N - 1
    
    def get_points(params):
        thetas = params[:n_opt]
        phis = params[n_opt:]
        
        x = np.sin(thetas) * np.cos(phis)
        y = np.sin(thetas) * np.sin(phis)
        z = np.cos(thetas)
        
        opt_points = np.column_stack([x, y, z])
        return np.vstack([fixed_point, opt_points])

    def energy(params):
        points = get_points(params)
        # Pairwise distances
        # Use scipy.spatial.distance.pdist for efficiency? 
        # Or manual vectorization.
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        i_idx, j_idx = np.triu_indices(N, k=1)
        pair_dists = dists[i_idx, j_idx]
        
        if np.any(pair_dists < 1e-12):
            return 1e10
            
        return -np.sum(np.log(pair_dists))

    def floyd_repulsion(n_iter=1000):
        # Initialize with random points or Fibonacci
        # Let's use Fibonacci for stability
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
        # Rotate so first point is at North Pole? 
        # Or just run repulsion on all N points and then fix the first one?
        # The problem fixes point 0. 
        # Floyd's algorithm works on all points. 
        # If we fix point 0, we only move the others.
        
        # Let's initialize pts[1:] from Fibonacci[1:]
        # pts[0] is fixed at (0,0,1)
        
        # Convert pts[1:] to spherical params
        opt_pts = pts[1:]
        thetas = np.arccos(opt_pts[:, 2])
        phis = np.at2(opt_pts[:, 1], opt_pts[:, 0]) # Typo in thought, use arctan2
        phis = np.arctan2(opt_pts[:, 1], opt_pts[:, 0])
        phis = phis % (2 * np.pi)
        params = np.concatenate([thetas, phis])
        
        # Run repulsion on the 99 points
        # Force on point i is sum over j != i of (p_i - p_j) / ||p_i - p_j||^3 ?
        # For logarithmic potential, force is gradient of log(1/d) = 1/d * gradient(d)?
        # Energy E = sum log(1/d). 
        # dE/dp_i = sum_{j != i} d/dp_i log(1/||p_i - p_j||)
        # = sum_{j != i} - d/dp_i log(||p_i - p_j||)
        # = sum_{j != i} - (1/||p_i - p_j||) * (p_i - p_j)/||p_i - p_j||
        # = sum_{j != i} (p_j - p_i) / ||p_i - p_j||^2
        
        # We update p_i, then project back to sphere.
        
        p = get_points(params)
        p[0] = fixed_point # Fix first point
        
        for k in range(n_iter):
            # Calculate forces
            forces = np.zeros_like(p)
            for i in range(1, N):
                for j in range(N):
                    if i == j: continue
                    d = p[i] - p[j]
                    dist_sq = np.dot(d, d)
                    if dist_sq < 1e-12:
                        dist_sq = 1e-12
                    # Force is (p_j - p_i) / dist^2
                    forces[i] += (p[j] - p[i]) / dist_sq
            
            # Update
            step = 0.01 * (1.0 - k/n_iter) # Annealing step
            for i in range(1, N):
                p[i] += step * forces[i]
                # Normalize
                norm = np.linalg.norm(p[i])
                if norm > 1e-12:
                    p[i] /= norm
            p[0] = fixed_point
            
            # Convert back to params for next iteration? 
            # Or just keep p and convert at the end?
            # We need params for the optimizer later. 
            # But for Floyd, we can just work with p.
            
        # Convert final p[1:] to params
        opt_pts = p[1:]
        thetas = np.arccos(opt_pts[:, 2])
        phis = np.arctan2(opt_pts[:, 1], opt_pts[:, 0])
        phis = phis % (2 * np.pi)
        return np.concatenate([thetas, phis])

    # 1. Generate candidates
    candidates = []
    
    # Candidate 1: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    # Fix first point? Fibonacci point 0 is not exactly (0,0,1).
    # We need to map the 99 optimized points.
    # Let's just take pts[1:] as the initial guess for the 99 free points.
    opt_pts = pts[1:]
    thetas = np.arccos(opt_pts[:, 2])
    phis = np.arctan2(opt_pts[:, 1], opt_pts[:, 0])
    phis = phis % (2 * np.pi)
    candidates.append(np.concatenate([thetas, phis]))
    
    # Candidate 2: Floyd Repulsion
    try:
        candidates.append(floyd_repulsion(500))
    except Exception:
        pass
        
    # Candidate 3: Random
    np.random.seed(42)
    thetas = np.random.uniform(0, np.pi, n_opt)
    phis = np.random.uniform(0, 2*np.pi, n_opt)
    candidates.append(np.concatenate([thetas, phis]))

    # 2. Refine with L-BFGS-B
    bounds = [(0.0, np.pi)] * n_opt + [(0.0, 2.0 * np.pi)] * n_opt
    
    best_params = None
    best_energy = float('inf')
    
    for c in candidates:
        try:
            res = minimize(energy, c, method='L-BFGS-B', bounds=bounds, options={'maxiter': 2000, 'ftol': 1e-12})
            if res.success or res.fun < best_energy:
                if res.fun < best_energy:
                    best_energy = res.fun
                    best_params = res.x
        except Exception:
            pass
            
    # 3. If time allows, run DE on the best candidate found so far?
    # Or run DE from scratch with better settings?
    # Given 120s limit, and L-BFGS is fast, we can afford a DE run.
    
    # Let's try DE with the best candidate as part of the initial population?
    # scipy DE doesn't easily take a single x0.
    # But we can run DE with a smaller population and fewer iterations to polish?
    # No, DE is global.
    
    # Let's just run DE with standard good settings.
    try:
        # Use latin hypercube init
        res_de = differential_evolution(
            energy, 
            bounds, 
            seed=42, 
            maxiter=100, 
            popsize=15, 
            tol=1e-8,
            init='latinhypercube'
        )
        if res_de.fun < best_energy:
            best_energy = res_de.fun
            best_params = res_de.x
            # Polish DE result
            res_polish = minimize(energy, best_params, method='L-BFGS-B', bounds=bounds, options={'maxiter': 2000, 'ftol': 1e-12})
            if res_polish.fun < best_energy:
                best_energy = res_polish.fun
                best_params = res_polish.x
    except Exception:
        pass

    final_points = get_points(best_params)
    norms = np.linalg.norm(final_points, axis=1, keepdims=True)
    final_points = final_points / norms
    
    # Final check
    diff = final_points[:, np.newaxis, :] - final_points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    i_idx, j_idx = np.triu_indices(N, k=1)
    pair_dists = dists[i_idx, j_idx]
    
    if np.any(pair_dists < 1e-9) or not np.all(np.isfinite(final_points)):
        # Fallback to Fibonacci
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
    return final_points
