import numpy as np
import itertools
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time

def min_max_dist_dim3_14() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 570  # 9.5 minutes, safety margin
    
    # 1. Generate Dodecahedron vertices
    phi = (1 + np.sqrt(5)) / 2.0
    verts = []
    for x in [-1, 1]:
        for y in [-1, 1]:
            for z in [-1, 1]:
                verts.append([x, y, z])
    for a in [-1/phi, 1/phi]:
        for b in [-phi, phi]:
            verts.append([0, a, b])
            verts.append([a, b, 0])
            verts.append([b, 0, a])
    verts = np.array(verts)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    
    # 2. Find best subset of 14
    best_score = 0
    best_subset_idx = None
    n_verts = 20
    
    for combo in itertools.combinations(range(n_verts), 14):
        pts = verts[list(combo)]
        dists = pdist(pts)
        d_min = np.min(dists)
        d_max = np.max(dists)
        score = (d_min / d_max) ** 2
        if score > best_score:
            best_score = score
            best_subset_idx = combo
            
    initial_points = verts[list(best_subset_idx)].copy()
    
    def compute_score(pts):
        dists = pdist(pts)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2
    
    def objective_soft(x, beta):
        """Soft objective using log-sum-exp for stability"""
        pts = x.reshape(14, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        pts = pts / norms
        dists = pdist(pts)
        
        # Stabilized log-sum-exp
        max_d = np.max(dists)
        min_d = np.min(dists)
        
        # Soft max: log(sum(exp(dists/beta))) * beta
        # = (beta * max_d + log(sum(exp((dists - max_d)/beta))))
        soft_max = beta * max_d + np.log(np.sum(np.exp((dists - max_d) / beta)))
        
        # Soft min: -log(sum(exp(-dists/beta))) * beta
        # = -(beta * (-min_d) + log(sum(exp((-dists - (-min_d))/beta))))
        # = beta * min_d - log(sum(exp((min_d - dists)/beta)))
        soft_min = beta * min_d - np.log(np.sum(np.exp((min_d - dists) / beta)))
        
        if soft_min < 1e-10:
            return 1e10
        return (soft_max - soft_min) / soft_min

    best_final_points = initial_points.copy()
    best_final_score = best_score
    x0 = initial_points.flatten()
    
    # Define annealing schedule for beta
    # Start large (smooth), end small (sharp)
    betas = [100.0, 50.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1]
    
    def run_annealing_schedule(start_x, schedule_betas):
        """Run SLSQP optimization with decreasing beta"""
        current_x = start_x
        for beta in schedule_betas:
            if time.time() > deadline - 10:
                break
            
            try:
                res = minimize(
                    lambda x, b=beta: objective_soft(x, b),
                    current_x,
                    method='SLSQP',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                if res.fun < 1e6:
                    opt_pts = res.x.reshape(14, 3)
                    norms = np.linalg.norm(opt_pts, axis=1, keepdims=True)
                    norms[norms == 0] = 1.0
                    opt_pts = opt_pts / norms
                    current_x = opt_pts.flatten()
                    
                    score = compute_score(opt_pts)
                    if score > best_final_score:
                        best_final_score = score
                        best_final_points = opt_pts.copy()
                        
            except Exception:
                pass
        return current_x

    # Phase 1: Main annealing from dodecahedron subset
    x0 = run_annealing_schedule(x0, betas)
    
    # Phase 2: Random restarts with annealing
    # We have used some time, let's check remaining budget
    remaining_time = deadline - time.time()
    
    # Estimate cost per restart. SLSQP is slower than L-BFGS-B.
    # Assume ~5-10 seconds per full annealing run.
    # Let's try to fit as many as possible, prioritizing perturbed best solutions.
    
    np.random.seed(42)
    
    # First, do some perturbed restarts from the best found
    for i in range(10):
        if time.time() > deadline - 20:
            break
        
        # Perturb best solution
        noise = np.random.randn(42) * 0.1
        x0_perturbed = best_final_points.flatten() + noise
        
        # Use a shorter schedule for restarts to save time
        restart_betas = [20.0, 5.0, 1.0, 0.5, 0.2]
        run_annealing_schedule(x0_perturbed, restart_betas)
        
    # Then, pure random restarts if time permits
    for i in range(20):
        if time.time() > deadline - 10:
            break
        
        rand_pts = np.random.randn(14, 3)
        rand_pts /= np.linalg.norm(rand_pts, axis=1, keepdims=True)
        x0_rand = rand_pts.flatten()
        
        # Even shorter schedule for random starts
        random_betas = [50.0, 10.0, 2.0, 0.5]
        run_annealing_schedule(x0_rand, random_betas)

    # Final polish with very small beta on the best point found
    if time.time() < deadline - 5:
        final_betas = [0.1, 0.05, 0.02, 0.01]
        run_annealing_schedule(best_final_points.flatten(), final_betas)

    return best_final_points
