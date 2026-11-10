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
    
    def objective_soft(x, beta=10.0):
        """Soft objective: -soft_min(dists) + soft_max(dists) scaled"""
        pts = x.reshape(14, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        pts = pts / norms
        dists = pdist(pts)
        # Soft min: -log-sum-exp(-dists/beta)
        min_d = -np.log(np.sum(np.exp(-dists / beta))) * beta
        # Soft max: log-sum-exp(dists/beta)
        max_d = np.log(np.sum(np.exp(dists / beta))) * beta
        # We want to maximize min_d / max_d
        # Minimize -log(min_d) + log(max_d) = log(max_d/min_d)
        # Or simpler: minimize (max_d - min_d) / min_d
        if min_d < 1e-10:
            return 1e10
        return (max_d - min_d) / min_d
    
    def objective_ratio(x):
        pts = x.reshape(14, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        pts = pts / norms
        dists = pdist(pts)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return -((d_min / d_max) ** 2)
    
    best_final_points = initial_points.copy()
    best_final_score = best_score
    
    # 3. Optimization using L-BFGS-B with soft objective
    # Start with higher beta for smoother landscape, then anneal
    
    x0 = initial_points.flatten()
    
    # Phase 1: Coarse optimization with large beta
    for beta in [50.0, 20.0, 10.0, 5.0]:
        if time.time() > deadline - 120:
            break
            
        def obj(x, b=beta):
            return objective_soft(x, b)
        
        try:
            res = minimize(obj, x0, method='L-BFGS-B', 
                          options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if res.fun < 1e6:
                opt_pts = res.x.reshape(14, 3)
                norms = np.linalg.norm(opt_pts, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                opt_pts = opt_pts / norms
                
                score = compute_score(opt_pts)
                if score > best_final_score:
                    best_final_score = score
                    best_final_points = opt_pts.copy()
                    x0 = opt_pts.flatten()
        except Exception:
            pass
    
    # Phase 2: Fine optimization with direct ratio objective
    if time.time() > deadline - 60:
        return best_final_points
    
    # Use Nelder-Mead for fine polishing
    for _ in range(3):
        if time.time() > deadline - 60:
            break
        try:
            res = minimize(objective_ratio, x0, method='Nelder-Mead', 
                          options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
            
            opt_pts = res.x.reshape(14, 3)
            norms = np.linalg.norm(opt_pts, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            opt_pts = opt_pts / norms
            
            score = compute_score(opt_pts)
            if score > best_final_score:
                best_final_score = score
                best_final_points = opt_pts.copy()
                x0 = opt_pts.flatten()
        except Exception:
            pass
    
    # 4. Random restarts with L-BFGS-B
    if time.time() > deadline - 30:
        return best_final_points
    
    np.random.seed(42)
    for i in range(20):
        if time.time() > deadline - 30:
            break
        
        # Generate random points on sphere
        rand_pts = np.random.randn(14, 3)
        rand_pts /= np.linalg.norm(rand_pts, axis=1, keepdims=True)
        x0_rand = rand_pts.flatten()
        
        # Quick L-BFGS-B with moderate beta
        try:
            res = minimize(lambda x: objective_soft(x, 10.0), x0_rand, method='L-BFGS-B', 
                          options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6})
            
            if res.fun < 1e6:
                opt_pts = res.x.reshape(14, 3)
                norms = np.linalg.norm(opt_pts, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                opt_pts = opt_pts / norms
                
                score = compute_score(opt_pts)
                if score > best_final_score:
                    best_final_score = score
                    best_final_points = opt_pts.copy()
        except Exception:
            pass
    
    return best_final_points
