import numpy as np
import itertools
from scipy.spatial.distance import pdist
import time

def min_max_dist_dim3_14() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 600  # 10 minutes
    
    # 1. Generate Dodecahedron vertices
    phi = (1 + np.sqrt(5)) / 2.0
    # Vertices
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
    # Normalize to unit sphere
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    
    # 2. Find best subset of 14
    best_score = 0
    best_subset_idx = None
    n_verts = 20
    
    # Exhaustive search over combinations of 14
    # 38760 combinations. Fast.
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
    
    # 3. Local Optimization
    # Objective: Maximize d_min on the sphere
    # We minimize -d_min
    
    def objective(x):
        # x is shape (14, 3)
        pts = x.reshape(14, 3)
        # Project to sphere
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms == 0] = 1.0
        pts = pts / norms
        dists = pdist(pts)
        d_min = np.min(dists)
        return -d_min
        
    def objective_ratio(x):
        pts = x.reshape(14, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        pts = pts / norms
        dists = pdist(pts)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0: return 0
        return -((d_min / d_max) ** 2)

    # Use Nelder-Mead
    # Flatten initial points
    x0 = initial_points.flatten()
    
    # Run optimization
    # To save time, limit iterations
    # Nelder-Mead can be slow for 42 dims.
    # Let's try a few restarts or just one good run.
    
    best_final_points = initial_points
    best_final_score = best_score
    
    # Try optimizing from the initial guess
    try:
        from scipy.optimize import minimize
        # Use the ratio objective directly for better alignment with task
        res = minimize(objective_ratio, x0, method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
        
        if res.success or res.fun < objective_ratio(x0):
            opt_pts = res.x.reshape(14, 3)
            # Project to sphere
            norms = np.linalg.norm(opt_pts, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            opt_pts = opt_pts / norms
            
            dists = pdist(opt_pts)
            d_min = np.min(dists)
            d_max = np.max(dists)
            score = (d_min / d_max) ** 2
            if score > best_final_score:
                best_final_score = score
                best_final_points = opt_pts
                
    except Exception:
        pass
        
    # Check time
    if time.time() > deadline - 60:
        return best_final_points
        
    # If we have time, try a few random restarts
    np.random.seed(42)
    for i in range(10):
        if time.time() > deadline - 60:
            break
        # Random initial points on sphere
        rand_pts = np.random.randn(14, 3)
        rand_pts /= np.linalg.norm(rand_pts, axis=1, keepdims=True)
        x0_rand = rand_pts.flatten()
        
        try:
            res = minimize(objective_ratio, x0_rand, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-5})
            if res.success or res.fun < objective_ratio(x0_rand):
                opt_pts = res.x.reshape(14, 3)
                norms = np.linalg.norm(opt_pts, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                opt_pts = opt_pts / norms
                
                dists = pdist(opt_pts)
                d_min = np.min(dists)
                d_max = np.max(dists)
                score = (d_min / d_max) ** 2
                if score > best_final_score:
                    best_final_score = score
                    best_final_points = opt_pts
        except Exception:
            pass
            
    return best_final_points
