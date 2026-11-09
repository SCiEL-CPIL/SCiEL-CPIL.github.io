import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    n = 8
    
    def compute_score(points):
        # points is shape (n, 2)
        # 1. Calculate pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        
        # Avoid log(0)
        dists = np.maximum(dists, 1e-10)
        
        # 2. Find diameter
        mask = ~np.eye(n, dtype=bool)
        max_dist = np.max(dists[mask])
        
        if max_dist == 0:
            return -np.inf
            
        # 3. Compute log of product of ordered pair distances
        # Product is over i != j
        # log(product) = sum(log(dists[i, j])) for i != j
        log_prod_ordered = np.sum(np.log(dists[mask]))
        
        # 4. Objective to maximize:
        # K = log_prod_ordered - 56 * log(max_dist)
        # We minimize -K
        return log_prod_ordered - 56 * np.log(max_dist)

    def objective(x):
        points = x.reshape(n, 2)
        s = compute_score(points)
        if not np.isfinite(s):
            return 1e10
        return -s

    # Initial candidates
    candidates = []
    
    # 1. Regular Octagon
    angles = 2 * np.pi * np.arange(n) / n
    p_oct = np.column_stack([np.cos(angles), np.sin(angles)])
    candidates.append(p_oct.flatten())
    
    # 2. Symmetric Configuration (D2 symmetry)
    # Parameters: x1, y1, x2, y2
    # Points: (±x1, ±y1) and (±x2, ±y2)
    def sym_points(params):
        x1, y1, x2, y2 = params
        # Ensure positive
        x1, y1, x2, y2 = abs(x1), abs(y1), abs(x2), abs(y2)
        p1 = np.array([x1, y1])
        p2 = np.array([-x1, y1])
        p3 = np.array([-x1, -y1])
        p4 = np.array([x1, -y1])
        p5 = np.array([x2, y2])
        p6 = np.array([-x2, y2])
        p7 = np.array([-x2, -y2])
        p8 = np.array([x2, -y2])
        return np.vstack([p1, p2, p3, p4, p5, p6, p7, p8])

    # Try some symmetric initial guesses
    # Close to octagon: x1 ~ cos(22.5), y1 ~ sin(22.5) etc?
    # Octagon points: (1,0), (0.92, 0.38), (0.7, 0.7), (0.38, 0.92)...
    # D2 symmetry doesn't fit octagon perfectly (octagon has D4).
    # But we can start near it.
    rng = np.random.default_rng(42)
    
    # Generate random symmetric candidates
    for _ in range(50):
        x1 = rng.uniform(0.5, 1.5)
        y1 = rng.uniform(0.5, 1.5)
        x2 = rng.uniform(0.5, 1.5)
        y2 = rng.uniform(0.5, 1.5)
        p_sym = sym_points([x1, y1, x2, y2])
        candidates.append(p_sym.flatten())
        
    # 3. Random perturbations of octagon
    for _ in range(50):
        noise = rng.normal(0, 0.1, size=(n, 2))
        p_rand = p_oct + noise
        candidates.append(p_rand.flatten())
        
    # 4. Random points
    for _ in range(50):
        p_rand = rng.uniform(-1, 1, size=(n, 2))
        candidates.append(p_rand.flatten())

    best_score = -np.inf
    best_points = None

    # Shuffle candidates
    idx = np.arange(len(candidates))
    rng.shuffle(idx)
    
    for i in idx:
        if time.time() > deadline:
            break
            
        x0 = candidates[i]
        try:
            # Use Nelder-Mead for robustness
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10})
            
            final_points = res.x.reshape(n, 2)
            final_score = compute_score(final_points)
            
            if final_score > best_score:
                best_score = final_score
                best_points = final_points
                
        except Exception:
            pass

    if best_points is None:
        return [[float(x), float(y)] for x, y in p_oct]
        
    return [[float(x), float(y)] for x, y in best_points]
