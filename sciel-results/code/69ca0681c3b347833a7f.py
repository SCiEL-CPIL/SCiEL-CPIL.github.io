import numpy as np
import time
from scipy.optimize import differential_evolution, minimize, basinhopping

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s for cleanup

    n = 8
    pair_indices = []
    for i in range(n):
        for j in range(i + 1, n):
            pair_indices.append((i, j))
    pair_indices = np.array(pair_indices)
    idx_i = pair_indices[:, 0]
    idx_j = pair_indices[:, 1]
    
    def objective(x):
        # Check time
        if time.time() > deadline:
            return 1e10
            
        points = x.reshape(n, 2)
        
        # Calculate pairwise distances
        diff = points[idx_i, :] - points[idx_j, :]
        dists = np.sqrt(np.sum(diff**2, axis=1))
        
        # Check for zero distances (distinct points)
        if np.any(dists < 1e-10):
            return 1e10
            
        D = np.max(dists)
        if D < 1e-10:
            return 1e10
            
        # Sum of logs of scaled distances
        # log(d_scaled) = log(d) + log(2) - log(D)
        # Product of ordered pairs = (Product of unordered pairs)^2
        # Score = (Product of scaled unordered pairs)^2 / 8^8
        # log(Score) = 2 * sum(log(d_scaled)) - 8*log(8)
        # We want to minimize -log(Score)
        
        sum_log_d = np.sum(np.log(dists))
        # sum(log(d_scaled)) = sum(log(d)) + 28*log(2) - 28*log(D)
        sum_log_scaled = sum_log_d + 28.0 * (np.log(2.0) - np.log(D))
        
        log_score = 2.0 * sum_log_scaled - 8.0 * np.log(8.0)
        
        return -log_score

    # Bounds
    bounds = [(-10.0, 10.0)] * 16
    
    # Initial guesses
    # 1. Regular Octagon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x0_oct = np.column_stack([np.cos(angles), np.sin(angles)]).flatten()
    
    # 2. Two Squares (Concentric, Rotated)
    # Vary radii r1, r2 and rotation theta
    best_x = x0_oct
    best_val = objective(x0_oct)
    
    starts = [x0_oct]
    
    # Generate diverse starts
    np.random.seed(42)
    
    # Random perturbations of octagon
    for _ in range(5):
        noise = np.random.normal(0, 0.2, 16)
        starts.append(x0_oct + noise)
        
    # Two squares configurations
    # 4 points at (r1*cos, r1*sin), 4 points at (r2*cos(theta+pi/4), r2*sin(theta+pi/4))
    for r1 in [0.5, 0.8, 1.0, 1.2, 1.5]:
        for r2 in [0.5, 0.8, 1.0, 1.2, 1.5]:
            for theta in [np.pi/4, np.pi/8, 3*np.pi/8, np.pi/3]:
                p1 = []
                p2 = []
                for k in range(4):
                    a1 = k * np.pi / 2
                    p1.append([r1 * np.cos(a1), r1 * np.sin(a1)])
                    a2 = theta + k * np.pi / 2
                    p2.append([r2 * np.cos(a2), r2 * np.sin(a2)])
                pts = np.array(p1 + p2)
                # Shuffle to ensure all pairs are considered? No, objective is symmetric.
                # But we need to flatten.
                starts.append(pts.flatten())
                
    # Random starts
    for _ in range(10):
        starts.append(np.random.uniform(-5, 5, 16))

    # Local Refinement
    for s in starts:
        if time.time() > deadline - 10.0:
            break
        try:
            res2 = minimize(
                objective, 
                s, 
                method='Powell', 
                options={'maxiter': 200, 'xatol': 1e-12, 'fatol': 1e-12}
            )
            if res2.fun < best_val:
                best_val = res2.fun
                best_x = res2.x
        except Exception:
            pass

    # Global Search with DE
    # If time remains, run DE
    if time.time() < deadline - 20.0:
        try:
            # Use a smaller population to fit in time, or rely on the local refinements
            # The current best might already be good. DE is expensive.
            # Let's try a short DE run
            res = differential_evolution(
                objective, 
                bounds, 
                maxiter=50, 
                popsize=15, 
                tol=1e-8, 
                seed=42, 
                init='sobol'
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass
            
    # Final Polish
    if time.time() < deadline - 2.0:
        try:
            res3 = minimize(
                objective, 
                best_x, 
                method='Powell', 
                options={'maxiter': 500, 'xatol': 1e-15, 'fatol': 1e-15}
            )
            if res3.fun < best_val:
                best_val = res3.fun
                best_x = res3.x
        except Exception:
            pass

    best_points = best_x.reshape(n, 2)
    return [[float(p[0]), float(p[1])] for p in best_points]
