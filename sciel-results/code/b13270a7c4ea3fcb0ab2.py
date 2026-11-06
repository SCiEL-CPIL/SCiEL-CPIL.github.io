import numpy as np
from scipy.optimize import minimize
import math

def construct():
    n = 8
    
    # Objective function: Minimize negative log of product of distances
    # Add penalty if diameter > 2
    def objective(params):
        pts = params.reshape(n, 2)
        # Calculate pairwise distances
        diff = pts[:, np.newaxis] - pts[np.newaxis, :]
        dists = np.linalg.norm(diff, axis=2)
        
        # Avoid log(0)
        dists = np.maximum(dists, 1e-10)
        
        # Mask diagonal
        mask = ~np.eye(n, dtype=bool)
        
        # Log product of all ordered pairs (sum of logs)
        log_prod = np.sum(np.log(dists[mask]))
        
        # Penalty for diameter > 2
        max_dist = np.max(dists)
        penalty = 0.0
        if max_dist > 2.0:
            penalty = 100.0 * (max_dist - 2.0) ** 2
        
        return -log_prod + penalty

    # Initial guesses
    starts = []
    
    # 1. Regular Octagon
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    starts.append(np.array([np.cos(angles), np.sin(angles)]).T.flatten())
    
    # 2. Square + Square (Concentric, rotated 45 deg)
    # Outer square: (1,0), (-1,0), (0,1), (0,-1)
    # Inner square: (r,r), (-r,r), (-r,-r), (r,-r)
    r = 0.6
    outer = np.array([[1,0], [-1,0], [0,1], [0,-1]])
    inner = np.array([[r,r], [-r,r], [-r,-r], [r,-r]])
    pts_sq = np.vstack([outer, inner])
    starts.append(pts_sq.flatten())
    
    # 3. Regular Octagon with noise
    noise = np.random.normal(0, 0.1, (n, 2))
    pts_noise = np.array([np.cos(angles), np.sin(angles)]).T + noise
    starts.append(pts_noise.flatten())
    
    best_score = -np.inf
    best_pts = None
    
    for x0 in starts:
        # Use COBYLA for inequality constraints (implicitly via penalty here)
        # Or SLSQP with explicit constraints?
        # Penalty method is easier to code robustly for this time limit.
        
        res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
        
        if res.success:
            pts = res.x.reshape(n, 2)
            
            # Scale to diameter 2
            diff = pts[:, np.newaxis] - pts[np.newaxis, :]
            dists = np.linalg.norm(diff, axis=2)
            max_d = np.max(dists)
            
            if max_d > 1e-6:
                pts = pts * (2.0 / max_d)
                
                # Calculate true score
                diff = pts[:, np.newaxis] - pts[np.newaxis, :]
                dists = np.linalg.norm(diff, axis=2)
                mask = ~np.eye(n, dtype=bool)
                # Product of ordered pairs
                # Delta = prod(d_ij for i!=j)
                # Score = Delta / n^n
                # log(Score) = sum(log(d_ij)) - n*log(n)
                log_delta = np.sum(np.log(dists[mask]))
                log_score = log_delta - n * math.log(n)
                score = math.exp(log_score)
                
                if score > best_score:
                    best_score = score
                    best_pts = pts
    
    # If optimization failed or returned bad results, fallback to regular octagon
    if best_pts is None:
        return [[math.cos(2 * math.pi * k / 8), math.sin(2 * math.pi * k / 8)] for k in range(8)]
        
    return best_pts.tolist()
