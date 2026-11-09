import numpy as np
from scipy.optimize import minimize
import time
import math

def construct():
    start_time = time.time()
    time_limit = 140.0  # Leave 10s margin
    
    n = 8
    
    # Function to compute the negative log product of distances
    def neg_log_prod(x):
        pts = x.reshape(n, 2)
        # Compute all pairwise distances
        # Using broadcasting
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        # Mask diagonal
        mask = ~np.eye(n, dtype=bool)
        dists = dists[mask]
        # Avoid log(0)
        dists = np.maximum(dists, 1e-10)
        return -np.sum(np.log(dists))
    
    # Constraints: diameter <= 2
    # We can use squared distances to avoid sqrt in constraints
    def diam_constraint(x):
        pts = x.reshape(n, 2)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        mask = ~np.eye(n, dtype=bool)
        dists_sq = dists_sq[mask]
        # We want dists_sq <= 4
        # SLSQP constraints are of the form g(x) >= 0
        return 4.0 - dists_sq

    # Initial guesses
    # 1. Regular octagon
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    x0_regular = np.concatenate([np.cos(angles), np.sin(angles)]) * 2.0  # Diameter 2
    
    guesses = [x0_regular]
    
    # Add some noisy versions
    rng = np.random.default_rng(42)
    for i in range(20):
        noise = rng.normal(0, 0.1, size=2*n)
        x0 = x0_regular + noise
        guesses.append(x0)
        
    # Add a "two squares" guess
    # Square 1: radius 1.4, Square 2: radius 0.6, rotated 45 deg
    r1, r2 = 1.4, 0.6
    ang1 = np.linspace(0, 2*np.pi, 4, endpoint=False)
    ang2 = np.linspace(np.pi/4, 2*np.pi + np.pi/4, 4, endpoint=False)
    sq1 = np.concatenate([np.cos(ang1)*r1, np.sin(ang1)*r1])
    sq2 = np.concatenate([np.cos(ang2)*r2, np.sin(ang2)*r2])
    guesses.append(np.concatenate([sq1, sq2]))
    
    best_score = -np.inf
    best_x = None
    
    for x0 in guesses:
        if time.time() - start_time > time_limit:
            break
            
        # Define constraints for SLSQP
        constraints = []
        for i in range(n):
            for j in range(i+1, n):
                # This creates a lot of constraints, might be slow.
                # Alternative: use a single constraint for the max distance?
                # SLSQP can handle a vector of constraints.
                pass
        
        # Let's use a penalty method with L-BFGS-B instead, it's often faster and more robust for this.
        # Penalty: -lambda * sum(max(0, d^2 - 4)^2)
        
        def objective_with_penalty(x, lam):
            pts = x.reshape(n, 2)
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists_sq = np.sum(diff**2, axis=2)
            mask = ~np.eye(n, dtype=bool)
            dists_sq = dists_sq[mask]
            dists = np.sqrt(dists_sq)
            dists = np.maximum(dists, 1e-10)
            log_prod = np.sum(np.log(dists))
            penalty = lam * np.sum(np.maximum(0, dists_sq - 4.0)**2)
            return -log_prod + penalty

        # Start with a small lambda and increase
        lam = 1.0
        x_cur = x0.copy()
        
        # Optimize
        for _ in range(5):
            res = minimize(objective_with_penalty, x_cur, args=(lam,), method='L-BFGS-B', 
                           options={'maxiter': 100, 'ftol': 1e-12})
            x_cur = res.x
            lam *= 10.0
            
        # Check if the result is valid (diameter <= 2 + eps)
        pts = x_cur.reshape(n, 2)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        max_d = np.max(dists)
        
        if max_d > 2.0 + 1e-5:
            # Scale down
            scale = 2.0 / max_d
            x_cur = x_cur * scale
            pts = x_cur.reshape(n, 2)
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            max_d = np.max(dists)
            
        # Calculate score
        # Score = Delta / 8^8 where Delta is product of distances
        # We have -log_prod from the optimizer, but let's recalculate for accuracy
        mask = ~np.eye(n, dtype=bool)
        dists = dists[mask]
        log_prod = np.sum(np.log(dists))
        # The score is exp(log_prod) / 8^8
        # But we want to maximize log_prod
        # Let's compare log_prod
        if log_prod > best_score:
            best_score = log_prod
            best_x = x_cur.copy()
            
    # Convert best_x to list of lists
    best_pts = best_x.reshape(n, 2)
    return best_pts.tolist()
