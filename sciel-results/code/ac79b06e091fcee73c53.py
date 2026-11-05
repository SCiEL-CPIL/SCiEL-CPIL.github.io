import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s margin
    
    n = 8
    # Number of pairs
    m = n * (n - 1) // 2
    
    def calculate_score(points):
        # points: (n, 2) array
        # Calculate all pairwise distances
        # Using broadcasting
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        # Zero out diagonal
        np.fill_diagonal(dists, 0.0)
        # Get upper triangle
        iu = np.triu_indices(n, 1)
        d = dists[iu]
        
        if np.any(d < 1e-12):
            return -1e100
            
        # Log product of distances
        log_prod = np.sum(np.log(d))
        
        # Diameter (max distance)
        # Use soft-max for smoothness in optimization, but for final score we need exact max?
        # The task says "after scaling ... to diameter 2". 
        # If we use soft-max in optimization, we need to evaluate the true score at the end.
        # For the optimization objective, we use soft-max.
        # For the final return, we calculate the exact score.
        
        return log_prod, d

    def objective(x, tau=50.0):
        points = x.reshape(n, 2)
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, 0.0)
        iu = np.triu_indices(n, 1)
        d = dists[iu]
        
        if np.any(d < 1e-12):
            return 1e100
            
        log_prod = np.sum(np.log(d))
        
        # Soft-max approximation for log(max(d))
        # log(max(d)) approx (1/tau) * log(sum(d^tau))
        # To avoid overflow, use log-sum-exp trick:
        # log(sum(d^tau)) = tau * max(d) + log(sum(exp(tau*(d - max(d)))))
        d_max = np.max(d)
        # If d_max is 0, handled above
        # tau * d might be large? d is usually O(1). tau=50 is fine.
        # Let's use a stable logsumexp
        # logsumexp(tau * d) = tau * max(d) + log(sum(exp(tau * (d - max(d)))))
        # We want (1/tau) * logsumexp(tau * d) = max(d) + (1/tau) * log(sum(exp(tau * (d - max(d)))))
        # This is a smooth approximation of max(d).
        
        # Actually, we want to minimize -log_prod + 28 * log_D
        # log_D approx (1/tau) * log(sum(d^tau))
        
        # Let's compute log_D_smooth
        # d > 0
        # log_D_smooth = (1/tau) * np.log(np.sum(d**tau))
        # d**tau can overflow if d > 1 and tau is large.
        # Scale d by max(d)
        d_scaled = d / d_max
        # sum(d^tau) = d_max^tau * sum(d_scaled^tau)
        # log(sum(d^tau)) = tau * log(d_max) + log(sum(d_scaled^tau))
        # (1/tau) * log(sum(d^tau)) = log(d_max) + (1/tau) * log(sum(d_scaled^tau))
        
        log_sum_scaled = np.log(np.sum(d_scaled**tau))
        log_D_smooth = np.log(d_max) + (1.0 / tau) * log_sum_scaled
        
        val = -log_prod + 28.0 * log_D_smooth
        return val

    def true_score(points):
        diff = points[:, np.newaxis, :] - points[np.newaxis, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, 0.0)
        iu = np.triu_indices(n, 1)
        d = dists[iu]
        
        if np.any(d < 1e-12):
            return 0.0
            
        D = np.max(d)
        if D < 1e-12:
            return 0.0
            
        # Scale to diameter 2
        # Product of scaled distances
        # prod(d_scaled) = prod(d) / D^28
        log_prod_scaled = np.sum(np.log(d)) - 28.0 * np.log(D)
        # Score = Delta / n^n
        # Delta = exp(log_prod_scaled)
        # Score = exp(log_prod_scaled) / (8**8)
        # We want to return the score value.
        # The prompt says "score is Delta/n^n".
        # Regular octagon scores 1.
        
        score = np.exp(log_prod_scaled) / (8.0 ** 8)
        return score

    best_score = 0.0
    best_points = None
    
    # Initial guesses
    inits = []
    
    # 1. Regular Octagon
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    octagon = np.column_stack([np.cos(angles), np.sin(angles)])
    inits.append(octagon.flatten())
    
    # 2. Perturbed Octagon
    for _ in range(5):
        noise = np.random.normal(0, 0.1, (n, 2))
        inits.append((octagon + noise).flatten())
        
    # 3. Random points on circle
    for _ in range(10):
        r = np.ones(n)
        theta = np.sort(np.random.uniform(0, 2*np.pi, n))
        pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        inits.append(pts.flatten())
        
    # 4. "Squashed" or asymmetric distributions
    # Try points clustered in two groups?
    # Or points on two concentric circles?
    for _ in range(5):
        # 4 points inner, 4 outer
        r_inner = 0.5
        r_outer = 1.0
        theta_in = np.linspace(0, 2*np.pi, 4, endpoint=False) + np.pi/4
        theta_out = np.linspace(0, 2*np.pi, 4, endpoint=False)
        pts_in = np.column_stack([r_inner * np.cos(theta_in), r_inner * np.sin(theta_in)])
        pts_out = np.column_stack([r_outer * np.cos(theta_out), r_outer * np.sin(theta_out)])
        pts = np.vstack([pts_in, pts_out])
        noise = np.random.normal(0, 0.1, (n, 2))
        inits.append((pts + noise).flatten())

    # Optimization loop
    for i, x0 in enumerate(inits):
        if time.time() > deadline:
            break
            
        try:
            # Use BFGS or L-BFGS-B
            # We don't have bounds, but we can add a small penalty for large coords?
            # The scaling is invariant, so magnitude doesn't matter for the score,
            # but numerically it's better to keep coords O(1).
            # We can re-center and scale after each step? No, minimize handles the landscape.
            # The function is scale invariant (score is scale invariant).
            # This creates a flat direction. To fix this, we can constrain the centroid to 0
            # and the radius of gyration to be fixed?
            # Or just let it run. BFGS might drift.
            # Let's add a small penalty to keep points near origin.
            
            def obj_with_penalty(x, tau=50.0):
                val = objective(x, tau)
                points = x.reshape(n, 2)
                # Penalty for large coordinates to prevent drift
                pen = 0.01 * np.sum(points**2)
                return val + pen

            res = minimize(obj_with_penalty, x0, method='L-BFGS-B', 
                           options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if res.success or res.fun < 1e10:
                pts_opt = res.x.reshape(n, 2)
                # Evaluate true score
                s = true_score(pts_opt)
                if s > best_score:
                    best_score = s
                    best_points = pts_opt
                    
        except Exception:
            pass

    # If we didn't find anything better, return octagon
    if best_points is None:
        return octagon.tolist()
        
    return best_points.tolist()
