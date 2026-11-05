import numpy as np
import time
from scipy.optimize import minimize

def compute_log_score(points):
    """
    Compute the log of the score Delta/n^n for n=8.
    Delta is the product of ordered pair distances after scaling to diameter 2.
    Score = Delta / 8^8.
    We maximize log(Score) = log(Delta) - 8*log(8).
    log(Delta) = sum_{i != j} log(d_{ij} / diameter)
    = sum_{i != j} log(d_{ij}) - 8*7*log(diameter)
    Note: There are 8*7 = 56 ordered pairs.
    """
    points = np.array(points)
    n = len(points)
    
    # Compute pairwise distances
    # diff[i, j] = points[i] - points[j]
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Diameter is the max distance
    diameter = np.max(dists)
    if diameter <= 0:
        return -np.inf
        
    # Mask diagonal to avoid log(0)
    mask = ~np.eye(n, dtype=bool)
    valid_dists = dists[mask]
    
    if np.any(valid_dists <= 0):
        return -np.inf
        
    # log(Delta) = sum(log(valid_dists)) - n*(n-1)*log(diameter)
    # Wait, the definition says: "Delta is the product over ordered pairs of distances after scaling the set to diameter 2"
    # Scaled distance d' = d / (diameter/2) = 2d/diameter
    # So Delta = prod_{i!=j} (2 * d_{ij} / diameter)
    # log(Delta) = sum_{i!=j} [log(2) + log(d_{ij}) - log(diameter)]
    # log(Delta) = n*(n-1)*log(2) + sum_{i!=j} log(d_{ij}) - n*(n-1)*log(diameter)
    
    n_pairs = n * (n - 1)
    log_delta = n_pairs * np.log(2) + np.sum(np.log(valid_dists)) - n_pairs * np.log(diameter)
    
    # Score = Delta / n^n
    # log(Score) = log(Delta) - n*log(n)
    log_score = log_delta - n * np.log(n)
    
    return log_score

def negative_log_score(points_flat):
    """
    Objective function for minimization.
    Minimizes -log_score.
    """
    points = points_flat.reshape(8, 2)
    log_score = compute_log_score(points)
    if not np.isfinite(log_score):
        return 1e10
    return -log_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    best_score = 1.0  # Regular octagon score
    # Regular octagon points
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    rng = np.random.default_rng(42)
    
    # List of initializations to try
    initializations = []
    
    # 1. Regular octagon
    initializations.append(best_points.copy())
    
    # 2. Two concentric squares
    # Square 1: radius r1, angles 0, pi/2, pi, 3pi/2
    # Square 2: radius r2, angles pi/4, 3pi/4, 5pi/4, 7pi/4
    for r1 in [0.5, 0.7, 1.0, 1.2, 1.5]:
        for r2 in [0.5, 0.7, 1.0, 1.2, 1.5]:
            if abs(r1 - r2) < 1e-6: continue
            p1 = np.column_stack([r1 * np.cos(np.array([0, np.pi/2, np.pi, 3*np.pi/2])), 
                                  r1 * np.sin(np.array([0, np.pi/2, np.pi, 3*np.pi/2]))])
            p2 = np.column_stack([r2 * np.cos(np.array([np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4])), 
                                  r2 * np.sin(np.array([np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4]))])
            initializations.append(np.vstack([p1, p2]))
    
    # 3. Perturbed regular octagons
    for _ in range(20):
        angles_pert = angles + rng.normal(0, 0.2, 8)
        radii_pert = np.ones(8) + rng.normal(0, 0.2, 8)
        radii_pert = np.abs(radii_pert) # Ensure positive
        p = np.column_stack([radii_pert * np.cos(angles_pert), radii_pert * np.sin(angles_pert)])
        initializations.append(p)
        
    # 4. Random points
    for _ in range(20):
        radii = rng.uniform(0.1, 1.5, 8)
        angles_r = rng.uniform(0, 2 * np.pi, 8)
        p = np.column_stack([radii * np.cos(angles_r), radii * np.sin(angles_r)])
        initializations.append(p)
        
    # 5. Specific known good configurations if any? 
    # The paper "Eight planar points maximizing the product of distances" suggests non-regular solutions.
    # A common candidate is a "distorted" octagon or two squares.
    
    # Sort initializations by their initial score to prioritize good starts? 
    # Or just iterate through them.
    
    for init_p in initializations:
        if time.time() > deadline:
            break
            
        x0 = init_p.flatten()
        
        try:
            # Use Nelder-Mead for robustness, it doesn't need gradients
            # maxiter can be high since it's fast for 16 vars
            res = minimize(negative_log_score, x0, method='Nelder-Mead', 
                           options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-8})
            
            if res.success or res.fun < 1e10:
                optimal_points = res.x.reshape(8, 2)
                log_score_val = -res.fun
                if np.isfinite(log_score_val):
                    score = np.exp(log_score_val)
                    if score > best_score:
                        best_score = score
                        best_points = optimal_points
        except Exception:
            pass

    # Refine the best point found with a tighter tolerance
    if time.time() < deadline - 10:
        x_best = best_points.flatten()
        try:
            res_refined = minimize(negative_log_score, x_best, method='Nelder-Mead', 
                           options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-10})
            if res_refined.success or res_refined.fun < 1e10:
                refined_points = res_refined.x.reshape(8, 2)
                log_score_val = -res_refined.fun
                if np.isfinite(log_score_val):
                    score = np.exp(log_score_val)
                    if score > best_score:
                        best_score = score
                        best_points = refined_points
        except Exception:
            pass

    # Scale best_points to diameter 2 for final return
    diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    diameter = np.max(dists)
    if diameter > 0:
        best_points = best_points * (2.0 / diameter)
        
    return best_points.tolist()

if __name__ == "__main__":
    pts = construct()
    # Calculate final score for debugging
    print(f"Best score: {np.exp(compute_log_score(np.array(pts)))}")
