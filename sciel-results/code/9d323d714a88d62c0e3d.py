import numpy as np
from scipy.optimize import minimize
import time

def compute_score(points):
    """
    Computes the score for a set of 8 points.
    Score is (Product of pairwise distances after scaling to diameter 2) / 8^8.
    """
    points = np.array(points)
    n = len(points)
    if n != 8:
        return 0.0
    
    # Calculate all pairwise distances
    # dist[i, j] = ||p_i - p_j||
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(points[i] - points[j])
            dists[i, j] = d
            dists[j, i] = d
            
    max_dist = np.max(dists)
    if max_dist < 1e-12:
        return 0.0
        
    # Scale to diameter 2
    scale_factor = 2.0 / max_dist
    scaled_points = points * scale_factor
    
    # Recalculate distances for scaled points (or just scale the distances)
    # d' = d * scale_factor
    scaled_dists = dists * scale_factor
    
    # Product of distances for i < j
    # log_product = sum(log(d_ij)) for i < j
    # Using log to avoid underflow
    # Exclude diagonal and double count
    log_prod = 0.0
    for i in range(n):
        for j in range(i+1, n):
            if scaled_dists[i, j] > 1e-15:
                log_prod += np.log(scaled_dists[i, j])
            else:
                return 0.0
                
    # Score = exp(log_prod) / 8^8
    # We want to maximize this. 
    # The regular octagon scores 1.0.
    # So we return exp(log_prod) / (8**8)
    
    score = np.exp(log_prod) / (8 ** 8)
    return score

def objective(x):
    """
    Objective function for minimization (negative log score).
    x is a flat array of 16 floats.
    """
    points = x.reshape(8, 2)
    # Compute log of the product of distances after scaling
    # This is equivalent to maximizing the product
    n = 8
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(points[i] - points[j])
            dists[i, j] = d
            dists[j, i] = d
            
    max_dist = np.max(dists)
    if max_dist < 1e-12:
        return 1e100 # Large penalty
        
    # We want to maximize sum(log(d_ij)) - binom(8,2)*log(max_dist) + binom(8,2)*log(2)
    # Actually, let's just use the score formula directly in log space to be safe.
    # Score = (prod d_ij_scaled) / 8^8
    # log(Score) = sum(log(d_ij_scaled)) - 8*8*log(8)
    # d_ij_scaled = d_ij * (2 / max_dist)
    # log(d_ij_scaled) = log(d_ij) + log(2) - log(max_dist)
    
    # Sum over i < j
    log_sum = 0.0
    for i in range(n):
        for j in range(i+1, n):
            if dists[i, j] < 1e-15:
                return 1e100
            log_sum += np.log(dists[i, j])
            
    # There are C(8,2) = 28 pairs
    num_pairs = 28
    log_score = log_sum + num_pairs * (np.log(2.0) - np.log(max_dist)) - 8.0 * 8.0 * np.log(8.0)
    
    # We minimize -log_score
    return -log_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    best_score = 0.0
    best_points = None
    
    # Generate initial guesses
    def regular_octagon():
        angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
        return np.column_stack((np.cos(angles), np.sin(angles)))
    
    def two_squares():
        # Square 1
        a1 = np.linspace(0, 2*np.pi, 4, endpoint=False)
        p1 = np.column_stack((np.cos(a1), np.sin(a1)))
        # Square 2 rotated by 45 degrees (pi/4)
        a2 = a1 + np.pi/4
        p2 = np.column_stack((np.cos(a2), np.sin(a2)))
        return np.vstack((p1, p2))
        
    def random_circle():
        angles = np.sort(np.random.uniform(0, 2*np.pi, 8))
        return np.column_stack((np.cos(angles), np.sin(angles)))
        
    def random_distorted():
        # Regular octagon with random radial perturbations
        angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
        radii = np.random.uniform(0.8, 1.2, 8)
        return np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))

    initial_guesses = [
        regular_octagon(),
        two_squares(),
        random_circle(),
        random_distorted(),
        random_distorted(),
        random_distorted()
    ]
    
    # Add some rotated versions of the regular octagon to break symmetry if needed? 
    # The objective is rotationally invariant, so rotation doesn't change the score.
    # But it might help escape local minima if the landscape is complex? 
    # Actually, since it's invariant, the optimizer might struggle with the flat direction.
    # We can fix the first point to break rotational symmetry, but that's complicated.
    # Let's just rely on multi-start.
    
    for i, guess in enumerate(initial_guesses):
        if time.time() > deadline:
            break
            
        # Flatten
        x0 = guess.flatten()
        
        # Optimization
        # Use L-BFGS-B or SLSQP. SLSQP is slower but more robust for small problems.
        # Given 16 variables, it's small.
        try:
            res = minimize(objective, x0, method='SLSQP', options={'maxiter': 100, 'ftol': 1e-12})
        except Exception:
            continue
            
        if res.success or res.fun < 1e99:
            x_opt = res.x
            points_opt = x_opt.reshape(8, 2)
            score = compute_score(points_opt)
            
            if score > best_score:
                best_score = score
                best_points = points_opt
                
        # Check time
        if time.time() > deadline - 10:
            break
            
    if best_points is None:
        # Fallback to regular octagon
        best_points = regular_octagon()
        
    # Ensure distinct points and valid coordinates
    # The optimizer might produce very close points. 
    # If score is 0 or very low, fallback.
    if best_score < 0.9:
        best_points = regular_octagon()
        
    return [[float(p[0]), float(p[1])] for p in best_points]
