# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import time
from scipy.optimize import minimize

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    start_time = time.time()
    deadline = start_time + 600.0  # Safety margin for 600s limit, leave 30s buffer

    n_points = 16

    def compute_score(points):
        if points.shape != (n_points, 2):
            return -np.inf
        d = pdist(points)
        if d.size == 0:
            return 0.0
        d_min = np.min(d)
        d_max = np.max(d)
        if d_max == 0:
            return 0.0
        ratio = d_min / d_max
        return ratio * ratio

    def normalize_points(points):
        """Scale points such that the maximum pairwise distance is 1."""
        d = pdist(points)
        d_max = np.max(d)
        if d_max > 1e-12:
            return points / d_max
        return points

    def soft_min_objective(x, beta):
        """
        Objective function to minimize: -soft_min(distances).
        The points are normalized to unit diameter internally.
        """
        points = x.reshape(n_points, 2)
        points = normalize_points(points)
        d = pdist(points)
        if np.max(d) < 1e-12:
            return 0.0
        
        # Soft min: -1/beta * log(sum(exp(-beta * d)))
        # Use log-sum-exp trick for stability if beta is large
        # d is in [0, 1]
        # exp(-beta * d)
        # If beta is large, exp(-beta * d_min) dominates.
        # To avoid underflow/overflow, we can subtract the min exponent.
        # exponents = -beta * d
        # min_exp = np.min(exponents) # which is -beta * max(d) ? No, -beta * min(d) is largest.
        # We want log(sum(exp(exponents))).
        # Let m = max(exponents). sum(exp(exponents)) = exp(m) * sum(exp(exponents - m)).
        # log(sum(exp(exponents))) = m + log(sum(exp(exponents - m))).
        
        exponents = -beta * d
        m = np.max(exponents)
        if m < -700: # exp(m) underflows to 0
            return 0.0
        sum_exp = np.sum(np.exp(exponents - m))
        if sum_exp == 0:
            return 0.0
        soft_min_val = -(m + np.log(sum_exp)) / beta
        return -soft_min_val

    def optimize_candidate(points, time_budget=10.0):
        """
        Optimizes a candidate using Powell with increasing beta (temperature annealing).
        """
        points = points.copy()
        points = normalize_points(points)
        
        best_points = points.copy()
        best_score = compute_score(best_points)
        
        # Schedule of betas to sharpen the minimum
        # Start smooth, end sharp
        betas = [10, 20, 50, 100, 200, 500, 1000]
        
        start_opt = time.time()
        
        for beta in betas:
            if time.time() - start_opt > time_budget:
                break
            
            def obj(x):
                return soft_min_objective(x, beta)
            
            try:
                # Powell is derivative-free and robust
                res = minimize(
                    obj, 
                    points.flatten(), 
                    method='Powell', 
                    options={'maxiter': 1000, 'maxfev': 20000, 'xtol': 1e-10, 'ftol': 1e-10}
                )
                
                opt_points = res.x.reshape(n_points, 2)
                opt_points = normalize_points(opt_points)
                opt_score = compute_score(opt_points)
                
                if opt_score > best_score:
                    best_score = opt_score
                    best_points = opt_points.copy()
                    
                # Update current points for next beta stage
                points = best_points.copy()
                
            except Exception:
                pass
                
            if time.time() - start_opt > time_budget:
                break
                
        return best_points, best_score

    def optimize_candidate_nm(points, time_budget=5.0):
        """
        Refine using Nelder-Mead on the soft-min with high beta.
        """
        points = points.copy()
        points = normalize_points(points)
        best_points = points.copy()
        best_score = compute_score(best_points)
        
        betas = [500, 1000, 2000]
        start_opt = time.time()
        
        for beta in betas:
            if time.time() - start_opt > time_budget:
                break
            
            def obj(x):
                return soft_min_objective(x, beta)
            
            try:
                res = minimize(
                    obj, 
                    points.flatten(), 
                    method='Nelder-Mead', 
                    options={'maxiter': 5000, 'maxfev': 10000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
                )
                
                opt_points = res.x.reshape(n_points, 2)
                opt_points = normalize_points(opt_points)
                opt_score = compute_score(opt_points)
                
                if opt_score > best_score:
                    best_score = opt_score
                    best_points = opt_points.copy()
                    points = best_points.copy()
            except Exception:
                pass
                
            if time.time() - start_opt > time_budget:
                break
                
        return best_points, best_score

    best_points = None
    best_score = -np.inf

    # Generate diverse initial candidates
    candidates = []

    # 1. Random Points (Multiple seeds)
    for seed in range(40):
        np.random.seed(seed * 100 + 7)
        random_points = np.random.randn(n_points, 2)
        candidates.append(random_points)

    # 2. Concentric Circles
    # Try various splits
    splits = [(4, 12), (6, 10), (8, 8), (2, 14)]
    for n_inner, n_outer in splits:
        inner_angles = np.linspace(0, 2 * np.pi, n_inner, endpoint=False)
        outer_angles = np.linspace(0, 2 * np.pi, n_outer, endpoint=False) + np.pi/n_outer
        # Optimal radius for inner circle?
        # For 16 points, a common configuration is 1 center, 6 hex, 9 outer? Or 12 outer, 4 inner?
        # Let's try a few radii
        for r_inner in [0.3, 0.4, 0.5, 0.6]:
            inner_points = np.column_stack([r_inner * np.cos(inner_angles), r_inner * np.sin(inner_angles)])
            outer_points = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
            two_circle = np.vstack([inner_points, outer_points])
            candidates.append(two_circle)

    # 3. Triangular Lattice (subset)
    s = 1.0
    h = s * np.sqrt(3) / 2
    lattice_points = []
    for i in range(-5, 6):
        for j in range(-5, 6):
            x = i * s + (j % 2) * s / 2
            y = j * h
            lattice_points.append([x, y])
    lattice_points = np.array(lattice_points)
    centroid = np.mean(lattice_points, axis=0)
    dists_to_centroid = np.sum((lattice_points - centroid) ** 2, axis=1)
    idx = np.argsort(dists_to_centroid)[:n_points]
    candidates.append(lattice_points[idx])

    # 4. Square Grid
    grid = np.array([np.meshgrid(np.linspace(-1.5, 1.5, 4), np.linspace(-1.5, 1.5, 4))]).reshape(2, -1).T
    candidates.append(grid)

    # 5. Regular 16-gon
    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    poly_points = np.column_stack([np.cos(angles), np.sin(angles)])
    candidates.append(poly_points)

    # 6. Known good structure: 1 center, 6 around it, 9 outer?
    # Or 12 outer, 3 inner?
    # Let's add a "flower" shape
    center = np.array([[0.0, 0.0]])
    ring1_angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    ring1_points = np.column_stack([0.5 * np.cos(ring1_angles), 0.5 * np.sin(ring1_angles)])
    ring2_angles = np.linspace(0, 2 * np.pi, 9, endpoint=False) + np.pi/9
    ring2_points = np.column_stack([np.cos(ring2_angles), np.sin(ring2_angles)])
    flower = np.vstack([center, ring1_points, ring2_points])
    candidates.append(flower)

    # Evaluate and optimize candidates
    processed = []
    
    # Budget per candidate: ~5s for Powell. 
    # Total candidates ~ 40 + 12 + 1 + 1 + 1 + 1 = 56.
    # 56 * 5s = 280s.
    # Remaining time for NM polishing.
    
    for i, cand in enumerate(candidates):
        if time.time() > deadline - 60:
            break
            
        # Optimize each candidate
        opt_cand, opt_score = optimize_candidate(cand, time_budget=5.0)
        
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_cand.copy()
            
        processed.append((opt_score, opt_cand))

    # Sort processed by score
    processed.sort(key=lambda x: x[0], reverse=True)
    
    # Refine the top candidates with Nelder-Mead
    top_candidates = processed[:10]
    for score, cand in top_candidates:
        if time.time() > deadline - 10:
            break
            
        # Run NM polishing
        opt_cand, opt_score = optimize_candidate_nm(cand, time_budget=5.0)
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_cand.copy()

    if best_points is None:
        # Fallback
        best_points = np.random.randn(n_points, 2)

    return best_points

# EVOLVE-BLOCK-END
