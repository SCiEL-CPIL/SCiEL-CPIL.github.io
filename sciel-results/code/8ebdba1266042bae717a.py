# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import time
from scipy.optimize import minimize
import random

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the (x,y) coordinates of the 16 points.
    """
    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin for 630s limit, leave 40s buffer

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
        
        exponents = -beta * d
        m = np.max(exponents)
        if m < -700:
            return 0.0
        sum_exp = np.sum(np.exp(exponents - m))
        if sum_exp == 0:
            return 0.0
        soft_min_val = -(m + np.log(sum_exp)) / beta
        return -soft_min_val

    def ratio_objective(x):
        """
        Direct objective: minimize -(d_min/d_max)^2 using smoothed min
        """
        points = x.reshape(n_points, 2)
        points = normalize_points(points)
        d = pdist(points)
        if d.size == 0 or np.max(d) < 1e-12:
            return 0.0
        
        d_max = np.max(d)
        beta = 5000
        exponents = -beta * d
        m = np.max(exponents)
        if m < -700:
            return 0.0
        sum_exp = np.sum(np.exp(exponents - m))
        if sum_exp == 0:
            return 0.0
        soft_min_val = -(m + np.log(sum_exp)) / beta
        
        ratio = soft_min_val / d_max
        return -(ratio * ratio)

    def optimize_candidate_powell(points, time_budget=5.0):
        """
        Optimizes a candidate using Powell with increasing beta.
        """
        points = points.copy()
        points = normalize_points(points)
        
        best_points = points.copy()
        best_score = compute_score(best_points)
        
        betas = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
        
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
                    method='Powell', 
                    options={'maxiter': 300, 'maxfev': 8000, 'xtol': 1e-12, 'ftol': 1e-12}
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

    def optimize_candidate_nm(points, time_budget=5.0):
        """
        Refine using Nelder-Mead on the soft-min with high beta.
        """
        points = points.copy()
        points = normalize_points(points)
        best_points = points.copy()
        best_score = compute_score(best_points)
        
        betas = [1000, 2000, 5000, 10000]
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
                    options={'maxiter': 8000, 'maxfev': 15000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True}
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

    def final_polish(points, time_budget=30.0):
        """
        Final high-precision polish using the direct ratio objective.
        """
        points = points.copy()
        points = normalize_points(points)
        best_points = points.copy()
        best_score = compute_score(best_points)
        
        start_opt = time.time()
        
        for iteration in range(5):
            if time.time() - start_opt > time_budget:
                break
            
            def obj(x):
                return ratio_objective(x)
            
            try:
                res = minimize(
                    obj, 
                    best_points.flatten(), 
                    method='Nelder-Mead', 
                    options={'maxiter': 10000, 'maxfev': 20000, 'xatol': 1e-15, 'fatol': 1e-15, 'adaptive': True}
                )
                
                opt_points = res.x.reshape(n_points, 2)
                opt_points = normalize_points(opt_points)
                opt_score = compute_score(opt_points)
                
                if opt_score > best_score:
                    best_score = opt_score
                    best_points = opt_points.copy()
            except Exception:
                pass
            
            if time.time() - start_opt > time_budget:
                break
        
        return best_points, best_score

    best_points = None
    best_score = -np.inf

    # Generate diverse initial candidates
    candidates = []

    # 1. Structured Candidates: Symmetry-Constrained Constructions
    
    # A. Radial Distortion of 4x4 Grid
    grid_x, grid_y = np.meshgrid(np.linspace(-1, 1, 4), np.linspace(-1, 1, 4))
    grid = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    grid -= np.mean(grid, axis=0)
    
    for alpha in [0.5, 0.7, 0.8, 1.0, 1.2, 1.5]:
        r = np.linalg.norm(grid, axis=1)
        theta = np.arctan2(grid[:, 1], grid[:, 0])
        r_new = r ** alpha
        warped = np.column_stack((r_new * np.cos(theta), r_new * np.sin(theta)))
        candidates.append(warped)

    # B. Multi-Ring Structures (Flower Patterns)
    
    # B1. 1 Center, 6 Inner, 9 Outer
    center = np.array([[0.0, 0.0]])
    for r_inner in [0.4, 0.5, 0.6, 0.7, 0.8]:
        ring1_angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
        ring1_points = np.column_stack([r_inner * np.cos(ring1_angles), r_inner * np.sin(ring1_angles)])
        
        for phase in [0, np.pi/9, np.pi/18]:
            ring2_angles = np.linspace(0, 2 * np.pi, 9, endpoint=False) + phase
            ring2_points = np.column_stack([np.cos(ring2_angles), np.sin(ring2_angles)])
            flower = np.vstack([center, ring1_points, ring2_points])
            candidates.append(flower)

    # B2. 1 Center, 7 Inner, 8 Outer
    for r_inner in [0.4, 0.5, 0.6]:
        ring1_angles = np.linspace(0, 2 * np.pi, 7, endpoint=False)
        ring1_points = np.column_stack([r_inner * np.cos(ring1_angles), r_inner * np.sin(ring1_angles)])
        
        for phase in [0, np.pi/8, np.pi/16]:
            ring2_angles = np.linspace(0, 2 * np.pi, 8, endpoint=False) + phase
            ring2_points = np.column_stack([np.cos(ring2_angles), np.sin(ring2_angles)])
            flower = np.vstack([center, ring1_points, ring2_points])
            candidates.append(flower)

    # B3. 2 Rings: 8 Inner, 8 Outer
    for r_inner in [0.5, 0.6, 0.7, 0.8]:
        ring1_angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        ring1_points = np.column_stack([r_inner * np.cos(ring1_angles), r_inner * np.sin(ring1_angles)])
        
        for phase in [0, np.pi/8, np.pi/16]:
            ring2_angles = np.linspace(0, 2 * np.pi, 8, endpoint=False) + phase
            ring2_points = np.column_stack([np.cos(ring2_angles), np.sin(ring2_angles)])
            two_rings = np.vstack([ring1_points, ring2_points])
            candidates.append(two_rings)

    # C. Triangular Lattice Subset
    s = 1.0
    h = s * np.sqrt(3) / 2
    lattice_points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = i * s + (j % 2) * s / 2
            y = j * h
            lattice_points.append([x, y])
    lattice_points = np.array(lattice_points)
    centroid = np.mean(lattice_points, axis=0)
    dists_to_centroid = np.sum((lattice_points - centroid) ** 2, axis=1)
    idx = np.argsort(dists_to_centroid)[:n_points]
    candidates.append(lattice_points[idx])

    # D. Random Points (Reduced count, as structured seeds are stronger)
    for seed in range(15):
        np.random.seed(seed * 100 + 7)
        random_points = np.random.randn(n_points, 2)
        candidates.append(random_points)

    # Evaluate and optimize candidates
    processed = []
    
    for i, cand in enumerate(candidates):
        if time.time() > deadline - 60:
            break
            
        # Optimize each candidate
        opt_cand, opt_score = optimize_candidate_powell(cand, time_budget=4.0)
        
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_cand.copy()
            
        processed.append((opt_score, opt_cand))

    # Sort processed by score
    processed.sort(key=lambda x: x[0], reverse=True)
    
    # Refine the top candidates with Nelder-Mead
    top_candidates = processed[:20]
    for score, cand in top_candidates:
        if time.time() > deadline - 10:
            break
            
        # Run NM polishing
        opt_cand, opt_score = optimize_candidate_nm(cand, time_budget=4.0)
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_cand.copy()
            
    # If we still have time, try a few more random restarts with NM directly
    while time.time() < deadline - 30:
        np.random.seed(int(time.time() * 1000) % 10000)
        random_points = np.random.randn(n_points, 2)
        opt_cand, opt_score = optimize_candidate_nm(random_points, time_budget=3.0)
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_cand.copy()

    # Final polish on the best candidate found
    if best_points is not None:
        remaining_time = max(0, deadline - 10 - time.time())
        if remaining_time > 5:
            opt_cand, opt_score = final_polish(best_points, time_budget=remaining_time)
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_cand.copy()

    if best_points is None:
        # Fallback
        best_points = np.random.randn(n_points, 2)

    return best_points

# EVOLVE-BLOCK-END
