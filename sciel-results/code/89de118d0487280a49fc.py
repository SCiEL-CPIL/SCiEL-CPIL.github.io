import numpy as np
import time
from scipy.optimize import differential_evolution

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin for 150s limit
    
    n = 8
    
    def objective(params):
        """
        params: 16 floats (8 x, 8 y)
        Returns: -log_score (to be minimized)
        """
        if time.time() > deadline:
            return 1e10
            
        points = params.reshape((n, 2))
        
        # Check for distinct points
        # If any two points are too close, return a large penalty
        # We can check pairwise distances
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(points[i] - points[j])
                dists[i, j] = d
                dists[j, i] = d
                if d < 1e-8:
                    return 1e10
        
        # Calculate product of distances for ordered pairs?
        # The task says "product over ordered pairs".
        # Usually this means sum over i!=j of log(dist_ij).
        # Or product over i<j of dist_ij^2?
        # "Delta is the product over ordered pairs of distances"
        # Ordered pairs (i, j) with i != j. There are n*(n-1) = 56 such pairs.
        # If it meant unordered, it would be n*(n-1)/2 = 28.
        # Let's check the regular octagon score.
        # Regular octagon on diameter 2 (radius 1).
        # Distances between vertices:
        # 1 step: 2*sin(pi/8)
        # 2 steps: 2*sin(2pi/8)
        # 3 steps: 2*sin(3pi/8)
        # 4 steps: 2*sin(4pi/8) = 2
        # Each distance appears 8 times in ordered pairs (for each starting point).
        # Wait, for a regular n-gon, the distance d_k (k steps away) appears n times in the set of ordered pairs?
        # No, for each point i, there is exactly one point at distance d_k (for k=1..n/2) in each direction?
        # For n=8:
        # d1 = 2 sin(pi/8). Appears 8 times (i to i+1) and 8 times (i to i-1). Total 16?
        # Let's just compute the product directly.
        
        # Product over ordered pairs i!=j of dist(i,j)
        # log_product = sum_{i!=j} log(dist(i,j))
        
        # Diameter D = max_{i,j} dist(i,j)
        D = np.max(dists)
        
        if D < 1e-8:
            return 1e10
            
        # Scaling factor: scale points so diameter is 2.
        # Original points P. Scaled points P' = (2/D) * (P - centroid)? 
        # No, scaling is usually uniform. "scaling the set to diameter 2".
        # This implies we take the set, compute its diameter D, and multiply all coordinates by 2/D.
        # Does it center them? Usually "scaling" implies homothety. 
        # The product of distances scales by (2/D)^n if we scale by 2/D? 
        # Wait, distance between two points scales by factor s.
        # Product of N distances scales by s^N.
        # N = n*(n-1) = 56.
        # So Delta_scaled = Delta_original * (2/D)^56.
        
        # Score = Delta_scaled / n^n
        # log_score = log(Delta_original) + 56*log(2) - 56*log(D) - n*log(n)
        
        # We want to maximize log_score.
        # Minimize -log_score.
        
        # Calculate log(Delta_original)
        # We have the dists matrix. We need sum of log(dists[i,j]) for i!=j.
        # dists[i,i] is 0, so we must exclude those.
        # Mask diagonal.
        mask = ~np.eye(n, dtype=bool)
        valid_dists = dists[mask]
        
        # Check for zero distances again just in case
        if np.any(valid_dists < 1e-8):
            return 1e10
            
        log_delta = np.sum(np.log(valid_dists))
        
        log_score = log_delta + 56 * np.log(2) - 56 * np.log(D) - n * np.log(n)
        
        return -log_score

    # Bounds for differential evolution
    # We can bound the coordinates to a reasonable range, e.g., [-10, 10]
    # Since the problem is scale invariant (after normalization), the absolute scale doesn't matter,
    # but DE needs bounds.
    bounds = [(-10, 10)] * (2 * n)
    
    # Strategy:
    # 1. Run DE with a moderate population size and maxiter to find a good global candidate.
    # 2. If time permits, run a second DE with different parameters or refine the best result.
    
    best_result = None
    best_val = np.inf
    
    # Parameters for DE
    # popsize: 15 (150 individuals)
    # maxiter: 100
    # tol: 1e-8
    
    try:
        # Use differential_evolution
        # strategy: 'best1bin' or 'best1exp'
        # maxiter: limit based on time
        # We estimate time per iteration. 
        # DE with 150 individuals, 16 dims.
        # Each iteration evaluates 150 individuals.
        # Objective function is fast (8x8 dist calc).
        # Let's try maxiter=200.
        
        result = differential_evolution(
            objective, 
            bounds, 
            maxiter=200, 
            popsize=15, 
            tol=1e-8, 
            seed=42,
            polish=True,  # Use L-BFGS-B to polish the result
            maxfev=10000 # Safety cap on function evaluations
        )
        
        if result.success or result.fun < best_val:
            best_val = result.fun
            best_result = result.x
            
    except Exception as e:
        pass
    
    # If time remains, try another run with different seed or parameters
    if time.time() < deadline - 20 and best_result is not None:
        try:
            # Refine with a tighter search or different strategy
            # Maybe use the previous best as a starting point for a local search?
            # Or just run DE again with a different seed.
            
            # Let's try a run with 'rand1bin' strategy which is more exploratory
            result2 = differential_evolution(
                objective, 
                bounds, 
                maxiter=100, 
                popsize=10, 
                tol=1e-8, 
                seed=123,
                strategy='rand1bin',
                polish=True,
                maxfev=5000
            )
            
            if result2.fun < best_val:
                best_val = result2.fun
                best_result = result2.x
                
        except Exception:
            pass
            
    # Fallback: Regular Octagon if optimization failed or returned bad result
    if best_result is None or best_val > 1e9:
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        best_result = np.column_stack((np.cos(angles), np.sin(angles))).flatten()
    
    # Final check: ensure distinct points
    points = best_result.reshape((n, 2))
    
    # Verify distinctness
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(points[i] - points[j]) < 1e-6:
                # Perturb slightly
                points[j] += np.random.normal(0, 1e-5, 2)
                
    # Return as list of lists
    return [[float(p[0]), float(p[1])] for p in points]
