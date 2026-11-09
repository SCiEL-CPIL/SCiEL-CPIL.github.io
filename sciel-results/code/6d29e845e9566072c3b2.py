# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from itertools import combinations

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 13.

    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    start_time = time.time()
    deadline = start_time + 600.0  # 600 second deadline with safety margin
    
    # Precompute all triangle indices
    tri_indices = list(combinations(range(n), 3))
    n_tris = len(tri_indices)
    
    def compute_score(points):
        """Compute A_min / A_hull ratio."""
        # Compute all triangle areas
        areas = np.zeros(n_tris)
        for idx, (i, j, k) in enumerate(tri_indices):
            p1, p2, p3 = points[i], points[j], points[k]
            area = abs(p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1])) / 2.0
            areas[idx] = area
        
        a_min = np.min(areas)
        if a_min <= 1e-15:
            return 0.0
        
        # Compute convex hull area
        try:
            hull = ConvexHull(points)
            a_hull = hull.volume
        except:
            return 0.0
        
        if a_hull <= 1e-15:
            return 0.0
        
        return a_min / a_hull
    
    # Start with a regular polygon-like arrangement
    # Use a grid-based initialization that avoids collinearity
    best_points = None
    best_score = 0.0
    
    # Try several different initial configurations
    # Configuration 1: Regular 13-gon
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    init1 = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Configuration 2: Concentric rings (inner points + outer ring)
    # 1 center + 4 inner ring + 8 outer ring
    init2 = np.zeros((n, 2))
    init2[0] = [0, 0]
    inner_angles = np.linspace(0, 2*np.pi, 4, endpoint=False)
    init2[1:5] = np.column_stack([0.5*np.cos(inner_angles), 0.5*np.sin(inner_angles)])
    outer_angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
    init2[5:] = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    
    # Configuration 3: Two concentric rings with offset
    init3 = np.zeros((n, 2))
    # 7 points on outer ring
    outer_angles = np.linspace(0, 2*np.pi, 7, endpoint=False)
    init3[:7] = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    # 6 points on inner ring, rotated
    inner_angles = np.linspace(np.pi/6, 2*np.pi + np.pi/6, 6, endpoint=False)
    init3[7:] = np.column_stack([0.6*np.cos(inner_angles), 0.6*np.sin(inner_angles)])
    
    # Configuration 4: Grid-like pattern
    init4 = np.array([
        [-1, -1], [0, -1], [1, -1],
        [-1, 0], [0, 0], [1, 0],
        [-1, 1], [0, 1], [1, 1],
        [-0.5, -1.5], [0.5, -1.5],
        [-0.5, 1.5], [0.5, 1.5]
    ])
    
    # Configuration 5: Random with good seed
    rng = np.random.default_rng(seed=12345)
    init5 = rng.random((n, 2)) * 2 - 1
    
    inits = [init1, init2, init3, init4, init5]
    
    # Evaluate initial configurations
    for init in inits:
        score = compute_score(init)
        if score > best_score:
            best_score = score
            best_points = init.copy()
    
    # Local optimization using gradient-free methods
    # Use Nelder-Mead or similar to refine
    from scipy.optimize import minimize
    
    def negative_score(params):
        """Objective to minimize (negative of score)."""
        points = params.reshape(n, 2)
        # Check for invalid points
        if not np.all(np.isfinite(points)):
            return 1e10
        score = compute_score(points)
        return -score
    
    # Try multiple restarts with different perturbations
    n_restarts = 20
    for restart in range(n_restarts):
        if time.time() > deadline - 30:  # Safety margin
            break
        
        # Create perturbed initial point
        if best_points is not None:
            rng_r = np.random.default_rng(seed=1000 + restart)
            perturbation = rng_r.random((n, 2)) * 0.3 - 0.15
            current = best_points + perturbation
        else:
            current = inits[restart % len(inits)]
        
        # Flatten for optimization
        x0 = current.flatten()
        
        # Run Nelder-Mead optimization
        try:
            result = minimize(
                negative_score,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            
            if result.success or result.fun < best_score:
                new_points = result.x.reshape(n, 2)
                new_score = -result.fun
                if new_score > best_score:
                    best_score = new_score
                    best_points = new_points.copy()
        except Exception:
            pass
    
    # Additional refinement: coordinate descent on the best points
    if best_points is not None and time.time() < deadline - 60:
        current = best_points.copy()
        current_score = compute_score(current)
        
        # Simple coordinate descent
        step_size = 0.05
        for iteration in range(200):
            if time.time() > deadline - 30:
                break
            
            improved = False
            for i in range(n):
                for d in range(2):
                    # Try moving in positive and negative directions
                    for direction in [1, -1]:
                        trial = current.copy()
                        trial[i, d] += direction * step_size
                        
                        # Keep within bounds (optional, helps stability)
                        if np.any(np.abs(trial) > 10):
                            continue
                        
                        trial_score = compute_score(trial)
                        if trial_score > current_score:
                            current = trial
                            current_score = trial_score
                            improved = True
            
            if not improved:
                # Reduce step size
                step_size *= 0.7
                if step_size < 1e-6:
                    break
            
            if current_score > best_score:
                best_score = current_score
                best_points = current.copy()
    
    # Ensure valid output
    if best_points is None:
        # Fallback to regular polygon
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Verify the result is valid
    if not np.all(np.isfinite(best_points)):
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_points

# EVOLVE-BLOCK-END
