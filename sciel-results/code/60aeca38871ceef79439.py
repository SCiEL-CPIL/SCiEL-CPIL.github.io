# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
import itertools
from scipy.optimize import minimize

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 13.

    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    start_time = time.time()
    deadline = start_time + 580.0  # 20s safety margin
    
    # Precompute triangle indices
    tri_indices = np.array(list(itertools.combinations(range(n), 3)))
    
    def compute_scores(points):
        """
        Compute the minimum triangle area and the convex hull area.
        Returns (min_area, hull_area).
        """
        # Extract coordinates for triangle vertices
        p1 = points[tri_indices[:, 0]]
        p2 = points[tri_indices[:, 1]]
        p3 = points[tri_indices[:, 2]]
        
        # Area = 0.5 * |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
        # Vectorized calculation
        areas = 0.5 * np.abs(
            p1[:, 0] * (p2[:, 1] - p3[:, 1]) +
            p2[:, 0] * (p3[:, 1] - p1[:, 1]) +
            p3[:, 0] * (p1[:, 1] - p2[:, 1])
        )
        
        min_area = np.min(areas)
        
        # Convex Hull Area
        try:
            hull = ConvexHull(points)
            hull_area = hull.volume
        except Exception:
            # If hull fails (e.g., collinear), return 0 area or a small value
            # But we want to maximize ratio, so if hull is 0, score is 0 or inf?
            # Task says "three collinear points give 0". 
            # If all collinear, hull area is 0. Ratio undefined. 
            # We should avoid this.
            return 0.0, 0.0
            
        return min_area, hull_area

    def score(points):
        min_area, hull_area = compute_scores(points)
        if hull_area == 0:
            return 0.0
        return min_area / hull_area

    def soft_min_score(points, tau):
        """
        Compute a smooth approximation of the score to allow gradient descent.
        Objective to minimize: -soft_min_area / hull_area
        """
        p1 = points[tri_indices[:, 0]]
        p2 = points[tri_indices[:, 1]]
        p3 = points[tri_indices[:, 2]]
        
        areas = 0.5 * np.abs(
            p1[:, 0] * (p2[:, 1] - p3[:, 1]) +
            p2[:, 0] * (p3[:, 1] - p1[:, 1]) +
            p3[:, 0] * (p1[:, 1] - p2[:, 1])
        )
        
        # Handle zero areas (collinear points) in soft min
        # Adding a small epsilon to avoid log(0) or exp(inf)
        areas_safe = np.maximum(areas, 1e-12)
        
        # Soft min: -tau * log(sum(exp(-A/tau)))
        # To avoid overflow, we can use log-sum-exp trick or just clip if tau is small
        # For simplicity and stability with BFGS, we might use a power mean if tau is too small
        # But log-sum-exp is fine if we scale areas.
        
        # Normalize areas by max area to keep exponents manageable?
        # Or just use a fixed tau.
        
        try:
            log_sum_exp = np.log(np.sum(np.exp(-areas_safe / tau)))
            soft_min = -tau * log_sum_exp
        except OverflowError:
            # Fallback to hard min if overflow
            soft_min = np.min(areas_safe)

        try:
            hull = ConvexHull(points)
            hull_area = hull.volume
        except Exception:
            return 1e6  # Large penalty for invalid hull
            
        if hull_area < 1e-12:
            return 1e6
            
        return -soft_min / hull_area

    def simulate_annealing(points, iters=50000, t_start=0.05, t_end=1e-6, step_start=0.5):
        current_points = points.copy()
        current_score = score(current_points)
        best_points = current_points.copy()
        best_score = current_score
        
        # Normalize scale to unit hull area? 
        # The score is scale invariant, so we don't strictly need to, 
        # but it helps with step size stability.
        
        for i in range(iters):
            if time.time() > deadline - 10:
                break
                
            # Temperature
            t = t_start * (t_end / t_start) ** (i / iters)
            
            # Step size
            step = step_start * (0.9999 ** i) # Slow decay
            
            # Perturb a random subset of points
            num_perturbed = np.random.randint(1, n + 1)
            indices = np.random.choice(n, num_perturbed, replace=False)
            
            new_points = current_points.copy()
            new_points[indices] += np.random.normal(0, step, (num_perturbed, 2))
            
            new_score = score(new_points)
            
            # Acceptance
            if new_score > current_score:
                current_points = new_points
                current_score = new_score
                if current_score > best_score:
                    best_points = new_points.copy()
                    best_score = current_score
            else:
                delta = new_score - current_score
                if np.random.random() < np.exp(delta / t):
                    current_points = new_points
                    current_score = new_score
                    
        return best_points, best_score

    def polish(points):
        """
        Use scipy.optimize.minimize on the soft-min objective.
        """
        if time.time() > deadline - 10:
            return points
            
        # Try a few different taus
        best_points = points
        best_score = score(points)
        
        taus = [0.01, 0.005, 0.001]
        
        for tau in taus:
            if time.time() > deadline - 5:
                break
                
            # Flatten points for optimizer
            x0 = points.flatten()
            
            def objective(x):
                pts = x.reshape(n, 2)
                return soft_min_score(pts, tau)
            
            try:
                res = minimize(objective, x0, method='BFGS', options={'maxiter': 1000, 'gtol': 1e-6})
                if res.success or res.fun < best_score: # Note: objective is negative score
                    # Wait, soft_min_score returns -score. So lower is better.
                    # We want to maximize score, so minimize -score.
                    new_pts = res.x.reshape(n, 2)
                    new_score = score(new_pts)
                    if new_score > best_score:
                        best_points = new_pts
                        best_score = new_score
            except Exception:
                pass
                
        return best_points

    # 1. Initialization: Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # 2. Simulated Annealing
    best_points, best_score = simulate_annealing(points, iters=100000)
    
    # 3. Polish with Gradient Descent
    polished_points = polish(best_points)
    polished_score = score(polished_points)
    
    if polished_score > best_score:
        best_points = polished_points
        best_score = polished_score
        
    # 4. One more round of SA from the polished point with smaller steps?
    # Or just return.
    
    # Ensure shape and finiteness
    if not np.all(np.isfinite(best_points)):
        # Fallback to regular polygon if something went wrong
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_points

# EVOLVE-BLOCK-END
