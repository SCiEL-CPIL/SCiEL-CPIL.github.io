# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    n = 16
    d = 2
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin for 600s limit

    def get_score(points):
        """Calculate the score R = (d_min / d_max)^2"""
        dists = pdist(points)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def objective(x):
        """
        Objective function for minimization.
        We minimize log(d_max) - log(d_min) which is equivalent to minimizing -log(d_min/d_max).
        This is numerically more stable than directly minimizing (d_max/d_min) or maximizing (d_min/d_max).
        """
        points = x.reshape(n, d)
        dists = pdist(points)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_min < 1e-12 or d_max < 1e-12:
            return 1e10
        return np.log(d_max) - np.log(d_min)

    def local_search(points, max_iters=20000):
        """
        Local search with adaptive step size and random perturbations.
        This helps escape shallow local minima that gradient-based methods might get stuck in.
        """
        best_points = points.copy()
        best_score = get_score(best_points)
        step_size = 0.1
        stagnant = 0
        
        for i in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            improved = False
            # Try to improve each point
            for j in range(n):
                # Generate a few random perturbations for point j
                for _ in range(5):
                    perturbation = np.random.normal(0, step_size, size=d)
                    candidate = best_points.copy()
                    candidate[j] += perturbation
                    
                    new_score = get_score(candidate)
                    if new_score > best_score:
                        best_score = new_score
                        best_points = candidate
                        improved = True
            
            if not improved:
                stagnant += 1
                if stagnant > 5:
                    step_size *= 0.7
                    stagnant = 0
                    if step_size < 1e-6:
                        break
            else:
                stagnant = 0
                
        return best_points, best_score

    # Generate diverse starting points
    starts = []
    
    # 1. Random starts
    np.random.seed(42)
    for _ in range(5):
        starts.append(np.random.randn(n, d))
        
    # 2. Regular 16-gon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    starts.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 3. 4x4 Grid
    x = np.linspace(-1.5, 1.5, 4)
    y = np.linspace(-1.5, 1.5, 4)
    xx, yy = np.meshgrid(x, y)
    starts.append(np.column_stack([xx.ravel(), yy.ravel()]))
    
    # 4. Hexagonal cluster approximation
    hex_points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x_h = i + 0.5 * j
            y_h = np.sqrt(3) / 2 * j
            hex_points.append([x_h, y_h])
    hex_points = np.array(hex_points)
    dists_center = np.sum(hex_points**2, axis=1)
    indices = np.argsort(dists_center)
    starts.append(hex_points[indices[:n]])
    
    # 5. Concentric rings (1 center, 7 inner, 8 outer)
    p_center = [[0, 0]]
    angles_inner = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    p_inner = np.column_stack([np.cos(angles_inner), np.sin(angles_inner)])
    angles_outer = np.linspace(0, 2 * np.pi, 8, endpoint=False) + np.pi/8
    p_outer = np.column_stack([1.5 * np.cos(angles_outer), 1.5 * np.sin(angles_outer)])
    starts.append(np.vstack([p_center, p_inner, p_outer]))

    # 6. Two rings of 8
    angles1 = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    p1 = np.column_stack([np.cos(angles1), np.sin(angles1)])
    angles2 = np.linspace(0, 2 * np.pi, 8, endpoint=False) + np.pi/8
    p2 = np.column_stack([1.8 * np.cos(angles2), 1.8 * np.sin(angles2)])
    starts.append(np.vstack([p1, p2]))

    best_score = -1.0
    best_points = starts[0].copy()

    for i, start in enumerate(starts):
        if time.time() > deadline - 10:
            break
            
        try:
            # Phase 1: Nelder-Mead optimization
            res = minimize(objective, start.ravel(), method='Nelder-Mead', 
                           options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
            opt_points = res.x.reshape(n, d)
            
            s = get_score(opt_points)
            if s > best_score:
                best_score = s
                best_points = opt_points.copy()
                
            # Phase 2: Local search refinement on the result of Nelder-Mead
            # Only do this if we have time and the score is promising
            if time.time() < deadline - 20:
                np.random.seed(1000 + i)
                refined_points, refined_score = local_search(opt_points, max_iters=5000)
                if refined_score > best_score:
                    best_score = refined_score
                    best_points = refined_points.copy()
                    
        except Exception:
            pass

    # Final check and return
    return best_points

# EVOLVE-BLOCK-END
