import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import itertools

def _score(points: np.ndarray) -> float:
    """Compute A_min / A_hull for a set of points using vectorized operations."""
    n = points.shape[0]
    if n < 3:
        return 0.0
    
    # Check for finite points
    if not np.all(np.isfinite(points)):
        return 0.0
    
    # Compute convex hull
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 0.0
    
    if A_hull <= 0:
        return 0.0
    
    # Vectorized triangle area calculation
    # Get all combinations of indices i < j < k
    # n=13, C(13,3) = 286 triangles. 
    # Using itertools is fast enough for 286 triangles.
    idx = np.arange(n)
    # Pre-compute indices for speed if called repeatedly? 
    # For n=13, generating indices is negligible.
    i_idx, j_idx, k_idx = [], [], []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                i_idx.append(i)
                j_idx.append(j)
                k_idx.append(k)
    
    i_arr = np.array(i_idx)
    j_arr = np.array(j_idx)
    k_arr = np.array(k_idx)
    
    p1 = points[i_arr]
    p2 = points[j_arr]
    p3 = points[k_arr]
    
    # Area = 0.5 * |cross|
    # Cross product in 2D: (x2-x1)*(y3-y1) - (y2-y1)*(x3-x1)
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = np.abs(cross) * 0.5
    
    min_area = np.min(areas)
    
    if min_area < 1e-12:
        return 0.0
        
    return min_area / A_hull

def _soft_min_score(points: np.ndarray, temperature=0.01) -> float:
    """
    A differentiable approximation of the score for optimization.
    Uses soft-min to approximate the minimum area.
    """
    n = points.shape[0]
    if n < 3:
        return 0.0
    if not np.all(np.isfinite(points)):
        return -1e6 # Penalty
    
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return -1e6
        
    if A_hull <= 0:
        return -1e6

    idx = np.arange(n)
    i_idx, j_idx, k_idx = [], [], []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                i_idx.append(i)
                j_idx.append(j)
                k_idx.append(k)
    
    i_arr = np.array(i_idx)
    j_arr = np.array(j_idx)
    k_arr = np.array(k_idx)
    
    p1 = points[i_arr]
    p2 = points[j_arr]
    p3 = points[k_arr]
    
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = np.abs(cross) * 0.5
    
    # Soft min: -t * log(sum(exp(-area/t)))
    # To avoid overflow, subtract max
    min_area_approx = -temperature * np.log(np.sum(np.exp(-areas / temperature)))
    
    if min_area_approx < 1e-12:
        return -1e6
        
    return min_area_approx / A_hull

def _nelder_mead_polish(points: np.ndarray, max_iter=200) -> np.ndarray:
    """
    Refine points using Nelder-Mead on the negative soft-min score.
    """
    # We want to maximize score, so minimize -score
    # Use a small temperature for the soft-min to make it closer to hard-min
    t_soft = 0.005 
    
    def objective(flat_points):
        pts = flat_points.reshape(13, 2)
        # Clip to valid range to avoid weird behavior during optimization steps
        pts = np.clip(pts, 0.0, 1.0)
        # If points are collinear or invalid, return high penalty
        s = _soft_min_score(pts, temperature=t_soft)
        if s < 0:
            return 1e6
        return -s

    initial_flat = points.flatten()
    
    try:
        res = minimize(
            objective, 
            initial_flat, 
            method='Nelder-Mead', 
            options={'maxiter': max_iter, 'xatol': 1e-6, 'fatol': 1e-6}
        )
        if res.success or res.fun < 1e5:
            best_pts = res.x.reshape(13, 2)
            best_pts = np.clip(best_pts, 0.0, 1.0)
            return best_pts
    except Exception:
        pass
        
    return points

def _simulated_annealing(points: np.ndarray, start_temp=0.05, end_temp=0.001, iterations=5000, time_limit=None, start_time=None):
    """
    Perform Simulated Annealing to improve the point arrangement.
    """
    if start_time is None:
        start_time = time.time()
    
    current = points.copy()
    current_score = _score(current)
    best = current.copy()
    best_score = current_score
    
    n = current.shape[0]
    T = start_temp
    cooling_rate = (end_temp / start_temp) ** (1.0 / iterations)
    
    for i in range(iterations):
        if time_limit and time.time() > time_limit:
            break
            
        # Pick a random point to move
        idx = np.random.randint(n)
        # Gaussian perturbation
        # Scale step size with temperature
        step_size = T * 2.0
        delta = np.random.normal(0, step_size, 2)
        new_point = current[idx] + delta
        
        # Keep points within [0, 1]^2
        if new_point[0] < 0 or new_point[0] > 1 or new_point[1] < 0 or new_point[1] > 1:
            # Reflect or reject? Reject is simpler
            continue
        
        candidate = current.copy()
        candidate[idx] = new_point
        candidate_score = _score(candidate)
        
        # Acceptance probability
        if candidate_score > current_score:
            current = candidate
            current_score = candidate_score
            if candidate_score > best_score:
                best_score = candidate_score
                best = candidate.copy()
        else:
            # Metropolis criterion
            # delta_E = current_score - candidate_score (positive means candidate is worse)
            delta_E = current_score - candidate_score
            if delta_E > 0:
                probability = np.exp(-delta_E / T)
                if np.random.random() < probability:
                    current = candidate
                    current_score = candidate_score
        
        # Cool down
        T *= cooling_rate
        
    return best, best_score

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 13.

    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    start_time = time.time()
    deadline = start_time + 590  # 10s safety margin
    
    best_points = None
    best_score = -1.0
    
    rng = np.random.default_rng(42)
    
    # Initial configurations
    # 1. Grid
    grid = np.array([[i/3, j/3] for i in range(4) for j in range(4) if not (i==3 and j==3)])
    grid = np.clip(grid, 0.01, 0.99)
    
    # 2. Hexagonal
    hex_pts = np.array([
        [0.1, 0.1], [0.3, 0.1], [0.5, 0.1], [0.7, 0.1],
        [0.2, 0.3], [0.4, 0.3], [0.6, 0.3],
        [0.1, 0.5], [0.3, 0.5], [0.5, 0.5], [0.7, 0.5],
        [0.2, 0.7], [0.4, 0.7]
    ])
    
    # 3. Random seeds
    random_starts = [rng.random((13, 2)) for _ in range(5)]
    random_starts = [np.clip(p, 0.01, 0.99) for p in random_starts]
    
    starts = [grid, hex_pts] + random_starts
    
    # Evaluate starts
    for pts in starts:
        score = _score(pts)
        if score > best_score:
            best_score = score
            best_points = pts.copy()
    
    # Run Simulated Annealing from the best start
    if best_points is not None:
        sa_deadline = deadline - 30 # Leave 30s for polish and other restarts
        improved, improved_score = _simulated_annealing(
            best_points, 
            start_temp=0.05, 
            end_temp=0.001, 
            iterations=10000, 
            time_limit=sa_deadline,
            start_time=start_time
        )
        if improved_score > best_score:
            best_score = improved_score
            best_points = improved.copy()
    
    # Additional random restarts if time permits
    while time.time() < deadline - 60:
        pts = rng.random((13, 2))
        pts = np.clip(pts, 0.01, 0.99)
        
        # Quick local search / SA with shorter time
        restart_deadline = min(time.time() + 15, deadline - 30)
        improved, improved_score = _simulated_annealing(
            pts, 
            start_temp=0.1, 
            end_temp=0.005, 
            iterations=3000, 
            time_limit=restart_deadline,
            start_time=start_time
        )
        
        if improved_score > best_score:
            best_score = improved_score
            best_points = improved.copy()
    
    # Final Polish with Nelder-Mead
    if best_points is not None and time.time() < deadline - 10:
        polished = _nelder_mead_polish(best_points, max_iter=100)
        polished_score = _score(polished)
        if polished_score > best_score:
            best_score = polished_score
            best_points = polished
    
    if best_points is None:
        best_points = np.random.default_rng(42).random((13, 2))
    
    return best_points

# EVOLVE-BLOCK-END
