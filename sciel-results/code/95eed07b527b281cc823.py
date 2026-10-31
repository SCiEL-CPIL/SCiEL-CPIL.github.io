import numpy as np
import time
from scipy.optimize import minimize

N = 13

def compute_hull_area_fast(points):
    """
    Computes the area of the convex hull assuming points are in convex position.
    For robustness, we sort by angle around the centroid.
    """
    centroid = np.mean(points, axis=0)
    # Calculate angles
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
    # Sort indices by angle
    sorted_idx = np.argsort(angles)
    sorted_points = points[sorted_idx]
    
    # Shoelace formula
    x = sorted_points[:, 0]
    y = sorted_points[:, 1]
    # Shift x, y for cyclic calculation
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)
    
    area = 0.5 * np.abs(np.sum(x * y_next - x_next * y))
    return area

def compute_min_triangle_area(points):
    """
    Computes the minimum area of all triangles formed by triples of points.
    """
    n = points.shape[0]
    min_area = float('inf')
    
    # Vectorized approach for speed
    # i, j, k
    # We can use broadcasting for all pairs, but memory might be an issue for N=13?
    # 13*12*11 = 1716 triples. Very small.
    
    # Efficient loop
    for i in range(n - 2):
        p1 = points[i]
        for j in range(i + 1, n - 1):
            p2 = points[j]
            # Vectorize over k
            p3s = points[j + 1:]
            # Area = 0.5 * |(p2-p1) x (p3-p1)|
            # Cross product in 2D: (x2-x1)*(y3-y1) - (y2-y1)*(x3-x1)
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            
            # p3s are (m, 2)
            # dx * (p3s[:,1] - p1[1]) - dy * (p3s[:,0] - p1[0])
            areas = 0.5 * np.abs(dx * (p3s[:, 1] - p1[1]) - dy * (p3s[:, 0] - p1[0]))
            current_min = np.min(areas)
            if current_min < min_area:
                min_area = current_min
    return min_area

def objective(points):
    """
    Returns the negative of the score (to minimize).
    Score = min_triangle_area / hull_area
    """
    hull_area = compute_hull_area_fast(points)
    if hull_area < 1e-10:
        return 0.0 # Degenerate
    
    min_area = compute_min_triangle_area(points)
    score = min_area / hull_area
    return -score

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    # Initial guess: Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    best_points = points.copy()
    best_score = -objective(points)
    
    # Simulated Annealing
    current_points = points.copy()
    current_score = -objective(current_points)
    
    T = 0.05
    T_min = 0.001
    cooling_rate = 0.999
    
    max_iter = 5000
    iter_count = 0
    
    while T > T_min and iter_count < max_iter:
        if time.time() > deadline:
            break
            
        # Perturb
        perturbed = current_points + np.random.normal(0, T, size=(N, 2))
        
        new_score = -objective(perturbed)
        
        # Acceptance criterion
        if new_score < current_score: # Improvement (since we minimize negative score)
            current_points = perturbed
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_points = current_points.copy()
        else:
            # Metropolis criterion
            delta = current_score - new_score # Positive if worse
            if np.random.random() < np.exp(-delta / T):
                current_points = perturbed
                current_score = new_score
        
        T *= cooling_rate
        iter_count += 1
        
    # Nelder-Mead Polish
    if time.time() < deadline - 10: # Ensure enough time for NM
        try:
            res = minimize(objective, best_points, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.fun < best_score:
                best_score = res.fun
                best_points = res.x
        except Exception:
            pass
            
    # Final check
    return best_points

# EVOLVE-BLOCK-START
# The construct function is defined above. 
# To strictly follow the "rewrite the program" and "keep function names" instruction,
# I will place the code inside the evolve block or ensure the file structure is valid.
# The prompt asks for the COMPLETE program in a single code block.
# The original code had EVOLVE-BLOCK markers. I will provide the full script.
# EVOLVE-BLOCK-END
