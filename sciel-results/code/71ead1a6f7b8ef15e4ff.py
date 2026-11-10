# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 13.

    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    start_time = time.time()
    deadline = start_time + 590  # 10s safety margin
    
    def get_min_triangle_area(points):
        # points shape (n, 2)
        min_area = np.inf
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    p1 = points[i]
                    p2 = points[j]
                    p3 = points[k]
                    area = 0.5 * abs(p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1]))
                    if area < min_area:
                        min_area = area
        return min_area

    def objective(x):
        points = x.reshape(n, 2)
        if not np.all(np.isfinite(points)):
            return 1e10
        return -get_min_triangle_area(points)

    best_points = None
    best_score = -np.inf

    rng = np.random.default_rng(42)
    
    # Generate initial points
    initial_points_list = []
    
    # 1. Random
    for _ in range(10):
        initial_points_list.append(rng.random((n, 2)))
        
    # 2. Hexagonal-ish
    # Create a hexagonal lattice
    # Center at 0.5, 0.5
    # Radius 0.4
    angles = np.linspace(0, 2*np.pi, 7)[:-1]
    hex_pts = np.array([[0.5 + 0.3*np.cos(a), 0.5 + 0.3*np.sin(a)] for a in angles])
    hex_pts = np.vstack([hex_pts, [[0.5, 0.5]]])
    # Add more points
    # Just use random for now, hex is tricky to get exactly 13 non-overlapping
    # Let's add a grid
    grid_x, grid_y = np.meshgrid(np.linspace(0.1, 0.9, 4))
    grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    if len(grid_points) >= n:
        idx = rng.choice(len(grid_points), n, replace=False)
        initial_points_list.append(grid_points[idx])

    # Optimization
    for i, init_pts in enumerate(initial_points_list):
        if time.time() > deadline:
            break
            
        x0 = init_pts.flatten()
        
        try:
            # Powell is good for non-smooth
            res = minimize(objective, x0, method='Powell', 
                           options={'maxiter': 200, 'ftol': 1e-8, 'xtol': 1e-6})
        except Exception:
            continue
            
        candidate = res.x.reshape(n, 2)
        if np.all(np.isfinite(candidate)):
            try:
                hull = ConvexHull(candidate)
                if hull.volume > 1e-9:
                    min_area = get_min_triangle_area(candidate)
                    score = min_area / hull.volume
                    if score > best_score:
                        best_score = score
                        best_points = candidate
            except Exception:
                pass

    if best_points is None:
        best_points = rng.random((n, 2))
        
    return best_points

# EVOLVE-BLOCK-END
