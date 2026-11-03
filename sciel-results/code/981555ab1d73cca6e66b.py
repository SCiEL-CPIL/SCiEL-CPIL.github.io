import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14
N_FIXED = 11
N_FREE = 3

def min_triangle_area(points):
    # points: (N, 2)
    min_area = np.inf
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                p1, p2, p3 = points[i], points[j], points[k]
                v1 = p2 - p1
                v2 = p3 - p1
                area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
                if area < min_area:
                    min_area = area
    return min_area

def hull_area(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def score(points):
    a_min = min_triangle_area(points)
    a_hull = hull_area(points)
    if a_hull == 0: return 0
    return a_min / a_hull

def construct():
    # Fixed points: Regular 11-gon
    angles_fixed = np.linspace(0, 2*np.pi, N_FIXED, endpoint=False)
    fixed_pts = np.column_stack([np.cos(angles_fixed), np.sin(angles_fixed)])
    
    best_val = -np.inf
    best_pts = None
    
    # Multi-start optimization for the 3 free points
    np.random.seed(42)
    for _ in range(10):
        # Initialize free points at random angles on the unit circle
        angles_free = np.random.uniform(0, 2*np.pi, N_FREE)
        free_init = np.column_stack([np.cos(angles_free), np.sin(angles_free)])
        
        def objective(free_flat):
            free_pts = free_flat.reshape(N_FREE, 2)
            all_pts = np.vstack([fixed_pts, free_pts])
            h_area = hull_area(all_pts)
            if h_area < 1e-6:
                return 0
            scale = np.sqrt(1.0 / h_area)
            all_scaled = all_pts * scale
            return -min_triangle_area(all_scaled)
        
        try:
            result = minimize(objective, free_init.flatten(), method='Nelder-Mead', 
                              options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-6})
            
            val = -result.fun
            if val > best_val:
                best_val = val
                free_best = result.x.reshape(N_FREE, 2)
                all_best = np.vstack([fixed_pts, free_best])
                h_area = hull_area(all_best)
                if h_area > 0:
                    best_pts = all_best * np.sqrt(1.0 / h_area)
        except Exception:
            continue
            
    if best_pts is None:
        # Fallback to regular 14-gon
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
        h_area = hull_area(best_pts)
        if h_area > 0:
            best_pts = best_pts * np.sqrt(1.0 / h_area)
            
    return best_pts
