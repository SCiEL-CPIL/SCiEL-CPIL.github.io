import numpy as np
import time
import random
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 13

def compute_score(points):
    n = points.shape[0]
    # Efficiently get indices
    # For N=13, simple loops are also fine, but vectorized is safer for speed
    # 286 triangles.
    min_area = float('inf')
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                # Area = 0.5 * |det(p2-p1, p3-p1)|
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                if area < min_area:
                    min_area = area
                    
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0
        
    if hull_area < 1e-10:
        return 0.0
        
    return min_area / hull_area

def generate_grid_init():
    # 4x4 grid
    xs = np.linspace(0, 1, 4)
    ys = np.linspace(0, 1, 4)
    grid = []
    for x in xs:
        for y in ys:
            grid.append([x, y])
    # Remove 3 points to get 13
    # Remove corners (0,0), (1,1), (0,1) ? Or center?
    # Let's remove 3 corners/edges to keep symmetry-ish
    # Indices: 
    # (0,0) idx 0, (0,1) idx 4, (0,2) idx 8, (0,3) idx 12
    # (1,0) idx 1, ...
    # Let's just remove the first 3 points
    grid = grid[3:]
    return np.array(grid)

def generate_random_init():
    return np.random.rand(N, 2)

def generate_circle_init():
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    # Add a center point? No, N=13.
    # Maybe 12 on circle, 1 in center?
    # Or 13 on circle?
    # Let's try 13 on circle
    return np.column_stack((np.cos(angles), np.sin(angles)))

def construct():
    start_time = time.time()
    timeout = 145.0 # 5s margin
    
    best_score = -1
    best_points = None
    
    # List of initializers
    inits = [
        generate_grid_init(),
        generate_random_init(),
        generate_circle_init(),
        # Perturbed grid
        generate_grid_init() + np.random.normal(0, 0.05, (13, 2)),
        # Another random
        generate_random_init()
    ]
    
    for init_pts in inits:
        if time.time() - start_time > timeout:
            break
            
        current_points = init_pts.copy()
        current_score = compute_score(current_points)
        
        if current_score > best_score:
            best_score = current_score
            best_points = current_points.copy()
            
        # Simulated Annealing
        T = 0.01
        T_min = 0.0001
        cooling = 0.995
        step_size = 0.05
        
        # Run SA for a fixed number of steps or time
        steps = 2000
        
        for step in range(steps):
            if time.time() - start_time > timeout:
                break
                
            # Random move
            idx = np.random.randint(N)
            delta = np.random.normal(0, step_size, 2)
            new_points = current_points.copy()
            new_points[idx] += delta
            
            new_score = compute_score(new_points)
            
            # Acceptance
            if new_score > current_score:
                current_points = new_points
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_points.copy()
            else:
                # Metropolis
                if np.random.random() < np.exp((new_score - current_score) / T):
                    current_points = new_points
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_points = current_points.copy()
            
            T *= cooling
            if T < T_min:
                T = T_min
                # Restart with higher T? Or just continue with small T
                # Let's reset T to restart exploration
                T = 0.005
                step_size = 0.02

        # Polish with Nelder-Mead
        if time.time() - start_time > timeout - 5:
            continue
            
        # Flatten points
        x0 = current_points.flatten()
        
        def objective(x):
            pts = x.reshape(N, 2)
            # Return negative score to maximize
            return -compute_score(pts)
            
        try:
            res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-6})
            if res.success:
                polished_points = res.x.reshape(N, 2)
                polished_score = compute_score(polished_points)
                if polished_score > best_score:
                    best_score = polished_score
                    best_points = polished_points
        except Exception:
            pass

    if best_points is None:
        # Fallback
        return np.random.rand(N, 2)
        
    return best_points
