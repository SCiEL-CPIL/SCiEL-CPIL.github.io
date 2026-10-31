import numpy as np
from scipy.optimize import minimize, basinhopping
import time

N = 11
SQRT3 = np.sqrt(3)
H = SQRT3 / 2.0

# Precompute indices for all triplets (i, j, k) with i < j < k
idx_i, idx_j, idx_k = [], [], []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            idx_i.append(i)
            idx_j.append(j)
            idx_k.append(k)
idx_i = np.array(idx_i)
idx_j = np.array(idx_j)
idx_k = np.array(idx_k)

def compute_min_area(points):
    """
    Vectorized calculation of the minimum area among all triangles formed by points.
    """
    # points shape: (N, 2)
    # Get vectors for all triplets
    pi = points[idx_i]
    pj = points[idx_j]
    pk = points[idx_k]
    
    # Vector differences
    v1 = pj - pi
    v2 = pk - pi
    
    # Cross product (2D)
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def penalty(points):
    """
    Calculate penalty for points outside the triangle.
    Triangle vertices: (0,0), (1,0), (0.5, H)
    Inequalities:
    1. y >= 0
    2. y <= SQRT3 * x  (Line from (0,0) to (0.5, H) is y = sqrt(3)x? No.
       Line from (0,0) to (1,0) is y=0.
       Line from (0,0) to (0.5, H) is y = (H/0.5)x = 2H x = SQRT3 x.
       Line from (1,0) to (0.5, H) is y - 0 = (H - 0)/(0.5 - 1) (x - 1) => y = -SQRT3 (x - 1) => y = -SQRT3 x + SQRT3.
    """
    x = points[:, 0]
    y = points[:, 1]
    
    # d1: distance to y=0 (inside is y >= 0)
    d1 = y
    
    # d2: distance to y = SQRT3 * x (inside is y <= SQRT3 * x)
    # Normal vector pointing inward? 
    # Let's use signed distance or just linear constraint violation.
    # Violation if y > SQRT3 * x.
    d2 = SQRT3 * x - y
    
    # d3: distance to y = -SQRT3 * x + SQRT3 (inside is y <= -SQRT3 * x + SQRT3)
    d3 = -SQRT3 * x + SQRT3 - y
    
    # Sum of squared violations
    pen = np.sum(np.maximum(0, -d1)**2) + np.sum(np.maximum(0, -d2)**2) + np.sum(np.maximum(0, -d3)**2)
    return pen

def objective(flat_params):
    points = flat_params.reshape(N, 2)
    min_area = compute_min_area(points)
    pen = penalty(points)
    # Strong penalty to ensure feasibility
    return -min_area + 1000.0 * pen

def construct():
    start_time = time.time()
    time_limit = 110.0
    
    best_points = None
    best_score = -np.inf
    
    # Define initializations
    inits = []
    
    # 1. Vertices + Random
    for seed in range(20):
        rng = np.random.RandomState(seed)
        init = np.zeros((N, 2))
        # Place vertices
        init[0] = [0, 0]
        init[1] = [1, 0]
        init[2] = [0.5, H]
        # Place rest randomly
        for i in range(3, N):
            while True:
                u = rng.rand()
                v = rng.rand()
                if u + v <= 1:
                    break
            w = 1 - u - v
            init[i, 0] = v * 1.0 + w * 0.5
            init[i, 1] = w * H
        inits.append(init)
        
    # 2. Grid-like initialization
    # Try to spread points evenly
    # Simple heuristic: place points in rows
    init_grid = np.zeros((N, 2))
    # Row 1 (bottom): 4 points
    # Row 2: 3 points
    # Row 3: 2 points
    # Row 4 (top): 1 point (vertex)
    # This is a rough guess, let's just use a jittered grid
    
    # Let's try a specific known good configuration heuristic:
    # Place points near the center and vertices.
    
    # 3. Pure Random (without fixed vertices, to allow optimization to move them)
    for seed in range(20):
        rng = np.random.RandomState(seed + 100)
        init = np.zeros((N, 2))
        for i in range(N):
            while True:
                u = rng.rand()
                v = rng.rand()
                if u + v <= 1:
                    break
            w = 1 - u - v
            init[i, 0] = v * 1.0 + w * 0.5
            init[i, 1] = w * H
        inits.append(init)

    for init_points in inits:
        if time.time() - start_time > time_limit:
            break
        
        flat_init = init_points.flatten()
        
        try:
            # Use Powell for local refinement
            result = minimize(
                objective, 
                flat_init, 
                method='Powell', 
                options={'maxiter': 100, 'maxfev': 5000, 'disp': False}
            )
            
            opt_points = result.x.reshape(N, 2)
            min_area = compute_min_area(opt_points)
            pen = penalty(opt_points)
            
            if pen < 1e-6:
                score = min_area
            else:
                score = min_area - 1000 * pen
            
            if score > best_score:
                best_score = score
                best_points = opt_points.copy()
                
        except Exception:
            pass

    if best_points is not None and time.time() - start_time < time_limit - 10:
        # Basinhopping for global improvement
        try:
            bh_result = basinhopping(
                objective,
                best_points.flatten(),
                minimizer_kwargs={'method': 'Powell', 'maxiter': 50, 'maxfev': 2000},
                niter=15,
                seed=42
            )
            bh_points = bh_result.x.reshape(N, 2)
            bh_min_area = compute_min_area(bh_points)
            bh_pen = penalty(bh_points)
            
            if bh_pen < 1e-6:
                bh_score = bh_min_area
            else:
                bh_score = bh_min_area - 1000 * bh_pen
                
            if bh_score > best_score:
                best_score = bh_score
                best_points = bh_points.copy()
        except Exception:
            pass

    if best_points is None:
        # Fallback
        return np.array([[0.5, H/3]] * N)
        
    return best_points
