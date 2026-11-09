import numpy as np
from scipy.optimize import minimize, LinearConstraint
import time
import itertools

N = 11
h = np.sqrt(3) / 2
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, h])

def project_to_triangle(p):
    """
    p: array of shape (N, 2)
    Returns projected points inside the triangle (0,0), (1,0), (0.5, h)
    """
    p = p.copy()
    # Clip to bounding box first
    p[:, 0] = np.clip(p[:, 0], 0.0, 1.0)
    p[:, 1] = np.clip(p[:, 1], 0.0, h)
    
    # Iterate a few times to handle corners properly
    for _ in range(5):
        # Edge AB: y >= 0
        mask_ab = p[:, 1] < 0.0
        if np.any(mask_ab):
            p[mask_ab, 1] = 0.0
            
        # Edge AC: y <= sqrt(3) x
        mask_ac = p[:, 1] > np.sqrt(3) * p[:, 0]
        if np.any(mask_ac):
            px = p[mask_ac, 0]
            py = p[mask_ac, 1]
            dot = px + py * np.sqrt(3)
            px_proj = dot / 4.0
            py_proj = dot * np.sqrt(3) / 4.0
            px_proj = np.clip(px_proj, 0.0, 0.5)
            py_proj = px_proj * np.sqrt(3)
            p[mask_ac, 0] = px_proj
            p[mask_ac, 1] = py_proj
            
        # Edge BC: y <= sqrt(3)(1-x)
        mask_bc = p[:, 1] > np.sqrt(3) * (1.0 - p[:, 0])
        if np.any(mask_bc):
            px = p[mask_bc, 0]
            py = p[mask_bc, 1]
            d = (np.sqrt(3) * px + py - np.sqrt(3)) / 4.0
            px_proj = px - d * np.sqrt(3)
            py_proj = py - d * 1.0
            px_proj = np.clip(px_proj, 0.5, 1.0)
            py_proj = np.sqrt(3) * (1.0 - px_proj)
            p[mask_bc, 0] = px_proj
            p[mask_bc, 1] = py_proj
            
    return p

def min_triangle_area(points):
    """
    points: (N, 2)
    Returns the minimum area of any triangle formed by 3 points.
    Uses vectorized operations for speed.
    """
    n = len(points)
    # Create indices
    indices = np.arange(n)
    i_idx, j_idx, k_idx = np.meshgrid(indices[:-2], indices[1:-1], indices[2:], indexing='ij')
    # Filter valid triples i < j < k
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_valid = i_idx[mask]
    j_valid = j_idx[mask]
    k_valid = k_idx[mask]
    
    p1 = points[i_valid]
    p2 = points[j_valid]
    p3 = points[k_valid]
    
    v12 = p2 - p1
    v13 = p3 - p1
    
    crosses = v12[:, 0] * v13[:, 1] - v12[:, 1] * v13[:, 0]
    areas = 0.5 * np.abs(crosses)
    
    if len(areas) == 0:
        return 0.0
    return np.min(areas)

def objective(x):
    """
    x: flattened array of shape (2*N,)
    Returns negative min area.
    """
    points = x.reshape(N, 2)
    return -min_triangle_area(points)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_score = -1
    best_points = None
    
    rng = np.random.default_rng(42)
    
    # Define constraints for the triangle
    # y >= 0
    # y <= sqrt(3) x  =>  -sqrt(3) x + y <= 0
    # y <= sqrt(3)(1-x) => sqrt(3) x + y <= sqrt(3)
    
    # Linear constraints: A_ub x <= b_ub
    # For each point i (2 variables x_i, y_i)
    A_ub = []
    b_ub = []
    
    for i in range(N):
        # y_i >= 0 => -y_i <= 0
        row = np.zeros(2 * N)
        row[2 * i + 1] = -1.0
        A_ub.append(row)
        b_ub.append(0.0)
        
        # -sqrt(3) x_i + y_i <= 0
        row = np.zeros(2 * N)
        row[2 * i] = -np.sqrt(3)
        row[2 * i + 1] = 1.0
        A_ub.append(row)
        b_ub.append(0.0)
        
        # sqrt(3) x_i + y_i <= sqrt(3)
        row = np.zeros(2 * N)
        row[2 * i] = np.sqrt(3)
        row[2 * i + 1] = 1.0
        A_ub.append(row)
        b_ub.append(np.sqrt(3))
        
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # Bounds: x in [0, 1], y in [0, h]
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, h))
    
    # Initial guesses
    guesses = []
    
    # Guess 1: Symmetric rows
    y_levels = [0.1*h, 0.35*h, 0.6*h, 0.85*h]
    counts = [5, 3, 2, 1]
    pts = []
    for y, c in zip(y_levels, counts):
        x_min = y / np.sqrt(3)
        x_max = 1.0 - y / np.sqrt(3)
        if c > 1:
            margin = 0.05 * (x_max - x_min)
            xs = np.linspace(x_min + margin, x_max - margin, c)
        else:
            xs = np.array([(x_min + x_max) / 2])
        for x in xs:
            pts.append([x, y])
    guesses.append(np.array(pts))
    
    # Guess 2: Random
    guesses.append(rng.random((N, 2)) * np.array([1.0, h]))
    guesses.append(rng.random((N, 2)) * np.array([1.0, h]))
    
    # Optimization loop
    for i, start_points in enumerate(guesses):
        if time.time() > deadline - 10:
            break
            
        start_points = project_to_triangle(start_points)
        x0 = start_points.flatten()
        
        try:
            # Use trust-constr for better handling of constraints
            res = minimize(objective, x0, method='trust-constr', 
                           bounds=bounds, 
                           constraints={'type': 'ineq', 'fun': lambda x: b_ub - A_ub @ x},
                           options={'maxiter': 50, 'gtol': 1e-4, 'xtol': 1e-4})
            
            curr_points = project_to_triangle(res.x.reshape(N, 2))
            curr_area = min_triangle_area(curr_points)
            if curr_area > best_score:
                best_score = curr_area
                best_points = curr_points
        except Exception:
            pass
            
    # Additional random restarts if time permits
    if time.time() < deadline - 20:
        rng2 = np.random.default_rng(123)
        for _ in range(10):
            if time.time() > deadline - 5:
                break
            start_points = rng2.random((N, 2)) * np.array([1.0, h])
            start_points = project_to_triangle(start_points)
            x0 = start_points.flatten()
            
            try:
                res = minimize(objective, x0, method='trust-constr', 
                               bounds=bounds, 
                               constraints={'type': 'ineq', 'fun': lambda x: b_ub - A_ub @ x},
                               options={'maxiter': 30, 'gtol': 1e-3, 'xtol': 1e-3})
                
                curr_points = project_to_triangle(res.x.reshape(N, 2))
                curr_area = min_triangle_area(curr_points)
                if curr_area > best_score:
                    best_score = curr_area
                    best_points = curr_points
            except Exception:
                pass
                
    # Final local refinement with Nelder-Mead if we have a good candidate
    if best_points is not None and time.time() < deadline - 5:
        x0 = best_points.flatten()
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-6})
            curr_points = project_to_triangle(res.x.reshape(N, 2))
            curr_area = min_triangle_area(curr_points)
            if curr_area > best_score:
                best_score = curr_area
                best_points = curr_points
        except Exception:
            pass
            
    if best_points is None:
        # Fallback
        pts = []
        y_levels = [0.08*h, 0.35*h, 0.62*h, 0.88*h]
        counts = [5, 3, 2, 1]
        for y, c in zip(y_levels, counts):
            x_min = y / np.sqrt(3)
            x_max = 1.0 - y / np.sqrt(3)
            if c > 1:
                margin = 0.03 * (x_max - x_min)
                xs = np.linspace(x_min + margin, x_max - margin, c)
            else:
                xs = np.array([(x_min + x_max) / 2])
            for x in xs:
                pts.append([x, y])
        best_points = np.array(pts)
        
    return best_points
