import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

N = 11
H = np.sqrt(3) / 2
S3 = np.sqrt(3)

# Precompute indices for vectorized area calculation
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

def min_triangle_area_vec(points):
    p1 = points[idx_i]
    p2 = points[idx_j]
    p3 = points[idx_k]
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def is_inside(points):
    x = points[:, 0]
    y = points[:, 1]
    if np.any(y < -1e-9):
        return False
    if np.any(y > S3 * x + 1e-9):
        return False
    if np.any(y > S3 * (1 - x) + 1e-9):
        return False
    return True

def centroid():
    return np.array([0.5, S3 / 6])

def rotate120(p, c):
    """Rotate point p by 120 degrees around center c."""
    dx = p[0] - c[0]
    dy = p[1] - c[1]
    # 120 degree rotation matrix
    cos120 = -0.5
    sin120 = S3 / 2
    rx = cos120 * dx - sin120 * dy
    ry = sin120 * dx + cos120 * dy
    return np.array([c[0] + rx, c[1] + ry])

def build_symmetric_points(params):
    """
    params: 10 values
    - First 6: (r1, theta1, r2, theta2, r3, theta3) for 3 orbits of 3 points
    - Last 4: (x4, y4, x5, y5) for 2 free points
    
    Returns 11 points.
    """
    c = centroid()
    points = []
    
    # Point 0: centroid
    points.append(c.copy())
    
    # Orbits 1-3: 3 points each
    for i in range(3):
        r = params[2*i]
        theta = params[2*i + 1]
        # Generate 3 points at 120 degree rotations
        for k in range(3):
            angle = theta + k * 2 * np.pi / 3
            px = c[0] + r * np.cos(angle)
            py = c[1] + r * np.sin(angle)
            points.append(np.array([px, py]))
    
    # 2 free points
    points.append(np.array([params[6], params[7]]))
    points.append(np.array([params[8], params[9]]))
    
    return np.array(points)

def objective_sym(params):
    """Objective for symmetric parameterization."""
    points = build_symmetric_points(params)
    
    if not is_inside(points):
        return 1.0
    
    # Check for near-coincident points (degenerate triangles)
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(points[i] - points[j])
            if d < 1e-6:
                return 1.0
    
    area = min_triangle_area_vec(points)
    return -area

def construct():
    start_time = time.time()
    timeout = 140
    
    best_points = None
    best_score = -float('inf')
    
    # Bounds for symmetric parameters
    # r in [0, 0.5], theta in [0, 2*pi]
    # Free points: x in [0,1], y in [0,H]
    bounds_sym = [
        (0.0, 0.5), (0.0, 2*np.pi),  # orbit 1
        (0.0, 0.5), (0.0, 2*np.pi),  # orbit 2
        (0.0, 0.5), (0.0, 2*np.pi),  # orbit 3
        (0.0, 1.0), (0.0, H),        # free point 1
        (0.0, 1.0), (0.0, H),        # free point 2
    ]
    
    # Try multiple DE runs with different seeds
    for seed in [42, 123, 456]:
        if time.time() - start_time > timeout - 30:
            break
        
        try:
            result = differential_evolution(
                objective_sym,
                bounds_sym,
                maxiter=300,
                popsize=12,
                tol=0.01,
                seed=seed,
                polish=True,
                init='sobol'
            )
            
            if result.x is not None:
                points = build_symmetric_points(result.x)
                if is_inside(points):
                    # Check for near-coincident points
                    valid = True
                    for i in range(N):
                        for j in range(i+1, N):
                            if np.linalg.norm(points[i] - points[j]) < 1e-6:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        score = min_triangle_area_vec(points)
                        if score > best_score:
                            best_score = score
                            best_points = points.copy()
                            
        except Exception:
            pass
    
    if best_points is None:
        # Fallback: simple valid configuration
        c = centroid()
        pts = [c]
        for i in range(3):
            r = 0.15
            theta = i * 2 * np.pi / 3
            pts.append(c + r * np.array([np.cos(theta), np.sin(theta)]))
            pts.append(rotate120(pts[-1], c))
            pts.append(rotate120(pts[-1], c))
        # Add 2 more points
        pts.append(np.array([0.25, S3/6]))
        pts.append(np.array([0.75, S3/6]))
        best_points = np.array(pts)
        # Ensure validity
        if not is_inside(best_points):
            # Clip to triangle
            for i in range(N):
                x, y = best_points[i]
                if y < 0: y = 0
                if y > S3 * x: y = S3 * x
                if y > S3 * (1-x): y = S3 * (1-x)
                best_points[i] = np.array([x, y])
        return best_points
    
    # Local refinement
    if time.time() - start_time < timeout - 10:
        def refine_local(points, deadline):
            best_pts = points.copy()
            best_sc = min_triangle_area_vec(best_pts)
            rng = np.random.default_rng(42)
            
            for attempt in range(10):
                if time.time() > deadline:
                    break
                
                if attempt == 0:
                    x0 = best_pts.flatten()
                else:
                    x0 = best_pts.flatten() + rng.normal(0, 0.005, size=2*N)
                    # Project inside
                    for i in range(N):
                        x, y = x0[2*i], x0[2*i+1]
                        if y < 0: y = 0
                        if y > S3 * x: y = S3 * x
                        if y > S3 * (1-x): y = S3 * (1-x)
                        x0[2*i] = x
                        x0[2*i+1] = y
                
                def obj_local(params):
                    pts = params.reshape(N, 2)
                    if not is_inside(pts):
                        return 1.0
                    return -min_triangle_area_vec(pts)
                
                try:
                    res = minimize(obj_local, x0, method='Nelder-Mead', 
                                   options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-11})
                    if res.x is not None:
                        pts = res.x.reshape(N, 2)
                        if is_inside(pts):
                            sc = min_triangle_area_vec(pts)
                            if sc > best_sc:
                                best_sc = sc
                                best_pts = pts.copy()
                except Exception:
                    pass
            
            return best_pts
        
        best_points = refine_local(best_points, timeout)
    
    return best_points
