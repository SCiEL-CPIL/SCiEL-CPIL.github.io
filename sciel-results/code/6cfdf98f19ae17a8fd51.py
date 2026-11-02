"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _area_matrix(points):
    """Compute all triangle areas from n points. Returns array of shape (n, n, n) with area[i,j,k]."""
    n = len(points)
    areas = np.zeros((n, n, n))
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a = points[i]
                b = points[j]
                c = points[k]
                area = 0.5 * abs((b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]))
                areas[i,j,k] = area
                areas[j,i,k] = area
                areas[j,k,i] = area
                areas[i,k,j] = area
                areas[k,i,j] = area
                areas[k,j,i] = area
    return areas

def _min_area(points):
    """Compute minimum triangle area."""
    n = len(points)
    min_a = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a = points[i]
                b = points[j]
                c = points[k]
                area = 0.5 * abs((b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]))
                if area < min_a:
                    min_a = area
    return min_a

def _softmin_objective(flat_points, temperature=0.01):
    """Negative soft-min of triangle areas. We minimize this to maximize soft-min."""
    n = len(flat_points) // 2
    points = flat_points.reshape(n, 2)
    
    # Compute all triangle areas
    min_a = float('inf')
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a = points[i]
                b = points[j]
                c = points[k]
                area = 0.5 * abs((b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]))
                areas.append(area)
    
    areas = np.array(areas)
    
    # Soft-min using log-sum-exp
    # softmin(x) = -logsumexp(-x/t) * t
    # To avoid overflow, use max trick
    neg_areas = -areas / temperature
    max_neg = np.max(neg_areas)
    logsumexp_val = max_neg + np.log(np.sum(np.exp(neg_areas - max_neg)))
    softmin_val = -logsumexp_val * temperature
    
    # Also add penalty if points go outside triangle
    # Triangle: (0,0), (1,0), (0.5, sqrt(3)/2)
    # Constraints: y >= 0, y <= sqrt(3)/2 * x, y <= sqrt(3)/2 * (1-x)
    penalty = 0.0
    h = np.sqrt(3)/2
    for p in points:
        x, y = p
        # y >= 0
        if y < 0:
            penalty += (100 - y) ** 2
        # y <= h * x
        if y > h * x:
            penalty += (y - h * x) ** 2
        # y <= h * (1-x)
        if y > h * (1 - x):
            penalty += (y - h * (1 - x)) ** 2
        # x in [0,1]
        if x < 0:
            penalty += x ** 2
        if x > 1:
            penalty += (x - 1) ** 2
    
    return -softmin_val + penalty

def _in_triangle(point, eps=1e-9):
    x, y = point
    h = np.sqrt(3)/2
    return (y >= -eps) and (x >= -eps) and (x <= 1+eps) and (y <= h*x + eps) and (y <= h*(1-x) + eps)

def _clamp_to_triangle(points):
    """Clamp points to be inside the triangle."""
    h = np.sqrt(3)/2
    clamped = []
    for p in points:
        x, y = p
        # Project to triangle if outside
        # Simple approach: clamp and check
        if y < 0:
            y = 0
        if x < 0:
            x = 0
        if x > 1:
            x = 1
        if y > h * x:
            # Project onto line y = h*x
            # Closest point on line y=hx to (x,y)
            # Line: hx - y = 0, normal (h, -1)
            dist = (h*x - y) / (h*h + 1)
            x = x - dist * h
            y = y + dist
        if y > h * (1-x):
            # Project onto line y = h*(1-x) => hx + y = h
            dist = (h*x + y - h) / (h*h + 1)
            x = x - dist * h
            y = y - dist
        clamped.append([x, y])
    return np.array(clamped)

def _generate_initial_configs():
    """Generate multiple good initial configurations."""
    configs = []
    h = np.sqrt(3)/2
    
    # Config 1: 3 vertices + 3 edge midpoints + 5 interior (pentagon-like)
    # Vertices
    v1 = [0, 0]
    v2 = [1, 0]
    v3 = [0.5, h]
    # Edge midpoints
    m1 = [0.5, 0]
    m2 = [0.75, h/2]
    m3 = [0.25, h/2]
    # Interior: center and 4 points
    cx, cy = 0.5, h/3  # centroid
    configs.append([v1, v2, v3, m1, m2, m3, [cx, cy],
                    [0.5, h/6], [0.5, h/2], [0.25, h/3], [0.75, h/3]])
    
    # Config 2: More symmetric interior
    # 3 vertices, 3 near-edge, 5 interior in hexagonal pattern
    configs.append([v1, v2, v3,
                    [0.3, 0.1], [0.7, 0.1], [0.5, 0.3],
                    [0.4, 0.2], [0.6, 0.2], [0.5, 0.4],
                    [0.35, 0.35], [0.65, 0.35]])
    
    # Config 3: Based on known good arrangements
    # Use a grid-like pattern
    pts = []
    # Bottom row
    for i in range(4):
        x = 0.1 + i * 0.2
        y = 0.05
        pts.append([x, y])
    # Middle row
    for i in range(3):
        x = 0.2 + i * 0.2
        y = 0.25
        pts.append([x, y])
    # Top
    pts.append([0.5, 0.5])
    pts.append([0.3, 0.4])
    configs.append(pts)
    
    # Config 4: Random but constrained
    import random
    rng = random.Random(42)
    for trial in range(20):
        pts = []
        for _ in range(N):
            while True:
                u, v = rng.random(), rng.random()
                if u + v < 0.95:
                    x = u + 0.5 * v
                    y = h * v
                    pts.append([x, y])
                    break
        configs.append(pts)
    
    return configs

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5 second safety margin
    
    best_points = None
    best_score = -float('inf')
    
    # Get initial configurations
    configs = _generate_initial_configs()
    
    # First, evaluate all initial configs
    for cfg in configs:
        pts = np.array(cfg, dtype=float)
        if len(pts) != N:
            continue
        score = _min_area(pts)
        if score > best_score:
            best_score = score
            best_points = pts.copy()
    
    # Now refine each config with optimization
    for cfg_idx, cfg in enumerate(configs):
        if time.time() > deadline - 5:
            break
        
        pts = np.array(cfg, dtype=float)
        if len(pts) != N:
            continue
        
        flat = pts.flatten()
        
        # Try optimization with different temperatures
        for temp in [0.05, 0.02, 0.01, 0.005]:
            if time.time() > deadline - 5:
                break
            
            try:
                result = minimize(
                    _softmin_objective,
                    flat,
                    args=(temp,),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                if result.success or result.fun < _softmin_objective(flat, temp):
                    opt_pts = result.x.reshape(N, 2)
                    # Clamp to triangle
                    opt_pts = _clamp_to_triangle(opt_pts)
                    score = _min_area(opt_pts)
                    if score > best_score:
                        best_score = score
                        best_points = opt_pts.copy()
            except Exception:
                pass
        
        # Also try a direct local search / gradient-free refinement
        if time.time() > deadline - 5:
            break
        
        # Simple hill climbing: try small perturbations
        current = np.array(cfg, dtype=float)
        current_score = _min_area(current)
        
        step_size = 0.05
        for iteration in range(200):
            if time.time() > deadline - 5:
                break
            
            improved = False
            for i in range(N):
                for dx in [-step_size, 0, step_size]:
                    for dy in [-step_size, 0, step_size]:
                        if dx == 0 and dy == 0:
                            continue
                        new_pts = current.copy()
                        new_pts[i, 0] += dx
                        new_pts[i, 1] += dy
                        if _in_triangle(new_pts[i]):
                            s = _min_area(new_pts)
                            if s > current_score:
                                current = new_pts
                                current_score = s
                                improved = True
            
            if not improved:
                step_size *= 0.7
                if step_size < 0.001:
                    break
        
        if current_score > best_score:
            best_score = current_score
            best_points = current.copy()
    
    # Final polish: use scipy with tight tolerances on the best solution
    if best_points is not None and time.time() < deadline - 10:
        flat = best_points.flatten()
        for temp in [0.002, 0.001, 0.0005]:
            if time.time() > deadline - 5:
                break
            try:
                result = minimize(
                    _softmin_objective,
                    flat,
                    args=(temp,),
                    method='L-BFGS-B',
                    options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if result.success:
                    opt_pts = result.x.reshape(N, 2)
                    opt_pts = _clamp_to_triangle(opt_pts)
                    score = _min_area(opt_pts)
                    if score > best_score:
                        best_score = score
                        best_points = opt_pts.copy()
                        flat = best_points.flatten()
            except Exception:
                pass
    
    if best_points is None:
        # Fallback: return a valid configuration
        h = np.sqrt(3)/2
        best_points = np.array([[0.5, h/3]] * N)
    
    # Final validation and clamping
    best_points = _clamp_to_triangle(best_points)
    
    return best_points
# EVOLVE-BLOCK-END
