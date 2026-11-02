import numpy as np
import time
from scipy.optimize import minimize

N = 11
SQRT3 = np.sqrt(3)
TRI_AREA = SQRT3 / 4.0  # Area of unit equilateral triangle
DEADLINE = 110.0  # Safety margin from 120s limit

def get_cartesian(params):
    # params shape (2N,)
    s = params[0::2]
    t = params[1::2]
    # u = (1-t)(1-s), v = t, w = s(1-t)
    w = s * (1 - t)
    v = t
    # x = v + 0.5 w
    # y = SQRT3/2 * w
    x = v + 0.5 * w
    y = (SQRT3 / 2) * w
    return np.column_stack((x, y))

def compute_areas(points):
    # points shape (N, 2)
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                xi, yi = points[i]
                xj, yj = points[j]
                xk, yk = points[k]
                a = 0.5 * abs((xj - xi) * (yk - yi) - (xk - xi) * (yj - yi))
                areas.append(a)
    return np.array(areas)

def min_area_normalized(points):
    """Return the minimum triangle area normalized by triangle area."""
    areas = compute_areas(points)
    return np.min(areas) / TRI_AREA

def hexagonal_initialization(n):
    """Generate hexagonal grid points inside the triangle."""
    points = []
    # Hexagonal packing in triangular region
    # Use rows of hexagonal points
    for row in range(4):
        y_frac = row * 0.25
        x_start = 0.1 if row % 2 == 0 else 0.15
        for col in range(3):
            x_frac = x_start + col * 0.25
            # Check if point is inside triangle
            x = x_frac
            y = y_frac * SQRT3 / 2
            # Triangle vertices: (0,0), (1,0), (0.5, sqrt(3)/2)
            # Check barycentric coords
            if x >= 0 and x <= 1 and y >= 0 and y <= SQRT3/2:
                # Check if inside triangle
                # y <= SQRT3 * x for left edge
                # y <= SQRT3 * (1-x) for right edge
                if y <= SQRT3 * x + 1e-9 and y <= SQRT3 * (1 - x) + 1e-9:
                    points.append([x, y])
    
    # Add remaining points randomly if we don't have enough
    while len(points) < n:
        x = np.random.rand()
        y = np.random.rand() * SQRT3 / 2
        if y <= SQRT3 * x + 1e-9 and y <= SQRT3 * (1 - x) + 1e-9:
            points.append([x, y])
    
    points = np.array(points[:n])
    return points

def simulated_annealing(start_points, max_iter=5000, t_start=0.1, t_end=0.001, deadline=DEADLINE):
    """Simulated annealing to maximize minimum triangle area."""
    current_points = start_points.copy()
    current_score = min_area_normalized(current_points)
    best_points = current_points.copy()
    best_score = current_score
    
    n = len(current_points)
    start_time = time.time()
    
    for i in range(max_iter):
        if time.time() - start_time > deadline:
            break
        
        # Temperature schedule
        progress = i / max_iter
        temperature = t_start * (t_end / t_start) ** progress
        
        # Generate perturbation: move one point randomly
        idx = np.random.randint(n)
        dx = np.random.randn() * temperature
        dy = np.random.randn() * temperature * 0.5  # Scale y due to triangle shape
        
        new_points = current_points.copy()
        new_points[idx] = [current_points[idx][0] + dx, current_points[idx][1] + dy]
        
        # Check if new point is inside triangle
        x, y = new_points[idx]
        if x < -1e-9 or x > 1 + 1e-9 or y < -1e-9 or y > SQRT3/2 + 1e-9:
            continue
        if y > SQRT3 * x + 1e-9 or y > SQRT3 * (1 - x) + 1e-9:
            continue
        
        new_score = min_area_normalized(new_points)
        delta = new_score - current_score
        
        # Accept if better, or with probability exp(delta/temperature)
        if delta > 0 or np.random.rand() < np.exp(delta / temperature):
            current_points = new_points
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
    
    return best_points, best_score

def construct():
    start_time = time.time()
    
    # Try multiple initialization strategies
    initializations = []
    
    # Hexagonal grid
    np.random.seed(42)
    initializations.append(hexagonal_initialization(N))
    
    # Random
    for _ in range(3):
        pts = np.random.rand(N, 2)
        pts[:, 1] *= SQRT3 / 2
        # Filter to inside triangle
        mask = (pts[:, 1] <= SQRT3 * pts[:, 0] + 1e-9) & (pts[:, 1] <= SQRT3 * (1 - pts[:, 0]) + 1e-9)
        valid_pts = pts[mask]
        while len(valid_pts) < N:
            x = np.random.rand()
            y = np.random.rand() * SQRT3 / 2
            if y <= SQRT3 * x + 1e-9 and y <= SQRT3 * (1 - x) + 1e-9:
                valid_pts = np.vstack([valid_pts, [[x, y]]])
        initializations.append(valid_pts[:N])
    
    # Run simulated annealing on each initialization
    best_points = None
    best_score = -np.inf
    
    for init_pts in initializations:
        if time.time() - start_time > DEADLINE - 10:
            break
        
        sa_points, sa_score = simulated_annealing(
            init_pts, 
            max_iter=3000, 
            t_start=0.15, 
            t_end=0.0005,
            deadline=min(DEADLINE - 5, time.time() + 15)
        )
        
        if sa_score > best_score:
            best_score = sa_score
            best_points = sa_points
    
    # Final local optimization with L-BFGS-B on the best result
    if best_points is not None and time.time() - start_time < DEADLINE - 20:
        # Convert to barycentric-like parameters for optimization
        def objective_local(params):
            points = get_cartesian(params)
            # Check validity
            x = points[:, 0]
            y = points[:, 1]
            valid = (x >= -1e-9) & (x <= 1 + 1e-9) & (y >= -1e-9) & (y <= SQRT3/2 + 1e-9)
            valid &= (y <= SQRT3 * x + 1e-9) & (y <= SQRT3 * (1 - x) + 1e-9)
            if not np.all(valid):
                return 1000.0
            areas = compute_areas(points)
            return -np.min(areas)
        
        # Initialize parameters from best_points
        # Invert get_cartesian: x = v + 0.5*w, y = SQRT3/2 * w where w = s*(1-t), v = t
        # y = SQRT3/2 * s * (1-t) => s*(1-t) = 2y/SQRT3
        # x = t + 0.5*s*(1-t) => t = x - 0.5*s*(1-t) = x - y/SQRT3
        # So t = x - y/SQRT3, s*(1-t) = 2y/SQRT3 => s = (2y/SQRT3)/(1-t)
        
        t_vals = best_points[:, 0] - best_points[:, 1] / SQRT3
        s_vals = np.zeros(N)
        for i in range(N):
            one_minus_t = 1 - t_vals[i]
            if one_minus_t > 1e-9:
                s_vals[i] = (2 * best_points[i][1] / SQRT3) / one_minus_t
            else:
                s_vals[i] = 0
        
        params = np.zeros(2 * N)
        params[0::2] = np.clip(s_vals, 0, 1)
        params[1::2] = np.clip(t_vals, 0, 1)
        
        res = minimize(objective_local, params, method='L-BFGS-B',
                       bounds=[(0, 1)] * (2 * N),
                       options={'maxiter': 200, 'time_limit': min(20, DEADLINE - (time.time() - start_time))})
        
        if res.success:
            local_points = get_cartesian(res.x)
            local_score = min_area_normalized(local_points)
            if local_score > best_score:
                best_score = local_score
                best_points = local_points
    
    # Return best points
    if best_points is not None:
        return best_points
    else:
        # Fallback
        return hexagonal_initialization(N)
