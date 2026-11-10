# EVOLVE-BLOCK-START
import numpy as np
import scipy.spatial
import scipy.optimize
import time
import itertools

# Precompute all triangle indices for n=13
_N = 13
_TRI_INDICES = np.array(list(itertools.combinations(range(_N), 3)))

def _compute_score(points: np.ndarray) -> float:
    """
    Compute A_min / A_hull for a set of 13 points.
    """
    n = points.shape[0]
    if n != _N:
        return 0.0
    
    if not np.all(np.isfinite(points)):
        return 0.0
    
    # Compute convex hull area
    try:
        hull = scipy.spatial.ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 0.0
    
    if A_hull <= 1e-12:
        return 0.0
        
    # Compute all triangle areas vectorized
    p1 = points[_TRI_INDICES[:, 0]]
    p2 = points[_TRI_INDICES[:, 1]]
    p3 = points[_TRI_INDICES[:, 2]]
    
    # v1 = p2 - p1, v2 = p3 - p1
    v1 = p2 - p1
    v2 = p3 - p1
    
    # Cross product z-component: v1_x * v2_y - v1_y * v2_x
    cross_z = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross_z)
    
    min_area = np.min(areas)
    
    if min_area <= 1e-12:
        return 0.0
        
    return min_area / A_hull

def _local_search(points: np.ndarray, deadline: float) -> np.ndarray:
    """
    Perform local search (hill climbing) to improve the point configuration.
    Uses adaptive step size and multiple directions.
    """
    best_points = points.copy()
    best_score = _compute_score(best_points)
    n = len(best_points)
    
    # Initial step size
    step = 0.05
    min_step = 1e-7
    no_improve_count = 0
    max_no_improve = 15
    
    # Directions: 8 neighbors + 4 axis-aligned with larger steps
    directions = [
        [0, 1], [0, -1], [1, 0], [-1, 0],
        [1, 1], [1, -1], [-1, 1], [-1, -1]
    ]
    
    while time.time() < deadline - 20:
        improved = False
        for i in range(n):
            for dx, dy in directions:
                candidate = best_points.copy()
                candidate[i, 0] += dx * step
                candidate[i, 1] += dy * step
                
                if not np.all(np.isfinite(candidate)):
                    continue
                
                score = _compute_score(candidate)
                if score > best_score:
                    best_points = candidate
                    best_score = score
                    improved = True
                    no_improve_count = 0
                else:
                    no_improve_count += 1
        
        if not improved:
            step *= 0.7
            if step < min_step:
                break
        else:
            # Keep step size stable when improving
            pass
            
    return best_points

def _refine(points: np.ndarray, deadline: float) -> np.ndarray:
    """
    Use scipy.optimize.minimize to polish the solution.
    """
    n = len(points)
    
    def objective(flat_points):
        pts = flat_points.reshape(n, 2)
        if not np.all(np.isfinite(pts)):
            return 1e6
        score = _compute_score(pts)
        if score == 0.0:
            return 1e6
        return -score
    
    try:
        res = scipy.optimize.minimize(
            objective,
            points.flatten(),
            method='Powell',
            options={'maxiter': 300, 'ftol': 1e-12, 'xatol': 1e-12, 'disp': False}
        )
        candidate = res.x.reshape(n, 2)
        if _compute_score(candidate) > _compute_score(points):
            return candidate
    except Exception:
        pass
    
    return points

def _simulated_annealing(points: np.ndarray, deadline: float) -> np.ndarray:
    """
    Simulated annealing to escape local minima.
    """
    current_points = points.copy()
    current_score = _compute_score(current_points)
    best_points = current_points.copy()
    best_score = current_score
    
    temperature = 0.1
    cooling_rate = 0.95
    n = len(current_points)
    
    while time.time() < deadline - 30 and temperature > 1e-8:
        # Pick a random point and move it randomly
        i = np.random.randint(n)
        dx = np.random.randn() * temperature
        dy = np.random.randn() * temperature
        
        candidate = current_points.copy()
        candidate[i, 0] += dx
        candidate[i, 1] += dy
        
        if not np.all(np.isfinite(candidate)):
            temperature *= cooling_rate
            continue
        
        new_score = _compute_score(candidate)
        
        # Accept if better, or with probability exp((new - current) / temp)
        if new_score > current_score or np.random.rand() < np.exp((new_score - current_score) / max(temperature, 1e-10)):
            current_points = candidate
            current_score = new_score
            
            if new_score > best_score:
                best_points = current_points.copy()
                best_score = new_score
        
        temperature *= cooling_rate
    
    return best_points

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the ratio of the smallest
    triangle area to the convex hull area.
    
    Returns:
        points: np.ndarray of shape (13,2) of finite point coordinates.
    """
    n = 13
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    best_points = None
    best_score = 0.0
    
    # 1. Try structured initializations
    initializations = []
    
    # Regular polygon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    init_poly = np.column_stack([np.cos(angles), np.sin(angles)])
    initializations.append(init_poly)
    
    # Grid-like (4x4 grid, take 13 points)
    xs = np.linspace(-1, 1, 4)
    ys = np.linspace(-1, 1, 4)
    grid_pts = []
    for x in xs:
        for y in ys:
            grid_pts.append([x, y])
    init_grid = np.array(grid_pts[:n])
    initializations.append(init_grid)
    
    # Hexagonal packing-like
    hex_pts = []
    for row in range(3):
        for col in range(3):
            x = col * 1.0 + (0.5 if row % 2 else 0)
            y = row * 0.866
            hex_pts.append([x - 1.0, y - 0.866])
    # Add 4 more points to make 13
    for i in range(4):
        hex_pts.append([np.random.uniform(-1, 1), np.random.uniform(-1, 1)])
    init_hex = np.array(hex_pts[:n])
    initializations.append(init_hex)
    
    # 2. Run local search + refinement on structured starts
    for init in initializations:
        if time.time() > deadline - 120:
            break
            
        # Local search
        pts = _local_search(init, deadline)
        # Refine
        pts = _refine(pts, deadline)
        
        score = _compute_score(pts)
        if score > best_score:
            best_score = score
            best_points = pts.copy()
            
    # 3. Random restarts with simulated annealing
    rng = np.random.default_rng(42)
    restart_count = 0
    
    while time.time() < deadline - 60:
        # Generate random points in a square
        init_rand = rng.uniform(-1, 1, (n, 2))
        
        # Local search first to get to a local optimum
        pts = _local_search(init_rand, deadline)
        
        # Simulated annealing to potentially escape
        pts_sa = _simulated_annealing(pts, deadline)
        
        # Final refinement
        pts_final = _refine(pts_sa, deadline)
        
        score = _compute_score(pts_final)
        if score > best_score:
            best_score = score
            best_points = pts_final.copy()
            
        restart_count += 1
        
    # 4. If we have time, do a final polish on the best solution
    if best_points is not None and time.time() < deadline - 30:
        best_points = _refine(best_points, deadline)
        best_score = _compute_score(best_points)
        
    if best_points is None:
        # Fallback
        best_points = init_poly
        
    return best_points

# EVOLVE-BLOCK-END
