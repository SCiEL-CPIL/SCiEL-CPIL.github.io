import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

N = 11
H = np.sqrt(3) / 2
S3 = np.sqrt(3)

# Pre-compute indices for all 165 triangles
tri_indices = list(combinations(range(N), 3))
tri_indices = np.array(tri_indices)
I = tri_indices[:, 0]
J = tri_indices[:, 1]
K = tri_indices[:, 2]

def _min_area(points):
    """
    points: (N, 2)
    Returns the minimum area of any triangle formed by 3 points.
    """
    Pi = points[I]
    Pj = points[J]
    Pk = points[K]
    
    v1 = Pj - Pi
    v2 = Pk - Pi
    
    # Cross product z-component
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _penalty(points):
    """
    Calculate penalty for points outside the unit equilateral triangle.
    Triangle vertices: (0,0), (1,0), (0.5, H)
    """
    x = points[:, 0]
    y = points[:, 1]
    
    # Distance to boundary violations
    v1 = np.maximum(0, -y)
    v2 = np.maximum(0, y - S3 * x)
    v3 = np.maximum(0, y - S3 * (1 - x))
    
    return np.sum(v1**2 + v2**2 + v3**2)

def _objective(points):
    """
    Objective to maximize: min_area - penalty
    Returns negative for minimization algorithms if needed, 
    but here we use it directly for SA (maximization).
    """
    area = _min_area(points)
    pen = _penalty(points)
    return area - 1000.0 * pen

def _random_valid_point(rng):
    """
    Generate a random point inside the triangle.
    """
    while True:
        u = rng.random()
        v = rng.random()
        if u + v < 1:
            x = u + 0.5 * v
            y = H * v
            return np.array([x, y])

def _initial_population(rng, count=1):
    """
    Generate initial valid configurations.
    """
    configs = []
    for _ in range(count):
        pts = []
        for _ in range(N):
            pts.append(_random_valid_point(rng))
        configs.append(np.array(pts))
    return configs

def _simulated_annealing(initial_points, time_limit, rng):
    """
    Run Simulated Annealing.
    """
    current_points = initial_points.copy()
    current_score = _objective(current_points)
    
    best_points = current_points.copy()
    best_score = current_score
    
    T = 0.05  # Initial temperature
    cooling_rate = 0.999
    min_T = 1e-6
    
    start_time = time.time()
    
    while T > min_T:
        if time.time() - start_time > time_limit:
            break
        
        # Propose a move
        # Pick a random point to perturb
        idx = rng.integers(N)
        # Gaussian noise
        noise = rng.normal(0, 0.01, size=2)
        
        new_points = current_points.copy()
        new_points[idx] += noise
        
        # Check validity (simple bounds check for speed, penalty handles the rest)
        # Actually, let's just evaluate the objective. The penalty is strong.
        new_score = _objective(new_points)
        
        delta = new_score - current_score
        
        if delta > 0 or rng.random() < np.exp(delta / T):
            current_points = new_points
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
        
        T *= cooling_rate
        
        # Occasionally, if stuck, do a larger jump
        if rng.random() < 0.01:
             idx = rng.integers(N)
             new_points = current_points.copy()
             new_points[idx] = _random_valid_point(rng)
             new_score = _objective(new_points)
             delta = new_score - current_score
             if delta > 0 or rng.random() < np.exp(delta / T):
                 current_points = new_points
                 current_score = new_score
                 if current_score > best_score:
                     best_score = current_score
                     best_points = current_points.copy()
    
    return best_points, best_score

def construct():
    start_time = time.time()
    timeout = 150.0 # Safety margin from 150s limit
    
    rng = np.random.default_rng(42)
    
    best_points = None
    best_score = -np.inf
    
    # Multi-start SA
    num_starts = 5
    time_per_start = (timeout - 20) / num_starts # Leave 20s for polishing
    
    for i in range(num_starts):
        if time.time() - start_time > timeout - 20:
            break
            
        # Generate initial config
        init_pts = _initial_population(rng, 1)[0]
        
        # Run SA
        sa_pts, sa_score = _simulated_annealing(init_pts, time_per_start, rng)
        
        if sa_score > best_score:
            best_score = sa_score
            best_points = sa_pts
    
    # Polish the best result found
    if best_points is not None and time.time() - start_time < timeout - 5:
        # Use Nelder-Mead for local refinement
        # We need a minimization objective
        def _min_obj(flat_pts):
            pts = flat_pts.reshape(-1, 2)
            return -_objective(pts)
        
        try:
            res = minimize(
                _min_obj, 
                best_points.flatten(), 
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-7}
            )
            polished_pts = res.x.reshape(-1, 2)
            polished_score = _objective(polished_pts)
            
            if polished_score > best_score:
                best_score = polished_score
                best_points = polished_pts
        except Exception:
            pass
    
    # Final validity check and projection if needed
    if best_points is None:
        # Fallback
        best_points = _initial_population(rng, 1)[0]
    
    # Ensure points are strictly inside or on boundary
    # The penalty should have kept them inside, but let's project if slightly out
    def project_point(p):
        x, y = p
        # If y < 0, y=0
        if y < 0:
            y = 0
        # If y > sqrt(3)*x, project
        if y > S3 * x:
            # Project onto y = sqrt(3)x
            # Line: -sqrt(3)x + y = 0. Normal: (-sqrt(3), 1).
            # Distance d = (-sqrt(3)x + y) / 2.
            # Projection: (x,y) - d * normal? No, move towards origin.
            # t = (x*sqrt(3) + y) / 4? 
            # Let's use the formula from before
            dot = x * 0.5 + y * (S3/2)
            x = dot * 0.5
            y = dot * (S3/2)
        # If y > sqrt(3)*(1-x), project
        if y > S3 * (1 - x):
            # Line: sqrt(3)x + y - sqrt(3) = 0. Normal: (sqrt(3), 1).
            # Distance d = (sqrt(3)x + y - sqrt(3)) / 2.
            # Move in direction -normal.
            v = np.array([x-1, y])
            u = np.array([0.5, -S3/2])
            dot = np.dot(v, u)
            x = 1 + dot * 0.5
            y = 0 + dot * (-S3/2)
            
        return np.array([x, y])

    for i in range(N):
        best_points[i] = project_point(best_points[i])
        
    return best_points
