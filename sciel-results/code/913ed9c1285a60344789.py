import numpy as np
import time
from scipy.optimize import minimize

N = 11

def _min_triangle_area_vec(points):
    """
    Vectorized calculation of the minimum area of any triangle formed by the points.
    points: array of shape (N, 2)
    Returns: float, the minimum area.
    """
    n = len(points)
    # We need to compute areas for all triples (i, j, k) with i < j < k
    # Number of triples is C(11, 3) = 165. This is small enough for a loop,
    # but we can vectorize the inner operations for speed.
    
    # To be very fast, we can precompute indices or just use a simple loop
    # but vectorize the cross product calculation.
    # Actually, for N=11, a pure Python loop over 165 items is fast,
    # but let's try to make it as efficient as possible.
    
    min_area = float('inf')
    # Precompute indices for i, j, k
    # i from 0 to N-3
    # j from i+1 to N-2
    # k from j+1 to N-1
    
    # A slightly faster approach:
    # Iterate i, j, and vectorize k
    for i in range(n - 2):
        p1 = points[i]
        for j in range(i + 1, n - 1):
            p2 = points[j]
            v12 = p2 - p1
            # p3 are points[j+1:]
            p3s = points[j+1:]
            v13s = p3s - p1
            # Cross product: v12 x v13
            # v12 is (dx, dy), v13s is (N, 2)
            # Cross = dx * y3 - dy * x3
            cross_vals = v12[0] * v13s[:, 1] - v12[1] * v13s[:, 0]
            areas = 0.5 * np.abs(cross_vals)
            min_local = np.min(areas)
            if min_local < min_area:
                min_area = min_local
                # Early exit if area is very small?
                # if min_area < 1e-9: return min_area
    return min_area

def _project_to_triangle(points):
    """
    Project points onto the interior of the equilateral triangle (0,0), (1,0), (0.5, sqrt(3)/2).
    This ensures constraints are strictly satisfied.
    """
    eps = 1e-9
    pts = np.array(points, dtype=float)
    
    # Clip to bounding box first
    pts[:, 0] = np.clip(pts[:, 0], eps, 1 - eps)
    pts[:, 1] = np.clip(pts[:, 1], eps, np.sqrt(3)/2 - eps)
    
    # Check triangle inequalities
    # 1. y >= 0 (handled by clip)
    # 2. y <= -sqrt(3)(x - 1)  =>  y + sqrt(3)x - sqrt(3) <= 0
    # 3. y <= sqrt(3)x         =>  y - sqrt(3)x <= 0
    
    # If a point violates 2: y > sqrt(3)(1-x)
    # We need to project it back onto the line y = sqrt(3)(1-x)
    # The line is sqrt(3)x + y - sqrt(3) = 0
    # Normal vector n = (sqrt(3), 1)
    # Distance from point p to line is (sqrt(3)x + y - sqrt(3)) / norm(n)
    # norm(n) = sqrt(3+1) = 2
    
    # For each point, check violation
    # Violation 2: val = sqrt(3)*x + y - sqrt(3)
    # If val > 0, project.
    # Projection: p_new = p - (val / 4) * (2*sqrt(3), 2)  ? 
    # Wait, projection of p onto line ax+by+c=0 is p - (ax+by+c)/(a^2+b^2) * (a,b)
    # Here a=sqrt(3), b=1, c=-sqrt(3). a^2+b^2=4.
    # p_new = p - (val/4) * (sqrt(3), 1)
    
    # Violation 3: val = -sqrt(3)*x + y
    # Line: -sqrt(3)x + y = 0. a=-sqrt(3), b=1. a^2+b^2=4.
    # If val > 0, project.
    # p_new = p - (val/4) * (-sqrt(3), 1)
    
    for i in range(len(pts)):
        x, y = pts[i]
        # Check right edge: sqrt(3)*x + y - sqrt(3) <= 0
        val_right = np.sqrt(3)*x + y - np.sqrt(3)
        if val_right > 0:
            # Project
            pts[i, 0] = x - (val_right / 4.0) * np.sqrt(3)
            pts[i, 1] = y - (val_right / 4.0)
        
        # Check left edge: -sqrt(3)*x + y <= 0
        val_left = -np.sqrt(3)*x + y
        if val_left > 0:
            # Project
            pts[i, 0] = x - (val_left / 4.0) * (-np.sqrt(3))
            pts[i, 1] = y - (val_left / 4.0)
            
        # Re-clip to be safe against numerical drift
        pts[i, 0] = np.clip(pts[i, 0], eps, 1 - eps)
        pts[i, 1] = np.clip(pts[i, 1], eps, np.sqrt(3)/2 - eps)
        
    return pts

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s safety margin
    
    # Initial guess:
    # Try a few heuristic configurations
    # 1. Regular grid
    # 2. Random
    # 3. Centroid + Circle
    
    candidates = []
    
    # Heuristic 1: Grid-like
    # For 11 points, maybe 3x3 grid + 2 extra?
    # Let's just use random restarts with Simulated Annealing.
    
    rng = np.random.default_rng(42)
    
    best_points = None
    best_area = 0.0
    
    # Number of restarts
    # Each SA run takes ~T seconds.
    # We will run SA until time limit.
    
    num_restarts = 10
    
    for r in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # Initialize points
        # Strategy: Mix of random and structured
        if r == 0:
            # Start with a regular pattern
            # 3x3 grid in a smaller triangle?
            # Let's just use random for simplicity, but maybe biased towards center?
            pass
        
        # Random initialization inside triangle
        pts = np.zeros((N, 2))
        for i in range(N):
            while True:
                x = rng.random()
                y = rng.random() * (np.sqrt(3)/2)
                if y <= np.sqrt(3)*(1-x) - 1e-9 and y >= 1e-9 and x >= 1e-9 and x <= 1-1e-9:
                    pts[i] = [x, y]
                    break
        pts = _project_to_triangle(pts)
        
        # Simulated Annealing
        current_pts = pts.copy()
        current_area = _min_triangle_area_vec(current_pts)
        
        if current_area > best_area:
            best_area = current_area
            best_points = current_pts.copy()
            
        T = 0.05 # Initial temperature
        T_min = 1e-6
        cooling = 0.995
        step_size = 0.05
        
        while T > T_min:
            if time.time() > deadline - 2.0:
                break
                
            # Generate neighbor
            # Perturb a random subset of points
            num_perturb = rng.integers(1, 4) # 1 to 3 points
            indices = rng.choice(N, size=num_perturb, replace=False)
            
            new_pts = current_pts.copy()
            for idx in indices:
                # Gaussian perturbation
                delta = rng.normal(0, step_size, size=2)
                new_pts[idx] += delta
            
            # Project to triangle
            new_pts = _project_to_triangle(new_pts)
            
            new_area = _min_triangle_area_vec(new_pts)
            
            # Acceptance
            if new_area > current_area:
                current_pts = new_pts
                current_area = new_area
                if current_area > best_area:
                    best_area = current_area
                    best_points = current_pts.copy()
            else:
                # Accept with probability exp((new_area - current_area) / T)
                # Note: We are maximizing, so delta = new - current < 0
                delta = new_area - current_area
                prob = np.exp(delta / T)
                if rng.random() < prob:
                    current_pts = new_pts
                    current_area = new_area
            
            T *= cooling
            # Adapt step size?
            # If we are stuck, maybe increase step size?
            # If we are improving, maybe decrease?
            # Simple: keep constant or slow decay.
            # Let's decay step size slightly
            step_size *= 0.999

    # Final polish with Nelder-Mead on the best point found
    if best_points is not None and time.time() < deadline - 1.0:
        def obj_local(p):
            pts = _project_to_triangle(p.reshape((N, 2)))
            return -_min_triangle_area_vec(pts)
        
        try:
            res = minimize(obj_local, best_points.flatten(), method='Nelder-Mead', 
                           options={'maxiter': 100, 'xatol': 1e-5, 'fatol': 1e-6})
            refined_pts = _project_to_triangle(res.x.reshape((N, 2)))
            refined_area = _min_triangle_area_vec(refined_pts)
            if refined_area > best_area:
                best_area = refined_area
                best_points = refined_pts
        except Exception:
            pass

    if best_points is None:
        # Fallback
        cx, cy = 0.5, np.sqrt(3)/6
        r = 0.1
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.array([[cx + r*np.cos(a), cy + r*np.sin(a)] for a in angles])
        best_points = _project_to_triangle(best_points)

    return best_points

# EVOLVE-BLOCK-END
