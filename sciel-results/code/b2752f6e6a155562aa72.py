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
    
    # y >= 0
    v1 = np.maximum(0, -y)
    # y <= sqrt(3) * x
    v2 = np.maximum(0, y - S3 * x)
    # y <= sqrt(3) * (1 - x)
    v3 = np.maximum(0, y - S3 * (1 - x))
    
    return np.sum(v1**2 + v2**2 + v3**2)

def _clamp_to_triangle(points):
    """
    Project points to the nearest point inside the triangle.
    """
    x = points[:, 0].copy()
    y = points[:, 1].copy()
    
    # y >= 0
    y = np.maximum(y, 0)
    
    # y <= sqrt(3) * x  =>  x >= y / sqrt(3)
    viol1 = y > S3 * x
    if np.any(viol1):
        x[viol1] = y[viol1] / S3
    
    # y <= sqrt(3) * (1 - x)  =>  x <= 1 - y / sqrt(3)
    viol2 = y > S3 * (1 - x)
    if np.any(viol2):
        x[viol2] = 1 - y[viol2] / S3
        
    # Re-check y >= 0
    y = np.maximum(y, 0)
    
    # Safety clip
    x = np.clip(x, 0, 1)
    y = np.clip(y, 0, H)
    
    # Final check
    viol1 = y > S3 * x
    if np.any(viol1):
        x[viol1] = y[viol1] / S3
    viol2 = y > S3 * (1 - x)
    if np.any(viol2):
        x[viol2] = 1 - y[viol2] / S3
        
    return np.column_stack((x, y))

def _smooth_objective(flat_pts, beta=50.0):
    """
    Smooth approximation of -min_area for gradient-based optimization.
    Uses log-sum-exp to approximate the minimum.
    We want to MINIMIZE this function.
    """
    points = flat_pts.reshape(-1, 2)
    points = _clamp_to_triangle(points)
    
    Pi = points[I]
    Pj = points[J]
    Pk = points[K]
    
    v1 = Pj - Pi
    v2 = Pk - Pi
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Avoid log(0)
    areas = np.maximum(areas, 1e-12)
    
    # Approximate min(areas) using log-sum-exp
    # min(A) ~ -1/beta * log(sum(exp(-beta * A)))
    # We want to maximize min(A), so minimize -min(A).
    # -min(A) ~ 1/beta * log(sum(exp(-beta * A)))
    
    m = np.min(areas)
    shifted = areas - m
    log_sum_exp = np.log(np.sum(np.exp(-beta * shifted)))
    smooth_min = m - log_sum_exp / beta
    
    # Add penalty for boundary violations to keep points inside
    pen = _penalty(points)
    
    # We return the negative of the smooth min plus a large penalty
    # So minimizing this function maximizes the smooth min and keeps points inside
    return -smooth_min + 1000.0 * pen

def _simulated_annealing(initial_points, max_time, rng):
    """
    Simulated Annealing with adaptive step size.
    Returns the best points found.
    """
    points = initial_points.copy()
    best_points = points.copy()
    best_score = _min_area(best_points)
    
    T = 0.05 # Initial temperature
    T_min = 1e-6
    cooling_rate = 0.995
    step_scale = 0.05 # Initial step size scale
    
    start_time = time.time()
    
    while T > T_min:
        if time.time() - start_time > max_time:
            break
            
        # Generate new candidate
        # Adaptive step: larger when T is high, smaller when T is low
        current_step = step_scale * T
        new_points = points + rng.normal(0, current_step, size=points.shape)
        new_points = _clamp_to_triangle(new_points)
        
        new_score = _min_area(new_points)
        delta = new_score - _min_area(points) # We want to maximize score
        
        # Accept if better, or with probability exp(delta / T) if worse
        if delta > 0:
            points = new_points
            if new_score > best_score:
                best_score = new_score
                best_points = new_points.copy()
        else:
            if rng.random() < np.exp(delta / T):
                points = new_points
        
        # Cool down
        T *= cooling_rate
        
        # Occasionally restart step scale if stuck? 
        # For now, simple cooling is robust.
        
    return best_points, best_score

def _polish(points, max_time=10.0):
    """
    Local polish using L-BFGS-B on the smooth objective.
    """
    start_time = time.time()
    try:
        # Use a higher beta for more accurate min approximation during polish
        def obj_func(flat_pts):
            return _smooth_objective(flat_pts, beta=100.0)
            
        # L-BFGS-B is good for this scale
        res = minimize(
            obj_func,
            points.flatten(),
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        polished_points = _clamp_to_triangle(res.x.reshape(-1, 2))
        polished_score = _min_area(polished_points)
        
        if time.time() - start_time > max_time:
            return points, _min_area(points)
            
        if polished_score > _min_area(points):
            return polished_points, polished_score
        else:
            return points, _min_area(points)
            
    except Exception:
        return points, _min_area(points)

def construct():
    start_time = time.time()
    timeout = 150.0
    safety_margin = 5.0
    effective_timeout = timeout - safety_margin
    
    rng = np.random.default_rng(42)
    
    best_points = None
    best_score = -np.inf
    
    # Number of SA restarts
    n_restarts = 10
    
    # Allocate time per restart
    # Reserve time for final polishing
    time_per_sa = (effective_timeout * 0.8) / n_restarts
    time_for_polish = effective_timeout * 0.2
    
    for i in range(n_restarts):
        current_time = time.time()
        elapsed = current_time - start_time
        if elapsed > effective_timeout - 1.0:
            break
            
        # Generate random initial points
        # Mix of random and structured
        if i % 2 == 0:
            # Random uniform
            u = rng.random((N, 2))
            # Map to triangle
            # Use rejection sampling or affine transform
            # Simple rejection
            points = np.zeros((N, 2))
            for j in range(N):
                while True:
                    x = rng.random()
                    y = rng.random()
                    if x + y <= 1:
                        points[j, 0] = x + 0.5 * y
                        points[j, 1] = H * y
                        break
        else:
            # Structured perturbation
            # Start with a grid-like distribution
            xs = np.linspace(0.1, 0.9, N)
            ys = np.linspace(0.1, H*0.9, N)
            points = np.column_stack([xs, ys])
            points = points + rng.normal(0, 0.1, size=points.shape)
            points = _clamp_to_triangle(points)
            
        # Run SA
        sa_points, sa_score = _simulated_annealing(points, time_per_sa, rng)
        
        if sa_score > best_score:
            best_score = sa_score
            best_points = sa_points
            
    # Final Polish Phase
    if best_points is not None:
        time_left = effective_timeout - (time.time() - start_time)
        if time_left > 1.0:
            # Polish the best point
            polished_points, polished_score = _polish(best_points, max_time=min(time_left, 20.0))
            if polished_score > best_score:
                best_score = polished_score
                best_points = polished_points
                
            # If we have time, try polishing a few other good candidates?
            # We only kept the best one. Let's try to re-run SA on the best point with a very short time to see if it jumps?
            # Or just polish again with different beta?
            time_left2 = effective_timeout - (time.time() - start_time)
            if time_left2 > 5.0:
                # Try a second polish with different parameters or a slight perturbation
                perturbed = best_points + rng.normal(0, 0.01, size=best_points.shape)
                perturbed = _clamp_to_triangle(perturbed)
                p2_points, p2_score = _polish(perturbed, max_time=min(time_left2, 10.0))
                if p2_score > best_score:
                    best_score = p2_score
                    best_points = p2_points

    # Fallback
    if best_points is None:
        # Very simple fallback
        points = np.array([
            [0.0, 0.0], [1.0, 0.0], [0.5, H],
            [0.5, 0.0], [0.25, H/2], [0.75, H/2],
            [0.5, H/3],
            [0.3, H/3], [0.7, H/3], [0.5, 2*H/3], [0.5, H/6]
        ])
        best_points = _clamp_to_triangle(points)
        
    return best_points

if __name__ == "__main__":
    construct()
