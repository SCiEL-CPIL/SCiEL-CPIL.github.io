import numpy as np
import time
from itertools import combinations
from scipy.optimize import minimize, minimize_scalar

N = 11
SQRT3 = np.sqrt(3.0)
H = SQRT3 / 2.0  # height of the equilateral triangle with side 1

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, H])

# Precompute triangle indices
TRI_IDX = np.array(list(combinations(range(N), 3)))
M = len(TRI_IDX)  # C(11,3) = 165

def min_triangle_area(points):
    """Compute the minimum area over all C(n,3) triangles using vectorization."""
    pi = points[TRI_IDX[:, 0]]  # (M, 2)
    pj = points[TRI_IDX[:, 1]]  # (M, 2)
    pk = points[TRI_IDX[:, 2]]  # (M, 2)
    
    # Area = 0.5 * |cross(pj - pi, pk - pi)|
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    return float(np.min(np.abs(cross)) * 0.5)

def is_inside_triangle(p, eps=1e-9):
    """Check if point p is inside the equilateral triangle."""
    x, y = p
    return (y >= -eps) and (y <= H + eps) and (x >= y/SQRT3 - eps) and (x <= 1 - y/SQRT3 + eps)

def project_to_triangle(p):
    """Project a point onto the triangle if it's outside."""
    x, y = p
    # Clip y
    if y < 0:
        y = 0
    if y > H:
        y = H
    # Clip x based on y
    x_min = y / SQRT3
    x_max = 1 - y / SQRT3
    if x < x_min:
        x = x_min
    if x > x_max:
        x = x_max
    return np.array([x, y])

def generate_structured_initial():
    """Generate a structured initial guess with good symmetry."""
    points = []
    # Vertices
    points.append([0.0, 0.0])
    points.append([1.0, 0.0])
    points.append([0.5, H])
    
    # Edge midpoints
    points.append([0.5, 0.0])
    points.append([0.25, H/2])
    points.append([0.75, H/2])
    
    # Center
    points.append([0.5, H/3])
    
    # Additional points
    points.append([0.25, 0.0])
    points.append([0.75, 0.0])
    points.append([0.35, H/4])
    points.append([0.65, H/4])
    
    return np.array(points[:N])

def generate_random_points(seed):
    """Generate random points inside the triangle using rejection sampling."""
    rng = np.random.RandomState(seed)
    points = []
    while len(points) < N:
        u, v = rng.random(2)
        if u + v < 1:
            # Barycentric coordinates
            a = 1 - u - v
            b = u
            c = v
            x = b * 1.0 + c * 0.5
            y = c * H
            points.append([x, y])
    return np.array(points)

def soft_min_surrogate(points, t):
    """
    Compute a smooth approximation of the negative minimum triangle area.
    We minimize this function.
    f(x) = - (1/t) * log( sum( exp(-t * area_k) ) )
    As t -> inf, this approaches -min(area_k).
    So minimizing this is equivalent to maximizing min(area_k).
    """
    pi = points[TRI_IDX[:, 0]]
    pj = points[TRI_IDX[:, 1]]
    pk = points[TRI_IDX[:, 2]]
    
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    
    # Avoid log(0) and numerical instability
    areas = np.maximum(areas, 1e-12)
    
    # Log-sum-exp trick for stability
    max_area = np.max(areas)
    log_sum = max_area + np.log(np.sum(np.exp(-t * (areas - max_area))))
    return -log_sum / t

def soft_min_grad(points, t):
    """
    Gradient of the soft-min surrogate.
    This is complex to derive analytically for the bilinear area terms,
    so we might rely on finite differences if L-BFGS-B is too slow with it,
    or use Nelder-Mead which doesn't need gradients.
    Given the complexity, Nelder-Mead on the true objective is often more reliable
    for small N like 11.
    """
    # We will use Nelder-Mead instead for robustness, so this gradient is not strictly needed.
    # But if we were to implement it:
    # The derivative of area w.r.t. points is linear.
    # The derivative of the soft-min is a weighted sum of the area gradients.
    pass

def nelder_mead_optimize(points, deadline, maxiter=2000):
    """
    Optimize points using Nelder-Mead on the true negative min area objective.
    """
    def objective(flat_points):
        if time.time() > deadline:
            return 1e6 # Penalty for timeout
        pts = flat_points.reshape(-1, 2)
        # Project points to stay inside triangle to help convergence
        for i in range(N):
            pts[i] = project_to_triangle(pts[i])
        # Small penalty if points are too close (degenerate triangles)
        min_area = min_triangle_area(pts)
        return -min_area
    
    x0 = points.flatten()
    
    # Nelder-Mead parameters
    options = {
        'maxiter': maxiter,
        'xatol': 1e-6,
        'fatol': 1e-6,
        'adaptive': True
    }
    
    try:
        res = minimize(objective, x0, method='Nelder-Mead', options=options)
        if time.time() > deadline:
             # Return best found so far if we hit deadline mid-optimization? 
             # Minimize doesn't easily expose intermediate bests. 
             # We rely on the deadline check inside objective to stop early or just let it finish quickly.
             # Better to check time before starting.
             pass
        best_pts = res.x.reshape(-1, 2)
        # Final projection
        for i in range(N):
            best_pts[i] = project_to_triangle(best_pts[i])
        return best_pts, -res.fun
    except Exception:
        return points, min_triangle_area(points)

def lbfgs_optimize(points, deadline, t_init=100, t_mult=1.5, t_max=1000):
    """
    Optimize points using L-BFGS-B on the soft-min surrogate.
    This can be faster for initial large-scale moves.
    """
    # We need a gradient. Since deriving it is tricky, we can use a finite difference approximation
    # or just stick to Nelder-Mead. Nelder-Mead is derivative-free and robust.
    # Let's try a hybrid: use Nelder-Mead.
    pass

def local_optimize(points, max_iter=300, step_size=0.05, deadline=None):
    """Local optimization using coordinate descent / hill climbing with Nelder-Mead-like refinement."""
    points = points.copy()
    best_val = min_triangle_area(points)
    
    # Try to improve each point using a simple hill-climbing with shrinking steps
    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
            
        improved = False
        for i in range(N):
            # Try moving point i in different directions
            best_local = points[i].copy()
            best_local_val = best_val
            
            # Try 8 directional moves + random perturbations
            for _ in range(30):
                if deadline is not None and time.time() > deadline:
                    break
                    
                # Mix of directional and random moves
                if _ < 8:
                    # 8 compass directions
                    angle = _ * np.pi / 4
                    dx = step_size * np.array([np.cos(angle), np.sin(angle)])
                else:
                    dx = np.random.uniform(-step_size, step_size, 2)
                
                new_p = best_local + dx
                new_p = project_to_triangle(new_p)
                
                # Check if it's still inside (with small tolerance)
                if not is_inside_triangle(new_p, eps=1e-6):
                    continue
                    
                new_points = points.copy()
                new_points[i] = new_p
                
                val = min_triangle_area(new_points)
                if val > best_local_val + 1e-12:
                    best_local_val = val
                    best_local = new_p
                    improved = True
            
            points[i] = best_local
        
        # Update best value
        current_val = min_triangle_area(points)
        if current_val > best_val:
            best_val = current_val
        
        # If no improvement in a full pass, reduce step size
        if not improved:
            step_size *= 0.7
            if step_size < 1e-7:
                break
    
    return points, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_points = None
    best_value = -1
    
    # Strategy 1: Structured initial guess
    init_points = generate_structured_initial()
    opt_points, opt_val = nelder_mead_optimize(init_points, deadline, maxiter=1000)
    if opt_val > best_value:
        best_value = opt_val
        best_points = opt_points.copy()
        
    # Strategy 2: Multiple random restarts with Nelder-Mead
    # Use remaining time for more restarts
    rng = np.random.RandomState(123)
    
    # Estimate time per Nelder-Mead run. Let's say ~5-10s.
    # We have ~145s.
    # Let's do as many as we can.
    
    while time.time() < deadline - 5.0:
        # Generate a random starting point
        # We can bias it towards the previous best to refine, or start fresh.
        # Starting fresh from random is good for global exploration.
        # Starting near best is good for local refinement.
        # Let's do a mix.
        
        if rng.random() < 0.3 and best_points is not None:
            # Perturb best points
            perturbation = rng.normal(0, 0.05, (N, 2))
            init_points = best_points + perturbation
            for i in range(N):
                init_points[i] = project_to_triangle(init_points[i])
        else:
            # Random points
            init_points = generate_random_points(rng.randint(10000))
            
        # Run Nelder-Mead
        # Limit maxiter to ensure we don't spend too long on one run
        # If we have lots of time, allow more iterations.
        time_left = deadline - time.time()
        max_iter = int(min(2000, max(100, time_left * 50))) # Heuristic: ~20ms per iter? No, NM is slower.
        # Nelder-Mead with 11x2=22 variables is fast. 1000 iterations is usually < 1s.
        # Let's cap at 1000 iterations per run.
        
        opt_points, opt_val = nelder_mead_optimize(init_points, deadline, maxiter=1000)
        
        if opt_val > best_value:
            best_value = opt_val
            best_points = opt_points.copy()
            
            # Optional: Fine-tune with hill climbing if time permits
            if time.time() < deadline - 10.0:
                fine_points, fine_val = local_optimize(opt_points, max_iter=100, step_size=0.01, deadline=deadline)
                if fine_val > best_value:
                    best_value = fine_val
                    best_points = fine_points.copy()
    
    # Final check and projection
    if best_points is None:
        best_points = generate_structured_initial()
    else:
        for i in range(N):
            best_points[i] = project_to_triangle(best_points[i])
            
    return best_points

if __name__ == "__main__":
    # Quick test
    pts = construct()
    print(f"Score: {min_triangle_area(pts)}")
