"""Heilbronn triangle problem: maximize minimum triangle area for 11 points."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 11
S3H = np.sqrt(3) / 2
DEADLINE = 115.0  # seconds, safety margin before 120s limit

def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by three points."""
    n = len(pts)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                dx1 = pts[j, 0] - pts[i, 0]
                dy1 = pts[j, 1] - pts[i, 1]
                dx2 = pts[k, 0] - pts[i, 0]
                dy2 = pts[k, 1] - pts[i, 1]
                cross = abs(dx1 * dy2 - dy1 * dx2)
                area = 0.5 * cross
                if area < min_area:
                    min_area = area
    return min_area

def _check_inside(pts, eps=1e-9):
    """Check if all points are inside or on the boundary of the equilateral triangle."""
    for p in pts:
        x, y = p
        if y < -eps:
            return False
        if y > np.sqrt(3) * x + eps:
            return False
        if y > np.sqrt(3) * (1 - x) + eps:
            return False
        if x < -eps or x > 1 + eps:
            return False
    return True

def _project_to_triangle(pts):
    """Project points that are slightly outside the triangle back inside."""
    pts = np.array(pts, dtype=float)
    for i in range(len(pts)):
        x, y = pts[i]
        if y < 0:
            y = 0
        max_y = min(np.sqrt(3) * x, np.sqrt(3) * (1 - x))
        if y > max_y:
            y = max_y
        if x < 0:
            x = 0
        if x > 1:
            x = 1
        pts[i] = [x, y]
    return pts

def _random_point_in_triangle(rng):
    """Generate a random point inside the equilateral triangle."""
    u = rng.random()
    v = rng.random()
    while u + v > 1:
        u = rng.random()
        v = rng.random()
    x = u + 0.5 * v
    y = S3H * v
    return [x, y]

def _initial_guess(rng):
    """Generate an initial set of 11 points using symmetry-breaking."""
    # Place points in a grid-like pattern with noise to break symmetry
    # This helps escape the "all points cluster in center" local minimum
    pts = []
    # 3 points near vertices (slightly inside)
    pts.append([0.1, 0.05])
    pts.append([0.9, 0.05])
    pts.append([0.5, 0.8])
    
    # 4 points along mid-edges
    pts.append([0.3, 0.2])
    pts.append([0.7, 0.2])
    pts.append([0.5, 0.4])
    pts.append([0.2, 0.5])
    
    # 4 points in center region
    for _ in range(4):
        pts.append(_random_point_in_triangle(rng))
        
    return np.array(pts)

def _epigraph_objective_and_grad(x_flat, t):
    """
    Objective: maximize t subject to area(i,j,k) >= t for all i<j<k.
    Here we minimize -t.
    Constraints are handled by SLSQP, but we can also use a penalty or 
    simply rely on the constraint definition.
    For SLSQP, we define constraints as area - t >= 0.
    """
    # We don't need to compute gradient of objective (-t) explicitly if we let SLSQP estimate it,
    # but providing it is faster.
    # Objective is -t, so grad is 0 for points, -1 for t.
    # However, SLSQP handles the constraints.
    # The "objective" passed to minimize is just -t.
    # The constraints are the hard part.
    
    # This function is just a placeholder to match signature if we used it directly,
    # but SLSQP needs separate constraint functions.
    # We will define constraints separately.
    pass

def _area_constraints(x_flat, t_idx=N*2):
    """
    Define the constraints for the epigraph problem.
    We want area(i,j,k) - t >= 0 for all triples.
    """
    pts = x_flat[:N*2].reshape(N, 2)
    t = x_flat[N*2]
    
    cons = []
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                dx1 = pts[j, 0] - pts[i, 0]
                dy1 = pts[j, 1] - pts[i, 1]
                dx2 = pts[k, 0] - pts[i, 0]
                dy2 = pts[k, 1] - pts[i, 1]
                area = 0.5 * abs(dx1 * dy2 - dy1 * dx2)
                # Constraint: area - t >= 0
                cons.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i, j=j, k=k: _area_single(x, i, j, k) - x[N*2],
                    'jac': lambda x, i=i, j=j, k=k: _jac_area_single(x, i, j, k)
                })
    return cons

def _area_single(x, i, j, k):
    pts = x[:N*2].reshape(N, 2)
    dx1 = pts[j, 0] - pts[i, 0]
    dy1 = pts[j, 1] - pts[i, 1]
    dx2 = pts[k, 0] - pts[i, 0]
    dy2 = pts[k, 1] - pts[i, 1]
    return 0.5 * abs(dx1 * dy2 - dy1 * dx2)

def _jac_area_single(x, i, j, k):
    # This is complex to derive analytically for abs(cross).
    # Since area is piecewise linear in the coordinates (with a kink at area=0),
    # and we are maximizing area, we are usually in the region where the sign of the cross product is fixed.
    # However, for SLSQP, providing an approximate or finite-difference jac is safer if analytic is tricky.
    # Let's skip providing 'jac' in constraints and let SLSQP use finite differences, 
    # OR provide a simplified version.
    # Actually, for speed, let's not provide jac for constraints initially.
    # If we do, it must be accurate.
    # Let's remove 'jac' from the constraint dict to avoid errors.
    return None

def _triangle_boundary_constraints(x_flat):
    """
    Constraints to keep points inside the triangle.
    1. y >= 0
    2. y <= sqrt(3)*x
    3. y <= sqrt(3)*(1-x)
    4. x >= 0 (implied by others mostly, but good to have)
    5. x <= 1
    """
    cons = []
    for i in range(N):
        # y >= 0
        cons.append({
            'type': 'ineq',
            'fun': lambda x, i=i: x[i*2+1]
        })
        # y - sqrt(3)*x <= 0  =>  sqrt(3)*x - y >= 0
        cons.append({
            'type': 'ineq',
            'fun': lambda x, i=i: np.sqrt(3)*x[i*2] - x[i*2+1]
        })
        # y - sqrt(3)*(1-x) <= 0 => sqrt(3)*(1-x) - y >= 0
        cons.append({
            'type': 'ineq',
            'fun': lambda x, i=i: np.sqrt(3)*(1-x[i*2]) - x[i*2+1]
        })
        # x >= 0
        cons.append({
            'type': 'ineq',
            'fun': lambda x, i=i: x[i*2]
        })
        # x <= 1
        cons.append({
            'type': 'ineq',
            'fun': lambda x, i=i: 1 - x[i*2]
        })
    return cons

def _solve_epigraph(x0, t0):
    """
    Solve the epigraph problem using SLSQP.
    x0: initial points (22 values)
    t0: initial t value
    """
    x0_full = np.concatenate([x0.flatten(), [t0]])
    
    # Constraints
    # 1. Triangle boundaries
    cons = _triangle_boundary_constraints(x0_full)
    
    # 2. Area constraints: area(i,j,k) - t >= 0
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                cons.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i, j=j, k=k: _area_single(x, i, j, k) - x[N*2]
                })
    
    # Objective: maximize t => minimize -t
    def objective(x):
        return -x[N*2]
    
    # Bounds: t >= 0
    bounds = [(None, None)] * (N*2) + [(0, None)]
    
    try:
        res = minimize(
            objective,
            x0_full,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 200, 'ftol': 1e-9}
        )
        if res.success:
            return res.x[:N*2].reshape(N, 2), res.x[N*2]
        else:
            # Return best feasible point found?
            # SLSQP might fail but still improve t.
            # Let's check if the result is better than start.
            return res.x[:N*2].reshape(N, 2), res.x[N*2]
    except Exception:
        return x0, t0

def construct():
    """Construct 11 points in the unit equilateral triangle to maximize min triangle area."""
    start_time = time.time()
    rng = random.Random(42)
    
    best_pts = None
    best_area = 0.0
    
    # Phase 1: Simulated Annealing to find a good starting point
    # This helps escape local minima better than random starts.
    
    def _sa_step(pts, temperature, rng):
        """Perform one SA step: perturb a random point."""
        new_pts = pts.copy()
        idx = rng.randint(0, N-1)
        # Gaussian perturbation
        step_size = 0.05 * temperature
        new_pts[idx, 0] += rng.gauss(0, step_size)
        new_pts[idx, 1] += rng.gauss(0, step_size)
        
        # Project to triangle
        new_pts = _project_to_triangle(new_pts)
        
        # Check inside (strict)
        if not _check_inside(new_pts, eps=1e-9):
            return pts, _min_triangle_area(pts)
            
        return new_pts, _min_triangle_area(new_pts)

    # Initialize with symmetry-breaking guess
    current_pts = _initial_guess(rng)
    current_area = _min_triangle_area(current_pts)
    
    # SA parameters
    T = 1.0
    cooling_rate = 0.99
    steps_per_temp = 50
    
    for step in range(2000): # Max SA steps
        if time.time() - start_time > DEADLINE - 30:
            break
            
        if step % steps_per_temp == 0:
            T *= cooling_rate
            if T < 1e-4:
                break
                
        new_pts, new_area = _sa_step(current_pts, T, rng)
        
        # Acceptance criterion
        if new_area > current_area or rng.random() < np.exp((new_area - current_area) / T):
            current_pts = new_pts
            current_area = new_area
            
        if current_area > best_area:
            best_area = current_area
            best_pts = current_pts.copy()
            
    # Phase 2: Polish the best SA result using SLSQP Epigraph
    # We can try multiple polishes from slightly perturbed versions of the best SA point
    
    # First, try polishing the best SA point directly
    if best_pts is not None:
        # Estimate initial t
        t0 = best_area
        polished_pts, polished_t = _solve_epigraph(best_pts, t0)
        
        if polished_pts is not None and _check_inside(polished_pts):
            polished_area = _min_triangle_area(polished_pts)
            if polished_area > best_area:
                best_area = polished_area
                best_pts = polished_pts.copy()
                
    # Try a few random starts with SLSQP as well, in case SA got stuck
    for trial in range(10):
        if time.time() - start_time > DEADLINE - 10:
            break
            
        # Random start
        pts = np.array([_random_point_in_triangle(rng) for _ in range(N)])
        pts = _project_to_triangle(pts)
        if not _check_inside(pts):
            continue
            
        area = _min_triangle_area(pts)
        if area > best_area:
            best_area = area
            best_pts = pts.copy()
            
        # Polish this random start
        polished_pts, polished_t = _solve_epigraph(pts, area)
        if polished_pts is not None and _check_inside(polished_pts):
            polished_area = _min_triangle_area(polished_pts)
            if polished_area > best_area:
                best_area = polished_area
                best_pts = polished_pts.copy()
                
    if best_pts is None:
        # Fallback
        pts = _initial_guess(rng)
        return pts
        
    return best_pts
