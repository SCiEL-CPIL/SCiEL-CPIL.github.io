import numpy as np
import time
from scipy.optimize import minimize

N = 50
_START_TIME = None

def _compute_radius(x):
    """
    Compute the maximum radius r for a given configuration x.
    x: (N, 2) array of centers.
    """
    x = x.reshape(N, 2)
    # Pairwise distances
    dx = x[:, 0:1] - x[:, 0:1].T
    dy = x[:, 1:2] - x[:, 1:2].T
    dists_sq = dx**2 + dy**2
    np.fill_diagonal(dists_sq, 1.0) # Avoid self-distance 0
    min_dist = np.sqrt(np.min(dists_sq))
    
    # Wall distances
    # Distance to boundary is min(0.5 - x, x + 0.5) for each coord? 
    # Actually, for square [-0.5, 0.5]^2, dist to wall is min(0.5 - |x|, 0.5 - |y|)
    wx = 0.5 - np.abs(x[:, 0])
    wy = 0.5 - np.abs(x[:, 1])
    min_wall = np.min(np.minimum(wx, wy))
    
    return min(min_dist / 2.0, min_wall)

def _violation_obj_and_grad(x, r_target):
    """
    Objective: Sum of squared violations of constraints for radius r_target.
    Constraints:
    1. ||x_i - x_j|| >= 2*r_target
    2. ||x_i||_inf <= 0.5 - r_target  (i.e., x_i in [-0.5+r, 0.5-r])
    
    Returns: (objective, gradient)
    """
    x = x.reshape(N, 2)
    obj = 0.0
    grad = np.zeros_like(x)
    
    # 1. Pairwise constraints
    # We only need to check pairs that are close? For N=50, O(N^2) is 2500, which is fine.
    for i in range(N):
        for j in range(i + 1, N):
            d = x[i] - x[j]
            dist = np.linalg.norm(d)
            min_dist = 2.0 * r_target
            if dist < min_dist:
                # Violation = min_dist - dist
                # Grad wrt x_i: - (d / dist) * 2 * (min_dist - dist) ? 
                # Let v = min_dist - dist. Obj += v^2. 
                # d(v^2)/dx_i = 2v * dv/dx_i. 
                # dv/dx_i = - d(dist)/dx_i = - (x_i - x_j) / dist = - d / dist.
                # So term = 2 * v * (-d/dist) = -2 * v * d / dist.
                if dist > 1e-8:
                    v = min_dist - dist
                    term = -2.0 * v * (d / dist)
                    grad[i] += term
                    grad[j] -= term
                    obj += v * v
                else:
                    # If they overlap completely, push them apart arbitrarily
                    # This should be rare if we have good init
                    obj += 1e6
                    grad[i] += np.array([1.0, 1.0])
                    grad[j] -= np.array([1.0, 1.0])

    # 2. Wall constraints
    # x_k in [-L, L] where L = 0.5 - r_target
    L = 0.5 - r_target
    for i in range(N):
        for k in range(2):
            val = x[i, k]
            if val > L:
                v = val - L
                # Obj += v^2. d(v^2)/dx = 2v.
                grad[i, k] += 2.0 * v
                obj += v * v
            elif val < -L:
                v = -L - val
                # Obj += v^2. d(v^2)/dx = 2v * (-1) = -2v.
                grad[i, k] -= 2.0 * v
                obj += v * v
                
    return obj, grad

def _init_hex(r_guess=0.07):
    """Initialize points in a hexagonal pattern."""
    dx = 2 * r_guess
    dy = np.sqrt(3) * r_guess
    
    points = []
    y = -0.5 + dy/2
    row_idx = 0
    while y <= 0.5:
        x_start = -0.5 + dx/2
        if row_idx % 2 == 1:
            x_start += dx/2
        x = x_start
        while x <= 0.5:
            points.append((x, y))
            x += dx
        y += dy
        row_idx += 1
        
    points = np.array(points)
    
    if len(points) >= N:
        # Randomly select N points to avoid bias
        idx = np.random.choice(len(points), N, replace=False)
        return points[idx]
    else:
        # Pad with random points if needed
        remaining = N - len(points)
        random_pts = np.random.uniform(-0.45, 0.45, (remaining, 2))
        return np.vstack([points, random_pts])

def _simulated_annealing(x0, r_target, T_init=0.05, T_min=0.0001, alpha=0.995, max_iter=2000, deadline=None):
    """
    Simulated Annealing to minimize violation for a fixed r_target.
    """
    x = x0.copy()
    T = T_init
    current_obj, _ = _violation_obj_and_grad(x, r_target)
    best_x = x.copy()
    best_obj = current_obj
    
    for i in range(max_iter):
        if deadline and time.time() > deadline:
            break
            
        # Propose move
        # Random step size
        step_size = 0.02 * (T / T_init) + 0.001
        x_new = x + np.random.normal(0, step_size, (N, 2))
        
        # Keep within bounds roughly to avoid huge wall violations
        L = 0.5 - r_target
        x_new = np.clip(x_new, -L - 0.1, L + 0.1)
        
        new_obj, _ = _violation_obj_and_grad(x_new, r_target)
        
        delta = new_obj - current_obj
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            x = x_new
            current_obj = new_obj
            if current_obj < best_obj:
                best_obj = current_obj
                best_x = x.copy()
                
        T *= alpha
        if T < T_min:
            break
            
    return best_x, best_obj

def _lbfgs_polish(x0, r_target, deadline=None):
    """
    L-BFGS polish to minimize violation for a fixed r_target.
    """
    try:
        res = minimize(
            _violation_obj_and_grad,
            x0,
            args=(r_target,),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if res.success or res.fun < _violation_obj_and_grad(x0, r_target)[0]:
            return res.x.reshape(N, 2)
    except Exception:
        pass
    return x0

def construct():
    global _START_TIME
    _START_TIME = time.time()
    # Safety margin: 145s limit to ensure we return before 150s
    deadline = _START_TIME + 145.0
    
    best_x = None
    best_score = -np.inf
    
    # We will try to increase the radius gradually.
    # Start with a conservative radius and try to go up.
    # Known best is ~0.07137. Let's start at 0.070 and try to reach higher.
    
    # Phase 1: Find a good feasible solution for r=0.070
    r_current = 0.070
    
    # Multiple restarts for r_current
    num_restarts = 10
    
    for k in range(num_restarts):
        if time.time() > deadline - 30.0: # Leave time for higher radii
            break
            
        if k == 0:
            x0 = _init_hex(0.070)
        elif k % 2 == 0:
            x0 = _init_hex(0.065)
        else:
            x0 = np.random.uniform(-0.45, 0.45, (N, 2))
            
        # SA
        x_sa, obj_sa = _simulated_annealing(x0, r_current, deadline=deadline)
        
        # Polish
        x_pol = _lbfgs_polish(x_sa, r_current, deadline=deadline)
        
        score = _compute_radius(x_pol)
        
        if score > best_score:
            best_score = score
            best_x = x_pol.copy()
            
        # If we found a valid solution for r_current, we can try to increase r
        # But we need to be careful with time.
        
    # Phase 2: Binary search / Incremental increase for higher radii
    # Use the best solution found so far as a starting point
    if best_x is not None:
        # Try to push the radius up
        # We know the theoretical max is around 0.07137
        # Let's try a few steps up from the current best
        r_low = best_score
        r_high = 0.0720 # Slightly above known best
        
        # We have limited time, so do a coarse binary search or incremental steps
        # Let's try 3 specific higher radii
        target_radii = [0.0705, 0.0710, 0.07137]
        
        # Sort by difficulty (lowest first)
        for r_target in target_radii:
            if time.time() > deadline - 10.0:
                break
                
            # Use best_x as start, maybe perturb it slightly
            x_start = best_x + np.random.normal(0, 0.001, (N, 2))
            
            # SA for this higher radius
            x_sa, obj_sa = _simulated_annealing(x_start, r_target, T_init=0.02, max_iter=1000, deadline=deadline)
            
            # Polish
            x_pol = _lbfgs_polish(x_sa, r_target, deadline=deadline)
            
            score = _compute_radius(x_pol)
            
            if score > best_score:
                best_score = score
                best_x = x_pol.copy()
                
    # Final Polish: If we have a best_x, do one more L-BFGS polish on the ACTUAL radius
    # Wait, the objective function depends on r_target.
    # To polish the actual radius, we need to maximize min(dist/2, wall).
    # This is non-smooth.
    # However, if we assume the current configuration is "tight" in terms of distance,
    # we can try to optimize for the specific constraint that is active.
    
    # A simpler final polish: just return best_x.
    # Or, do a "soft" polish using the soft-min from the original code?
    # Let's stick to the robustness of the feasibility approach.
    
    if best_x is None:
        # Fallback
        best_x = _init_hex(0.07)
        
    return best_x
