import numpy as np
from scipy.optimize import minimize
import time

N = 21

def _validate(pos, radii, w, h, tol=1e-7):
    """Check if configuration is valid (no overlaps, within bounds)."""
    n = len(radii)
    # Check bounds and wall constraints
    for i in range(n):
        if pos[i, 0] < -tol or pos[i, 0] > w + tol:
            return False
        if pos[i, 1] < -tol or pos[i, 1] > h + tol:
            return False
        if radii[i] < -tol:
            return False
        if radii[i] > pos[i, 0] + tol:
            return False
        if radii[i] > w - pos[i, 0] + tol:
            return False
        if radii[i] > pos[i, 1] + tol:
            return False
        if radii[i] > h - pos[i, 1] + tol:
            return False
    
    # Check circle-circle non-overlap
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            d = np.sqrt(dx * dx + dy * dy)
            if d < radii[i] + radii[j] - tol:
                return False
    return True

def _solve_packing(w, h, init_pos, time_limit=10.0):
    """
    Solve the packing problem for a given rectangle w x h.
    Returns (pos, radii, success).
    """
    # Variables: x_0, y_0, r_0, x_1, y_1, r_1, ...
    # Total variables: 3 * N
    n_vars = 3 * N
    
    # Initial guess
    x0 = np.zeros(n_vars)
    for i in range(N):
        x0[3*i] = init_pos[i, 0]
        x0[3*i+1] = init_pos[i, 1]
        # Initial radius estimate: small
        x0[3*i+2] = min(w, h) / (2 * np.sqrt(N)) * 0.8
        
    # Objective: Maximize sum of radii -> Minimize -sum(radii)
    def objective(vars):
        radii = vars[2::3]
        return -np.sum(radii)
    
    # Constraints
    # 1. Wall constraints (inequality: g(vars) >= 0)
    #    x_i - r_i >= 0
    #    w - x_i - r_i >= 0
    #    y_i - r_i >= 0
    #    h - y_i - r_i >= 0
    # 2. Non-overlap: sqrt((x_i-x_j)^2 + (y_i-y_j)^2) - r_i - r_j >= 0
    
    # We use inequality constraints in the form g(vars) >= 0
    # SLSQP handles this.
    
    constraints = []
    
    # Wall constraints
    for i in range(N):
        # Left wall
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right wall
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: w - v[3*i] - v[3*i+2]})
        # Bottom wall
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top wall
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: h - v[3*i+1] - v[3*i+2]})
        
    # Non-overlap constraints
    for i in range(N):
        for j in range(i + 1, N):
            def non_overlap(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                d = np.sqrt(dx*dx + dy*dy)
                return d - v[3*i+2] - v[3*j+2]
            constraints.append({'type': 'ineq', 'fun': non_overlap})
            
    # Bounds: radii must be positive
    bounds = []
    for i in range(N):
        bounds.append((0, w/2)) # x
        bounds.append((0, h/2)) # y
        bounds.append((0, min(w, h)/2)) # r
        
    # Run optimization
    start_time = time.time()
    
    # Use SLSQP
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints,
                   options={'maxiter': 200, 'ftol': 1e-9})
    
    # Check time
    if time.time() - start_time > time_limit:
        return None, None, False
        
    if res.success:
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        for i in range(N):
            pos[i, 0] = res.x[3*i]
            pos[i, 1] = res.x[3*i+1]
            radii[i] = res.x[3*i+2]
            
        # Validate strictly
        if _validate(pos, radii, w, h):
            return pos, radii, True
        else:
            # If slightly invalid, try to fix by shrinking
            for _ in range(10):
                if _validate(pos, radii, w, h):
                    return pos, radii, True
                radii *= 0.98
            return pos, radii, _validate(pos, radii, w, h)
    else:
        return None, None, False

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    best_score = 0
    best_result = None
    
    # Search aspect ratios
    # Perimeter 4 -> 2(w+h)=4 -> h=2-w
    # w ranges from 0.5 to 1.5 (symmetric around 1.0)
    # We will test a set of aspect ratios
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5, 2.0, 0.4]
    
    # Sort by how close to square (usually good for packing) but include extremes
    # Actually, let's just iterate through them.
    
    for ar in aspect_ratios:
        if time.time() > deadline - 20:
            break
            
        # Calculate w and h
        # h = w * (1/ar) ? No, aspect ratio is usually w/h.
        # Let's define ar = w/h.
        # w + h = 2
        # w + w/ar = 2 => w(1 + 1/ar) = 2 => w = 2 * ar / (1 + ar)
        w = 2.0 * ar / (1.0 + ar)
        h = 2.0 / (1.0 + ar)
        
        # Try multiple initializations for this aspect ratio
        # 1. Grid initialization
        # 2. Random initialization (if time permits)
        
        inits = []
        
        # Grid init
        cols = int(np.ceil(np.sqrt(N * w / h)))
        rows = int(np.ceil(N / cols))
        if cols * rows < N:
            cols += 1
            rows = int(np.ceil(N / cols))
        
        pos_grid = np.zeros((N, 2))
        for i in range(N):
            col = i % cols
            row = i // cols
            pos_grid[i, 0] = (col + 0.5) * w / cols
            pos_grid[i, 1] = (row + 0.5) * h / rows
        inits.append(pos_grid)
        
        # Add a few random perturbations to grid
        if time.time() < deadline - 30:
            for _ in range(3):
                pos_rand = pos_grid + np.random.randn(N, 2) * 0.1 * min(w, h)
                pos_rand[:, 0] = np.clip(pos_rand[:, 0], 0, w)
                pos_rand[:, 1] = np.clip(pos_rand[:, 1], 0, h)
                inits.append(pos_rand)
        
        for init_pos in inits:
            if time.time() > deadline - 15:
                break
                
            pos, radii, success = _solve_packing(w, h, init_pos, time_limit=10.0)
            
            if success and pos is not None:
                score = np.sum(radii)
                if score > best_score:
                    best_score = score
                    best_result = np.column_stack([pos, radii])
                    
    # Fallback if no valid result found (should not happen with grid init)
    if best_result is None:
        w = 1.0
        h = 1.0
        side = int(np.ceil(np.sqrt(N)))
        pos = np.zeros((N, 2))
        rad = np.zeros(N)
        r_val = 0.5 / side * 0.8
        for i in range(N):
            col = i % side
            row = i // side
            pos[i, 0] = (col + 0.5) / side
            pos[i, 1] = (row + 0.5) / side
            rad[i] = r_val
        best_result = np.column_stack([pos, rad])
        
    return best_result

