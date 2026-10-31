"""Rewritten program: circle packing in perimeter-4 rectangle, maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START
def _perimeter_constraint(w, h):
    """Perimeter is 2*(w+h). We want 2*(w+h) <= 4 => w+h <= 2."""
    return w + h - 2.0

def _packing_objective(x, w, h):
    """
    x: array of shape (2*N,) with [x0, y0, x1, y1, ...]
    Returns a scalar objective to MINIMIZE. We want to maximize sum of radii.
    But we need to handle the non-smooth max(0, overlap) constraint.
    
    Actually, let's reformulate: we fix the rectangle dimensions (w, h) with w+h=2,
    and we have N circles. The radii are NOT independent variables here - wait, the task says
    "maximize the sum of their radii" and the output is (n,3) with (cx, cy, r).
    
    So both positions AND radii are variables. But if radii are free variables, we'd make them
    infinite unless constrained by non-overlap and boundary.
    
    Let me re-read: "Pack 21 circles with disjoint interiors in a variable rectangle of perimeter 4
    and maximize the sum of their radii."
    
    So we have:
    - Rectangle with perimeter 4, so if width=w, height=h, then 2(w+h)=4 => w+h=2.
    - 21 circles with centers (cx_i, cy_i) and radii r_i > 0.
    - Constraints:
      1. Circles are inside the rectangle: cx_i - r_i >= 0, cx_i + r_i <= w, cy_i - r_i >= 0, cy_i + r_i <= h.
      2. Non-overlap: (cx_i - cx_j)^2 + (cy_i - cy_j)^2 >= (r_i + r_j)^2 for all i<j.
    - Objective: maximize sum(r_i).
    
    This is a non-convex optimization problem. The current seed program just places circles in a grid
    with fixed small radii, which is suboptimal.
    
    Strategy: Use a penalty-based approach or SLSQP with proper constraints.
    Variables: x = [w, cx_0, cy_0, r_0, cx_1, cy_1, r_1, ..., cx_20, cy_20, r_20]
    That's 1 + 3*21 = 64 variables.
    
    Constraints:
    - w + h = 2, but h is not a variable... wait, we have w and h as the rectangle dimensions.
      Actually, we can parameterize by w (and h = 2 - w). So w is one variable, h is derived.
    - For each circle i: 
      cx_i - r_i >= 0
      cx_i + r_i <= w
      cy_i - r_i >= 0
      cy_i + r_i <= h = 2 - w
    - For each pair (i,j): (cx_i - cx_j)^2 + (cy_i - cy_j)^2 >= (r_i + r_j)^2
    
    Objective: minimize -sum(r_i)
    
    Let's use SLSQP with these constraints.
    """
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Parameterize: w (width), h = 2 - w
    # Variables: [w, cx_0, cy_0, r_0, cx_1, cy_1, r_1, ...]
    # Total: 1 + 3*N = 64 variables
    
    # Initial guess: hexagonal lattice
    # Start with a reasonable rectangle, say w=1, h=1 (square, perimeter 4)
    w_init = 1.0
    h_init = 1.0
    
    # Generate hexagonal lattice points in [0, w] x [0, h]
    # For a hex lattice, spacing s, rows offset
    # Let's just use a grid for simplicity initially, then optimize
    
    # Simple grid initialization
    rows = 5
    cols = 5
    xs = np.linspace(0.1, w_init - 0.1, cols)
    ys = np.linspace(0.1, h_init - 0.1, rows)
    
    # Take first N points
    points = []
    for i in range(rows):
        for j in range(cols):
            if len(points) < N:
                points.append((xs[j], ys[i]))
    
    # Initial radii: small
    r_init = 0.05
    
    x0 = np.array([w_init] + [p[0] for p in points] + [p[1] for p in points] + [r_init]*N)
    # Wait, that's not the right ordering. Let me fix:
    # x = [w, cx_0, cy_0, r_0, cx_1, cy_1, r_1, ...]
    x0 = np.zeros(1 + 3*N)
    x0[0] = w_init
    for i in range(N):
        x0[1 + 3*i] = points[i][0]
        x0[2 + 3*i] = points[i][1]
        x0[3 + 3*i] = r_init
    
    # Define objective and constraints
    def objective(x):
        w = x[0]
        h = 2.0 - w
        r_sum = 0.0
        for i in range(N):
            r_sum += x[3 + 3*i]
        return -r_sum
    
    def constraints(x):
        cons = []
        w = x[0]
        h = 2.0 - w
        
        # Rectangle aspect ratio bounds: w in [0.3, 1.7] to keep h in [0.3, 1.7]
        cons.append({'type': 'ineq', 'fun': lambda x: x[0] - 0.3, 'jac': None})
        cons.append({'type': 'ineq', 'fun': lambda x: 1.7 - x[0], 'jac': None})
        
        for i in range(N):
            cx = x[1 + 3*i]
            cy = x[2 + 3*i]
            r = x[3 + 3*i]
            
            # Inside rectangle
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[1+3*i] - x[3+3*i], 'jac': None})
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[0] - x[1+3*i] - x[3+3*i], 'jac': None})
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2+3*i] - x[3+3*i], 'jac': None})
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: (2.0 - x[0]) - x[2+3*i] - x[3+3*i], 'jac': None})
            
            # Positive radius
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[3+3*i] - 1e-6, 'jac': None})
        
        # Non-overlap
        for i in range(N):
            for j in range(i+1, N):
                def nonoverlap(x, i=i, j=j):
                    dx = x[1+3*i] - x[1+3*j]
                    dy = x[2+3*i] - x[2+3*j]
                    dist_sq = dx*dx + dy*dy
                    r_sum = x[3+3*i] + x[3+3*j]
                    return dist_sq - r_sum*r_sum
                cons.append({'type': 'ineq', 'fun': nonoverlap, 'jac': None})
        
        return cons
    
    # Bounds
    bounds = [(0.3, 1.7)]  # w
    for i in range(N):
        bounds.append((0, 2.0))  # cx
        bounds.append((0, 2.0))  # cy
        bounds.append((1e-6, 1.0))  # r
    
    # Run SLSQP
    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints,
                       options={'maxiter': 200, 'ftol': 1e-8})
        
        if time.time() > deadline:
            # Return best found so far
            best_x = x0
        else:
            best_x = res.x
    except Exception:
        best_x = x0
    
    # Extract result
    w = best_x[0]
    h = 2.0 - w
    
    circles = []
    for i in range(N):
        cx = best_x[1 + 3*i]
        cy = best_x[2 + 3*i]
        r = best_x[3 + 3*i]
        # Clamp to be inside rectangle
        r = max(1e-6, min(r, cx, w - cx, cy, h - cy))
        cx = min(max(cx, r), w - r)
        cy = min(max(cy, r), h - r)
        circles.append((cx, cy, r))
    
    result = np.array(circles)
    
    # Verify perimeter
    if w + h > 2.0 + 1e-9:
        # Scale down if needed
        scale = 2.0 / (w + h)
        result[:, 0] *= scale
        result[:, 1] *= scale
        result[:, 2] *= scale
    
    return result
# EVOLVE-BLOCK-END
