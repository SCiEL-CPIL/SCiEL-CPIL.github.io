"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill
    
    # We pack 21 circles in a rectangle of perimeter 4 (2*(w+h)=4 => w+h=2)
    # We want to maximize sum of radii subject to:
    # 1. Circles are inside the rectangle: r_i <= x_i, r_i <= w-x_i, r_i <= y_i, r_i <= h-y_i
    # 2. Circles are disjoint: (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
    # 3. w + h = 2
    
    # Strategy: Use a grid-based initial guess and then optimize using scipy's minimize
    # with a penalty method or direct constraint handling.
    
    # Let's parameterize the rectangle width w, and h = 2 - w.
    # We'll optimize positions (x_i, y_i) and radii r_i and width w.
    # Variables: w, x_1..x_21, y_1..y_21, r_1..r_21 -> total 1 + 21*3 = 64 variables
    
    # Objective: maximize sum(r_i) -> minimize -sum(r_i)
    # Constraints:
    #   r_i <= x_i
    #   r_i <= w - x_i
    #   r_i <= y_i
    #   r_i <= h - y_i = (2-w) - y_i
    #   (x_i-x_j)^2 + (y_i-y_j)^2 >= (r_i+r_j)^2 for i<j
    
    # Since scipy's SLSQP can handle nonlinear constraints, let's try that.
    # But 64 variables and ~210 pairwise constraints might be slow.
    
    # Alternative: Fix w and h initially, optimize positions and radii.
    # Or use a simpler approach: start with a good grid packing and iteratively improve.
    
    # Let's try a different approach: use the fact that for dense packings,
    # circles often touch the boundary or each other.
    
    # Initial guess: 5x5 grid with 21 circles (last row has 1)
    # Actually, let's use a more symmetric initial packing.
    
    # For 21 circles, a 4x5 grid (20 circles) + 1 in the middle? Or 5x4?
    # Let's use 5 columns, 4 rows = 20, plus 1 extra.
    
    # Better: use a hexagonal-like packing as initial guess.
    
    # Let's start with a simple approach:
    # 1. Initialize with a grid packing
    # 2. Use gradient-based optimization to maximize sum of radii
    
    n = N
    
    # Initial width and height
    w0 = 1.0
    h0 = 1.0
    
    # Create grid positions
    cols = 5
    rows = 5
    # We have 21 circles, so use 5x5 grid but only take 21
    positions = []
    for i in range(cols):
        for j in range(rows):
            x = (i + 0.5) / cols * w0
            y = (j + 0.5) / rows * h0
            positions.append([x, y])
    positions = positions[:n]
    
    # Initial radii: limited by nearest neighbor distance and boundary
    # For a grid, max radius is half the minimum cell size
    r0 = 0.5 / max(cols, rows) * 0.9  # small margin
    
    # Build initial variable vector: [w, x_1, y_1, r_1, x_2, y_2, r_2, ...]
    x0 = [w0]
    for i in range(n):
        x0.append(positions[i][0])
        x0.append(positions[i][1])
        x0.append(r0)
    x0 = np.array(x0)
    
    # Objective: minimize -sum(r_i)
    def objective(v):
        w = v[0]
        h = 2.0 - w
        # Ensure w and h are positive
        if w <= 0 or h <= 0:
            return 1e6
        r_sum = 0.0
        for i in range(n):
            r_sum += v[1 + 3*i + 2]  # r_i is at index 1 + 3*i + 2
        return -r_sum
    
    # Constraints (inequality constraints: g(v) >= 0)
    def constraints(v):
        w = v[0]
        h = 2.0 - w
        if w <= 0 or h <= 0:
            return [ -1.0 ]  # infeasible
        
        cons = []
        
        # Boundary constraints: r_i <= x_i, r_i <= w-x_i, r_i <= y_i, r_i <= h-y_i
        for i in range(n):
            xi = v[1 + 3*i]
            yi = v[1 + 3*i + 1]
            ri = v[1 + 3*i + 2]
            cons.append(xi - ri)
            cons.append(w - xi - ri)
            cons.append(yi - ri)
            cons.append(h - yi - ri)
        
        # Pairwise distance constraints: (xi-xj)^2 + (yi-yj)^2 >= (ri+rj)^2
        for i in range(n):
            for j in range(i+1, n):
                xi = v[1 + 3*i]
                yi = v[1 + 3*i + 1]
                ri = v[1 + 3*i + 2]
                xj = v[1 + 3*j]
                yj = v[1 + 3*j + 1]
                rj = v[1 + 3*j + 2]
                dx = xi - xj
                dy = yi - yj
                dist_sq = dx*dx + dy*dy
                rad_sum = ri + rj
                cons.append(dist_sq - rad_sum*rad_sum)
        
        return cons
    
    # Bounds: w in (0, 2), positions in (0, w) or (0, h) but w is variable...
    # This is tricky. Let's use loose bounds and rely on constraints.
    bounds = [(1e-3, 1.999)]  # w
    for i in range(n):
        bounds.append((1e-3, 1.999))  # x_i
        bounds.append((1e-3, 1.999))  # y_i
        bounds.append((1e-3, 1.0))     # r_i
    
    # Use SLSQP
    try:
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=[{'type': 'ineq', 'fun': constraints}],
            options={'maxiter': 200, 'ftol': 1e-6}
        )
        
        if result.success:
            v = result.x
            w = v[0]
            h = 2.0 - w
            circles = []
            for i in range(n):
                xi = v[1 + 3*i]
                yi = v[1 + 3*i + 1]
                ri = v[1 + 3*i + 2]
                circles.append([xi, yi, ri])
            return np.array(circles)
        
    except Exception as e:
        pass
    
    # Fallback: return the initial grid packing
    circles = []
    for i in range(n):
        circles.append([positions[i][0], positions[i][1], r0])
    return np.array(circles)
# EVOLVE-BLOCK-END
