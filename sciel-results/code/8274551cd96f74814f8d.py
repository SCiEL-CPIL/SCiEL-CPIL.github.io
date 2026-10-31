"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
import numpy as np
from scipy.optimize import minimize, Bounds
import time

N = 21

# Deadline guard
DEADLINE = 150.0
START_TIME = time.time()

def elapsed():
    return time.time() - START_TIME

def deadline_exceeded(margin=5.0):
    return elapsed() > DEADLINE - margin

def compute_rect_dims(radii):
    """Given radii, find best rectangle dimensions with perimeter 4.
    For a set of circles, we need a rectangle W x H with 2(W+H)=4, so W+H=2.
    We want to find W, H such that all circles fit.
    """
    pass  # Not directly used; we optimize positions and radii jointly.

def objective_and_constraints(x, W):
    """
    x = [x_1, y_1, r_1, x_2, y_2, r_2, ..., x_N, y_N, r_N]
    H = 2 - W
    Minimize -sum(r_i)
    Constraints:
      - Circle i inside rectangle: 
        x_i - r_i >= 0, x_i + r_i <= W
        y_i - r_i >= 0, y_i + r_i <= H
      - Circle i, j disjoint: (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
      - r_i > 0
    """
    H = 2.0 - W
    n = N
    x = x.reshape(n, 3)
    cx = x[:, 0]
    cy = x[:, 1]
    r = x[:, 2]
    
    # Objective: maximize sum of radii -> minimize negative
    obj = -np.sum(r)
    
    # Inequality constraints (each must be >= 0)
    cons = []
    
    # Wall constraints
    for i in range(n):
        # x_i - r_i >= 0
        cons.append({'type': 'ineq', 'fun': lambda i=i, cx=cx, r=r: cx[i] - r[i]})
        # W - x_i - r_i >= 0
        cons.append({'type': 'ineq', 'fun': lambda i=i, cx=cx, r=r: W - cx[i] - r[i]})
        # y_i - r_i >= 0
        cons.append({'type': 'ineq', 'fun': lambda i=i, cy=cy, r=r: cy[i] - r[i]})
        # H - y_i - r_i >= 0
        cons.append({'type': 'ineq', 'fun': lambda i=i, cy=cy, r=r: H - cy[i] - r[i]})
    
    # Circle-circle constraints
    for i in range(n):
        for j in range(i+1, n):
            cons.append({
                'type': 'ineq',
                'fun': lambda i=i, j=j, cx=cx, cy=cy, r=r: 
                    (cx[i] - cx[j])**2 + (cy[i] - cy[j])**2 - (r[i] + r[j])**2
            })
    
    return obj, cons

def solve_packing(W, initial_positions, initial_radii, max_iter=500):
    """Solve the packing problem for a given width W."""
    H = 2.0 - W
    n = N
    
    # Initial guess
    x0 = np.zeros(n * 3)
    for i in range(n):
        x0[i*3] = initial_positions[i, 0]
        x0[i*3+1] = initial_positions[i, 1]
        x0[i*3+2] = initial_radii[i]
    
    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((0.001, W))      # x_i
        bounds.append((0.001, H))      # y_i
        bounds.append((0.001, min(W, H)/2))  # r_i
    
    # Constraints
    cons = []
    for i in range(n):
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3] - x[i*3+2]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: W - x[i*3] - x[i*3+2]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3+1] - x[i*3+2]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: H - x[i*3+1] - x[i*3+2]})
    
    for i in range(n):
        for j in range(i+1, n):
            cons.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: 
                    (x[i*3] - x[j*3])**2 + (x[i*3+1] - x[j*3+1])**2 - (x[i*3+2] + x[j*3+2])**2
            })
    
    # Optimize
    result = minimize(
        lambda x: -np.sum(x[2::3]),
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=cons,
        options={'maxiter': max_iter, 'ftol': 1e-8}
    )
    
    if result.success:
        x_opt = result.x.reshape(n, 3)
        total_r = np.sum(x_opt[:, 2])
        return x_opt, total_r
    else:
        return None, 0.0

def initial_grid_positions(W, H):
    """Create initial grid positions for N circles."""
    side = int(np.ceil(np.sqrt(N)))
    positions = np.zeros((N, 2))
    for i in range(N):
        row = i // side
        col = i % side
        positions[i, 0] = (col + 0.5) * W / side
        positions[i, 1] = (row + 0.5) * H / side
    radii = np.full(N, 0.5 * min(W, H) / side * 0.9)
    return positions, radii

def polish(x, W, H, max_iters=100):
    """Local polish: try to increase radii while maintaining feasibility."""
    n = N
    x = x.copy()
    for _ in range(max_iters):
        if deadline_exceeded(margin=10.0):
            break
        improved = False
        # Try small radial expansions
        for i in range(n):
            for dr in [0.005, 0.002, 0.001]:
                x_new = x.copy()
                x_new[i, 2] += dr
                # Check feasibility
                feasible = True
                # Wall checks
                if x_new[i, 0] - x_new[i, 2] < -1e-9 or x_new[i, 0] + x_new[i, 2] > W + 1e-9:
                    feasible = False
                if x_new[i, 1] - x_new[i, 2] < -1e-9 or x_new[i, 1] + x_new[i, 2] > H + 1e-9:
                    feasible = False
                # Circle-circle checks
                if feasible:
                    for j in range(n):
                        if j == i:
                            continue
                        dist2 = (x_new[i, 0] - x_new[j, 0])**2 + (x_new[i, 1] - x_new[j, 1])**2
                        min_dist = x_new[i, 2] + x_new[j, 2]
                        if dist2 < min_dist**2 - 1e-9:
                            feasible = False
                            break
                if feasible:
                    x = x_new
                    improved = True
                    break
    return x

def construct():
    global START_TIME
    START_TIME = time.time()
    
    best_x = None
    best_total_r = 0.0
    
    # Try different aspect ratios
    # W + H = 2, so W in (0, 2)
    # Square: W=1, H=1
    # Wide: W=1.5, H=0.5
    # Tall: W=0.5, H=1.5
    
    widths_to_try = [1.0, 0.9, 1.1, 0.8, 1.2, 0.7, 1.3]
    
    for W in widths_to_try:
        if deadline_exceeded(margin=20.0):
            break
        
        H = 2.0 - W
        if H <= 0.1:
            continue
        
        positions, radii = initial_grid_positions(W, H)
        x_opt, total_r = solve_packing(W, positions, radii, max_iter=300)
        
        if x_opt is not None and total_r > best_total_r:
            # Polish
            x_polished = polish(x_opt, W, H, max_iters=50)
            total_r_polished = np.sum(x_polished[:, 2])
            if total_r_polished > best_total_r:
                best_total_r = total_r_polished
                best_x = x_polished
    
    if best_x is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(N):
            row = i // side
            col = i % side
            x = (col + 0.5) / side
            y = (row + 0.5) / side
            r = 0.5 / side - 1e-3
            cells.append([x, y, r])
        return np.array(cells)
    
    return best_x
