"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _feasible(circles, eps=1e-9):
    """Check if circles are disjoint and inside [0,1]^2."""
    n = len(circles)
    for i in range(n):
        x, y, r = circles[i]
        if x - r < -eps or x + r > 1 + eps or y - r < -eps or y + r > 1 + eps:
            return False
    for i in range(n):
        for j in range(i+1, n):
            xi, yi, ri = circles[i]
            xj, yj, rj = circles[j]
            d2 = (xi-xj)**2 + (yi-yj)**2
            if d2 < (ri+rj - eps)**2:
                return False
    return True

def _objective(circles, eps=1e-9):
    """Objective: maximize sum of radii (we minimize negative sum)."""
    return -sum(c[2] for c in circles)

def _constraints(circles, eps=1e-9):
    """Return list of constraint functions for scipy minimize."""
    n = len(circles)
    cons = []
    # Boundary constraints: x >= r, x <= 1-r, y >= r, y <= 1-r
    for i in range(n):
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i] - x[3*i+2] - eps})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[3*i] - x[3*i+2] - eps})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i+1] - x[3*i+2] - eps})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[3*i+1] - x[3*i+2] - eps})
    # Disjointness constraints: dist >= ri + rj
    for i in range(n):
        for j in range(i+1, n):
            cons.append({'type': 'ineq', 'fun': lambda x, i=i, j=j: 
                         (x[3*i]-x[3*j])**2 + (x[3*i+1]-x[3*j+1])**2 - (x[3*i+2]+x[3*j+2])**2 - eps})
    return cons

def _grid_init():
    """Create a grid of circles as initial guess."""
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    return cells[:N]

def _local_search(circles, max_iter=500, lr=0.01):
    """Simple gradient-free local search."""
    circles = [list(c) for c in circles]
    best = sum(c[2] for c in circles)
    
    for it in range(max_iter):
        improved = False
        for i in range(len(circles)):
            x, y, r = circles[i]
            # Try increasing radius
            new_r = r * 1.01
            if x - new_r >= 0 and x + new_r <= 1 and y - new_r >= 0 and y + new_r <= 1:
                # Check disjointness
                valid = True
                for j in range(len(circles)):
                    if i == j: continue
                    xj, yj, rj = circles[j]
                    d2 = (x-xj)**2 + (y-yj)**2
                    if d2 < (new_r+rj)**2:
                        valid = False
                        break
                if valid:
                    circles[i][2] = new_r
                    new_sum = sum(c[2] for c in circles)
                    if new_sum > best:
                        best = new_sum
                        improved = True
        if not improved:
            break
    
    return [tuple(c) for c in circles], best

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s
    
    # Initial grid
    circles = _grid_init()
    best_circles = [tuple(c) for c in circles]
    best_sum = sum(c[2] for c in circles)
    
    # Local search to improve
    circles, best_sum = _local_search(circles, max_iter=200)
    best_circles = circles
    
    # Try scipy optimization with smooth constraints
    if time.time() < deadline - 30:
        try:
            x0 = np.array([c for circle in circles for c in circle])
            bounds = [(0, 1), (0, 1), (0, 0.5)] * N
            
            # Use SLSQP with constraints
            cons = _constraints(circles)
            res = minimize(_objective, x0, method='SLSQP', bounds=bounds, 
                          constraints=cons, options={'maxiter': 100, 'ftol': 1e-8})
            
            if res.success:
                new_circles = [(res.x[3*i], res.x[3*i+1], res.x[3*i+2]) for i in range(N)]
                if _feasible(new_circles):
                    new_sum = sum(c[2] for c in new_circles)
                    if new_sum > best_sum:
                        best_sum = new_sum
                        best_circles = new_circles
        except:
            pass
    
    # Final local search polish
    if time.time() < deadline - 10:
        circles, best_sum = _local_search(best_circles, max_iter=300)
        best_circles = circles
    
    return np.array(best_circles)
# EVOLVE-BLOCK-END
