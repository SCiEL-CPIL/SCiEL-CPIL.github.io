"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import itertools

def construct():
    """
    Pack 26 circles in the unit square [0,1]^2 maximizing the sum of radii.
    Uses gradient-based optimization with soft constraints for non-overlap
    and boundary constraints.
    """
    n = N
    np.random.seed(42)
    
    # Initial placement: grid-like with some randomization
    side = int(np.ceil(np.sqrt(n)))
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) < n:
                x = (i + 0.5) / side + np.random.uniform(-0.1, 0.1) / side
                y = (j + 0.5) / side + np.random.uniform(-0.1, 0.1) / side
                positions.append([x, y])
    
    # Ensure we have exactly n positions
    while len(positions) < n:
        positions.append([np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9)])
    
    positions = np.array(positions[:n])
    initial_radius = 0.5 / side - 0.02
    
    # Parameters: x, y, r for each circle
    # Total parameters: 3*n
    def unpack(params):
        x = params[0:n]
        y = params[n:2*n]
        r = params[2*n:3*n]
        return x, y, r
    
    def objective(params):
        x, y, r = unpack(params)
        # Maximize sum of radii -> minimize negative sum
        return -np.sum(r)
    
    # Constraints: 
    # 1. No overlap: for each pair (i,j), dist(i,j) >= r_i + r_j
    # 2. Boundary: r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
    
    def constraints(params):
        x, y, r = unpack(params)
        cons = []
        
        # Non-overlap constraints (use soft form for differentiability)
        for i in range(n):
            for j in range(i+1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy + 1e-12)
                # Constraint: dist - r_i - r_j >= 0
                cons.append(dist - r[i] - r[j])
        
        # Boundary constraints
        for i in range(n):
            cons.append(x[i] - r[i])
            cons.append(1 - x[i] - r[i])
            cons.append(y[i] - r[i])
            cons.append(1 - y[i] - r[i])
        
        return cons
    
    def constraint_functions(params):
        x, y, r = unpack(params)
        cons = []
        
        for i in range(n):
            for j in range(i+1, n):
                def c(p, i=i, j=j):
                    xi, yj, ri, rj = p[0:n], p[n:2*n], p[2*n:3*n], p[2*n:3*n]
                    dx = xi[i] - xi[j]
                    dy = yj[i] - yj[j]
                    dist = np.sqrt(dx*dx + dy*dy + 1e-12)
                    return dist - ri[i] - rj[j]
                cons.append(c)
        
        for i in range(n):
            def c_left(p, i=i):
                return p[i] - p[2*n + i]
            def c_right(p, i=i):
                return 1 - p[i] - p[2*n + i]
            def c_bottom(p, i=i):
                return p[n + i] - p[2*n + i]
            def c_top(p, i=i):
                return 1 - p[n + i] - p[2*n + i]
            cons.extend([c_left, c_right, c_bottom, c_top])
        
        return cons
    
    # Use SLSQP with penalty-based approach instead
    def penalized_objective(params, mu=100.0):
        x, y, r = unpack(params)
        
        # Ensure radii are positive
        r = np.maximum(r, 1e-6)
        
        penalty = 0.0
        
        # Non-overlap penalty
        for i in range(n):
            for j in range(i+1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy + 1e-12)
                overlap = max(0, r[i] + r[j] - dist)
                penalty += overlap**2
        
        # Boundary penalty
        for i in range(n):
            if x[i] < r[i]:
                penalty += (r[i] - x[i])**2
            if 1 - x[i] < r[i]:
                penalty += (r[i] - (1 - x[i]))**2
            if y[i] < r[i]:
                penalty += (r[i] - y[i])**2
            if 1 - y[i] < r[i]:
                penalty += (r[i] - (1 - y[i]))**2
        
        return -np.sum(r) + mu * penalty
    
    # Initial parameters
    x0 = np.concatenate([positions[:, 0], positions[:, 1], np.full(n, initial_radius)])
    
    # Bounds: x, y in [0, 1], r in [0, 0.5]
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))  # x
    for i in range(n):
        bounds.append((0.0, 1.0))  # y
    for i in range(n):
        bounds.append((1e-4, 0.5))  # r
    
    # Optimize
    best_params = x0.copy()
    best_val = penalized_objective(x0, mu=50.0)
    
    # Multiple restarts with different mu values
    for mu in [10.0, 50.0, 100.0, 200.0, 500.0]:
        res = minimize(
            penalized_objective,
            x0,
            args=(mu,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        if res.fun < best_val:
            best_val = res.fun
            best_params = res.x
    
    # Refine with higher mu
    for mu in [500.0, 1000.0, 2000.0]:
        res = minimize(
            penalized_objective,
            best_params,
            args=(mu,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        if res.fun < best_val:
            best_val = res.fun
            best_params = res.x
    
    x, y, r = unpack(best_params)
    
    # Final cleanup: ensure validity
    r = np.maximum(r, 1e-6)
    
    # Clip positions to ensure they're within bounds considering radius
    for i in range(n):
        x[i] = np.clip(x[i], r[i], 1 - r[i])
        y[i] = np.clip(y[i], r[i], 1 - r[i])
    
    # Check for any remaining overlaps and shrink if needed
    for _ in range(10):
        adjusted = False
        for i in range(n):
            for j in range(i+1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                required = r[i] + r[j]
                if dist < required:
                    # Shrink both radii slightly
                    scale = dist / (required + 1e-12)
                    r[i] *= scale
                    r[j] *= scale
                    adjusted = True
        if not adjusted:
            break
    
    result = np.column_stack([x, y, r])
    return result

# EVOLVE-BLOCK-END
