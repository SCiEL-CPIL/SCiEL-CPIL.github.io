"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def construct():
    """
    Pack 26 circles with disjoint interiors inside [0,1]^2 maximizing sum of radii.
    Returns array of shape (N, 3) with rows (x, y, radius).
    """
    start_time = time.time()
    deadline = 110.0  # seconds, safety margin before 120s limit
    
    # Initial guess: grid placement
    def initial_grid():
        # Use a 5x6 grid or similar
        cols, rows = 5, 6
        xs = []
        ys = []
        rs = []
        for i in range(cols):
            for j in range(rows):
                if len(xs) < N:
                    x = (i + 0.5) / cols
                    y = (j + 0.5) / rows
                    # Radius limited by grid spacing and boundary
                    r = min(0.5/cols, 0.5/rows) * 0.95
                    xs.append(x)
                    ys.append(y)
                    rs.append(r)
        # Pad if needed
        while len(xs) < N:
            xs.append(0.5)
            ys.append(0.5)
            rs.append(0.05)
        return np.array(xs[:N]), np.array(ys[:N]), np.array(rs[:N])
    
    x0, y0, r0 = initial_grid()
    
    # Objective function: maximize sum of radii
    # We'll use scipy.optimize.minimize with constraints
    # Variables: x (N), y (N), r (N) -> 3N variables
    
    def objective(vars):
        x = vars[:N]
        y = vars[N:2*N]
        r = vars[2*N:]
        # Penalty for invalid configurations
        penalty = 0.0
        # Boundary constraints
        penalty += 100.0 * np.sum(np.maximum(0, r - x))
        penalty += 100.0 * np.sum(np.maximum(0, r - (1 - x)))
        penalty += 100.0 * np.sum(np.maximum(0, r - y))
        penalty += 100.0 * np.sum(np.maximum(0, r - (1 - y)))
        # Overlap constraints
        for i in range(N):
            for j in range(i+1, N):
                dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                overlap = (r[i] + r[j]) - dist
                if overlap > 0:
                    penalty += 100.0 * overlap**2
        # Negative sum of radii (minimize)
        return -np.sum(r) + penalty
    
    # Initial variables
    vars0 = np.concatenate([x0, y0, r0])
    
    # Try to optimize
    best_vars = vars0.copy()
    best_obj = objective(vars0)
    
    # Use multiple restarts with different initial conditions
    np.random.seed(42)
    
    for restart in range(5):
        if time.time() - start_time > deadline:
            break
        
        if restart == 0:
            v = vars0.copy()
        else:
            # Perturb initial solution
            v = vars0 + np.random.randn(3*N) * 0.05
            # Ensure radii are positive
            v[2*N:] = np.maximum(v[2*N:], 0.01)
            # Ensure within bounds
            v[:N] = np.clip(v[:N], 0.05, 0.95)
            v[N:2*N] = np.clip(v[N:2*N], 0.05, 0.95)
        
        try:
            result = minimize(objective, v, method='L-BFGS-B', 
                             options={'maxiter': 100, 'ftol': 1e-10})
            if result.fun < best_obj:
                best_obj = result.fun
                best_vars = result.x
        except Exception:
            pass
    
    # Extract best solution
    x_best = best_vars[:N]
    y_best = best_vars[N:2*N]
    r_best = best_vars[2*N:]
    
    # Final cleanup: ensure validity
    # Clip to bounds
    r_best = np.minimum(r_best, np.minimum(x_best, 1-x_best))
    r_best = np.minimum(r_best, np.minimum(y_best, 1-y_best))
    r_best = np.maximum(r_best, 1e-6)
    
    # Adjust positions if needed to avoid boundary violations
    x_best = np.clip(x_best, r_best, 1-r_best)
    y_best = np.clip(y_best, r_best, 1-r_best)
    
    # Check for overlaps and reduce radii if necessary
    for _ in range(10):
        valid = True
        for i in range(N):
            for j in range(i+1, N):
                dist = np.sqrt((x_best[i]-x_best[j])**2 + (y_best[i]-y_best[j])**2)
                if dist < r_best[i] + r_best[j] - 1e-9:
                    valid = False
                    # Reduce both radii
                    scale = dist / (r_best[i] + r_best[j]) * 0.99
                    r_best[i] *= scale
                    r_best[j] *= scale
        if valid:
            break
    
    return np.column_stack([x_best, y_best, r_best])

# EVOLVE-BLOCK-END
