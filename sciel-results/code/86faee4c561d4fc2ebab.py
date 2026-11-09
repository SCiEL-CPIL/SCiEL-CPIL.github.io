"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def construct():
    """
    Pack 26 circles in the unit square maximizing sum of radii.
    Uses grid-based initialization with perturbation and L-BFGS-B optimization.
    """
    best_value = 0.0
    best_circles = None
    
    n = N
    # Number of random restarts to stay within time limit
    n_restarts = 12
    
    for restart in range(n_restarts):
        np.random.seed(restart * 999 + 42)
        
        # Initialize with a grid structure
        side = int(np.ceil(np.sqrt(n)))
        
        x = np.zeros(n)
        y = np.zeros(n)
        r = np.zeros(n)
        
        idx = 0
        for i in range(side):
            for j in range(side):
                if idx < n:
                    x[idx] = (i + 0.5) / side
                    y[idx] = (j + 0.5) / side
                    r[idx] = 0.5 / side
                    idx += 1
        
        # Add small random perturbation to break symmetry
        x += np.random.uniform(-0.02, 0.02, n)
        y += np.random.uniform(-0.02, 0.02, n)
        
        # Clip to be inside initially
        x = np.clip(x, r, 1 - r)
        y = np.clip(y, r, 1 - r)
        
        def objective(params):
            x = params[:n]
            y = params[n:2*n]
            r = params[2*n:]
            
            # Penalty for circles going outside the unit square
            penalty = 0.0
            for i in range(n):
                if x[i] - r[i] < 0:
                    penalty += (x[i] - r[i]) ** 2
                if x[i] + r[i] > 1:
                    penalty += (x[i] + r[i] - 1) ** 2
                if y[i] - r[i] < 0:
                    penalty += (y[i] - r[i]) ** 2
                if y[i] + r[i] > 1:
                    penalty += (y[i] + r[i] - 1) ** 2
            
            # Penalty for overlapping circles
            for i in range(n):
                for j in range(i+1, n):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist_sq = dx*dx + dy*dy
                    min_dist = r[i] + r[j]
                    if dist_sq < min_dist * min_dist:
                        dist = np.sqrt(dist_sq)
                        penalty += (min_dist - dist) ** 2
            
            # We want to maximize sum of radii, so minimize negative sum
            value = -np.sum(r)
            return value + 100.0 * penalty
        
        # Initial parameters
        params0 = np.concatenate([x, y, r])
        
        # Optimize
        try:
            result = minimize(objective, params0, method='L-BFGS-B', 
                            options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8})
        except Exception:
            continue
        
        # Check if the solution is valid
        x_opt = result.x[:n]
        y_opt = result.x[n:2*n]
        r_opt = result.x[2*n:]
        
        # Verify validity
        valid = True
        for i in range(n):
            if x_opt[i] - r_opt[i] < -1e-8 or x_opt[i] + r_opt[i] > 1 + 1e-8:
                valid = False
                break
            if y_opt[i] - r_opt[i] < -1e-8 or y_opt[i] + r_opt[i] > 1 + 1e-8:
                valid = False
                break
        
        if valid:
            for i in range(n):
                for j in range(i+1, n):
                    dx = x_opt[i] - x_opt[j]
                    dy = y_opt[i] - y_opt[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < r_opt[i] + r_opt[j] - 1e-8:
                        valid = False
                        break
                if not valid:
                    break
        
        if valid:
            # Clip to ensure exact feasibility
            r_opt = np.maximum(r_opt, 1e-9)
            x_opt = np.clip(x_opt, r_opt, 1 - r_opt)
            y_opt = np.clip(y_opt, r_opt, 1 - r_opt)
            
            # Re-verify after clipping
            valid2 = True
            for i in range(n):
                for j in range(i+1, n):
                    dx = x_opt[i] - x_opt[j]
                    dy = y_opt[i] - y_opt[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < r_opt[i] + r_opt[j] - 1e-8:
                        valid2 = False
                        break
                if not valid2:
                    break
            
            if valid2:
                value = np.sum(r_opt)
                if value > best_value:
                    best_value = value
                    best_circles = np.column_stack([x_opt, y_opt, r_opt])
    
    # If no valid solution found, fall back to a simple valid configuration
    if best_circles is None:
        # Simple grid arrangement
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    cx = (i + 0.5) / side
                    cy = (j + 0.5) / side
                    cr = 0.5 / side - 1e-4
                    cells.append([cx, cy, cr])
        best_circles = np.array(cells[:N])
    
    return best_circles
# EVOLVE-BLOCK-END
