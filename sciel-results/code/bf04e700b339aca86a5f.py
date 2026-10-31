"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 110s safety margin
    
    # Initial grid placement
    side = int(np.ceil(np.sqrt(N)))
    positions = []
    for i in range(side):
        for j in range(side):
            positions.append([(i + 0.5) / side, (j + 0.5) / side])
    positions = positions[:N]
    
    # Convert to numpy array
    x = np.array(positions)
    
    # Objective: maximize sum of radii
    # Constraint: circles must be disjoint and inside unit square
    
    def compute_radii(x):
        """Compute maximum possible radii for given positions."""
        n = len(x)
        radii = np.zeros(n)
        
        for i in range(n):
            # Distance to boundaries
            r_boundary = min(x[i, 0], 1 - x[i, 0], x[i, 1], 1 - x[i, 1])
            
            # Distance to other circles
            min_dist = r_boundary
            for j in range(n):
                if i == j:
                    continue
                dist = np.sqrt((x[i, 0] - x[j, 0])**2 + (x[i, 1] - x[j, 1])**2)
                min_dist = min(min_dist, dist / 2.0)
            
            radii[i] = min_dist
        
        return radii
    
    def objective(x_flat):
        """Negative sum of radii (we minimize negative sum)."""
        x = x_flat.reshape(-1, 2)
        radii = compute_radii(x)
        return -np.sum(radii)
    
    # Use gradient-free optimization with multiple restarts
    best_x = x.copy()
    best_value = -objective(x.flatten())
    
    # Try different initial configurations and optimize
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Perturb initial positions
        if restart == 0:
            x_current = x.copy()
        else:
            x_current = x + np.random.randn(N, 2) * 0.1
            # Ensure within bounds
            x_current = np.clip(x_current, 0.05, 0.95)
        
        # Optimize using Nelder-Mead
        result = minimize(
            objective,
            x_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8}
        )
        
        if time.time() > deadline:
            break
        
        x_opt = result.x.reshape(-1, 2)
        value = -result.fun
        
        if value > best_value:
            best_value = value
            best_x = x_opt
    
    # Compute final radii
    radii = compute_radii(best_x)
    
    # Build result array
    result = np.zeros((N, 3))
    for i in range(N):
        result[i, 0] = best_x[i, 0]
        result[i, 1] = best_x[i, 1]
        result[i, 2] = radii[i]
    
    return result
# EVOLVE-BLOCK-END
