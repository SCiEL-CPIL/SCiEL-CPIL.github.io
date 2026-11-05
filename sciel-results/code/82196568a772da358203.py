import time
import math
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin
    
    N = 256
    elements = list(range(1, N + 1))
    
    # Identify bad pairs
    bad_pairs = []
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            g = math.gcd(i, j)
            x = i // g
            y = j // g
            if g % (x + y) == 0:
                bad_pairs.append((i - 1, j - 1)) # 0-indexed for array
    
    # If no bad pairs, return all
    if not bad_pairs:
        return elements
        
    # Fallback if time is tight or MILP fails
    # Greedy heuristic: sort by degree, pick lowest degree first?
    # Or just return the seed solution if we run out of time.
    
    # Try MILP
    try:
        n_vars = N
        # Objective: maximize sum x_i -> minimize -sum x_i
        c = -np.ones(n_vars)
        
        # Constraints: x_i + x_j <= 1 for all bad pairs
        if len(bad_pairs) > 0:
            A_ub = np.zeros((len(bad_pairs), n_vars))
            for k, (i, j) in enumerate(bad_pairs):
                A_ub[k, i] = 1
                A_ub[k, j] = 1
            b_ub = np.ones(len(bad_pairs))
            
            constraints = [LinearConstraint(A_ub, -np.inf, b_ub)]
        else:
            constraints = []
            
        bounds = Bounds(lb=0, ub=1)
        integrality = np.ones(n_vars) # All integer
        
        res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds)
        
        if res.status == 0: # Optimal
            selected = [i + 1 for i in range(n_vars) if res.x[i] > 0.5]
            if selected:
                return selected
        else:
            # Fallback to seed
            return list(range(1, N + 1, 2))
            
    except Exception as e:
        # Fallback to seed
        return list(range(1, N + 1, 2))

    # If we reach here, return seed
    return list(range(1, N + 1, 2))
