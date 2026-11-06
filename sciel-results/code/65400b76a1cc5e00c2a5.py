import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from math import lcm

def construct():
    numbers = list(range(2, 257))
    n = len(numbers)
    
    # Map value to index
    val_to_idx = {v: i for i, v in enumerate(numbers)}
    
    # Objective coefficients
    c = np.array([-1.0 / v for v in numbers]) # Minimize -sum(1/v)
    
    # Constraints: x_i + x_j <= 1 for conflicts
    # We need to build the constraint matrix A_ub and b_ub
    # A_ub is sparse? milp accepts LinearConstraint which can take sparse matrices.
    # Let's use scipy.sparse.
    
    import scipy.sparse as sp
    
    rows = []
    cols = []
    data = []
    
    for i in range(n):
        for j in range(i + 1, n):
            if lcm(numbers[i], numbers[j]) <= 256:
                # Constraint: x_i + x_j <= 1
                rows.append(len(data))
                cols.append(i)
                data.append(1.0)
                
                rows.append(len(data))
                cols.append(j)
                data.append(1.0)
                
    A_ub = sp.csr_matrix((data, (rows, cols)), shape=(len(data), n))
    b_ub = np.ones(len(data))
    
    # Bounds: 0 <= x_i <= 1
    bounds = Bounds(lb=0, ub=1)
    
    # Integrality: all binary
    integrality = np.ones(n)
    
    # Solve
    res = milp(c=c, constraints=LinearConstraint(A_ub, ub=b_ub), bounds=bounds, integrality=integrality)
    
    if res.status != 0:
        # Fallback to greedy if solver fails
        selected = []
        for x in sorted(numbers):
            compatible = True
            for y in selected:
                if lcm(x, y) <= 256:
                    compatible = False
                    break
            if compatible:
                selected.append(x)
        return selected
        
    # Extract solution
    selected = [numbers[i] for i in range(n) if res.x[i] > 0.5]
    
    return selected
