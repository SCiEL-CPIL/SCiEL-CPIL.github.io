"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as opt
import time

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    N = len(demands)
    F = len(capacities)
    
    # Variable layout:
    # 0 .. N*F - 1 : x_{i,j}
    # N*F .. N*F + F - 1 : y_{j}
    
    total_vars = N * F + F
    
    # Objective coefficients
    c = np.zeros(total_vars)
    # x_{i,j} costs
    c[:N*F] = assign_cost.flatten()
    # y_{j} costs
    c[N*F:] = open_cost
    
    # Integrality
    integrality = np.ones(total_vars)
    
    # Bounds
    bounds = [(0, 1)] * total_vars
    
    # Constraints
    constraints = []
    
    # 1. Assignment constraints: sum_j x_{i,j} = 1 for each i
    # We can build this as a single sparse matrix or list of LinearConstraint.
    # For efficiency with SciPy, a sparse matrix is better if available, 
    # but for 100x20, dense is fine too. Let's use LinearConstraint objects.
    
    # Actually, building 100 separate LinearConstraint objects might be slow to construct?
    # No, 100 is small.
    
    # Alternatively, we can use a single LinearConstraint with a matrix.
    # Let's stick to the list for clarity, or build the matrix.
    # Building the matrix is likely faster for the solver interface.
    
    A_assign = np.zeros((N, N * F))
    for i in range(N):
        A_assign[i, i*F : (i+1)*F] = 1.0
        
    # This is dense. 100x2000 = 200,000 floats. Fine.
    
    # Capacity constraints: sum_i d_i x_{i,j} - C_j y_j <= 0
    # This couples x and y.
    # Let's build the full matrix A for all constraints.
    # Rows: N (assignment) + F (capacity)
    # Cols: N*F + F
    
    A = np.zeros((N + F, total_vars))
    
    # Assignment part
    A[:N, :N*F] = A_assign
    
    # Capacity part
    for j in range(F):
        row_idx = N + j
        # x coefficients: d_i for all i
        for i in range(N):
            A[row_idx, i*F + j] = demands[i]
        # y coefficient: -C_j
        A[row_idx, N*F + j] = -capacities[j]
        
    # Assignment bounds: [1, 1]
    # Capacity bounds: [-inf, 0]
    
    lower_bounds = np.zeros(N + F)
    upper_bounds = np.zeros(N + F)
    
    lower_bounds[:N] = 1.0
    upper_bounds[:N] = 1.0
    
    lower_bounds[N:] = -np.inf
    upper_bounds[N:] = 0.0
    
    cons = opt.LinearConstraint(A, lb=lower_bounds, ub=upper_bounds)
    
    # Solve
    try:
        res = opt.milp(
            c=c,
            constraints=cons,
            integrality=integrality,
            bounds=bounds,
            options={"time_limit": 110.0}
        )
        
        if res.status != 0 and res.status != 1:
            # Fallback to greedy if MILP fails
            return greedy_fallback(demands, capacities, open_cost, assign_cost)
            
        if res.x is None:
            return greedy_fallback(demands, capacities, open_cost, assign_cost)
            
        x = res.x[:N*F].reshape(N, F)
        
        # Determine assignment
        # x should be binary. Due to floating point, threshold at 0.5.
        assignment = np.argmax(x, axis=1)
        
        # Validate
        # Check capacities
        load = np.zeros(F)
        for i in range(N):
            load[assignment[i]] += demands[i]
            
        for j in range(F):
            if load[j] > capacities[j] + 1e-6:
                # Capacity violated, should not happen if MILP succeeded
                # But if it does, fallback
                return greedy_fallback(demands, capacities, open_cost, assign_cost)
                
        return assignment.astype(int)
        
    except Exception:
        return greedy_fallback(demands, capacities, open_cost, assign_cost)

def greedy_fallback(demands, capacities, open_cost, assign_cost):
    N = len(demands)
    F = len(capacities)
    result = np.zeros(N, dtype=int)
    load = np.zeros(F)
    for i in np.argsort(-demands):
        for f in np.argsort(assign_cost[i]):
            if load[f] + demands[i] <= capacities[f]:
                load[f] += demands[i]
                result[i] = f
                break
    return result

# EVOLVE-BLOCK-END
