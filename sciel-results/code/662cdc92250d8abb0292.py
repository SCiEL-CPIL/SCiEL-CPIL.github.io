"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assigns customers to facilities using MILP if available and fast, 
    otherwise falls back to a greedy heuristic.
    """
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Deadline guard: 150s limit, use 120s for safety margin
    start_time = time.time()
    deadline = 120.0
    
    # Try MILP first
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        import scipy.sparse as sp
        
        # Variables:
        # x_ij for i in 0..n_customers-1, j in 0..n_facilities-1
        # y_j for j in 0..n_facilities-1
        # Total variables: n_customers * n_facilities + n_facilities
        
        n_vars = n_customers * n_facilities + n_facilities
        
        # Objective coefficients
        # Minimize sum(open_cost[j] * y[j]) + sum(assign_cost[i][j] * x[i][j])
        c = np.zeros(n_vars)
        
        # Fill x coefficients
        for i in range(n_customers):
            for j in range(n_facilities):
                idx = i * n_facilities + j
                c[idx] = assign_cost[i][j]
        
        # Fill y coefficients
        for j in range(n_facilities):
            idx = n_customers * n_facilities + j
            c[idx] = open_cost[j]
        
        # Integrality: all binary
        integrality = np.ones(n_vars)
        
        # Constraints:
        # 1. Sum_j x_ij = 1 for each i
        # 2. Sum_i demand_i * x_ij <= capacity_j * y_j for each j
        
        # Constraint 1: Assignment
        # Shape: n_customers x n_vars
        A_assign = np.zeros((n_customers, n_vars))
        for i in range(n_customers):
            for j in range(n_facilities):
                idx = i * n_facilities + j
                A_assign[i, idx] = 1.0
        
        # Constraint 2: Capacity
        # Shape: n_facilities x n_vars
        # sum_i demand_i * x_ij - capacity_j * y_j <= 0
        A_cap = np.zeros((n_facilities, n_vars))
        for j in range(n_facilities):
            # Coefficients for x_ij
            for i in range(n_customers):
                idx = i * n_facilities + j
                A_cap[j, idx] = demands[i]
            # Coefficient for y_j
            idx_y = n_customers * n_facilities + j
            A_cap[j, idx_y] = -capacities[j]
        
        # Combine constraints
        A = np.vstack([A_assign, A_cap])
        
        # Bounds for linear constraints
        # A_assign * vars = 1 -> lb=1, ub=1
        # A_cap * vars <= 0 -> lb=-inf, ub=0
        lb = np.concatenate([np.ones(n_customers), -np.inf * np.ones(n_facilities)])
        ub = np.concatenate([np.ones(n_customers), np.zeros(n_facilities)])
        
        constraint = LinearConstraint(A, lb, ub)
        bounds = Bounds(0, 1)
        
        # Check time
        if time.time() - start_time > deadline - 5:
            raise TimeoutError("Time limit approaching")
            
        res = milp(c=c, integrality=integrality, constraints=constraint, bounds=bounds)
        
        if res.status == 0 and res.x is not None:
            # Extract solution
            x_flat = res.x[:n_customers * n_facilities].reshape(n_customers, n_facilities)
            # Find assigned facility for each customer
            result = np.argmax(x_flat, axis=1).astype(int)
            return result
            
    except Exception as e:
        # Fallback to greedy if MILP fails or is unavailable
        pass
        
    # Greedy Fallback: Largest demand first, cheapest feasible facility
    result = np.zeros(n_customers, dtype=int)
    load = np.zeros(n_facilities)
    
    # Sort customers by demand descending
    order = np.argsort(-demands)
    
    for i in order:
        # Sort facilities by assignment cost for this customer
        fac_order = np.argsort(assign_cost[i])
        assigned = False
        for f in fac_order:
            if load[f] + demands[i] <= capacities[f]:
                load[f] += demands[i]
                result[i] = f
                assigned = True
                break
        if not assigned:
            # If no facility has capacity, assign to the one with most remaining capacity
            # This might violate constraints, but it's a fallback for invalid inputs or bugs
            remaining = capacities - load
            f = np.argmax(remaining)
            result[i] = f
            load[f] += demands[i]
            
    return result
# EVOLVE-BLOCK-END
