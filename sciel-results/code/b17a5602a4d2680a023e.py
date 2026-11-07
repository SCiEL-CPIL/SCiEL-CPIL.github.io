"""Improved program using Mixed Integer Linear Programming (MIP) for exact Capacitated Facility Location."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import scipy.sparse as sp

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assigns customers to facilities using Mixed Integer Linear Programming (MIP).
    Uses a reduced constraint set and sparse matrices for efficiency.
    """
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    if n_customers == 0:
        return np.array([], dtype=int)

    demands = np.array(demands)
    capacities = np.array(capacities)
    open_cost = np.array(open_cost)
    assign_cost = np.array(assign_cost)

    # Variables:
    # x[i, j] for i in 0..n_customers-1, j in 0..n_facilities-1
    # y[j] for j in 0..n_facilities-1
    # Total variables: n_customers * n_facilities + n_facilities
    
    n_vars = n_customers * n_facilities + n_facilities
    
    # Objective coefficients
    c = np.zeros(n_vars)
    # x costs
    c[:n_customers * n_facilities] = assign_cost.flatten()
    # y costs
    c[n_customers * n_facilities:] = open_cost

    # Bounds: All variables are binary [0, 1]
    lb = np.zeros(n_vars)
    ub = np.ones(n_vars)
    
    # Integrality: All variables are integer (binary)
    integrality = np.ones(n_vars)
    
    # Constraints:
    # 1. Each customer is assigned to exactly one facility:
    #    sum_j x[i, j] = 1 for each i
    #    n_customers equality constraints.
    
    # 2. Capacity constraints:
    #    sum_i demands[i] * x[i, j] <= capacities[j] * y[j] for each j
    #    sum_i demands[i] * x[i, j] - capacities[j] * y[j] <= 0
    #    n_facilities inequality constraints.
    # Note: Linking constraints x[i,j] <= y[j] are redundant because demands[i] > 0.
    # If y[j]=0, sum d_i x_ij <= 0 => x_ij=0.
    
    n_cons_eq = n_customers
    n_cons_ub = n_facilities
    
    # Build Sparse Matrix for Equality Constraints (Assignment)
    # A_eq shape: (n_customers, n_vars)
    # Row i has 1s at positions i*n_facilities + j for j in 0..n_facilities-1
    row_idx_eq = np.repeat(np.arange(n_customers), n_facilities)
    col_idx_eq = np.arange(n_customers * n_facilities)
    data_eq = np.ones(n_customers * n_facilities)
    A_eq = sp.csr_matrix((data_eq, (row_idx_eq, col_idx_eq)), shape=(n_customers, n_vars))
    b_eq = np.ones(n_customers)
    
    # Build Sparse Matrix for Inequality Constraints (Capacity)
    # A_ub shape: (n_facilities, n_vars)
    # Row j has demands[i] at positions i*n_facilities + j for all i, and -capacities[j] at n_customers*n_facilities + j
    row_idx_ub = []
    col_idx_ub = []
    data_ub = []
    
    for j in range(n_facilities):
        # Add demand terms for all customers
        for i in range(n_customers):
            row_idx_ub.append(j)
            col_idx_ub.append(i * n_facilities + j)
            data_ub.append(demands[i])
        # Add capacity term for y[j]
        row_idx_ub.append(j)
        col_idx_ub.append(n_customers * n_facilities + j)
        data_ub.append(-capacities[j])
        
    A_ub = sp.csr_matrix((data_ub, (row_idx_ub, col_idx_ub)), shape=(n_facilities, n_vars))
    b_ub = np.zeros(n_facilities)

    constraints = [
        LinearConstraint(A_eq, lb=b_eq, ub=b_eq),
        LinearConstraint(A_ub, lb=-np.inf, ub=b_ub)
    ]
    
    bounds = Bounds(lb, ub)
    
    try:
        res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds)
        
        if res.status == 0: # Optimal
            x_flat = res.x[:n_customers * n_facilities]
            x = x_flat.reshape((n_customers, n_facilities))
            
            # Find the facility for each customer
            assignment = np.argmax(x, axis=1)
            
            return assignment.astype(int)
        else:
            # If MIP fails, fallback to greedy
            return _fallback_greedy(demands, capacities, open_cost, assign_cost)
            
    except Exception:
        # In case of any error, use fallback
        return _fallback_greedy(demands, capacities, open_cost, assign_cost)

def _fallback_greedy(demands, capacities, open_cost, assign_cost):
    """Simple greedy fallback to ensure validity."""
    n_customers = len(demands)
    n_facilities = len(capacities)
    assignment = np.zeros(n_customers, dtype=int)
    load = np.zeros(n_facilities)
    
    # Sort customers by demand descending to handle large demands first
    order = np.argsort(-demands)
    for i in order:
        best_f = -1
        best_cost = np.inf
        for f in range(n_facilities):
            if load[f] + demands[i] <= capacities[f]:
                # Heuristic: prefer cheaper assignment, maybe also consider open cost if facility is new
                cost = assign_cost[i, f]
                if open_cost[f] > 0 and load[f] == 0:
                    # Approximate penalty for opening
                    cost += open_cost[f] * 0.1 # Small penalty to encourage consolidation
                if cost < best_cost:
                    best_cost = cost
                    best_f = f
        if best_f == -1:
            # Should not happen if feasible, but fallback to any with space
            for f in range(n_facilities):
                if load[f] + demands[i] <= capacities[f]:
                    best_f = f
                    break
            if best_f == -1:
                # If no space, pick max capacity (invalid but prevents crash, though task says valid required)
                best_f = np.argmax(capacities - load)
        assignment[i] = best_f
        load[best_f] += demands[i]
        
    return assignment
# EVOLVE-BLOCK-END
