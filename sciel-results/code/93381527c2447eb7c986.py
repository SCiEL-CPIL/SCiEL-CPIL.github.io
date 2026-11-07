import numpy as np
from scipy.optimize import linprog
import time

def assign(demands, capacities, open_cost, assign_cost):
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    demands = np.asarray(demands, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.float64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)
    
    # Variables:
    # x[i, j] for i in 0..n-1, j in 0..m-1  -> index i*m + j
    # y[j] for j in 0..m-1 -> index n*m + j
    
    n_vars = n_customers * n_facilities + n_facilities
    
    # Objective coefficients
    c_obj = np.zeros(n_vars)
    c_obj[:n_customers * n_facilities] = assign_cost.flatten()
    c_obj[n_customers * n_facilities:] = open_cost
    
    # Inequality constraints A_ub x <= b_ub
    # 1. Capacity: sum_i d_i x_ij <= C_j y_j  => sum_i d_i x_ij - C_j y_j <= 0
    # 2. Linking: x_ij <= y_j  => x_ij - y_j <= 0
    
    n_constraints = n_facilities + n_customers * n_facilities
    A_ub = np.zeros((n_constraints, n_vars))
    b_ub = np.zeros(n_constraints)
    
    # Fill capacity constraints
    for j in range(n_facilities):
        row = j
        for i in range(n_customers):
            A_ub[row, i * n_facilities + j] = demands[i]
        A_ub[row, n_customers * n_facilities + j] = -capacities[j]
        b_ub[row] = 0.0
        
    # Fill linking constraints
    for i in range(n_customers):
        for j in range(n_facilities):
            row = n_facilities + i * n_facilities + j
            A_ub[row, i * n_facilities + j] = 1.0
            A_ub[row, n_customers * n_facilities + j] = -1.0
            b_ub[row] = 0.0
            
    # Equality constraints A_eq x = b_eq
    # sum_j x_ij = 1
    A_eq = np.zeros((n_customers, n_vars))
    b_eq = np.ones(n_customers)
    
    for i in range(n_customers):
        for j in range(n_facilities):
            A_eq[i, i * n_facilities + j] = 1.0
            
    # Bounds
    bounds = [(0, 1)] * n_vars
    
    # Solve LP
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        if res.status != 0:
            raise Exception("LP failed")
        
        x_flat = res.x[:n_customers * n_facilities].reshape(n_customers, n_facilities)
        y = res.x[n_customers * n_facilities:]
        
    except Exception:
        # Fallback to greedy if LP fails
        assignment = np.zeros(n_customers, dtype=int)
        load = np.zeros(n_facilities)
        order = np.argsort(-demands)
        for i in order:
            feasible = np.where(load + demands[i] <= capacities)[0]
            if len(feasible) == 0:
                f = np.argmax(capacities - load)
            else:
                costs = assign_cost[i, feasible]
                f = feasible[np.argmin(costs)]
            assignment[i] = f
            load[f] += demands[i]
        return assignment

    # Post-processing: Convert fractional LP solution to integer assignment
    # 1. Determine open facilities
    # Threshold 0.1 to include likely open facilities
    open_facilities = np.where(y > 0.1)[0]
    if len(open_facilities) == 0:
        open_facilities = np.array([np.argmax(y)])
        
    # 2. Assign customers
    assignment = np.zeros(n_customers, dtype=int)
    load = np.zeros(n_facilities)
    
    # Sort customers by decreasing demand to fill capacities efficiently
    order = np.argsort(-demands)
    
    for i in order:
        best_f = -1
        best_score = -np.inf
        
        # First try open facilities
        for j in open_facilities:
            if load[j] + demands[i] <= capacities[j]:
                score = x_flat[i, j]
                if best_f == -1 or score > best_score or (abs(score - best_score) < 1e-6 and assign_cost[i, j] < assign_cost[i, best_f]):
                    best_score = score
                    best_f = j
                    
        if best_f == -1:
            # No capacity in preferred open facilities.
            # Try to open a new facility or use any facility with capacity
            rem_cap = capacities - load
            candidates = np.argsort(-rem_cap)
            for j in candidates:
                if load[j] + demands[i] <= capacities[j]:
                    best_f = j
                    if j not in open_facilities:
                        open_facilities = np.append(open_facilities, j)
                    break
                    
        if best_f == -1:
            # Should not happen
            best_f = np.argmax(capacities - load)
            
        assignment[i] = best_f
        load[best_f] += demands[i]
        
    return assignment
