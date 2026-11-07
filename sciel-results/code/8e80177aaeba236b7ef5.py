"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog


def assign(demands, capacities, open_cost, assign_cost):
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Deadline guard: 150s wall-clock, use 140s for safety
    start_time = time.time()
    deadline = start_time + 140.0
    
    def greedy_fallback():
        result = np.zeros(n_customers, dtype=int)
        load = np.zeros(n_facilities)
        for i in np.argsort(-demands):
            for f in np.argsort(assign_cost[i]):
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    result[i] = f
                    break
        return result, load

    # 1. Formulate LP Relaxation
    # Variables:
    # x[i, j] for i in [0, N), j in [0, F) -> index i*F + j
    # y[j] for j in [0, F) -> index N*F + j
    
    N = n_customers
    F = n_facilities
    n_vars = N * F + F
    
    c = np.zeros(n_vars)
    for i in range(N):
        for j in range(F):
            c[i * F + j] = assign_cost[i, j]
    for j in range(F):
        c[N * F + j] = open_cost[j]
    
    # Constraints:
    # 1. Assignment: sum_j x[i, j] = 1 for all i
    # 2. Capacity: sum_i demands[i] * x[i, j] <= capacities[j] for all j
    # 3. Linking: x[i, j] - y[j] <= 0 for all i, j
    
    A_eq = []
    b_eq = []
    A_ub = []
    b_ub = []
    
    # Assignment constraints (Equality)
    for i in range(N):
        row = np.zeros(n_vars)
        for j in range(F):
            row[i * F + j] = 1.0
        A_eq.append(row)
        b_eq.append(1.0)
    A_eq = np.array(A_eq)
    b_eq = np.array(b_eq)
    
    # Capacity constraints (Inequality)
    for j in range(F):
        row = np.zeros(n_vars)
        for i in range(N):
            row[i * F + j] = demands[i]
        A_ub.append(row)
        b_ub.append(capacities[j])
        
    # Linking constraints (Inequality)
    for i in range(N):
        for j in range(F):
            row = np.zeros(n_vars)
            row[i * F + j] = 1.0
            row[N * F + j] = -1.0
            A_ub.append(row)
            b_ub.append(0.0)
            
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # Bounds
    bounds = [(0, 1)] * n_vars
    
    # Solve LP
    try:
        res = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    except Exception:
        res = None
    
    initial_result = None
    initial_load = None
    
    if res is not None and res.status == 0:
        # Extract x and y
        x_flat = res.x[:N * F]
        y_vals = res.x[N * F:]
        x_matrix = x_flat.reshape(N, F)
        
        # Heuristic Rounding:
        # Sort customers by demand descending
        # For each customer, pick the facility with highest x[i,j] that has capacity
        # If no capacity, pick best feasible by cost
        initial_result = np.zeros(N, dtype=int)
        initial_load = np.zeros(F)
        
        # Pre-sort facilities for each customer by x value descending, then cost ascending
        # Actually, we just iterate customers by demand desc
        order = np.argsort(-demands)
        
        for i in order:
            # Get x values for customer i
            x_i = x_matrix[i]
            # Sort facilities by x_i descending
            sorted_f = np.argsort(-x_i)
            
            assigned = False
            for f in sorted_f:
                if initial_load[f] + demands[i] <= capacities[f]:
                    initial_load[f] += demands[i]
                    initial_result[i] = f
                    assigned = True
                    break
            
            if not assigned:
                # Fallback to cheapest feasible facility ignoring LP values
                for f in np.argsort(assign_cost[i]):
                    if initial_load[f] + demands[i] <= capacities[f]:
                        initial_load[f] += demands[i]
                        initial_result[i] = f
                        assigned = True
                        break
            
            if not assigned:
                # Critical failure, use pure greedy fallback for whole thing
                initial_result, initial_load = greedy_fallback()
                break
    else:
        # LP failed, use greedy
        initial_result, initial_load = greedy_fallback()
        
    # 2. Local Search (same as before)
    def calc_cost(assignment):
        total_assign = np.sum(assign_cost[np.arange(n_customers), assignment])
        used = set(assignment.tolist())
        total_open = np.sum(open_cost[list(used)])
        return total_assign + total_open
    
    def is_valid(assignment):
        for f in range(n_facilities):
            mask = assignment == f
            if mask.any():
                if demands[mask].sum() > capacities[f]:
                    return False
        return True
    
    best_assignment = initial_result.copy()
    load = initial_load.copy()
    best_cost = calc_cost(best_assignment)
    
    # Local search with moves and swaps
    improved = True
    while improved and time.time() < deadline - 10.0:
        improved = False
        for i in range(n_customers):
            if time.time() > deadline - 10.0:
                break
            current_f = best_assignment[i]
            # Try moving customer i to another facility
            for f in range(n_facilities):
                if f == current_f:
                    continue
                # Check capacity constraint
                if load[f] + demands[i] <= capacities[f]:
                    new_cost = best_cost - assign_cost[i, current_f] + assign_cost[i, f]
                    # Account for open cost changes
                    if current_f != f:
                        # Check if current_f becomes empty
                        others_at_current = np.sum(best_assignment == current_f)
                        others_at_new = np.sum(best_assignment == f)
                        if others_at_current == 1:
                            new_cost -= open_cost[current_f]
                        if others_at_new == 0:
                            new_cost += open_cost[f]
                    if new_cost < best_cost - 1e-9:
                        # Make the move
                        load[current_f] -= demands[i]
                        load[f] += demands[i]
                        best_assignment[i] = f
                        best_cost = new_cost
                        improved = True
                        break
        
        # Try swaps between pairs of customers
        if time.time() > deadline - 10.0:
            break
        for i in range(n_customers):
            if time.time() > deadline - 10.0:
                break
            for j in range(i + 1, n_customers):
                if time.time() > deadline - 10.0:
                    break
                fi = best_assignment[i]
                fj = best_assignment[j]
                if fi == fj:
                    continue
                # Check capacity constraints for swap
                if load[fi] - demands[i] + demands[j] <= capacities[fi] and \
                   load[fj] - demands[j] + demands[i] <= capacities[fj]:
                    new_cost = best_cost - assign_cost[i, fi] - assign_cost[j, fj] + \
                              assign_cost[i, fj] + assign_cost[j, fi]
                    # Open cost doesn't change in a swap (both facilities still used)
                    if new_cost < best_cost - 1e-9:
                        # Make the swap
                        load[fi] = load[fi] - demands[i] + demands[j]
                        load[fj] = load[fj] - demands[j] + demands[i]
                        best_assignment[i] = fj
                        best_assignment[j] = fi
                        best_cost = new_cost
                        improved = True
    
    # Final verification
    if not is_valid(best_assignment):
        # Fallback to greedy result
        return initial_result
    
    return best_assignment
# EVOLVE-BLOCK-END
