"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import milp, Bounds, LinearConstraint

def _local_search(assignment, demands, capacities, open_cost, assign_cost, deadline):
    """
    Improve assignment via move and swap local search.
    Returns improved assignment.
    """
    n_customers = len(assignment)
    n_facilities = len(capacities)
    
    # Compute loads
    load = np.zeros(n_facilities, dtype=float)
    for i in range(n_customers):
        load[assignment[i]] += demands[i]
    
    # Compute current cost
    def compute_cost(assignment):
        total = 0.0
        for f in range(n_facilities):
            if load[f] > 0:
                total += open_cost[f]
        for i in range(n_customers):
            total += assign_cost[i, assignment[i]]
        return total
    
    current_cost = compute_cost(assignment)
    improved = True
    max_iter = 2000
    
    # Precompute customer indices by facility for faster swap search
    def get_facility_customers(assignment):
        fac_cust = {f: [] for f in range(n_facilities)}
        for i in range(n_customers):
            fac_cust[assignment[i]].append(i)
        return fac_cust
    
    for iteration in range(max_iter):
        if time.time() > deadline - 5.0:
            break
        improved = False
        
        # Try moves: move customer i from current facility to another
        for i in range(n_customers):
            if time.time() > deadline - 5.0:
                break
            current_f = assignment[i]
            for new_f in range(n_facilities):
                if new_f == current_f:
                    continue
                if load[new_f] + demands[i] > capacities[new_f]:
                    continue
                
                # Calculate cost change
                delta = assign_cost[i, new_f] - assign_cost[i, current_f]
                
                # If current facility becomes empty, we save open_cost
                if load[current_f] - demands[i] <= 0:
                    delta -= open_cost[current_f]
                
                # If new facility was empty, we add open_cost
                if load[new_f] <= 0:
                    delta += open_cost[new_f]
                
                if delta < -1e-9:
                    # Make the move
                    load[current_f] -= demands[i]
                    load[new_f] += demands[i]
                    assignment[i] = new_f
                    current_cost += delta
                    improved = True
                    break  # Re-evaluate from this customer
        
        # Try swaps: swap customers i and j between facilities
        if not improved:
            fac_cust = get_facility_customers(assignment)
            for fi in range(n_facilities):
                if time.time() > deadline - 5.0:
                    break
                for fj in range(fi + 1, n_facilities):
                    if time.time() > deadline - 5.0:
                        break
                    for i in fac_cust[fi]:
                        if time.time() > deadline - 5.0:
                            break
                        for j in fac_cust[fj]:
                            # Check capacity feasibility after swap
                            if load[fi] - demands[i] + demands[j] > capacities[fi]:
                                continue
                            if load[fj] - demands[j] + demands[i] > capacities[fj]:
                                continue
                            
                            # Calculate cost change
                            delta = (assign_cost[i, fj] - assign_cost[i, fi]) + \
                                    (assign_cost[j, fi] - assign_cost[j, fj])
                            
                            # Open cost changes
                            # If fi becomes empty
                            if load[fi] - demands[i] <= 0:
                                delta -= open_cost[fi]
                            # If fj becomes empty  
                            if load[fj] - demands[j] <= 0:
                                delta -= open_cost[fj]
                            
                            if delta < -1e-9:
                                # Make the swap
                                load[fi] -= demands[i]
                                load[fi] += demands[j]
                                load[fj] -= demands[j]
                                load[fj] += demands[i]
                                assignment[i] = fj
                                assignment[j] = fi
                                current_cost += delta
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        
        if not improved:
            break
    
    return assignment

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assigns customers to facilities using MILP (scipy.optimize.milp with HiGHS)
    followed by local search polish.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    n_customers = len(demands)
    n_facilities = len(capacities)

    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)

    # Variables:
    # x[i, f] for i in [0, n_customers), f in [0, n_facilities) - binary
    # y[f] for f in [0, n_facilities) - binary
    # Total variables: n_customers * n_facilities + n_facilities

    n_x = n_customers * n_facilities
    n_y = n_facilities
    n_vars = n_x + n_y

    # Objective: minimize sum(assign_cost[i,f] * x[i,f]) + sum(open_cost[f] * y[f])
    c = np.zeros(n_vars)
    for i in range(n_customers):
        for f in range(n_facilities):
            c[i * n_facilities + f] = assign_cost[i, f]
    for f in range(n_facilities):
        c[n_x + f] = open_cost[f]

    # Constraints:
    # 1. Each customer assigned to exactly one facility:
    #    sum_f x[i,f] = 1 for all i
    # 2. Capacity: sum_i demands[i] * x[i,f] <= capacities[f] * y[f] for all f
    #    => sum_i demands[i] * x[i,f] - capacities[f] * y[f] <= 0

    # Build A_ub and b_ub for inequality constraints (type <=)
    # And A_eq and b_eq for equality constraints (type ==)

    # Equality constraints: n_customers
    A_eq = np.zeros((n_customers, n_vars))
    b_eq = np.ones(n_customers)
    for i in range(n_customers):
        for f in range(n_facilities):
            A_eq[i, i * n_facilities + f] = 1.0

    # Inequality constraints: n_facilities (capacity)
    A_ub = np.zeros((n_facilities, n_vars))
    b_ub = np.zeros(n_facilities)
    for f in range(n_facilities):
        for i in range(n_customers):
            A_ub[f, i * n_facilities + f] = demands[i]
        A_ub[f, n_x + f] = -capacities[f]
        b_ub[f] = 0.0

    # Bounds: all variables in [0, 1]
    bounds = Bounds(lb=0.0, ub=1.0)

    # Integrality: all variables are binary (0 or 1)
    integrality = np.ones(n_vars)

    milp_result = None
    try:
        milp_result = milp(
            c=c,
            constraints=[
                LinearConstraint(A_eq, lb=b_eq, ub=b_eq),
                LinearConstraint(A_ub, lb=-np.inf, ub=b_ub)
            ],
            integrality=integrality,
            bounds=bounds
        )
    except Exception:
        milp_result = None

    if milp_result is not None and milp_result.success:
        x_flat = milp_result.x
        # Extract x[i, f]
        assignment = np.zeros(n_customers, dtype=int)
        for i in range(n_customers):
            for f in range(n_facilities):
                if x_flat[i * n_facilities + f] > 0.5:
                    assignment[i] = f
                    break
            else:
                # Fallback: assign to cheapest feasible
                assignment[i] = int(np.argmin(assign_cost[i]))
    else:
        # Fallback: simple greedy
        assignment = np.zeros(n_customers, dtype=int)
        load = np.zeros(n_facilities, dtype=float)
        order = np.argsort(-demands)
        for i in order:
            best_f = -1
            best_c = float('inf')
            for f in range(n_facilities):
                if load[f] + demands[i] <= capacities[f]:
                    if assign_cost[i, f] < best_c:
                        best_c = assign_cost[i, f]
                        best_f = f
            if best_f == -1:
                best_f = int(np.argmin(assign_cost[i]))
            assignment[i] = best_f
            load[best_f] += demands[i]
    
    # Local search polish - multiple passes for better convergence
    if time.time() < deadline - 10.0:
        # First pass with current deadline
        assignment = _local_search(assignment, demands, capacities, open_cost, assign_cost, deadline)
        
        # If we have time, do another pass to refine further
        if time.time() < deadline - 8.0:
            assignment = _local_search(assignment, demands, capacities, open_cost, assign_cost, deadline)
        
        # Final pass if time permits
        if time.time() < deadline - 5.0:
            assignment = _local_search(assignment, demands, capacities, open_cost, assign_cost, deadline)
    
    return assignment

# EVOLVE-BLOCK-END
