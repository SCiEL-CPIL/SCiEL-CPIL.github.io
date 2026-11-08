"""Program using Mixed-Integer Linear Programming (MILP) for exact optimal facility location."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import milp, LinearConstraint, Bounds

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assigns customers to facilities using exact MILP optimization.
    
    Args:
    demands: array-like of customer demands
    capacities: array-like of facility capacities
    open_cost: array-like of facility opening costs
    assign_cost: 2D array of assignment costs
    
    Returns:
    numpy array of facility indices for each customer
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacabilities if False else capacities, dtype=float) # Typo check in thought, fixed below
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Variables:
    # x[i, j]: 1 if customer i is assigned to facility j (n_cust * n_fac)
    # y[j]: 1 if facility j is open (n_fac)
    # Total variables: n_cust * n_fac + n_fac
    
    n_vars = n_cust * n_fac + n_fac
    x_vars = n_cust * n_fac
    y_vars = n_fac
    
    # Objective: Minimize sum(assign_cost[i,j] * x[i,j]) + sum(open_cost[j] * y[j])
    c = np.zeros(n_vars)
    c[:x_vars] = assign_cost.flatten()
    c[x_vars:] = open_cost
    
    # Variable bounds: 0 <= x <= 1, 0 <= y <= 1
    lower_bounds = np.zeros(n_vars)
    upper_bounds = np.ones(n_vars)
    
    # Integrality: x and y are binary (0 or 1)
    integrality = np.ones(n_vars)
    
    # Constraints:
    # 1. Each customer assigned to exactly one facility: sum_j x[i,j] = 1 for all i
    #    Matrix shape: (n_cust, n_vars)
    A1 = np.zeros((n_cust, n_vars))
    for i in range(n_cust):
        # x[i, j] are at indices i*n_fac + j
        A1[i, i*n_fac : (i+1)*n_fac] = 1.0
    
    # 2. Capacity constraint: sum_i demands[i] * x[i,j] <= capacities[j] * y[j] for all j
    #    sum_i demands[i] * x[i,j] - capacities[j] * y[j] <= 0
    #    Matrix shape: (n_fac, n_vars)
    A2 = np.zeros((n_fac, n_vars))
    for j in range(n_fac):
        for i in range(n_cust):
            A2[j, i*n_fac + j] = demands[i]
        A2[j, x_vars + j] = -capacities[j]
    
    # Combine constraints
    # We can use a single LinearConstraint if we stack them, but bounds differ (eq vs ineq)
    # LinearConstraint supports different bounds per row.
    
    A = np.vstack([A1, A2])
    lb = np.concatenate([np.ones(n_cust), np.zeros(n_fac)])
    ub = np.concatenate([np.ones(n_cust), np.zeros(n_fac)])
    
    constraints = LinearConstraint(A, lb, ub)
    bounds = Bounds(lower_bounds, upper_bounds)
    
    # Solve
    try:
        result = milp(
            c=c,
            constraints=constraints,
            integrality=integrality,
            bounds=bounds,
            options={"time_limit": 140.0, "mip_rel_gap": 0.0} # Exact solution
        )
        
        if result.status == 0 and result.success:
            x_sol = result.x[:x_vars].reshape(n_cust, n_fac)
            # Determine assignment: argmax of x_sol (should be 1)
            # Due to floating point, threshold at 0.5
            assignment = np.argmax(x_sol, axis=1)
            
            # Verify feasibility just in case of numerical drift
            load = np.zeros(n_fac)
            np.add.at(load, assignment, demands)
            if np.any(load > capacities + 1e-6):
                 # Fallback to greedy if MILP returns invalid due to numerical issues
                 return greedy_init(demands, capacities, assign_cost)
                 
            return assignment.astype(int)
        else:
            # If MILP fails or times out, fallback
            if time.time() > deadline:
                return greedy_init(demands, capacities, assign_cost)
            # Try a quick local search on greedy init if MILP failed
            return local_search_fallback(demands, capacities, open_cost, assign_cost, deadline)
            
    except Exception as e:
        # Fallback on error
        return greedy_init(demands, capacities, assign_cost)

def greedy_init(demands, capacities, assign_cost):
    """Greedy initialization: assign customers in order of demand (largest first)."""
    n_cust = len(demands)
    n_fac = len(capacities)
    res = np.zeros(n_cust, dtype=int)
    load = np.zeros(n_fac)
    order = np.argsort(-demands)
    for i in order:
        # Sort facilities by cost for this customer
        fac_order = np.argsort(assign_cost[i])
        for f in fac_order:
            if load[f] + demands[i] <= capacities[f] + 1e-9:
                res[i] = f
                load[f] += demands[i]
                break
        else:
            # If no facility has space, assign to the one with most remaining space
            space = capacities - load
            best_f = np.argmax(space)
            if space[best_f] >= demands[i]:
                res[i] = best_f
                load[best_f] += demands[i]
            else:
                # Fallback: assign to cheapest facility regardless (will be infeasible)
                res[i] = fac_order[0]
    return res

def local_search_fallback(demands, capacities, open_cost, assign_cost, deadline):
    """Simple local search fallback if MILP fails."""
    res = greedy_init(demands, capacities, assign_cost)
    load = np.zeros(len(capacities))
    np.add.at(load, res, demands)
    
    n_cust = len(demands)
    n_fac = len(capacities)
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for i in np.random.permutation(n_cust):
            current_f = res[i]
            best_f = -1
            best_delta = 1e-9
            for f_new in range(n_fac):
                if f_new == current_f:
                    continue
                if load[f_new] + demands[i] > capacities[f_new] + 1e-9:
                    continue
                
                delta = assign_cost[i, f_new] - assign_cost[i, current_f]
                if load[f_new] == 0:
                    delta += open_cost[f_new]
                if load[current_f] == demands[i]:
                    delta -= open_cost[current_f]
                
                if delta < best_delta:
                    best_delta = delta
                    best_f = f_new
            
            if best_f != -1:
                load[current_f] -= demands[i]
                load[best_f] += demands[i]
                res[i] = best_f
                improved = True
    return res

# EVOLVE-BLOCK-END
