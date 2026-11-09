"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linear_sum_assignment

def _min_cost_transportation(demands, capacities, costs, open_facilities):
    """
    Solve the transportation problem for a given set of open facilities.
    Returns the assignment and total cost, or None if infeasible.
    
    We use a min-cost flow approach. Since the constraint matrix is totally
    unimodular, the LP relaxation gives integer solutions.
    
    We model this as:
    - Source -> Facility f (f in open_facilities): capacity = capacities[f], cost = 0
    - Facility f -> Customer c: capacity = demands[c], cost = costs[c, f]
    - Customer c -> Sink: capacity = demands[c], cost = 0
    
    Total flow = sum(demands)
    """
    n_customers = len(demands)
    n_open = len(open_facilities)
    
    if n_open == 0:
        return None, None
    
    # Check if total capacity is sufficient
    total_demand = sum(demands)
    total_capacity = sum(capacities[f] for f in open_facilities)
    if total_capacity < total_demand:
        return None, None
    
    # Build the transportation problem
    # We'll use a successive shortest path algorithm for min-cost flow
    
    # Node indices:
    # 0: source
    # 1 to n_open: facilities
    # n_open+1 to n_open+n_customers: customers
    # n_open+n_customers+1: sink
    
    source = 0
    sink = n_open + n_customers + 1
    n_nodes = sink + 1
    
    # Build adjacency list for the flow network
    # Each edge: (to, capacity, cost)
    # We'll use a list of lists, where each entry is [to, cap, cost, reverse_index]
    
    # Actually, let's use a simpler approach: since this is a transportation problem,
    # we can use the Hungarian algorithm on an expanded matrix, or use scipy's
    # linear programming.
    
    # Let's use scipy.optimize.linprog for the LP relaxation
    # Since the constraint matrix is totally unimodular, the solution will be integer
    
    from scipy.optimize import linprog
    
    # Decision variables: x[c, f] for c in range(n_customers), f in open_facilities
    # We'll flatten: x[i] where i = c * n_open + fi_index
    
    m = n_open  # number of open facilities
    
    # Objective: minimize sum_{c, f} costs[c, f] * x[c, f]
    c_obj = np.zeros(n_customers * m)
    for c in range(n_customers):
        for fi, f in enumerate(open_facilities):
            c_obj[c * m + fi] = costs[c, f]
    
    # Constraints:
    # 1. For each customer c: sum_{f in open} x[c, f] = 1
    # 2. For each open facility f: sum_{c} demands[c] * x[c, f] <= capacities[f]
    
    A_ub = np.zeros((n_customers + m, n_customers * m))
    b_ub = np.zeros(n_customers + m)
    
    # Customer constraints (equality, but we'll use >= and <=)
    for c in range(n_customers):
        for fi, f in enumerate(open_facilities):
            A_ub[c, c * m + fi] = 1.0
        b_ub[c] = 1.0
    
    # Facility capacity constraints
    for fi, f in enumerate(open_facilities):
        for c in range(n_customers):
            A_ub[n_customers + fi, c * m + fi] = demands[c]
        b_ub[n_customers + fi] = capacities[f]
    
    # Bounds: x >= 0
    bounds = [(0, None)] * (n_customers * m)
    
    # We need equality constraints for customers
    A_eq = np.zeros((n_customers, n_customers * m))
    b_eq = np.ones(n_customers)
    for c in range(n_customers):
        for fi, f in enumerate(open_facilities):
            A_eq[c, c * m + fi] = 1.0
    
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    if res.status != 0:
        return None, None
    
    # Extract solution
    x = res.x.reshape(n_customers, m)
    # Round to nearest integer (should already be close to integer due to total unimodularity)
    x = np.round(x).astype(int)
    
    # Verify feasibility
    for c in range(n_customers):
        if x[c].sum() != 1:
            return None, None
    
    load = np.zeros(m)
    for c in range(n_customers):
        for fi in range(m):
            if x[c, fi] == 1:
                load[fi] += demands[c]
    
    for fi in range(m):
        if load[fi] > capacities[open_facilities[fi]]:
            return None, None
    
    # Compute total assignment cost
    assign_cost_total = 0.0
    for c in range(n_customers):
        for fi in range(m):
            if x[c, fi] == 1:
                assign_cost_total += costs[c, open_facilities[fi]]
    
    # Create result array
    result = np.zeros(n_customers, dtype=int)
    for c in range(n_customers):
        for fi in range(m):
            if x[c, fi] == 1:
                result[c] = open_facilities[fi]
    
    return result, assign_cost_total


def assign(demands, capacities, open_cost, assign_cost):
    """
    Assign each customer to a facility using a min-cost flow approach with
    facility selection heuristic.
    """
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Strategy: Try different subsets of open facilities
    # Start with all facilities, then try to remove facilities that are expensive
    # or underutilized
    
    # First, get a baseline solution with all facilities open
    all_facilities = list(range(n_facilities))
    result, assign_cost_total = _min_cost_transportation(demands, capacities, assign_cost, all_facilities)
    
    if result is None:
        # Fallback: greedy
        result = np.zeros(n_customers, dtype=int)
        load = np.zeros(n_facilities)
        order = np.argsort(-demands)
        for i in order:
            feasible = [f for f in range(n_facilities) if load[f] + demands[i] <= capacities[f]]
            if not feasible:
                f = np.argmax(capacities - load)
            else:
                f = min(feasible, key=lambda f: assign_cost[i, f])
            result[i] = f
            load[f] += demands[i]
        return result
    
    # Compute total cost with all facilities
    open_facilities_used = set(result)
    open_cost_total = sum(open_cost[f] for f in open_facilities_used)
    best_total = open_cost_total + assign_cost_total
    best_result = result.copy()
    
    # Try to improve by removing facilities
    # Sort facilities by open_cost / capacity ratio (cost per unit capacity)
    # Try removing facilities with high open cost relative to their usage
    
    current_open = set(range(n_facilities))
    current_result = result.copy()
    current_load = np.zeros(n_facilities)
    for c in range(n_customers):
        current_load[current_result[c]] += demands[c]
    
    # Iteratively try to remove facilities
    improved = True
    while improved:
        improved = False
        # Try removing each facility
        for f in list(current_open):
            if len(current_open) <= 1:
                break
            # Check if we can remove facility f
            # We need to reassign all customers from f to other open facilities
            new_open = current_open - {f}
            if not new_open:
                continue
            
            # Check if total capacity of remaining facilities is sufficient
            remaining_capacity = sum(capacities[g] for g in new_open)
            if remaining_capacity < sum(demands):
                continue
            
            # Try to solve with this set of facilities
            new_result, new_assign_cost = _min_cost_transportation(demands, capacities, assign_cost, list(new_open))
            if new_result is None:
                continue
            
            new_open_used = set(new_result)
            new_open_cost = sum(open_cost[g] for g in new_open_used)
            new_total = new_open_cost + new_assign_cost
            
            if new_total < best_total - 1e-9:
                best_total = new_total
                best_result = new_result
                current_open = new_open
                current_result = new_result
                improved = True
    
    return best_result

# EVOLVE-BLOCK-END
