"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linear_sum_assignment
import itertools

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assign each customer to a facility using a minimum cost flow formulation.
    
    Since we have 100 customers and 20 facilities, we can use a min-cost max-flow
    approach. However, scipy doesn't have a direct min-cost flow solver.
    
    Alternative approach: We can formulate this as an integer linear program, but
    that's complex. Instead, let's use a greedy approach with local search, or
    better yet, use the Hungarian algorithm on an expanded problem.
    
    Actually, let's think about this differently. The problem is a capacitated
    facility location problem. A common approach is:
    1. Decide which facilities to open
    2. Assign customers to open facilities respecting capacities
    
    Since there are only 20 facilities, we could enumerate subsets, but 2^20 = 1M
    which is feasible. For each subset of open facilities, we solve the assignment
    problem (which is a transportation problem) using min-cost flow or Hungarian.
    
    But 1M subsets × solving a 100×20 transportation problem might be too slow.
    
    Better approach: Use min-cost flow directly. We can model this as:
    - Source -> each facility node with capacity = capacities[f], cost = 0
    - Each facility node -> each customer node with capacity = demands[c], cost = assign_cost[c, f]
    - Each customer node -> sink with capacity = demands[c], cost = 0
    
    Wait, that's not quite right. The open cost is incurred if any customer is assigned to a facility.
    
    Let me reformulate:
    - Source -> facility f: capacity = capacities[f], cost = 0
    - Facility f -> customer c: capacity = demands[c], cost = assign_cost[c, f]
    - Customer c -> sink: capacity = demands[c], cost = 0
    
    But this doesn't capture the fixed open cost. The open cost is a fixed charge, which makes it a fixed-charge network flow problem, which is NP-hard in general.
    
    However, since we have only 20 facilities, we can use a different strategy:
    1. For each possible set of open facilities (or a subset of promising sets), solve the transportation problem.
    2. Use a heuristic to identify good candidate sets.
    
    Actually, let's try a simpler approach first: use a greedy algorithm with local search, or use scipy's linear programming.
    
    Let me try using scipy.optimize.linprog or milp. scipy.optimize.milp is available in newer versions.
    
    Let's check if we can use scipy.optimize.milp. If not, we'll use a heuristic.
    
    Actually, for the scope of this problem (100 customers, 20 facilities), we can:
    1. Enumerate all subsets of facilities that could be open (2^20 = 1,048,576)
    2. For each subset, check if the total capacity is sufficient
    3. Solve the transportation problem using min-cost flow
    4. Add the open cost
    5. Keep the best solution
    
    But 1M subsets × solving a 100×20 problem might be too slow in Python.
    
    Alternative: Use a branch-and-bound or a heuristic to prune.
    
    Let's try a different approach: use a greedy algorithm with multiple restarts and local search.
    
    Or, we can use the fact that the problem is small enough to use scipy.optimize.milp if available.
    
    Let's try using scipy.optimize.milp:
    """
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Try to use MILP if available
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        import numpy as np
        
        # Decision variables: x[c, f] = 1 if customer c is assigned to facility f
        # y[f] = 1 if facility f is open
        #
        # Minimize: sum_{c,f} assign_cost[c,f] * x[c,f] + sum_f open_cost[f] * y[f]
        #
        # Subject to:
        # 1. sum_f x[c,f] = 1 for all c (each customer assigned to exactly one facility)
        # 2. sum_c demands[c] * x[c,f] <= capacities[f] * y[f] for all f
        # 3. x[c,f] in {0,1}, y[f] in {0,1}
        
        n_vars = n_customers * n_facilities + n_facilities
        # x variables: indices 0 to n_customers*n_facilities - 1
        # y variables: indices n_customers*n_facilities to n_customers*n_facilities + n_facilities - 1
        
        # Objective coefficients
        c = np.zeros(n_vars)
        for ci in range(n_customers):
            for fi in range(n_facilities):
                c[ci * n_facilities + fi] = assign_cost[ci, fi]
        for fi in range(n_facilities):
            c[n_customers * n_facilities + fi] = open_cost[fi]
        
        # Constraints
        # Constraint 1: sum_f x[c,f] = 1 for all c
        A1 = np.zeros((n_customers, n_vars))
        for ci in range(n_customers):
            for fi in range(n_facilities):
                A1[ci, ci * n_facilities + fi] = 1.0
        lb1 = np.ones(n_customers)
        ub1 = np.ones(n_customers)
        
        # Constraint 2: sum_c demands[c] * x[c,f] <= capacities[f] * y[f] for all f
        # Rewrite: sum_c demands[c] * x[c,f] - capacities[f] * y[f] <= 0
        A2 = np.zeros((n_facilities, n_vars))
        for fi in range(n_facilities):
            for ci in range(n_customers):
                A2[fi, ci * n_facilities + fi] = demands[ci]
            A2[fi, n_customers * n_facilities + fi] = -capacities[fi]
        lb2 = np.full(n_facilities, -np.inf)
        ub2 = np.zeros(n_facilities)
        
        # Combine constraints
        A = np.vstack([A1, A2])
        lb = np.concatenate([lb1, lb2])
        ub = np.concatenate([ub1, ub2])
        
        # Bounds: x in [0,1], y in [0,1]
        bounds = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))
        
        # Integrality: all variables are binary
        integrality = np.ones(n_vars)
        
        res = milp(c=c, constraints=LinearConstraint(A, lb, ub), bounds=bounds, integrality=integrality)
        
        if res.status == 0:
            # Extract assignment
            x = res.x[:n_customers * n_facilities].reshape(n_customers, n_facilities)
            result = np.argmax(x, axis=1).astype(int)
            return result
        else:
            # Fallback to greedy
            pass
    
    except Exception:
        pass
    
    # Fallback: greedy algorithm with local search
    result = np.zeros(n_customers, dtype=int)
    load = np.zeros(n_facilities)
    
    # Sort customers by demand (largest first)
    order = np.argsort(-demands)
    
    for i in order:
        # Find feasible facilities sorted by cost
        feasible = []
        for f in range(n_facilities):
            if load[f] + demands[i] <= capacities[f]:
                feasible.append(f)
        
        if not feasible:
            # This shouldn't happen if total capacity is sufficient
            # Assign to facility with most remaining capacity
            f = np.argmax(capacities - load)
            result[i] = f
            load[f] += demands[i]
        else:
            # Sort by assign cost
            feasible_sorted = sorted(feasible, key=lambda f: assign_cost[i, f])
            result[i] = feasible_sorted[0]
            load[result[i]] += demands[i]
    
    # Local search: try to improve by reassigning customers
    # Simple 1-swap local search
    improved = True
    max_iters = 1000
    iter_count = 0
    while improved and iter_count < max_iters:
        improved = False
        iter_count += 1
        for i in range(n_customers):
            current_f = result[i]
            current_cost = assign_cost[i, current_f]
            # Try reassigning to a different facility
            for f in range(n_facilities):
                if f == current_f:
                    continue
                if load[f] + demands[i] <= capacities[f]:
                    new_cost = assign_cost[i, f]
                    if new_cost < current_cost - 1e-9:
                        # Check if this improves the total cost
                        # We need to consider that changing the assignment might affect the open cost
                        # For simplicity, ignore open cost change in local search
                        load[current_f] -= demands[i]
                        load[f] += demands[i]
                        result[i] = f
                        improved = True
                        break
    
    return result

# EVOLVE-BLOCK-END
