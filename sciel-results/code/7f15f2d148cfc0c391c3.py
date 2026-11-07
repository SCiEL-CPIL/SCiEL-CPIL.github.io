"""
Facility location solver using MIP (scipy.optimize.milp) with heuristic fallback.
Strategy: Formulate as an integer linear program and solve with HiGHS via scipy.
If MIP is unavailable or times out, fall back to greedy + local search.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assign each customer to a facility.
    Primary: MIP via scipy.optimize.milp.
    Fallback: greedy + local search.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    n_customers = len(demands)
    n_facilities = len(capacities)

    demands = np.asarray(demands, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.float64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)

    def compute_cost(assignment):
        """Compute total cost for an assignment, return inf if infeasible."""
        load = np.zeros(n_facilities)
        for c in range(n_customers):
            f = int(assignment[c])
            if f < 0 or f >= n_facilities:
                return np.inf
            load[f] += demands[c]
        for f in range(n_facilities):
            if load[f] > capacities[f] + 1e-9:
                return np.inf
        cost = 0.0
        used = np.zeros(n_facilities, dtype=bool)
        for c in range(n_customers):
            f = int(assignment[c])
            cost += assign_cost[c, f]
            used[f] = True
        for f in range(n_facilities):
            if used[f]:
                cost += open_cost[f]
        return cost

    def greedy_assignment(order=None):
        """Greedy: assign in given order to cheapest feasible facility."""
        if order is None:
            order = np.arange(n_customers)
        result = np.full(n_customers, -1, dtype=int)
        load = np.zeros(n_facilities)
        for c in order:
            costs = assign_cost[c].copy()
            feasible = load + demands[c] <= capacities
            costs[~feasible] = np.inf
            f = int(np.argmin(costs))
            if feasible[f]:
                result[c] = f
                load[f] += demands[c]
            else:
                return None
        return result

    def local_search(assignment, max_iter=500):
        """Improve assignment via single reassignments and 2-opt swaps."""
        assignment = assignment.copy()
        load = np.zeros(n_facilities)
        for c in range(n_customers):
            load[int(assignment[c])] += demands[c]
        current_cost = compute_cost(assignment)
        if current_cost == np.inf:
            return assignment
        
        for iteration in range(max_iter):
            if time.time() > deadline - 5:
                break
            improved = False
            
            # Single reassignment pass
            for c in range(n_customers):
                old_f = int(assignment[c])
                load[old_f] -= demands[c]
                best_new_f = old_f
                best_delta = 0.0
                for new_f in range(n_facilities):
                    if new_f == old_f:
                        continue
                    if load[new_f] + demands[c] > capacities[new_f] + 1e-9:
                        continue
                    delta = assign_cost[c, new_f] - assign_cost[c, old_f]
                    if load[new_f] == 0:
                        delta += open_cost[new_f]
                    if load[old_f] == 0:
                        delta -= open_cost[old_f]
                    if delta < best_delta - 1e-9:
                        best_delta = delta
                        best_new_f = new_f
                if best_new_f != old_f and best_delta < -1e-9:
                    assignment[c] = best_new_f
                    load[best_new_f] += demands[c]
                    current_cost += best_delta
                    improved = True
                else:
                    load[old_f] += demands[c]
            
            # 2-opt swap pass: try swapping assignments of two customers
            if not improved:
                for c1 in range(n_customers):
                    if time.time() > deadline - 5:
                        break
                    for c2 in range(c1 + 1, n_customers):
                        f1 = int(assignment[c1])
                        f2 = int(assignment[c2])
                        if f1 == f2:
                            continue
                        # Check if swapping is feasible
                        if load[f1] - demands[c1] + demands[c2] > capacities[f1] + 1e-9:
                            continue
                        if load[f2] - demands[c2] + demands[c1] > capacities[f2] + 1e-9:
                            continue
                        # Compute delta
                        delta = (assign_cost[c1, f2] - assign_cost[c1, f1]) + \
                                (assign_cost[c2, f1] - assign_cost[c2, f2])
                        # Open cost changes: if a facility becomes empty or gets first customer
                        old_load_f1 = load[f1]
                        old_load_f2 = load[f2]
                        new_load_f1 = old_load_f1 - demands[c1] + demands[c2]
                        new_load_f2 = old_load_f2 - demands[c2] + demands[c1]
                        
                        if old_load_f1 > 0 and new_load_f1 == 0:
                            delta -= open_cost[f1]
                        elif old_load_f1 == 0 and new_load_f1 > 0:
                            delta += open_cost[f1]
                        if old_load_f2 > 0 and new_load_f2 == 0:
                            delta -= open_cost[f2]
                        elif old_load_f2 == 0 and new_load_f2 > 0:
                            delta += open_cost[f2]
                        
                        if delta < -1e-9:
                            assignment[c1] = f2
                            assignment[c2] = f1
                            load[f1] = new_load_f1
                            load[f2] = new_load_f2
                            current_cost += delta
                            improved = True
                            break
                    if improved:
                        break
            
            if not improved:
                break
        return assignment

    # Try LP relaxation + deterministic rounding first (fast), then MIP if time permits
    try:
        from scipy.optimize import linprog, milp, LinearConstraint, Bounds
        from scipy.sparse import csr_matrix

        # Variables:
        # x[c, f] = 1 if customer c assigned to facility f (n_customers * n_facilities)
        # y[f] = 1 if facility f is open (n_facilities)
        # Total vars: n_customers * n_facilities + n_facilities

        n_vars = n_customers * n_facilities + n_facilities
        x_vars = n_customers * n_facilities

        # Objective: minimize sum_c,f assign_cost[c,f]*x[c,f] + sum_f open_cost[f]*y[f]
        c_obj = np.zeros(n_vars)
        for c in range(n_customers):
            for f in range(n_facilities):
                c_obj[c * n_facilities + f] = assign_cost[c, f]
        for f in range(n_facilities):
            c_obj[x_vars + f] = open_cost[f]

        # Build constraint matrix using sparse for efficiency
        # Constraint 1: Each customer assigned to exactly one facility: sum_f x[c,f] = 1
        # Constraint 2: Capacity: sum_c demands[c]*x[c,f] <= capacities[f] for each f
        # Constraint 3: y[f] >= x[c,f] for each c,f (if customer assigned to f, f must be open)

        # Use sparse matrices
        rows1 = []
        cols1 = []
        vals1 = []
        for c in range(n_customers):
            for f in range(n_facilities):
                rows1.append(c)
                cols1.append(c * n_facilities + f)
                vals1.append(1.0)

        A1 = csr_matrix((vals1, (rows1, cols1)), shape=(n_customers, n_vars))

        rows2 = []
        cols2 = []
        vals2 = []
        for f in range(n_facilities):
            for c in range(n_customers):
                rows2.append(f)
                cols2.append(c * n_facilities + f)
                vals2.append(demands[c])
        A2 = csr_matrix((vals2, (rows2, cols2)), shape=(n_facilities, n_vars))

        rows3 = []
        cols3 = []
        vals3 = []
        for c in range(n_customers):
            for f in range(n_facilities):
                rows3.append(c * n_facilities + f)
                cols3.append(c * n_facilities + f)
                vals3.append(1.0)
                cols3.append(x_vars + f)
                vals3.append(-1.0)
        A3 = csr_matrix((vals3, (rows3, cols3)), shape=(n_customers * n_facilities, n_vars))

        # Combine into single constraint matrix
        A_sparse = csr_matrix.vstack([A1, A2, A3], format='csr')

        lb = np.zeros(A_sparse.shape[0])
        ub = np.zeros(A_sparse.shape[0])

        # Equality constraints for customer assignment
        lb[:n_customers] = 1.0
        ub[:n_customers] = 1.0

        # Capacity constraints (<= capacities[f])
        lb[n_customers:n_customers + n_facilities] = -np.inf
        ub[n_customers:n_customers + n_facilities] = capacities

        # Openness constraints (x[c,f] - y[f] <= 0)
        lb[n_customers + n_facilities:] = -np.inf
        ub[n_customers + n_facilities:] = 0.0

        constraints = LinearConstraint(A_sparse, lb, ub)
        var_bounds = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))

        # First, solve LP relaxation (all continuous)
        lp_result = linprog(
            c=c_obj,
            A_ub=None,
            b_ub=None,
            A_eq=None,
            b_eq=None,
            bounds=var_bounds,
            method='highs',
            options={'time_limit': 30}
        )

        # Actually, let's use milp with integrality=0 for LP relaxation
        lp_result = milp(
            c=c_obj,
            constraints=constraints,
            integrality=np.zeros(n_vars),  # All continuous
            bounds=var_bounds,
            options={'time_limit': 30}
        )

        best_assignment = None
        best_cost = np.inf

        if lp_result.status == 0 and lp_result.x is not None:
            # Extract LP solution
            x_lp = lp_result.x[:x_vars].reshape(n_customers, n_facilities)
            y_lp = lp_result.x[x_vars:]

            # Deterministic rounding: sort customers by "fractionality" 
            # (max fractional value), assign most fractional first
            # For each customer, pick facility with highest LP value, 
            # but respect capacity constraints
            
            # Compute fractionality score for each customer
            # Higher score = more fractional = should be assigned first
            frac_scores = np.zeros(n_customers)
            for c in range(n_customers):
                vals = x_lp[c]
                # Fractionality: how spread out the assignment is
                # If one facility has value ~1, low fractionality
                # If values are spread, high fractionality
                sorted_vals = np.sort(vals)[::-1]
                frac_scores[c] = 1.0 - sorted_vals[0] if sorted_vals[0] < 1.0 else 0.0

            # Sort by fractionality descending, then by demand descending
            order = np.argsort(-frac_scores - 0.01 * demands)

            # Greedy rounding: assign customers in order to their best feasible facility
            assignment = np.full(n_customers, -1, dtype=int)
            load = np.zeros(n_facilities)
            facilities_open = np.zeros(n_facilities, dtype=bool)

            for c in order:
                # Find best facility considering LP guidance and feasibility
                best_f = -1
                best_score = np.inf
                for f in range(n_facilities):
                    if load[f] + demands[c] > capacities[f] + 1e-9:
                        continue
                    # Score: actual assignment cost + open cost if not open
                    score = assign_cost[c, f]
                    if not facilities_open[f]:
                        score += open_cost[f]
                    # Add small penalty based on LP value (prefer facilities with higher LP value)
                    score -= 0.01 * x_lp[c, f]
                    if score < best_score:
                        best_score = score
                        best_f = f
                if best_f >= 0:
                    assignment[c] = best_f
                    load[best_f] += demands[c]
                    facilities_open[best_f] = True
                else:
                    # Fallback: assign to facility with most remaining capacity
                    rem_cap = capacities - load
                    best_f = int(np.argmax(rem_cap))
                    if rem_cap[best_f] >= demands[c] - 1e-9:
                        assignment[c] = best_f
                        load[best_f] += demands[c]
                        facilities_open[best_f] = True

            # Check if all customers assigned
            if np.all(assignment >= 0):
                # Polish with local search
                assignment = local_search(assignment, max_iter=500)
                cost = compute_cost(assignment)
                if cost < best_cost:
                    best_cost = cost
                    best_assignment = assignment.copy()

        # If LP rounding didn't give good result or time permits, try MIP
        if time.time() < deadline - 20:
            try:
                # MIP with corrected capacity constraint
                # Capacity: sum_c demands[c]*x[c,f] <= capacities[f] (NOT *y[f])
                # This is correct because if no customer is assigned to f, 
                # the constraint is 0 <= capacities[f] which is always true
                
                # Reuse the constraint matrix from above (it's already correct)
                # The capacity constraint in A2 is: sum_c demands[c]*x[c,f] <= capacities[f]
                # which is what we want!
                
                integrality = np.ones(n_vars)
                
                mip_result = milp(
                    c=c_obj,
                    constraints=constraints,
                    integrality=integrality,
                    bounds=var_bounds,
                    options={'time_limit': max(10, int(deadline - time.time()) - 5)}
                )

                if mip_result.status == 0 and mip_result.x is not None:
                    assignment_mip = np.zeros(n_customers, dtype=int)
                    for c in range(n_customers):
                        for f in range(n_facilities):
                            if mip_result.x[c * n_facilities + f] > 0.5:
                                assignment_mip[c] = f
                                break
                    cost_mip = compute_cost(assignment_mip)
                    if cost_mip < best_cost:
                        best_cost = cost_mip
                        best_assignment = assignment_mip.copy()

            except Exception:
                pass

        if best_assignment is not None and best_cost < np.inf:
            return best_assignment.astype(int)

    except Exception as e:
        pass

    # Fallback: greedy + local search
    best_assignment = None
    best_cost = np.inf

    # Multiple greedy orderings
    orderings = [
        np.arange(n_customers),
        np.argsort(-demands),
        np.argsort(demands),
        np.argsort(assign_cost.sum(axis=1)),
    ]

    for order in orderings:
        if time.time() > deadline - 10:
            break
        sol = greedy_assignment(order)
        if sol is not None:
            sol = local_search(sol, max_iter=300)
            cost = compute_cost(sol)
            if cost < best_cost:
                best_cost = cost
                best_assignment = sol.copy()

    # Randomized greedy
    rng = np.random.RandomState(42)
    for i in range(30):
        if time.time() > deadline - 5:
            break
        order = rng.permutation(n_customers)
        sol = greedy_assignment(order)
        if sol is not None:
            sol = local_search(sol, max_iter=200)
            cost = compute_cost(sol)
            if cost < best_cost:
                best_cost = cost
                best_assignment = sol.copy()

    if best_assignment is None:
        # Last resort fallback
        best_assignment = np.zeros(n_customers, dtype=int)
        load = np.zeros(n_facilities)
        for c in range(n_customers):
            f = int(np.argmax(capacities - load))
            best_assignment[c] = f
            load[f] += demands[c]

    return best_assignment.astype(int)
# EVOLVE-BLOCK-END
