"""
Facility location solver using minimum-cost flow with iterative facility closing.
Strategy: Solve min-cost flow for assignment (ignoring fixed costs), then iteratively
close facilities with lowest marginal benefit, re-solving each time.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assign each customer to a facility.
    Primary: min-cost flow + iterative facility closing.
    Fallback: greedy + local search.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    n_customers = len(demands)
    n_facilities = len(capacities)

    demands = np.asarray(demands, dtype=np.int64)
    capacities = np.asarray(capacities, dtype=np.int64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)

    def compute_cost(assignment):
        """Compute total cost for an assignment, return inf if infeasible."""
        load = np.zeros(n_facilities, dtype=np.int64)
        for c in range(n_customers):
            f = int(assignment[c])
            if f < 0 or f >= n_facilities:
                return np.inf
            load[f] += demands[c]
        for f in range(n_facilities):
            if load[f] > capacities[f]:
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

    def solve_min_cost_flow(open_facilities):
        """
        Solve min-cost flow: assign each customer to an open facility.
        Returns assignment array or None if infeasible.
        Uses scipy's linear_sum_assignment on a transformed problem,
        or falls back to a custom min-cost flow.
        """
        from scipy.optimize import linear_sum_assignment
        
        # For min-cost flow with capacities, we can use a transformation.
        # Since each customer has demand 1 (unit supply) and facilities have capacities,
        # we need a proper min-cost flow solver.
        # Let's use scipy's min_cost_flow if available, or implement via LP.
        
        # Try using scipy's min_cost_flow
        try:
            from scipy.sparse import csr_matrix
            from scipy.optimize import linprog
            
            # Build min-cost flow as LP
            # Nodes: source (0), customers (1..n_customers), facilities (n_customers+1..n_customers+n_facilities), sink (n_customers+n_facilities+1)
            # But this is complex. Let's use a simpler approach.
            
            # Alternative: use the fact that this is a transportation problem.
            # We can solve it as a min-cost flow using network simplex via scipy.
            
            # Actually, let's use a direct approach with scipy.optimize.milp for the flow
            # but only for the assignment part (no fixed costs).
            
            # Simpler: use scipy.optimize.linear_sum_assignment for the case where 
            # capacities are large enough, or use a custom approach.
            
            # For proper min-cost flow with capacities, let's use a different approach:
            # Formulate as an LP and solve with linprog.
            
            # Variables: x[c, f] for c in customers, f in open_facilities
            n_open = len(open_facilities)
            if n_open == 0:
                return None
            
            n_vars = n_customers * n_open
            if n_vars == 0:
                return None
            
            # Objective: minimize sum assign_cost[c, f] * x[c, f]
            c_obj = np.zeros(n_vars)
            for i, f in enumerate(open_facilities):
                for c in range(n_customers):
                    c_obj[c * n_open + i] = assign_cost[c, f]
            
            # Constraints:
            # 1. Each customer assigned to exactly one open facility: sum_f x[c,f] = 1
            # 2. Capacity: sum_c demands[c] * x[c,f] <= capacities[f]
            
            n_cons = n_customers + n_open
            A = np.zeros((n_cons, n_vars))
            lb = np.zeros(n_cons)
            ub = np.zeros(n_cons)
            
            # Customer assignment constraints (equality)
            for c in range(n_customers):
                for i, f in enumerate(open_facilities):
                    A[c, c * n_open + i] = 1.0
                lb[c] = 1.0
                ub[c] = 1.0
            
            # Capacity constraints (inequality)
            for i, f in enumerate(open_facilities):
                row = n_customers + i
                for c in range(n_customers):
                    A[row, c * n_open + i] = demands[c]
                lb[row] = 0.0
                ub[row] = capacities[f]
            
            # Variable bounds: x[c,f] >= 0 (continuous is fine for transportation)
            bounds = [(0, None)] * n_vars
            
            result = linprog(c_obj, A_ub=A, b_ub=ub, A_eq=None, b_eq=None, bounds=bounds, method='highs')
            
            if result.status == 0 and result.x is not None:
                assignment = np.zeros(n_customers, dtype=int)
                for c in range(n_customers):
                    # Find which facility gets the most flow
                    best_f = open_facilities[0]
                    best_val = -1
                    for i, f in enumerate(open_facilities):
                        val = result.x[c * n_open + i]
                        if val > best_val:
                            best_val = val
                            best_f = f
                    assignment[c] = best_f
                return assignment
            
        except Exception:
            pass
        
        # Fallback: greedy assignment to open facilities
        result = np.full(n_customers, -1, dtype=int)
        load = np.zeros(n_facilities, dtype=np.int64)
        for c in range(n_customers):
            best_f = -1
            best_cost = np.inf
            for f in open_facilities:
                if load[f] + demands[c] <= capacities[f]:
                    if assign_cost[c, f] < best_cost:
                        best_cost = assign_cost[c, f]
                        best_f = f
            if best_f >= 0:
                result[c] = best_f
                load[best_f] += demands[c]
            else:
                return None
        return result

    def local_search(assignment, max_iter=300):
        """Improve assignment via single reassignments."""
        assignment = assignment.copy()
        load = np.zeros(n_facilities, dtype=np.int64)
        for c in range(n_customers):
            load[int(assignment[c])] += demands[c]
        current_cost = compute_cost(assignment)
        if current_cost == np.inf:
            return assignment
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            improved = False
            for c in range(n_customers):
                old_f = int(assignment[c])
                load[old_f] -= demands[c]
                best_new_f = old_f
                best_delta = 0.0
                for new_f in range(n_facilities):
                    if new_f == old_f:
                        continue
                    if load[new_f] + demands[c] > capacities[new_f]:
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
            if not improved:
                break
        return assignment

    # Main algorithm: min-cost flow + iterative facility closing
    best_assignment = None
    best_cost = np.inf
    
    # Start with all facilities open
    open_set = list(range(n_facilities))
    
    for iteration in range(n_facilities + 1):
        if time.time() > deadline - 20:
            break
        
        # Solve min-cost flow with current open set
        assignment = solve_min_cost_flow(open_set)
        if assignment is None:
            break
        
        # Apply local search to improve
        assignment = local_search(assignment)
        cost = compute_cost(assignment)
        
        if cost < best_cost:
            best_cost = cost
            best_assignment = assignment.copy()
        
        # Determine which facility to close
        # Compute marginal benefit of closing each facility
        load = np.zeros(n_facilities, dtype=np.int64)
        for c in range(n_customers):
            load[int(assignment[c])] += demands[c]
        
        # For each open facility, compute:
        # - open_cost saved
        # - reassignment penalty (increase in assignment costs)
        
        best_close = -1
        best_marginal = -np.inf  # We want to close the facility with most negative marginal (best saving)
        
        for f in open_set:
            if load[f] == 0:
                # Closing saves open_cost with no penalty
                marginal = -open_cost[f]
                if marginal < best_marginal:
                    best_marginal = marginal
                    best_close = f
                continue
            
            # Estimate reassignment penalty
            # For each customer at f, find best alternative
            penalty = 0.0
            feasible = True
            for c in range(n_customers):
                if int(assignment[c]) == f:
                    best_alt_cost = np.inf
                    for alt_f in open_set:
                        if alt_f == f:
                            continue
                        if load[alt_f] + demands[c] <= capacities[alt_f]:
                            if assign_cost[c, alt_f] < best_alt_cost:
                                best_alt_cost = assign_cost[c, alt_f]
                    if best_alt_cost == np.inf:
                        feasible = False
                        break
                    penalty += best_alt_cost - assign_cost[c, f]
            
            if not feasible:
                continue
            
            # Marginal = -open_cost[f] + penalty (we save open_cost but pay penalty)
            marginal = -open_cost[f] + penalty
            if marginal < best_marginal:
                best_marginal = marginal
                best_close = f
        
        if best_close < 0:
            break
        
        # Close the facility
        open_set.remove(best_close)
    
    # Final local search polish
    if best_assignment is not None:
        best_assignment = local_search(best_assignment)
        final_cost = compute_cost(best_assignment)
        if final_cost < best_cost:
            best_cost = final_cost
    
    # Fallback if no valid assignment found
    if best_assignment is None:
        # Greedy fallback
        load = np.zeros(n_facilities, dtype=np.int64)
        best_assignment = np.zeros(n_customers, dtype=int)
        for c in range(n_customers):
            best_f = 0
            best_remaining = -1
            for f in range(n_facilities):
                remaining = capacities[f] - load[f]
                if remaining >= demands[c] and remaining > best_remaining:
                    best_remaining = remaining
                    best_f = f
            best_assignment[c] = best_f
            load[best_f] += demands[c]
        best_assignment = local_search(best_assignment)
    
    return best_assignment.astype(int)
# EVOLVE-BLOCK-END
