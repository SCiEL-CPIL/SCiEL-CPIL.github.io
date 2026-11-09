"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import milp, LinearConstraint, Bounds


def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Pre-convert to numpy arrays for speed
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    
    def calculate_cost(assignment):
        # assignment: array of facility indices
        if np.any(assignment < 0) or np.any(assignment >= n_fac):
            return float('inf')
        total_assign_cost = np.sum(assign_cost[np.arange(n_cust), assignment])
        # Open cost: sum of open_cost for unique facilities used
        used_facilities = np.unique(assignment)
        total_open_cost = np.sum(open_cost[used_facilities])
        return total_assign_cost + total_open_cost

    def is_valid(assignment):
        # Check capacity
        if np.any(assignment < 0) or np.any(assignment >= n_fac):
            return False
        loads = np.zeros(n_fac)
        np.add.at(loads, assignment, demands)
        if np.any(loads > capacities + 1e-9):
            return False
        return True

    def greedy_assign(order):
        """
        Assign customers in given order to cheapest feasible facility.
        """
        result = np.full(n_cust, -1, dtype=int)
        load = np.zeros(n_fac)
        
        for i in order:
            fac_indices = np.argsort(assign_cost[i])
            
            assigned = False
            for f in fac_indices:
                if load[f] + demands[i] <= capacities[f] + 1e-9:
                    result[i] = f
                    load[f] += demands[i]
                    assigned = True
                    break
            
            if not assigned:
                # Fallback: assign to facility with most remaining capacity
                rem_cap = capacities - load
                fac_indices_cap = np.argsort(-rem_cap)
                for f in fac_indices_cap:
                    if load[f] + demands[i] <= capacities[f] + 1e-9:
                        result[i] = f
                        load[f] += demands[i]
                        assigned = True
                        break
                        
            if not assigned:
                return None # Infeasible
                
        return result

    def solve_milp(time_limit=60.0):
        """
        Solve the capacitated facility location problem using MILP.
        """
        # Variables:
        # x[j] for j in range(n_fac): binary, 1 if facility j is open
        # y[i,j] for i in range(n_cust), j in range(n_fac): binary, 1 if customer i assigned to facility j
        # Total variables: n_fac + n_cust * n_fac
        
        n_vars = n_fac + n_cust * n_fac
        
        # Objective: minimize sum(open_cost[j] * x[j]) + sum(assign_cost[i,j] * y[i,j])
        c = np.zeros(n_vars)
        for j in range(n_fac):
            c[j] = open_cost[j]
        for i in range(n_cust):
            for j in range(n_fac):
                c[n_fac + i * n_fac + j] = assign_cost[i, j]
        
        # Constraints:
        # 1. Each customer assigned to exactly one facility: sum_j y[i,j] = 1 for all i
        # 2. Capacity: sum_i demands[i] * y[i,j] <= capacities[j] for all j
        # 3. Facility opening: y[i,j] <= x[j] for all i, j
        
        A_rows = []
        lb = []
        ub = []
        
        # Constraint 1: sum_j y[i,j] = 1 for each i
        for i in range(n_cust):
            row = np.zeros(n_vars)
            for j in range(n_fac):
                row[n_fac + i * n_fac + j] = 1.0
            A_rows.append(row)
            lb.append(1.0)
            ub.append(1.0)
        
        # Constraint 2: sum_i demands[i] * y[i,j] <= capacities[j] for each j
        for j in range(n_fac):
            row = np.zeros(n_vars)
            for i in range(n_cust):
                row[n_fac + i * n_fac + j] = demands[i]
            A_rows.append(row)
            lb.append(0.0)
            ub.append(capacities[j])
        
        # Constraint 3: y[i,j] <= x[j] for all i, j => y[i,j] - x[j] <= 0
        for i in range(n_cust):
            for j in range(n_fac):
                row = np.zeros(n_vars)
                row[j] = -1.0
                row[n_fac + i * n_fac + j] = 1.0
                A_rows.append(row)
                lb.append(-np.inf)
                ub.append(0.0)
        
        A = np.array(A_rows)
        
        # Bounds: all variables are binary (0 or 1)
        integrality = np.ones(n_vars)
        lbounds = np.zeros(n_vars)
        ubounds = np.ones(n_vars)
        
        try:
            res = milp(
                c=c,
                constraints=LinearConstraint(A, lb, ub),
                integrality=integrality,
                bounds=Bounds(lbounds, ubounds),
                options={"time_limit": time_limit, "mip_rel_gap": 0.01}
            )
            
            if res.status == 0 or res.status == 1:  # optimal or best feasible found
                # Extract assignment
                assignment = np.zeros(n_cust, dtype=int)
                for i in range(n_cust):
                    for j in range(n_fac):
                        if res.x[n_fac + i * n_fac + j] > 0.5:
                            assignment[i] = j
                            break
                    else:
                        # Fallback if no facility found
                        assignment[i] = 0
                
                if is_valid(assignment):
                    return assignment, calculate_cost(assignment)
        except Exception as e:
            pass
        
        return None, float('inf')

    def local_search(initial_assignment, max_time=20.0):
        """
        Perform local search starting from initial_assignment.
        Uses Relocate and Swap moves.
        Returns the best assignment found.
        """
        current_assignment = initial_assignment.copy()
        current_cost = calculate_cost(current_assignment)
        
        # Precompute loads
        load = np.zeros(n_fac)
        np.add.at(load, current_assignment, demands)
        
        # Track customer count per facility for fast open cost delta calculation
        cust_count = np.zeros(n_fac, dtype=int)
        for i in range(n_cust):
            cust_count[current_assignment[i]] += 1

        local_deadline = time.time() + max_time
        improved = True
        iter_count = 0
        max_iters = 10000
        
        while improved and iter_count < max_iters:
            if time.time() > local_deadline:
                break
            improved = False
            iter_count += 1
            
            # 1. Relocate moves
            for i in range(n_cust):
                if time.time() > local_deadline:
                    break
                f_from = current_assignment[i]
                
                best_delta = 0.0
                best_f_to = -1
                
                for f_to in range(n_fac):
                    if f_to == f_from:
                        continue
                    # Check capacity
                    if load[f_to] + demands[i] > capacities[f_to] + 1e-9:
                        continue
                    
                    # Calculate cost change
                    delta_assign = assign_cost[i, f_to] - assign_cost[i, f_from]
                    
                    # Open cost change
                    delta_open = 0.0
                    # If f_from becomes empty (cust_count[f_from] == 1)
                    if cust_count[f_from] == 1:
                        delta_open -= open_cost[f_from]
                    # If f_to was empty (cust_count[f_to] == 0)
                    if cust_count[f_to] == 0:
                        delta_open += open_cost[f_to]
                    
                    total_delta = delta_assign + delta_open
                    
                    if total_delta < best_delta:
                        best_delta = total_delta
                        best_f_to = f_to
                
                if best_f_to != -1:
                    # Accept best move
                    current_assignment[i] = best_f_to
                    load[f_from] -= demands[i]
                    load[best_f_to] += demands[i]
                    cust_count[f_from] -= 1
                    cust_count[best_f_to] += 1
                    current_cost += best_delta
                    improved = True
            
            # 2. Swap moves
            used_facs = [f for f in range(n_fac) if cust_count[f] > 0]
            
            if time.time() < local_deadline - 1.0:
                for idx1, f1 in enumerate(used_facs):
                    for f2 in used_facs[idx1+1:]:
                        if time.time() > local_deadline:
                            break
                        
                        # Get customers in f1 and f2
                        custs_f1 = [i for i in range(n_cust) if current_assignment[i] == f1]
                        custs_f2 = [i for i in range(n_cust) if current_assignment[i] == f2]
                        
                        best_swap_delta = 0.0
                        best_i, best_j = -1, -1
                        
                        for i in custs_f1:
                            for j in custs_f2:
                                # Check capacity after swap
                                if load[f2] - demands[j] + demands[i] > capacities[f2] + 1e-9:
                                    continue
                                if load[f1] - demands[i] + demands[j] > capacities[f1] + 1e-9:
                                    continue
                                
                                # Cost change
                                delta_assign = (assign_cost[i, f2] + assign_cost[j, f1]) - (assign_cost[i, f1] + assign_cost[j, f2])
                                
                                if delta_assign < best_swap_delta:
                                    best_swap_delta = delta_assign
                                    best_i, best_j = i, j
                        
                        if best_i != -1:
                            # Accept best swap
                            current_assignment[best_i] = f2
                            current_assignment[best_j] = f1
                            load[f1] -= demands[best_i]
                            load[f1] += demands[best_j]
                            load[f2] -= demands[best_j]
                            load[f2] += demands[best_i]
                            current_cost += best_swap_delta
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    continue

        return current_assignment, current_cost

    # 1. Try MILP first
    elapsed = time.time() - start_time
    milp_time = min(deadline - start_time - elapsed - 30.0, 60.0)
    
    if milp_time > 5.0:
        milp_assignment, milp_cost = solve_milp(time_limit=milp_time)
        if milp_assignment is not None:
            best_assignment = milp_assignment
            best_cost = milp_cost
        else:
            # Fallback to greedy
            order = np.argsort(-demands)
            best_assignment = greedy_assign(order)
            best_cost = calculate_cost(best_assignment)
    else:
        # Not enough time for MILP, use greedy
        order = np.argsort(-demands)
        best_assignment = greedy_assign(order)
        best_cost = calculate_cost(best_assignment)

    # 2. Local Search polish
    elapsed = time.time() - start_time
    remaining = deadline - start_time - elapsed
    
    ls_time = min(remaining - 5.0, 40.0)
    if ls_time > 1.0:
        ls_assignment, ls_cost = local_search(best_assignment, max_time=ls_time)
        if ls_cost < best_cost:
            best_assignment = ls_assignment
            best_cost = ls_cost

    return best_assignment

# EVOLVE-BLOCK-END
