import numpy as np
from scipy.optimize import linprog
import time

def assign(demands, capacities, open_cost, assign_cost):
    n_cust = len(demands)
    n_fac = len(capacities)
    
    start_time = time.time()
    time_limit = 120.0
    safety_margin = 5.0

    def get_elapsed():
        return time.time() - start_time

    def is_deadline_exceeded():
        return get_elapsed() > time_limit - safety_margin

    def solve_assignment_lp(fac_indices):
        """
        Solves the LP relaxation for assignment of all customers to facilities in fac_indices.
        Returns fractional assignment x (n_cust x len(fac_indices)) and LP cost.
        """
        if not fac_indices:
            return None, np.inf
            
        m = len(fac_indices)
        # Variables: x_ij for i in 0..n_cust-1, j in 0..m-1
        # Total vars: n_cust * m
        
        # Objective: min sum assign_cost[i, fac_indices[j]] * x_ij
        c = np.zeros(n_cust * m)
        for i in range(n_cust):
            for j_idx, j in enumerate(fac_indices):
                c[i * m + j_idx] = assign_cost[i, j]
                
        # Constraints:
        # 1. Assignment: sum_j x_ij = 1 for each i
        # A_eq: n_cust x n_cust*m
        A_eq = np.zeros((n_cust, n_cust * m))
        b_eq = np.ones(n_cust)
        for i in range(n_cust):
            for j_idx in range(m):
                A_eq[i, i * m + j_idx] = 1.0
                
        # 2. Capacity: sum_i demands[i] * x_ij <= capacities[fac_indices[j]]
        # A_ub: m x n_cust*m
        A_ub = np.zeros((m, n_cust * m))
        b_ub = np.zeros(m)
        for j_idx, j in enumerate(fac_indices):
            b_ub[j_idx] = capacities[j]
            for i in range(n_cust):
                A_ub[j_idx, i * m + j_idx] = demands[i]
                
        bounds = [(0, 1)] * (n_cust * m)
        
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
            if res.status == 0:
                x_flat = res.x
                x = x_flat.reshape(n_cust, m)
                return x, res.fun
            else:
                return None, np.inf
        except Exception:
            return None, np.inf

    def round_assignment(x, fac_indices):
        """
        Rounds fractional assignment x to integer assignment.
        Returns list of facility indices for each customer, or None if infeasible.
        """
        m = len(fac_indices)
        assignment = np.zeros(n_cust, dtype=int)
        load = np.zeros(m)
        
        # Greedy rounding: pick argmax x_ij
        # But we must respect capacities.
        # Sort customers by "difficulty" (e.g., max x_ij is close to 1, or demand is large)
        # Let's just pick argmax and check capacity.
        
        # To handle capacity conflicts, we might need to re-assign.
        # Simple approach:
        # 1. Assign each customer to argmax x_ij.
        # 2. Check capacities.
        # 3. If violated, move smallest demand customers from over-capacity facilities to others.
        
        for i in range(n_cust):
            best_j_idx = np.argmax(x[i])
            assignment[i] = best_j_idx
            load[best_j_idx] += demands[i]
            
        # Check and repair
        for j_idx in range(m):
            if load[j_idx] > capacities[fac_indices[j_idx]]:
                # Over capacity
                # Find customers assigned to j_idx
                custs = np.where(assignment == j_idx)[0]
                # Sort by demand ascending (move smallest first)
                custs_sorted = custs[np.argsort(demands[custs])]
                
                excess = load[j_idx] - capacities[fac_indices[j_idx]]
                for i in custs_sorted:
                    if excess <= 0:
                        break
                    # Find best alternative facility
                    best_alt = -1
                    best_cost = np.inf
                    for k_idx in range(m):
                        if k_idx == j_idx:
                            continue
                        if load[k_idx] + demands[i] <= capacities[fac_indices[k_idx]]:
                            # Cost increase: assign_cost[i, fac_indices[k_idx]] - assign_cost[i, fac_indices[j_idx]]
                            # We don't have original x cost here, just assign_cost
                            cost = assign_cost[i, fac_indices[k_idx]]
                            if cost < best_cost:
                                best_cost = cost
                                best_alt = k_idx
                    if best_alt != -1:
                        load[j_idx] -= demands[i]
                        load[best_alt] += demands[i]
                        assignment[i] = best_alt
                        excess -= demands[i]
                    else:
                        # No feasible alternative, assignment is invalid
                        return None
        
        return assignment

    def calc_total_cost(assignment, fac_indices):
        if assignment is None:
            return np.inf
        total_assign = 0
        for i in range(n_cust):
            j_idx = assignment[i]
            total_assign += assign_cost[i, fac_indices[j_idx]]
        
        total_open = 0
        # Only count open cost if facility has load?
        # The problem says "open_cost over facilities with at least one customer".
        # So we only pay for facilities that are actually used.
        used_facilities = set(assignment)
        for j_idx in used_facilities:
            total_open += open_cost[fac_indices[j_idx]]
            
        return total_assign + total_open

    # 1. Initial Solution: Greedy Set Construction
    S = []
    best_S = []
    best_cost = np.inf
    
    # Start with the facility that has the lowest open cost? Or just try all single facilities.
    # Actually, let's start with S = [0] and expand.
    # Better: Start with S = [] and add best facility.
    
    # To make it faster, we can pre-compute something?
    # For now, iterate.
    
    # Initial S: Pick facility with min open_cost + min avg assign cost?
    # Let's just pick the facility with min open_cost.
    init_fac = np.argmin(open_cost)
    S = [init_fac]
    
    # Calculate cost for initial S
    x, lp_cost = solve_assignment_lp(S)
    if x is not None:
        assignment = round_assignment(x, S)
        if assignment is not None:
            cost = calc_total_cost(assignment, S)
            best_cost = cost
            best_S = S.copy()
    
    # Greedy Expansion
    improved = True
    while improved and not is_deadline_exceeded():
        improved = False
        best_add_fac = -1
        best_add_cost = np.inf
        best_add_assignment = None
        
        for k in range(n_fac):
            if k in S:
                continue
            
            new_S = S + [k]
            x, lp_cost = solve_assignment_lp(new_S)
            if x is None:
                continue
                
            assignment = round_assignment(x, new_S)
            if assignment is None:
                continue
                
            cost = calc_total_cost(assignment, new_S)
            if cost < best_add_cost:
                best_add_cost = cost
                best_add_fac = k
                best_add_assignment = assignment
        
        if best_add_fac != -1 and best_add_cost < best_cost:
            S.append(best_add_fac)
            best_cost = best_add_cost
            best_S = S.copy()
            improved = True
        else:
            # Try removing?
            # If no improvement from adding, maybe we can remove?
            # But we started with 1, so removing would lead to empty.
            # Let's allow removal in the local search phase.
            break
            
    # 2. Local Search on Set S
    # Try adding or removing facilities
    S = best_S.copy()
    current_cost = best_cost
    current_assignment = None
    
    # Get current assignment for best_S
    x, _ = solve_assignment_lp(S)
    if x is not None:
        current_assignment = round_assignment(x, S)
        if current_assignment is not None:
            current_cost = calc_total_cost(current_assignment, S)
        else:
            # Fallback to greedy assignment if rounding failed
            current_assignment = np.zeros(n_cust, dtype=int)
            load = np.zeros(len(S))
            for i in range(n_cust):
                # Assign to cheapest in S
                costs = [assign_cost[i, S[j]] for j in range(len(S))]
                j_idx = np.argmin(costs)
                current_assignment[i] = j_idx
                load[j_idx] += demands[i]
            # Repair if needed
            # (Simplified repair for now)
            current_cost = calc_total_cost(current_assignment, S)
            
    improved = True
    while improved and not is_deadline_exceeded():
        improved = False
        
        # Try adding a facility
        for k in range(n_fac):
            if k in S:
                continue
            new_S = S + [k]
            x, _ = solve_assignment_lp(new_S)
            if x is None: continue
            assignment = round_assignment(x, new_S)
            if assignment is None: continue
            cost = calc_total_cost(assignment, new_S)
            if cost < current_cost:
                S = new_S
                current_cost = cost
                current_assignment = assignment
                improved = True
                break # Restart search with new S
                
        if improved: continue
        
        # Try removing a facility
        for idx, k in enumerate(list(S)):
            if len(S) <= 1:
                continue
            new_S = S[:idx] + S[idx+1:]
            x, _ = solve_assignment_lp(new_S)
            if x is None: continue
            assignment = round_assignment(x, new_S)
            if assignment is None: continue
            cost = calc_total_cost(assignment, new_S)
            if cost < current_cost:
                S = new_S
                current_cost = cost
                current_assignment = assignment
                improved = True
                break # Restart search with new S
                
    # 3. Polish Assignment for final S
    # Local search on assignment: Moves and Swaps
    # Convert S to actual facility indices
    final_fac_indices = S
    
    # Ensure we have a valid assignment
    x, _ = solve_assignment_lp(final_fac_indices)
    if x is not None:
        assignment = round_assignment(x, final_fac_indices)
        if assignment is None:
            # Fallback
            assignment = np.zeros(n_cust, dtype=int)
            for i in range(n_cust):
                costs = [assign_cost[i, final_fac_indices[j]] for j in range(len(final_fac_indices))]
                assignment[i] = np.argmin(costs)
    else:
        assignment = np.zeros(n_cust, dtype=int)
        for i in range(n_cust):
            costs = [assign_cost[i, final_fac_indices[j]] for j in range(len(final_fac_indices))]
            assignment[i] = np.argmin(costs)
            
    # Local Search Polish
    # We work with indices into final_fac_indices
    m = len(final_fac_indices)
    load = np.zeros(m)
    for i in range(n_cust):
        load[assignment[i]] += demands[i]
        
    current_cost = calc_total_cost(assignment, final_fac_indices)
    
    polish_improved = True
    while polish_improved and not is_deadline_exceeded():
        polish_improved = False
        
        # Try moves
        for i in range(n_cust):
            curr_j = assignment[i]
            for new_j in range(m):
                if new_j == curr_j:
                    continue
                if load[new_j] + demands[i] <= capacities[final_fac_indices[new_j]]:
                    # Check cost change
                    old_cost = assign_cost[i, final_fac_indices[curr_j]]
                    new_cost = assign_cost[i, final_fac_indices[new_j]]
                    # Open cost change?
                    # If curr_j becomes empty, we save open_cost[curr_j]
                    # If new_j was empty, we add open_cost[new_j]
                    
                    old_load_curr = load[curr_j]
                    old_load_new = load[new_j]
                    
                    open_cost_change = 0
                    if old_load_curr == demands[i]: # curr_j becomes empty
                        open_cost_change -= open_cost[final_fac_indices[curr_j]]
                    if old_load_new == 0: # new_j becomes used
                        open_cost_change += open_cost[final_fac_indices[new_j]]
                        
                    total_change = (new_cost - old_cost) + open_cost_change
                    
                    if total_change < 0:
                        # Accept
                        load[curr_j] -= demands[i]
                        load[new_j] += demands[i]
                        assignment[i] = new_j
                        current_cost += total_change
                        polish_improved = True
                        break # Restart loop for this customer? Or just continue?
                        # Restarting the whole loop is safer to catch interactions.
                        # But for speed, let's just break the inner loop and continue outer.
                        # Actually, breaking inner and continuing outer is standard.
                        break
                        
        if polish_improved:
            continue
            
        # Try swaps
        for i in range(n_cust):
            for l in range(i+1, n_cust):
                j_i = assignment[i]
                j_l = assignment[l]
                if j_i == j_l:
                    continue
                
                # Swap i to j_l and l to j_i
                # Check capacities
                # Load at j_i: loses d_i, gains d_l
                # Load at j_l: loses d_l, gains d_i
                if load[j_i] - demands[i] + demands[l] <= capacities[final_fac_indices[j_i]] and \
                   load[j_l] - demands[l] + demands[i] <= capacities[final_fac_indices[j_l]]:
                   
                    old_cost_i = assign_cost[i, final_fac_indices[j_i]]
                    old_cost_l = assign_cost[l, final_fac_indices[j_l]]
                    new_cost_i = assign_cost[i, final_fac_indices[j_l]]
                    new_cost_l = assign_cost[l, final_fac_indices[j_i]]
                    
                    # Open cost change?
                    # j_i load change: -d_i + d_l. If it becomes 0, save open.
                    # j_l load change: -d_l + d_i. If it becomes 0, save open.
                    # If it was 0 (impossible since i and l are there), add open.
                    
                    new_load_j_i = load[j_i] - demands[i] + demands[l]
                    new_load_j_l = load[j_l] - demands[l] + demands[i]
                    
                    open_cost_change = 0
                    if load[j_i] > 0 and new_load_j_i == 0:
                        open_cost_change -= open_cost[final_fac_indices[j_i]]
                    if load[j_l] > 0 and new_load_j_l == 0:
                        open_cost_change -= open_cost[final_fac_indices[j_l]]
                    # Note: Since i and l are assigned to j_i and j_l, load was > 0.
                    # If new load is 0, we save cost.
                    # We don't add cost because they were already open.
                    
                    total_change = (new_cost_i + new_cost_l - old_cost_i - old_cost_l) + open_cost_change
                    
                    if total_change < 0:
                        # Accept
                        load[j_i] = new_load_j_i
                        load[j_l] = new_load_j_l
                        assignment[i] = j_l
                        assignment[l] = j_i
                        current_cost += total_change
                        polish_improved = True
                        break
            if polish_improved:
                break
                
    # Map back to original facility indices
    final_assignment = np.zeros(n_cust, dtype=int)
    for i in range(n_cust):
        final_assignment[i] = final_fac_indices[assignment[i]]
        
    return final_assignment

