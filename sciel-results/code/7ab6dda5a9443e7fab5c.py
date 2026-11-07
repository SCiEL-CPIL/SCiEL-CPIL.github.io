# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import aslinearoperator

def assign(demands, capacities, open_cost, assign_cost):
    """
    Capacitated facility location via Lagrangian relaxation + min-cost flow.
    """
    start = time.time()
    deadline = start + 140.0  # safety margin before 150s kill

    demands = np.asarray(demands, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.float64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)

    n_c = len(demands)
    n_f = len(capacities)

    def total_cost(assignment):
        load = np.zeros(n_f)
        for c in range(n_c):
            f = int(assignment[c])
            load[f] += demands[c]
        if np.any(load > capacities + 1e-9):
            return np.inf
        cost = 0.0
        used = np.zeros(n_f, dtype=bool)
        for c in range(n_c):
            f = int(assignment[c])
            cost += assign_cost[c, f]
            used[f] = True
        cost += open_cost[used].sum()
        return cost

    def min_cost_flow(open_mask, max_time=10.0):
        """
        Solve transportation: customers -> open facilities.
        Returns assignment array or None if infeasible/too slow.
        Uses scipy linprog (HiGHS) in network form.
        """
        t0 = time.time()
        open_idx = np.where(open_mask)[0]
        m = len(open_idx)
        if m == 0:
            return None

        # Variables x[c, k] for each customer c and open facility k
        n_vars = n_c * m
        # Objective: assign_cost[c, open_idx[k]]
        c_obj = assign_cost[:, open_idx].ravel().copy()

        # Equality constraints:
        # For each customer: sum_k x[c,k] = 1  (n_c constraints)
        # For each open facility: sum_c x[c,k] <= cap  (m constraints, but as equality with dummy? 
        # Actually capacity is an upper bound. In transportation, supply=1 per customer, demand=cap per facility.
        # But total supply = n_c, total capacity = sum(caps). If sum(caps) >= n_c, we can add dummy customer.
        # Simpler: use linprog with equality for customer assignment and inequality for capacity.
        
        # A_eq: identity for customers (n_c x n_vars)
        A_eq = np.zeros((n_c, n_vars))
        for c in range(n_c):
            A_eq[c, c*m:(c+1)*m] = 1.0
        b_eq = np.ones(n_c)

        # A_ub: capacity constraints (m x n_vars)
        A_ub = np.zeros((m, n_vars))
        for k in range(m):
            for c in range(n_c):
                A_ub[k, c*m + k] = 1.0
        b_ub = capacities[open_idx]

        bounds = [(0, 1)] * n_vars

        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                          bounds=bounds, method='highs')
        except Exception:
            return None

        if res.status != 0:
            return None

        if time.time() - t0 > max_time:
            return None

        x = res.x.reshape(n_c, m)
        # Round to assignment
        assignment = np.zeros(n_c, dtype=int)
        for c in range(n_c):
            # Find the open facility with max x
            k_best = int(np.argmax(x[c]))
            assignment[c] = open_idx[k_best]
        
        # Verify feasibility (rounding might break capacity slightly if not integer, but x should be integral for network flow)
        load = np.zeros(n_f)
        for c in range(n_c):
            load[assignment[c]] += demands[c]
        if np.any(load > capacities + 1e-9):
            return None

        return assignment

    def lagrangian_iteration(open_mask, mu, max_flow_time=8.0):
        """
        Given open facilities and Lagrange multipliers mu (per facility),
        solve relaxed assignment, then flow.
        Returns (assignment, dual_update_info).
        """
        # Relaxed cost: assign_cost[c, f] + mu[f]
        # But only for open facilities? In Lagrangian, we relax capacity constraints.
        # Actually, let's just do:
        # 1. Solve flow on open_mask.
        # 2. If feasible, we have a solution.
        # 3. Update mu based on congestion? 
        # Standard CFLP Lagrangian: relax capacity. 
        # Min sum_c sum_f x_cf (d_cf + mu_f) + sum_f open_cost_f y_f
        # s.t. sum_f x_cf = 1, x_cf <= y_f
        # Dual: maximize L(mu)
        
        # This is complex. Let's use a simpler heuristic decomposition:
        # 1. Solve flow on current open set.
        # 2. If infeasible (shouldn't be if caps are enough), or if we want to change open set:
        #    Estimate "congestion" or use reduced costs to decide opening/closing.
        
        # For this implementation, we will:
        # 1. Start with a set of open facilities (e.g., all, or greedy).
        # 2. Solve flow.
        # 3. If flow is infeasible (capacity tight), try opening more.
        # 4. If a facility is lightly used and open_cost is high, try closing it and re-solving.
        
        # Let's implement a "branch and bound" style or just iterative improvement.
        # Given time limits, let's do:
        # 1. Find a feasible assignment using flow on a candidate open set.
        # 2. Local search on the open set:
        #    - Try closing a facility: remove it, solve flow on remaining. If feasible and cost lower, accept.
        #    - Try opening a facility: add it, solve flow on new set. If feasible and cost lower, accept.
        
        pass # Placeholder, logic moved to main loop

    # --- Main Strategy ---
    
    # 1. Initial Open Set: Greedy by cost-effectiveness
    # Calculate "average assignment cost" per facility
    avg_assign_cost = assign_cost.mean(axis=0)
    # Sort facilities by (open_cost + avg_assign_cost * (total_demand / n_f)) ?
    # Simpler: Open facilities with lowest open_cost + avg distance.
    # But we need enough capacity.
    
    # Let's start with ALL facilities open, then close them.
    open_mask = np.ones(n_f, dtype=bool)
    
    # Solve flow on all facilities
    assignment = min_cost_flow(open_mask, max_time=15.0)
    if assignment is None:
        # Fallback to greedy
        assignment = np.zeros(n_c, dtype=int)
        load = np.zeros(n_f)
        for c in np.argsort(-demands):
            feasible = load + demands[c] <= capacities
            if not feasible.any():
                feasible = np.ones(n_f, dtype=bool) # Should not happen
            costs = assign_cost[c].copy()
            costs[~feasible] = np.inf
            f = int(np.argmin(costs))
            assignment[c] = f
            load[f] += demands[c]
        return assignment.astype(int)
    
    current_cost = total_cost(assignment)
    
    # 2. Iteratively try to close facilities
    # Compute load on current assignment
    load = np.zeros(n_f)
    for c in range(n_c):
        load[assignment[c]] += demands[c]
    
    # Facilities with 0 load are definitely closeable
    # Facilities with low load might be closeable if we can re-route cheaply.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try closing each open facility
        for f in range(n_f):
            if not open_mask[f]:
                continue
            
            # Don't try to close if it's the only one? Or if load is critical?
            # Just try.
            new_open = open_mask.copy()
            new_open[f] = False
            
            # Check if remaining capacity is enough
            if capacities[new_open].sum() < demands.sum():
                continue
            
            # Solve flow on new_open
            new_assign = min_cost_flow(new_open, max_time=5.0)
            if new_assign is None:
                continue
            
            new_cost = total_cost(new_assign)
            
            if new_cost < current_cost - 1e-9:
                current_cost = new_cost
                assignment = new_assign
                open_mask = new_open
                improved = True
                # Recompute load
                load = np.zeros(n_f)
                for c in range(n_c):
                    load[assignment[c]] += demands[c]
                break # Restart loop with new state
    
    # 3. Try opening closed facilities
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        
        for f in range(n_f):
            if open_mask[f]:
                continue
            
            new_open = open_mask.copy()
            new_open[f] = True
            
            # Solve flow
            new_assign = min_cost_flow(new_open, max_time=5.0)
            if new_assign is None:
                continue
            
            new_cost = total_cost(new_assign)
            
            if new_cost < current_cost - 1e-9:
                current_cost = new_cost
                assignment = new_assign
                open_mask = new_open
                improved = True
                break
    
    # 4. Final local search on customer assignments (within open facilities)
    # This is a standard local search but restricted to open facilities.
    # This helps fix any suboptimality from the flow rounding or initial flow solution.
    
    def local_search_restricted(assignment, open_mask, max_passes=10):
        assignment = assignment.copy()
        load = np.zeros(n_f)
        for c in range(n_c):
            load[assignment[c]] += demands[c]
        
        current_cost = total_cost(assignment)
        
        for _ in range(max_passes):
            if time.time() > deadline - 5:
                break
            improved = False
            for c in range(n_c):
                old_f = int(assignment[c])
                load[old_f] -= demands[c]
                
                # Find best feasible open facility
                feasible = (load + demands[c] <= capacities + 1e-9) & open_mask
                if not feasible.any():
                    load[old_f] += demands[c]
                    continue
                
                # Calculate cost delta
                # Current cost contribution: assign_cost[c, old_f]
                # New cost contribution: assign_cost[c, new_f]
                # Open cost change:
                # If old_f becomes 0 load, we save open_cost[old_f]
                # If new_f was 0 load, we pay open_cost[new_f]
                
                best_f = -1
                best_delta = np.inf
                
                candidate_facs = np.where(feasible)[0]
                for new_f in candidate_facs:
                    delta = assign_cost[c, new_f] - assign_cost[c, old_f]
                    
                    # Open cost adjustments
                    if load[old_f] == 0:
                        delta -= open_cost[old_f]
                    if load[new_f] == 0:
                        delta += open_cost[new_f]
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_f = new_f
                
                if best_f != -1 and best_delta < -1e-9:
                    assignment[c] = best_f
                    load[best_f] += demands[c]
                    current_cost += best_delta
                    improved = True
                else:
                    load[old_f] += demands[c]
            
            if not improved:
                break
        
        return assignment, current_cost

    if time.time() < deadline - 5:
        assignment, current_cost = local_search_restricted(assignment, open_mask, max_passes=20)
    
    return assignment.astype(int)

# EVOLVE-BLOCK-END
