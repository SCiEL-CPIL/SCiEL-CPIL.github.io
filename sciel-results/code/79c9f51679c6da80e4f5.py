"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    """
    Capacitated facility location: assign customers to facilities minimizing
    open_cost + assign_cost, respecting capacities.
    
    Strategy:
    1. Use Lagrangian relaxation to find good facility subsets.
    2. For each candidate subset, solve the transportation problem exactly.
    3. Use local search (customer swaps) to refine.
    4. Track best solution within time budget.
    """
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    best_result = None
    best_cost = float('inf')
    
    def total_cost(assignment):
        """Compute total cost for an assignment."""
        loads = np.zeros(n_facilities)
        for ci in range(n_customers):
            loads[assignment[ci]] += demands[ci]
        if np.any(loads > capacities + 1e-9):
            return float('inf')
        cost = 0.0
        for ci in range(n_customers):
            cost += assign_cost[ci, assignment[ci]]
        for f in range(n_facilities):
            if loads[f] > 0:
                cost += open_cost[f]
        return cost
    
    def solve_transportation(open_facilities, demands, capacities, assign_cost):
        """
        Solve the transportation problem: assign customers to open facilities
        to minimize assignment cost, respecting capacities.
        Uses scipy's linear programming (HiGHS).
        """
        try:
            from scipy.optimize import linprog
        except ImportError:
            return None
        
        open_list = list(open_facilities)
        k = len(open_list)
        if k == 0:
            return None
        
        n_vars = n_customers * k
        
        # Objective
        c_obj = np.zeros(n_vars)
        for ci in range(n_customers):
            for fi_idx, f in enumerate(open_list):
                c_obj[ci * k + fi_idx] = assign_cost[ci, f]
        
        # Equality constraints: each customer assigned to exactly one facility
        A_eq = np.zeros((n_customers, n_vars))
        b_eq = np.ones(n_customers)
        for ci in range(n_customers):
            for fi_idx, f in enumerate(open_list):
                A_eq[ci, ci * k + fi_idx] = 1.0
        
        # Inequality constraints: capacity
        A_ub = np.zeros((k, n_vars))
        b_ub = np.array([capacities[f] for f in open_list])
        for fi_idx, f in enumerate(open_list):
            for ci in range(n_customers):
                A_ub[fi_idx, ci * k + fi_idx] = demands[ci]
        
        bounds = [(0, 1)] * n_vars
        
        res = linprog(c_obj, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, 
                      bounds=bounds, method='highs')
        
        if res.status == 0:
            x = res.x.reshape(n_customers, k)
            assignment = np.zeros(n_customers, dtype=int)
            for ci in range(n_customers):
                best_idx = np.argmax(x[ci])
                assignment[ci] = open_list[best_idx]
            # Verify feasibility
            loads = np.zeros(n_facilities)
            for ci in range(n_customers):
                loads[assignment[ci]] += demands[ci]
            if np.all(loads <= capacities + 1e-9):
                return assignment
        return None
    
    def lagrangian_relaxation(demands, capacities, open_cost, assign_cost, 
                               max_iters=50, step_size=0.5):
        """
        Lagrangian relaxation for facility location.
        Dualize the capacity constraints and solve the resulting
        separable subproblems.
        """
        n_customers = len(demands)
        n_facilities = len(capacities)
        
        # Dual variables (Lagrange multipliers) for capacity constraints
        lambda_ = np.zeros(n_facilities)
        
        best_assign = None
        best_dual_cost = float('inf')
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # For each customer, choose facility that minimizes:
            # assign_cost[c, f] + lambda_[f] * demand[c]
            # subject to: if facility f is "open" (has any customer assigned)
            #             add open_cost[f]
            
            # First pass: ignore open cost, just minimize adjusted cost
            adjusted_cost = assign_cost + np.outer(demands, lambda_)
            
            # For each customer, pick the facility with minimum adjusted cost
            tentative_assign = np.argmin(adjusted_cost, axis=1)
            
            # Check which facilities are used
            used_facilities = set(tentative_assign.tolist())
            
            # Compute the Lagrangian objective value
            # L(lambda) = sum_c min_f (assign_cost[c,f] + lambda_f * demand_c)
            #            + sum_f (open_cost_f if f used else 0)
            #            - sum_f lambda_f * capacity_f
            #            + sum_f lambda_f * (capacity_f - load_f)  [complementary slackness]
            # Actually: L = sum_c min_f (assign_cost[c,f] + lambda_f * demand_c)
            #             + sum_f open_cost_f * I(f used)
            #             - sum_f lambda_f * capacity_f
            
            min_costs = np.min(adjusted_cost, axis=1)
            sum_min = np.sum(min_costs)
            
            open_cost_sum = sum(open_cost[f] for f in used_facilities)
            cap_penalty = -np.sum(lambda_ * capacities)
            
            L = sum_min + open_cost_sum + cap_penalty
            
            # Update best
            if L < best_dual_cost:
                best_dual_cost = L
                best_assign = tentative_assign.copy()
            
            # Check if capacity constraints are satisfied
            loads = np.zeros(n_facilities)
            for ci in range(n_customers):
                loads[tentative_assign[ci]] += demands[ci]
            
            if np.all(loads <= capacities + 1e-9):
                # Feasible solution!
                return tentative_assign, L
            
            # Subgradient update
            # g_f = capacity_f - load_f (if lambda_f > 0)
            #     = capacity_f - load_f (always, for subgradient of max)
            # Actually for max of L(lambda), gradient is:
            # g_f = load_f - capacity_f (we want to increase lambda where overloaded)
            gradient = loads - capacities
            
            # Adaptive step size
            # Use a diminishing step size
            step = step_size / (1.0 + it * 0.1)
            
            # Update lambda
            lambda_ = np.maximum(0, lambda_ + step * gradient)
        
        return best_assign, best_dual_cost
    
    def local_search(assignment, max_iters=500):
        """Improve assignment via 1-swap local search."""
        loads = np.zeros(n_facilities)
        for ci in range(n_customers):
            loads[assignment[ci]] += demands[ci]
        
        current_cost = total_cost(assignment)
        if current_cost == float('inf'):
            return assignment, current_cost
        
        improved = True
        iter_count = 0
        while improved and iter_count < max_iters:
            improved = False
            iter_count += 1
            for ci in range(n_customers):
                current_f = assignment[ci]
                for f in range(n_facilities):
                    if f == current_f:
                        continue
                    # Check capacity
                    if loads[f] + demands[ci] <= capacities[f] + 1e-9:
                        # Compute cost delta
                        old_cost = assign_cost[ci, current_f]
                        new_cost = assign_cost[ci, f]
                        # Open cost change
                        open_delta = 0
                        if loads[current_f] == demands[ci]:
                            # This facility would be closed
                            open_delta -= open_cost[current_f]
                        if loads[f] == 0:
                            # This facility would be opened
                            open_delta += open_cost[f]
                        
                        delta = new_cost - old_cost + open_delta
                        if delta < -1e-9:
                            # Apply swap
                            loads[current_f] -= demands[ci]
                            loads[f] += demands[ci]
                            assignment[ci] = f
                            current_cost += delta
                            improved = True
                            break
        return assignment, current_cost
    
    def greedy_nearest(demands, capacities, open_cost, assign_cost):
        """Greedy: assign each customer to nearest feasible facility."""
        result = np.zeros(n_customers, dtype=int)
        load = np.zeros(n_facilities)
        order = np.argsort(-demands)
        for i in order:
            feasible = [f for f in range(n_facilities) if load[f] + demands[i] <= capacities[f]]
            if not feasible:
                f = np.argmax(capacities - load)
                result[i] = f
                load[f] += demands[i]
            else:
                f = min(feasible, key=lambda f: assign_cost[i, f])
                result[i] = f
                load[result[i]] += demands[i]
        return result
    
    # Phase 1: Generate diverse candidates
    candidates = []
    
    # Greedy nearest
    candidates.append(greedy_nearest(demands, capacities, open_cost, assign_cost))
    
    # Lagrangian relaxation
    lr_assign, lr_cost = lagrangian_relaxation(demands, capacities, open_cost, assign_cost,
                                                max_iters=30, step_size=0.5)
    if lr_assign is not None:
        candidates.append(lr_assign)
    
    # Phase 2: For each candidate, try to improve via local search
    for cand in candidates:
        if time.time() > deadline - 5:
            break
        
        # Validate
        loads = np.zeros(n_facilities)
        valid = True
        for ci in range(n_customers):
            if cand[ci] < 0 or cand[ci] >= n_facilities:
                valid = False
                break
            loads[cand[ci]] += demands[ci]
        if not valid or np.any(loads > capacities + 1e-9):
            continue
        
        cost = total_cost(cand)
        if cost < best_cost:
            best_cost = cost
            best_result = cand.copy()
        
        # Local search to improve
        if time.time() < deadline - 10:
            improved, improved_cost = local_search(cand.copy(), max_iters=200)
            if improved_cost < best_cost:
                best_cost = improved_cost
                best_result = improved.copy()
    
    # Phase 3: If we have time, do more local search on the best solution
    if best_result is not None and time.time() < deadline - 5:
        while time.time() < deadline - 2:
            improved, improved_cost = local_search(best_result.copy(), max_iters=100)
            if improved_cost < best_cost - 1e-9:
                best_cost = improved_cost
                best_result = improved.copy()
            else:
                break
    
    if best_result is None:
        # Fallback: simple greedy
        best_result = greedy_nearest(demands, capacities, open_cost, assign_cost)
    
    return best_result.astype(int)

# EVOLVE-BLOCK-END
