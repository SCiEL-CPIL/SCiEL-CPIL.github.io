"""
Solver for Capacitated Facility Location Problem (CFLP).
Strategy: 
1. Construct a cost matrix for the Hungarian algorithm (linear sum assignment).
2. The cost matrix incorporates assignment costs and a large penalty for capacity 
   violations (approximated via a "soft" capacity check or simply relying on 
   post-processing). However, standard Hungarian does not handle capacities directly.
   
   Revised Strategy:
   Since standard Hungarian minimizes sum of costs for a perfect matching (1-to-1), 
   but we have capacities (many-to-1), we cannot directly use it on the raw matrix 
   unless we expand facilities. Expanding facilities by capacity units is too expensive 
   (Capacities can be large, e.g., 1000s, and we have 100 customers).
   
   Alternative Strategy: 
   Use a Greedy-Local Search hybrid with a strong initialization.
   The previous strategy used Lagrangian. Let's try a different family:
   "Greedy Construction with Randomized Restarts and Intensive Local Search".
   
   Actually, the prompt suggests: "Formulate the problem as a minimum-cost maximum-flow network... solve it using SciPy's linear programming or a specialized network simplex".
   
   Let's implement a Min-Cost Flow approach using `scipy.optimize.linprog` or a custom network simplex?
   SciPy does not have a direct min-cost flow solver in the main API, but `scipy.optimize.linprog` can solve it.
   However, LP for 100x20 = 2000 variables is small.
   
   Wait, the fixed charge (open_cost) makes it an Integer Program (MIP). LP relaxation will give fractional results.
   We can solve the LP relaxation, round it, and repair.
   
   Let's try:
   1. Solve LP relaxation of CFLP using `scipy.optimize.linprog`.
      Variables: x_cf (100*20 = 2000), y_f (20).
      Objective: Min sum c_cf x_cf + sum o_f y_f
      Constraints:
      - sum_f x_cf = 1 for all c
      - sum_c d_c x_cf <= cap_f y_f for all f
      - 0 <= x_cf <= 1
      - 0 <= y_f <= 1
      
   2. The LP solution will likely be fractional.
   3. Use the LP solution to guide a heuristic: 
      - Sort facilities by "importance" or use the fractional y_f values.
      - Or, simply use the LP solution to initialize a local search.
      - A common heuristic: Assign customer c to facility f with highest x_cf in the LP solution, then repair capacity violations.
      
   4. Perform local search (moves and swaps) to improve the feasible solution.
   
   This is a materially different strategy: LP Relaxation + Rounding/Repair + Local Search.
"""

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assign each customer to a facility.
    Strategy: LP Relaxation to get fractional solution, round to feasible, then local search.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    n_customers = len(demands)
    n_facilities = len(capacities)

    demands = np.asarray(demands, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.float64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)

    # Helper to check time
    def time_left():
        return deadline - time.time()

    # Helper to compute cost
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

    # Helper to repair infeasible assignment
    def repair(assignment):
        """
        Greedy repair: Move customers from overloaded facilities to underloaded ones.
        """
        assignment = assignment.copy()
        load = np.zeros(n_facilities)
        for c in range(n_customers):
            load[int(assignment[c])] += demands[c]
        
        # Iterate until feasible
        for _ in range(100):
            if time_left() < 5:
                break
            # Find overloaded facilities
            overloaded = [f for f in range(n_facilities) if load[f] > capacities[f] + 1e-9]
            if not overloaded:
                break
            
            for f in overloaded:
                excess = load[f] - capacities[f]
                # Find customers in f
                custs_in_f = [c for c in range(n_customers) if assignment[c] == f]
                # Sort by "regret" or cost difference to move
                # We want to move customers that have cheap alternatives
                def move_penalty(c):
                    best_alt = np.inf
                    for f2 in range(n_facilities):
                        if f2 == f:
                            continue
                        if load[f2] + demands[c] <= capacities[f2] + 1e-9:
                            alt_cost = assign_cost[c, f2]
                            if load[f2] == 0:
                                alt_cost += open_cost[f2]
                            if alt_cost < best_alt:
                                best_alt = alt_cost
                    return best_alt - assign_cost[c, f]
                
                custs_in_f.sort(key=move_penalty)
                
                for c in custs_in_f:
                    if excess <= 1e-9:
                        break
                    # Find best alternative
                    best_f2 = -1
                    best_cost = np.inf
                    for f2 in range(n_facilities):
                        if f2 == f:
                            continue
                        if load[f2] + demands[c] <= capacities[f2] + 1e-9:
                            alt_cost = assign_cost[c, f2]
                            if load[f2] == 0:
                                alt_cost += open_cost[f2]
                            if alt_cost < best_cost:
                                best_cost = alt_cost
                                best_f2 = f2
                    if best_f2 != -1:
                        load[f] -= demands[c]
                        load[best_f2] += demands[c]
                        assignment[c] = best_f2
                        excess -= demands[c]
        
        # Final check
        load = np.zeros(n_facilities)
        for c in range(n_customers):
            load[int(assignment[c])] += demands[c]
        if any(load[f] > capacities[f] + 1e-9 for f in range(n_facilities)):
            return None
        return assignment

    # Helper for Local Search
    def local_search(assignment, max_iters=500):
        """
        Single-move local search.
        """
        assignment = assignment.copy()
        load = np.zeros(n_facilities)
        for c in range(n_customers):
            load[int(assignment[c])] += demands[c]
        
        current_cost = compute_cost(assignment)
        if np.isinf(current_cost):
            return assignment
            
        improved = True
        while improved and max_iters > 0 and time_left() > 10:
            improved = False
            max_iters -= 1
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
        
        return assignment

    # === Main Algorithm ===
    
    # Phase 1: Solve LP Relaxation
    # Variables: x_0 ... x_{C*F-1} (x_cf), y_0 ... y_{F-1} (y_f)
    # Total vars: C*F + F
    C = n_customers
    F = n_facilities
    num_vars = C * F + F
    
    # Objective: c^T x
    c_obj = np.zeros(num_vars)
    for c in range(C):
        for f in range(F):
            c_obj[c * F + f] = assign_cost[c, f]
    for f in range(F):
        c_obj[C * F + f] = open_cost[f]
    
    # Inequality constraints: A_ub x <= b_ub
    # Capacity constraints: sum_c d_c x_cf <= cap_f y_f  =>  sum_c d_c x_cf - cap_f y_f <= 0
    A_ub = np.zeros((F, num_vars))
    b_ub = np.zeros(F)
    for f in range(F):
        for c in range(C):
            A_ub[f, c * F + f] = demands[c]
        A_ub[f, C * F + f] = -capacities[f]
        b_ub[f] = 0
    
    # Equality constraints: A_eq x = b_eq
    # Sum_f x_cf = 1 for each c
    A_eq = np.zeros((C, num_vars))
    b_eq = np.ones(C)
    for c in range(C):
        for f in range(F):
            A_eq[c, c * F + f] = 1.0
    
    # Bounds
    bounds = [(0, 1)] * num_vars
    
    # Solve LP
    sol_lp = None
    try:
        # Use highs method for speed and reliability
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        if res.success:
            sol_lp = res.x
    except Exception:
        pass
    
    best_solution = None
    best_cost = np.inf
    
    # Phase 2: Generate initial solutions
    # Solution 1: From LP rounding
    if sol_lp is not None:
        x_lp = sol_lp[:C*F].reshape(C, F)
        # Greedy rounding: assign c to f with highest x_lp[c, f]
        init_assign = np.argmax(x_lp, axis=1)
        # Repair
        repaired = repair(init_assign)
        if repaired is not None:
            cost = compute_cost(repaired)
            if cost < best_cost:
                best_cost = cost
                best_solution = repaired.copy()
        
        # Solution 2: Greedy based on LP "openness" (y values)
        y_lp = sol_lp[C*F:]
        # Prioritize facilities with high y_lp
        # Sort customers by demand, assign to feasible facility with best (assign_cost - lambda*y_lp)
        # Or just use the LP x values to break ties in a greedy
        # Let's try a simple greedy using LP guidance
        init_assign2 = np.full(C, -1)
        load = np.zeros(F)
        order = np.argsort(-demands)
        for c in order:
            # Score: assign_cost[c, f] - 10 * x_lp[c, f] (bias towards LP solution)
            scores = assign_cost[c] - 10 * x_lp[c]
            feasible = load + demands[c] <= capacities
            scores[~feasible] = np.inf
            if np.all(np.isinf(scores)):
                f = int(np.argmin(scores + load))
            else:
                f = int(np.argmin(scores))
            init_assign2[c] = f
            load[f] += demands[c]
        
        repaired2 = repair(init_assign2)
        if repaired2 is not None:
            cost = compute_cost(repaired2)
            if cost < best_cost:
                best_cost = cost
                best_solution = repaired2.copy()

    # Solution 3: Standard Greedy (Largest First)
    if time_left() > 20:
        init_assign3 = np.full(C, -1)
        load = np.zeros(F)
        order = np.argsort(-demands)
        for c in order:
            costs = assign_cost[c].copy()
            feasible = load + demands[c] <= capacities
            costs[~feasible] = np.inf
            if np.all(np.isinf(costs)):
                f = int(np.argmin(costs + load))
            else:
                f = int(np.argmin(costs))
            init_assign3[c] = f
            load[f] += demands[c]
        
        repaired3 = repair(init_assign3)
        if repaired3 is not None:
            cost = compute_cost(repaired3)
            if cost < best_cost:
                best_cost = cost
                best_solution = repaired3.copy()
        
        # Solution 4: Randomized Greedy
        for _ in range(5):
            if time_left() < 10:
                break
            init_assign4 = np.full(C, -1)
            load = np.zeros(F)
            order = np.argsort(-demands)
            for c in order:
                costs = assign_cost[c].copy()
                feasible = load + demands[c] <= capacities
                costs[~feasible] = np.inf
                if np.all(np.isinf(costs)):
                    f = int(np.argmin(costs + load))
                else:
                    # Add noise
                    costs += np.random.uniform(0, 1, F) * 0.1
                    f = int(np.argmin(costs))
                init_assign4[c] = f
                load[f] += demands[c]
            
            repaired4 = repair(init_assign4)
            if repaired4 is not None:
                cost = compute_cost(repaired4)
                if cost < best_cost:
                    best_cost = cost
                    best_solution = repaired4.copy()

    # Phase 3: Local Search
    if best_solution is not None and time_left() > 10:
        improved = local_search(best_solution, max_iters=1000)
        cost = compute_cost(improved)
        if cost < best_cost:
            best_cost = cost
            best_solution = improved

    # Fallback if no solution found (should not happen with repair)
    if best_solution is None:
        best_solution = np.zeros(C, dtype=int)
        # Force feasibility with simple greedy
        load = np.zeros(F)
        for c in range(C):
            # Find any feasible facility
            for f in range(F):
                if load[f] + demands[c] <= capacities[f]:
                    best_solution[c] = f
                    load[f] += demands[c]
                    break
            else:
                # Should not happen if total capacity >= total demand
                best_solution[c] = 0
                load[0] += demands[c]
        
        # Final repair just in case
        repaired_final = repair(best_solution)
        if repaired_final is not None:
            best_solution = repaired_final

    return best_solution.astype(int)
# EVOLVE-BLOCK-END
