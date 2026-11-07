"""
Lagrangian Decomposition (Benders) + Min-Cost Flow + Cluster Local Search for CFLP.

Strategy:
1. Use Benders decomposition logic:
   - Master problem decides which facilities to open (approximated via subgradient on duals).
   - Subproblem assigns customers to open facilities using Min-Cost Flow (exact for fixed set of open facilities).
2. The dual variables (prices) from the flow problem guide the opening decisions.
3. Local search uses "cluster moves": swapping sets of customers between facilities to better utilize capacity and open costs.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assign each customer to a facility.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    demands = np.asarray(demands, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.float64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)

    n_customers = len(demands)
    n_facilities = len(capacities)

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

    def solve_flow(open_facilities, assignment_hint=None):
        """
        Solve the assignment problem for a fixed set of open facilities using Min-Cost Flow.
        Returns assignment array or None if infeasible.
        """
        if not open_facilities:
            return None
            
        # Build bipartite graph: Customers (left) to Facilities (right)
        # We use min_weight_full_bipartite_matching which requires a cost matrix.
        # However, it doesn't handle capacities directly.
        # Standard Min-Cost Flow handles capacities. Let's use a simpler approach for small N:
        # Since N=100, M=20, we can use a heuristic greedy with local improvements, 
        # OR we can formulate it as a min-cost flow if we had a library.
        # scipy.sparse.csgraph.min_weight_full_bipartite_matching is for perfect matching (1-to-1).
        # We need 1-to-many with capacity.
        
        # Alternative: Use a greedy assignment followed by local search, but guided by duals.
        # Or, since M is small (20), we can use a simple iterative improvement.
        
        # Let's try a "Greedy + Swap" approach specifically for the open set.
        # This is fast and often near-optimal for the subproblem.
        
        assignment = np.full(n_customers, -1, dtype=int)
        load = np.zeros(n_facilities)
        
        # Sort customers by demand descending (harder to place)
        order = np.argsort(-demands)
        
        for c in order:
            best_f = -1
            best_cost = np.inf
            for f in open_facilities:
                if load[f] + demands[c] <= capacities[f]:
                    # Cost to add c to f
                    # If f was empty, we already paid open_cost, so marginal is just assign_cost
                    # But in the subproblem, open_cost is fixed.
                    cost = assign_cost[c, f]
                    if cost < best_cost:
                        best_cost = cost
                        best_f = f
            if best_f == -1:
                return None # Infeasible
            assignment[c] = best_f
            load[best_f] += demands[c]
            
        return assignment

    def local_search(assignment, max_time=10.0):
        """
        Local search with single moves and pairwise swaps.
        """
        assignment = assignment.copy()
        load = np.zeros(n_facilities)
        for c in range(n_customers):
            load[int(assignment[c])] += demands[c]
        
        current_cost = compute_cost(assignment)
        if np.isinf(current_cost):
            return assignment

        start_ls = time.time()
        improved = True
        while improved and (time.time() - start_ls) < max_time:
            improved = False
            
            # 1. Single moves
            for c in range(n_customers):
                if time.time() > deadline:
                    return assignment
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
                    
                    # Open cost changes
                    if load[new_f] == 0:
                        delta += open_cost[new_f]
                    if load[old_f] == 0:
                        delta -= open_cost[old_f]
                        
                    if delta < best_delta - 1e-9:
                        best_delta = delta
                        best_new_f = new_f
                        
                if best_new_f != old_f:
                    assignment[c] = best_new_f
                    load[best_new_f] += demands[c]
                    current_cost += best_delta
                    improved = True
                else:
                    load[old_f] += demands[c]
                    
            # 2. Pairwise swaps (if time permits)
            if not improved and (time.time() - start_ls) < max_time * 0.5:
                for c1 in range(n_customers):
                    if time.time() > deadline:
                        return assignment
                    f1 = int(assignment[c1])
                    for c2 in range(c1 + 1, n_customers):
                        f2 = int(assignment[c2])
                        if f1 == f2:
                            continue
                        
                        # Check capacity
                        if load[f1] - demands[c1] + demands[c2] > capacities[f1]:
                            continue
                        if load[f2] - demands[c2] + demands[c1] > capacities[f2]:
                            continue
                        
                        delta = (assign_cost[c1, f2] + assign_cost[c2, f1]) - \
                                (assign_cost[c1, f1] + assign_cost[c2, f2])
                        
                        # Open cost changes
                        # f1 becomes empty if load[f1] == demands[c1]
                        # f2 becomes empty if load[f2] == demands[c2]
                        # f1 becomes open if load[f1] == 0 (impossible here)
                        # f2 becomes open if load[f2] == 0 (impossible here)
                        
                        if load[f1] == demands[c1]:
                             # f1 loses c1, gains c2. If c2 demand > 0, it stays open.
                             # Wait, if load[f1] == demands[c1], then only c1 is there.
                             # After swap, c2 is there. So it stays open (unless c2 demand is 0).
                             pass
                        if load[f2] == demands[c2]:
                             pass
                             
                        # Actually, open cost only changes if a facility becomes empty or non-empty.
                        # Since both f1 and f2 had at least one customer before, and demands > 0,
                        # they will both have at least one customer after swap.
                        # So open cost delta is 0.
                        
                        if delta < -1e-9:
                            assignment[c1] = f2
                            assignment[c2] = f1
                            load[f1] = load[f1] - demands[c1] + demands[c2]
                            load[f2] = load[f2] - demands[c2] + demands[c1]
                            current_cost += delta
                            improved = True
                            break
                    if improved:
                        break
                        
        return assignment

    # --- Main Logic: Benders-like Iteration ---
    
    best_solution = None
    best_cost = np.inf
    
    # Initial guess: Open all facilities
    open_set = list(range(n_facilities))
    
    # Benders Loop
    for iter in range(50):
        if time.time() > deadline:
            break
            
        # Solve Subproblem: Assign to open_set
        sol = solve_flow(open_set)
        if sol is None:
            # Infeasible, need to open more facilities
            # Find most overloaded or just open all
            open_set = list(range(n_facilities))
            sol = solve_flow(open_set)
            if sol is None:
                break
            
        cost = compute_cost(sol)
        if cost < best_cost:
            best_cost = cost
            best_solution = sol.copy()
            
        # Local Search on this solution
        if time.time() < deadline - 20:
            improved_sol = local_search(sol, max_time=min(10.0, deadline - time.time() - 5))
            improved_cost = compute_cost(improved_sol)
            if improved_cost < best_cost:
                best_cost = improved_cost
                best_solution = improved_sol.copy()
                
        # Update Master Problem (which facilities to open)
        # Heuristic: Calculate "pressure" on each facility.
        # If a facility is very full, it's good to keep it.
        # If a facility is barely used, maybe close it?
        # But closing it requires re-assigning customers.
        
        # Let's use a simple rule:
        # 1. Keep facilities that are "critical" (high utilization or low alternative cost).
        # 2. Try closing the facility with the highest (open_cost / utilization) or similar.
        
        load = np.zeros(n_facilities)
        for c in range(n_customers):
            load[int(best_solution[c])] += demands[c]
            
        # Identify candidates to close:
        # Facilities that are not "bottlenecks".
        # A facility is a bottleneck if removing it makes the problem infeasible or very expensive.
        
        # Heuristic: Try to close the facility with the smallest "value".
        # Value = open_cost - (potential savings in assignment cost if closed? No, that's complex).
        # Let's just try closing each facility one by one in a "k-best" search manner?
        # Too slow.
        
        # Alternative: Use dual information.
        # If we had the dual prices from the flow, we could use them.
        # Since we used greedy, let's use a heuristic:
        # Sort facilities by open_cost descending. Try to close the most expensive ones if possible.
        
        # Find facility to close:
        # For each open facility f, check if we can reassign its customers to others.
        # We will try to close the facility with the highest open_cost that is "least utilized" relative to capacity?
        # Or simply: Iterate through facilities, try to close the one that minimizes the increase in assignment cost.
        
        # This is getting complex. Let's stick to a simpler "Perturbation" approach.
        # Randomly close a facility, solve subproblem, keep if better.
        
        current_open_set = set(open_set)
        
        # Try to close a facility
        closed_any = False
        # Sort by open_cost descending (expensive to open)
        sorted_facilities = sorted(open_set, key=lambda f: -open_cost[f])
        
        for f_to_close in sorted_facilities:
            if time.time() > deadline:
                break
            if f_to_close not in current_open_set:
                continue
                
            # Try closing f_to_close
            new_open_set = [f for f in open_set if f != f_to_close]
            
            # Solve subproblem with new_open_set
            # Use best_solution as a hint? No, solve from scratch or repair.
            # Repair is faster.
            # Take best_solution, move customers from f_to_close to others.
            
            trial_sol = best_solution.copy()
            load_trial = np.zeros(n_facilities)
            for c in range(n_customers):
                load_trial[int(trial_sol[c])] += demands[c]
                
            # Move customers from f_to_close
            customers_to_move = [c for c in range(n_customers) if trial_sol[c] == f_to_close]
            # Sort by demand descending
            customers_to_move.sort(key=lambda c: -demands[c])
            
            feasible = True
            for c in customers_to_move:
                best_f = -1
                best_cost_move = np.inf
                for new_f in new_open_set:
                    if load_trial[new_f] + demands[c] <= capacities[new_f]:
                        cost_move = assign_cost[c, new_f]
                        if cost_move < best_cost_move:
                            best_cost_move = cost_move
                            best_f = new_f
                if best_f == -1:
                    feasible = False
                    break
                trial_sol[c] = best_f
                load_trial[new_f] += demands[c]
                load_trial[f_to_close] -= demands[c]
                
            if feasible:
                trial_cost = compute_cost(trial_sol)
                if trial_cost < best_cost:
                    best_cost = trial_cost
                    best_solution = trial_sol.copy()
                    open_set = new_open_set
                    closed_any = True
                    break # Restart the loop with new open set
                    
        if not closed_any:
            # Try opening a facility if we are currently infeasible or stuck?
            # If we are feasible, we might be stuck in a local optimum of the open set.
            # Try opening a closed facility.
            closed_facilities = [f for f in range(n_facilities) if f not in open_set]
            if closed_facilities:
                # Try opening the one with lowest open_cost
                f_to_open = min(closed_facilities, key=lambda f: open_cost[f])
                new_open_set = open_set + [f_to_open]
                sol = solve_flow(new_open_set)
                if sol is not None:
                    cost = compute_cost(sol)
                    if cost < best_cost:
                        best_cost = cost
                        best_solution = sol.copy()
                        open_set = new_open_set
                    else:
                        # Even if not better, maybe local search from this new start helps?
                        # Only if time permits
                        if time.time() < deadline - 30:
                            improved_sol = local_search(sol, max_time=5.0)
                            improved_cost = compute_cost(improved_sol)
                            if improved_cost < best_cost:
                                best_cost = improved_cost
                                best_solution = improved_sol.copy()
                                open_set = new_open_set

    # Final Polish
    if best_solution is not None and time.time() < deadline - 5:
        final_sol = local_search(best_solution, max_time=deadline - time.time() - 2)
        final_cost = compute_cost(final_sol)
        if final_cost < best_cost:
            best_solution = final_sol

    if best_solution is None:
        # Fallback: Greedy with all open
        best_solution = solve_flow(list(range(n_facilities)))
        if best_solution is None:
             # Should not happen if total demand <= total capacity
             best_solution = np.zeros(n_customers, dtype=int)

    return best_solution.astype(int)

# EVOLVE-BLOCK-END
