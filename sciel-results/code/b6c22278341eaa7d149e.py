"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog
import itertools

def assign(demands, capacities, open_cost, assign_cost):
    """
    Capacitated Facility Location with Fixed Charges via Lagrangian Relaxation.
    
    Structural approach:
    1. Dualize capacity constraints with Lagrange multipliers.
    2. Each iteration: solve separable customer subproblems (greedy assignment 
       to facility minimizing adjusted cost), then update multipliers via 
       subgradient method.
    3. For promising multipliers, solve the transportation subproblem exactly 
       via LP to get a feasible assignment for the open facilities.
    4. Strong local search with 1-move and 2-swap moves to polish.
    5. Multiple restarts with different initial multipliers for diversity.
    """
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    
    best_result = None
    best_cost = float('inf')
    
    def total_cost(assignment):
        """Compute total cost for an assignment."""
        loads = np.zeros(n_facilities)
        for ci in range(n_customers):
            loads[assignment[ci]] += demands[ci]
        if np.any(loads > capacities + 1e-9):
            return float('inf')
        cost = np.sum(assign_cost[np.arange(n_customers), assignment])
        used = loads > 0
        cost += np.sum(open_cost[used])
        return cost
    
    def solve_transportation_lp(open_facilities):
        """
        Solve the transportation subproblem: assign all customers to the given
        open facilities to minimize total assignment cost, respecting capacities.
        Returns a feasible assignment or None.
        """
        open_list = list(open_facilities)
        k = len(open_list)
        if k == 0:
            return None
        
        n_vars = n_customers * k
        
        # Objective: assignment costs
        c_obj = assign_cost[:, open_list].flatten()
        
        # Equality constraints: each customer assigned to exactly one facility
        A_eq = np.zeros((n_customers, n_vars))
        b_eq = np.ones(n_customers)
        for ci in range(n_customers):
            for fi_idx in range(k):
                A_eq[ci, ci * k + fi_idx] = 1.0
        
        # Inequality constraints: capacity
        A_ub = np.zeros((k, n_vars))
        b_ub = np.array([capacities[f] for f in open_list])
        for fi_idx, f in enumerate(open_list):
            for ci in range(n_customers):
                A_ub[fi_idx, ci * k + fi_idx] = demands[ci]
        
        bounds = [(0, 1)] * n_vars
        
        try:
            res = linprog(c_obj, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub,
                          bounds=bounds, method='highs', options={'presolve': True})
        except Exception:
            return None
        
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
    
    def lagrangian_relaxation(lambda_init=None, max_iters=80, step_base=1.0):
        """
        Lagrangian relaxation: dualize capacity constraints.
        
        At each iteration:
        - Compute adjusted cost: assign_cost[c,f] + lambda[f] * demand[c]
        - Assign each customer to argmin adjusted cost (greedy)
        - Compute loads, check feasibility
        - If feasible, return the assignment
        - Update lambda via subgradient: lambda += step * (load - capacity)
        - Track best feasible or near-feasible solution
        
        Returns (best_assignment, best_value) where best_assignment may be infeasible.
        """
        if lambda_init is None:
            lambda_ = np.zeros(n_facilities)
        else:
            lambda_ = np.array(lambda_init, dtype=float)
        
        best_assign = None
        best_val = float('inf')
        best_feasible = None
        best_feasible_cost = float('inf')
        
        for it in range(max_iters):
            if time.time() > deadline - 15:
                break
            
            # Adjusted cost matrix: assign_cost[c,f] + lambda[f] * demand[c]
            adjusted_cost = assign_cost + np.outer(demands, lambda_)
            
            # Greedy assignment: each customer picks argmin adjusted cost
            tentative_assign = np.argmin(adjusted_cost, axis=1)
            
            # Compute loads
            loads = np.zeros(n_facilities)
            for ci in range(n_customers):
                loads[tentative_assign[ci]] += demands[ci]
            
            # Check feasibility
            is_feasible = np.all(loads <= capacities + 1e-9)
            
            # Compute Lagrangian lower bound
            min_costs = np.min(adjusted_cost, axis=1)
            used_mask = loads > 0
            L = np.sum(min_costs) + np.sum(open_cost[used_mask]) - np.sum(lambda_ * capacities)
            
            # Track best Lagrangian value
            if L < best_val:
                best_val = L
                best_assign = tentative_assign.copy()
            
            # If feasible, compute actual cost and track
            if is_feasible:
                actual_cost = total_cost(tentative_assign)
                if actual_cost < best_feasible_cost:
                    best_feasible_cost = actual_cost
                    best_feasible = tentative_assign.copy()
                return tentative_assign, L
            
            # Subgradient update: gradient of L w.r.t. lambda is (load - capacity)
            gradient = loads - capacities
            
            # Adaptive step size using Polyak-style with best known upper bound
            # Use diminishing step: step_base / (1 + it * decay)
            step = step_base / (1.0 + 0.05 * it)
            
            # Project onto lambda >= 0
            lambda_ = np.maximum(0, lambda_ + step * gradient)
        
        # Return best feasible if found, else best Lagrangian solution
        if best_feasible is not None:
            return best_feasible, best_feasible_cost
        return best_assign, best_val
    
    def local_search_strong(assignment, max_rounds=50):
        """
        Strong local search with:
        - 1-move: reassign single customer to another facility
        - 2-swap: swap two customers between two facilities
        - 2-relocate: move two customers from one facility to another
        """
        assignment = assignment.copy()
        loads = np.zeros(n_facilities)
        for ci in range(n_customers):
            loads[assignment[ci]] += demands[ci]
        
        current_cost = total_cost(assignment)
        if current_cost == float('inf'):
            return assignment, current_cost
        
        improved = True
        round_count = 0
        while improved and round_count < max_rounds:
            improved = False
            round_count += 1
            if time.time() > deadline - 10:
                break
            
            # Pass 1: 1-move reassignments
            for ci in range(n_customers):
                if time.time() > deadline - 10:
                    break
                current_f = assignment[ci]
                best_delta = 0
                best_f = -1
                for f in range(n_facilities):
                    if f == current_f:
                        continue
                    if loads[f] + demands[ci] > capacities[f] + 1e-9:
                        continue
                    delta = assign_cost[ci, f] - assign_cost[ci, current_f]
                    # Open cost changes
                    if loads[current_f] == demands[ci]:
                        delta -= open_cost[current_f]
                    if loads[f] == 0:
                        delta += open_cost[f]
                    if delta < best_delta - 1e-9:
                        best_delta = delta
                        best_f = f
                if best_f >= 0:
                    loads[current_f] -= demands[ci]
                    loads[best_f] += demands[ci]
                    assignment[ci] = best_f
                    current_cost += best_delta
                    improved = True
            
            # Pass 2: 2-swap (swap two customers between two facilities)
            if time.time() > deadline - 15:
                break
            # Only try swaps among facilities that have at least one customer
            active_facs = [f for f in range(n_facilities) if loads[f] > 0]
            if len(active_facs) >= 2:
                # Sample a subset of customer pairs for efficiency
                n_pairs_checked = 0
                max_pairs = 2000
                for ci1 in range(n_customers):
                    if n_pairs_checked >= max_pairs:
                        break
                    if time.time() > deadline - 10:
                        break
                    f1 = assignment[ci1]
                    for ci2 in range(ci1 + 1, n_customers):
                        if n_pairs_checked >= max_pairs:
                            break
                        f2 = assignment[ci2]
                        if f1 == f2:
                            continue
                        n_pairs_checked += 1
                        # Check if swapping is feasible
                        if loads[f1] - demands[ci1] + demands[ci2] > capacities[f1] + 1e-9:
                            continue
                        if loads[f2] - demands[ci2] + demands[ci1] > capacities[f2] + 1e-9:
                            continue
                        # Cost delta
                        delta = (assign_cost[ci1, f2] + assign_cost[ci2, f1] -
                                 assign_cost[ci1, f1] - assign_cost[ci2, f2])
                        # Open cost changes
                        open_delta = 0
                        if loads[f1] == demands[ci1]:
                            open_delta -= open_cost[f1]
                        if loads[f2] == demands[ci2]:
                            open_delta -= open_cost[f2]
                        # After swap, both facilities still have customers unless one becomes empty
                        if loads[f1] - demands[ci1] + demands[ci2] == 0:
                            open_delta -= open_cost[f1]  # already counted above
                        if loads[f2] - demands[ci2] + demands[ci1] == 0:
                            open_delta -= open_cost[f2]
                        
                        if delta + open_delta < -1e-9:
                            # Apply swap
                            loads[f1] = loads[f1] - demands[ci1] + demands[ci2]
                            loads[f2] = loads[f2] - demands[ci2] + demands[ci1]
                            assignment[ci1] = f2
                            assignment[ci2] = f1
                            current_cost += delta + open_delta
                            improved = True
                            break
        
        return assignment, current_cost
    
    def greedy_open_aware():
        """
        Greedy that considers open costs: prefer facilities that are already 
        'open' (will have other customers) to avoid additional open costs.
        """
        result = np.zeros(n_customers, dtype=int)
        load = np.zeros(n_facilities)
        # Process customers in order of decreasing demand
        order = np.argsort(-demands)
        for i in order:
            best_f = -1
            best_cost = float('inf')
            for f in range(n_facilities):
                if load[f] + demands[i] > capacities[f] + 1e-9:
                    continue
                # Cost: assignment + marginal open cost
                marginal_open = open_cost[f] if load[f] == 0 else 0.0
                total = assign_cost[i, f] + marginal_open
                if total < best_cost:
                    best_cost = total
                    best_f = f
            if best_f < 0:
                # Fallback: facility with most remaining capacity
                remaining = capacities - load
                best_f = np.argmax(remaining)
                load[best_f] += demands[i]
                result[i] = best_f
            else:
                load[best_f] += demands[i]
                result[i] = best_f
        return result
    
    def random_feasible():
        """Generate a random feasible assignment."""
        result = np.zeros(n_customers, dtype=int)
        load = np.zeros(n_facilities)
        order = np.random.permutation(n_customers)
        for i in order:
            feasible = [f for f in range(n_facilities) 
                       if load[f] + demands[i] <= capacities[f] + 1e-9]
            if not feasible:
                # Shouldn't happen if total demand <= total capacity
                f = np.argmax(capacities - load)
            else:
                f = feasible[np.random.randint(len(feasible))]
            result[i] = f
            load[f] += demands[i]
        return result
    
    # Phase 1: Generate diverse initial candidates
    candidates = []
    
    # Greedy open-aware
    candidates.append(greedy_open_aware())
    
    # Lagrangian with zero initial multipliers
    lr1, _ = lagrangian_relaxation(lambda_init=np.zeros(n_facilities), max_iters=60, step_base=1.0)
    if lr1 is not None:
        candidates.append(lr1)
    
    # Lagrangian with random initial multipliers
    np.random.seed(42)
    for seed in range(3):
        if time.time() > deadline - 30:
            break
        lam_init = np.random.rand(n_facilities) * 2.0
        lr, _ = lagrangian_relaxation(lambda_init=lam_init, max_iters=40, step_base=0.8)
        if lr is not None:
            candidates.append(lr)
    
    # Random feasible
    np.random.seed(123)
    candidates.append(random_feasible())
    
    # Phase 2: For each candidate, solve transportation LP if facilities are well-chosen,
    # then polish with local search
    for idx, cand in enumerate(candidates):
        if time.time() > deadline - 20:
            break
        
        # Validate candidate
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
        
        # Try to improve: extract open facilities and solve LP
        if time.time() < deadline - 25:
            used_facs = sorted(set(cand.tolist()))
            if len(used_facs) >= 2:
                lp_result = solve_transportation_lp(used_facs)
                if lp_result is not None:
                    lp_cost = total_cost(lp_result)
                    if lp_cost < best_cost:
                        best_cost = lp_cost
                        best_result = lp_result.copy()
                    # Polish LP result
                    if time.time() < deadline - 15:
                        polished, pol_cost = local_search_strong(lp_result, max_rounds=30)
                        if pol_cost < best_cost:
                            best_cost = pol_cost
                            best_result = polished.copy()
        
        # Polish the candidate directly
        if time.time() < deadline - 15:
            polished, pol_cost = local_search_strong(cand, max_rounds=30)
            if pol_cost < best_cost:
                best_cost = pol_cost
                best_result = polished.copy()
    
    # Phase 3: More Lagrangian iterations on the best solution's facility set
    if best_result is not None and time.time() < deadline - 20:
        used_facs = sorted(set(best_result.tolist()))
        # Try LP on the current best's facility set
        lp_best = solve_transportation_lp(used_facs)
        if lp_best is not None:
            lp_best_cost = total_cost(lp_best)
            if lp_best_cost < best_cost:
                best_cost = lp_best_cost
                best_result = lp_best.copy()
        
        # Additional local search rounds
        while time.time() < deadline - 5:
            improved, improved_cost = local_search_strong(best_result, max_rounds=20)
            if improved_cost < best_cost - 1e-9:
                best_cost = improved_cost
                best_result = improved.copy()
            else:
                break
    
    # Phase 4: Try different facility subsets via Lagrangian with varied parameters
    if best_result is not None and time.time() < deadline - 30:
        np.random.seed(999)
        for attempt in range(5):
            if time.time() > deadline - 15:
                break
            # Perturb the multipliers
            lam = np.random.rand(n_facilities) * 3.0
            lr, _ = lagrangian_relaxation(lambda_init=lam, max_iters=30, step_base=1.2)
            if lr is not None:
                # Validate
                loads = np.zeros(n_facilities)
                valid = True
                for ci in range(n_customers):
                    if lr[ci] < 0 or lr[ci] >= n_facilities:
                        valid = False
                        break
                    loads[lr[ci]] += demands[ci]
                if not valid or np.any(loads > capacities + 1e-9):
                    continue
                lr_cost = total_cost(lr)
                if lr_cost < best_cost:
                    best_cost = lr_cost
                    best_result = lr.copy()
                # Polish
                if time.time() < deadline - 10:
                    polished, pol_cost = local_search_strong(lr, max_rounds=20)
                    if pol_cost < best_cost:
                        best_cost = pol_cost
                        best_result = polished.copy()
    
    # Final validation
    if best_result is None:
        best_result = greedy_open_aware()
    
    # Ensure the result is valid
    final_loads = np.zeros(n_facilities)
    for ci in range(n_customers):
        final_loads[best_result[ci]] += demands[ci]
    if np.any(final_loads > capacities + 1e-9):
        # Fallback to a simple greedy that's guaranteed feasible
        best_result = greedy_open_aware()
    
    return best_result.astype(int)

# EVOLVE-BLOCK-END
