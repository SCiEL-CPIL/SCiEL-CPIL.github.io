import numpy as np
import time


def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    n_customers = len(demands)
    n_facilities = len(capacities)

    demands = np.asarray(demands, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.float64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)

    best_result = None
    best_score = float('inf')

    def evaluate(result):
        load = np.zeros(n_facilities)
        for i, f in enumerate(result):
            if f < 0 or f >= n_facilities:
                return float('inf')
            load[f] += demands[i]
        if np.any(load > capacities + 1e-9):
            return float('inf')

        open_mask = np.zeros(n_facilities, dtype=bool)
        for f in result:
            open_mask[f] = True

        open_sum = np.sum(open_cost[open_mask])
        assign_sum = np.sum(assign_cost[np.arange(n_customers), result])
        return open_sum + assign_sum

    def greedy_init(randomize=False, seed=None):
        if seed is not None:
            np.random.seed(seed)

        result = np.zeros(n_customers, dtype=int)
        load = np.zeros(n_facilities)
        is_open = np.zeros(n_facilities, dtype=bool)

        # Sort customers by demand descending
        order = np.argsort(-demands)

        for i in order:
            d = demands[i]
            # Feasible facilities
            feasible = load + d <= capacities

            if not np.any(feasible):
                # Should not happen if total capacity >= total demand, but handle gracefully
                # Fall back to cheapest
                f = np.argmin(assign_cost[i])
                result[i] = f
                load[f] += d
                is_open[f] = True
                continue

            # Calculate marginal costs
            # If facility is open, cost is assign_cost[i, f]
            # If closed, cost is assign_cost[i, f] + open_cost[f]
            marginal_costs = assign_cost[i].copy()
            closed_mask = ~is_open
            marginal_costs[closed_mask] += open_cost[closed_mask]

            # Mask infeasible
            marginal_costs[~feasible] = np.inf

            if randomize:
                # Pick from top 3
                sorted_indices = np.argsort(marginal_costs)
                top_k = sorted_indices[:min(3, n_facilities)]
                f = top_k[np.random.randint(len(top_k))]
            else:
                f = np.argmin(marginal_costs)

            result[i] = f
            load[f] += d
            is_open[f] = True

        return result, load, is_open

    def local_search(result, load, is_open):
        # result: array of facility indices
        # load: array of current loads
        # is_open: array of boolean open status

        improved = True
        while improved:
            improved = False
            # Iterate customers in fixed order
            for i in np.arange(n_customers):
                f_old = result[i]
                d = demands[i]

                best_delta = 1e-9
                best_f_new = -1

                for f_new in range(n_facilities):
                    if f_new == f_old:
                        continue

                    # Check capacity
                    if load[f_new] + d > capacities[f_new]:
                        continue

                    # Calculate delta
                    delta_assign = assign_cost[i, f_new] - assign_cost[i, f_old]

                    delta_open = 0.0
                    # Check if old facility becomes empty
                    if load[f_old] - d == 0:
                        # It becomes empty, so we save its open cost
                        # But only if it was open (it must be, since it has customer i)
                        delta_open -= open_cost[f_old]

                    # Check if new facility was empty
                    if load[f_new] == 0:
                        # It becomes non-empty, so we incur its open cost
                        # But only if it was closed
                        if not is_open[f_new]:
                            delta_open += open_cost[f_new]

                    total_delta = delta_assign + delta_open

                    if total_delta < best_delta:
                        best_delta = total_delta
                        best_f_new = f_new

                if best_f_new != -1:
                    # Apply move
                    # Update old facility
                    load[f_old] -= d
                    if load[f_old] == 0:
                        is_open[f_old] = False

                    # Update new facility
                    load[best_f_new] += d
                    if load[best_f_new] == d:  # Was 0
                        is_open[best_f_new] = True

                    result[i] = best_f_new
                    improved = True

                    # Time check inside loop
                    if time.time() > deadline:
                        return result, load, is_open

        return result, load, is_open

    def swap_search(result, load, is_open):
        # Try swapping customers between facilities
        improved = True
        while improved:
            improved = False
            for i in range(n_customers):
                f_i = result[i]
                d_i = demands[i]

                for j in range(i + 1, n_customers):
                    f_j = result[j]
                    d_j = demands[j]

                    if f_i == f_j:
                        continue

                    # Check capacity after swap
                    # f_i loses d_i, gains d_j
                    # f_j loses d_j, gains d_i
                    if load[f_i] - d_i + d_j > capacities[f_i]:
                        continue
                    if load[f_j] - d_j + d_i > capacities[f_j]:
                        continue

                    # Calculate delta
                    # Assignment cost change
                    delta_assign = (assign_cost[i, f_j] + assign_cost[j, f_i]) - \
                                   (assign_cost[i, f_i] + assign_cost[j, f_j])

                    # Open cost change
                    delta_open = 0.0
                    # Check if f_i becomes empty after losing i and gaining j
                    # f_i load after: load[f_i] - d_i + d_j
                    # f_i becomes empty if load[f_i] - d_i + d_j == 0
                    # But since d_j > 0, this can only happen if load[f_i] - d_i == 0 and d_j == 0, which is impossible
                    # Wait, if load[f_i] == d_i (only customer i), then after swap, load becomes d_j > 0
                    # So f_i does NOT become empty if it was the only customer, unless d_j=0
                    # Actually, f_i becomes empty only if load[f_i] - d_i + d_j == 0
                    # Since d_j >= 5, this is never 0 if d_i <= load[f_i]
                    # So f_i never becomes empty in a swap?
                    # No, if load[f_i] == d_i and we swap, new load is d_j. If d_j > 0, it's not empty.
                    # So open cost for f_i doesn't change?
                    # Wait, if f_i was open, it stays open.
                    # What if f_i was closed? It can't be, since customer i is there.
                    # So f_i is always open before and after (since d_j >= 5 > 0).
                    # Same for f_j.
                    # So delta_open is always 0 for swaps?
                    # Let's verify.
                    # f_i has at least customer i. After swap, it has customer j. So it has at least one customer.
                    # f_j has at least customer j. After swap, it has customer i. So it has at least one customer.
                    # So no facility becomes empty or newly opened.
                    # Thus delta_open = 0.

                    total_delta = delta_assign

                    if total_delta < -1e-9:
                        # Apply swap
                        result[i] = f_j
                        result[j] = f_i
                        load[f_i] -= d_i
                        load[f_i] += d_j
                        load[f_j] -= d_j
                        load[f_j] += d_i
                        improved = True
                        break

                if time.time() > deadline:
                    return result, load, is_open

        return result, load, is_open

    def lagrangian_init(seed=None):
        """
        Lagrangian relaxation for capacity constraints.
        1. Relax capacity constraints with multipliers lambda.
        2. Solve decoupled problem: min sum_i (assign_cost[i, f] + lambda_f * demands[i]) + sum_f open_cost_f * y_f
           where y_f = 1 if any customer assigned to f.
           This is still coupled by y_f, but we can use a heuristic:
           For each customer, assign to facility minimizing assign_cost[i,f] + lambda_f * demands[i] + open_cost_f * (1 if f is first assigned)
           Actually, the dual problem is:
           min_{y, x} sum_f open_cost_f y_f + sum_i sum_f assign_cost[i,f] x_if
           s.t. sum_f x_if = 1 for all i
                sum_i demands_i x_if <= capacities_f y_f for all f
                y_f in {0,1}
                
           Lagrangian relaxation of capacity constraints:
           L(lambda) = min_{x, y} sum_f open_cost_f y_f + sum_i sum_f (assign_cost[i,f] + lambda_f demands_i) x_if
                      s.t. sum_f x_if = 1, x_if >= 0, y_f >= 0, and y_f >= x_if (implicitly)
                      
           The inner minimization over x and y is not fully decoupled because of y_f.
           However, if we fix y, it decouples. If we fix x, y_f = max(0, max_i x_if) is determined.
           
           A common approach is to alternate:
           1. Given lambda, solve the assignment problem with modified costs c_if' = assign_cost[i,f] + lambda_f demands_i.
              But we also have open costs. We can treat open costs as part of the decision.
              
           Let's try a simpler heuristic:
           1. Initialize lambda = 0.
           2. Repeat:
              a. For each customer i, compute cost to facility f: c_if = assign_cost[i,f] + lambda_f * demands[i].
              b. Assign each customer to the facility with minimum c_if.
              c. Check capacity. If violated, increase lambda for violated facilities.
              d. If capacity is satisfied, we have a feasible solution (possibly suboptimal).
              e. Also, we need to account for open costs. The Lagrangian doesn't directly handle open costs well in this simple form.
              
           Alternative: Use Lagrangian relaxation for the "at least one customer" constraint? No, that's not a constraint.
           
           Let's stick to the strategy: "Apply Lagrangian relaxation to the facility capacity constraints, solving the resulting decoupled assignment subproblem via vectorized argmin for each customer, and use a subgradient method to iteratively update multipliers until the dual gap is minimized, followed by a rounding heuristic to enforce capacity feasibility."
           
           The "decoupled assignment subproblem" implies that after relaxing capacity, each customer can be assigned independently. This suggests that the open cost is either ignored or handled separately.
           
           If we ignore open cost in the Lagrangian step, we get a lower bound on the assignment cost part.
           Then we can use a rounding heuristic to enforce capacity and add open costs.
           
           Steps:
           1. lambda = 0
           2. Iteration:
              a. For each i, f_i = argmin_f (assign_cost[i,f] + lambda_f * demands[i])
              b. Compute load_f = sum_{i: f_i=f} demands[i]
              c. If all load_f <= capacities_f, break (feasible)
              d. Update lambda: lambda_f = max(0, lambda_f + step_size * (load_f - capacities_f) / capacities_f)
           
           After convergence or max iterations:
           3. We have an assignment that may violate capacity.
           4. Rounding/Feasibility:
              - Identify facilities with load > capacity.
              - Move customers from overloaded facilities to underloaded ones, prioritizing those with lowest additional cost.
              - This is similar to the local search but starting from the Lagrangian solution.
              
           Let's implement this.
        """
        if seed is not None:
            np.random.seed(seed)

        lambda_vec = np.zeros(n_facilities)
        step_size = 1.0
        max_iters = 100
        
        result = np.zeros(n_customers, dtype=int)
        
        for it in range(max_iters):
            # Vectorized assignment
            # cost_matrix[i, f] = assign_cost[i, f] + lambda_vec[f] * demands[i]
            cost_matrix = assign_cost + np.outer(demands, lambda_vec)
            result = np.argmin(cost_matrix, axis=1)
            
            # Check capacity
            load = np.zeros(n_facilities)
            for i, f in enumerate(result):
                load[f] += demands[i]
                
            violations = load - capacities
            if np.all(violations <= 1e-9):
                break
            
            # Subgradient update
            # Gradient of L(lambda) wrt lambda_f is (load_f - capacities_f)
            # We want to maximize the dual function, so we go in the direction of the gradient?
            # L(lambda) is a concave function of lambda (max of linear functions).
            # To maximize L(lambda), we update lambda in the direction of the subgradient.
            # subgradient_f = load_f - capacities_f
            # lambda_new = lambda_old + step_size * subgradient
            # But lambda must be non-negative.
            
            step_size = 0.5 * step_size  # Decay step size
            lambda_vec = np.maximum(0, lambda_vec + step_size * violations / np.maximum(capacities, 1))
            
            if time.time() > deadline - 10:
                break
        
        # Now we have a result from the last iteration (which might be infeasible)
        # We need to make it feasible.
        # Use a heuristic: sort customers by "regret" or just use local search to fix infeasibility.
        
        # Let's initialize load and is_open
        load = np.zeros(n_facilities)
        is_open = np.zeros(n_facilities, dtype=bool)
        for i, f in enumerate(result):
            load[f] += demands[i]
            is_open[f] = True
            
        # Feasibility repair: 
        # While infeasible, find an overloaded facility and move a customer to a feasible one.
        # To minimize cost increase, move the customer with lowest "penalty".
        
        # We can just run local_search on this result.
        # But local_search assumes feasibility? No, it checks capacity.
        # However, if start is infeasible, local_search might not fix it if it only considers moves that preserve feasibility?
        # My local_search checks `if load[f_new] + d > capacities[f_new]: continue`.
        # It does NOT check if the current state is feasible.
        # If the current state is infeasible, `load[f_old]` is already > capacity.
        # Moving a customer OUT of f_old will reduce load[f_old].
        # So local_search CAN fix infeasibility if it allows moving to a facility with spare capacity.
        # But what if no facility has spare capacity? Then we are stuck.
        # Assuming total capacity >= total demand, there should be a feasible solution.
        
        return result, load, is_open

    # Main loop with restarts
    # 1. Deterministic greedy
    res, load, is_open = greedy_init(randomize=False)
    res, load, is_open = local_search(res, load, is_open)
    res, load, is_open = swap_search(res, load, is_open)
    score = evaluate(res)
    if score < best_score:
        best_score = score
        best_result = res.copy()

    # 2. Lagrangian initialization + Local Search
    for k in range(20):
        if time.time() > deadline - 10:
            break

        res, load, is_open = lagrangian_init(seed=k)
        res, load, is_open = local_search(res, load, is_open)
        res, load, is_open = swap_search(res, load, is_open)
        score = evaluate(res)
        if score < best_score:
            best_score = score
            best_result = res.copy()

    # 3. Randomized restarts
    # Reduce number of restarts to make room for Lagrangian
    for k in range(30):
        if time.time() > deadline - 10:  # Leave 10s buffer
            break

        res, load, is_open = greedy_init(randomize=True, seed=100 + k)
        res, load, is_open = local_search(res, load, is_open)
        res, load, is_open = swap_search(res, load, is_open)
        score = evaluate(res)
        if score < best_score:
            best_score = score
            best_result = res.copy()

    return best_result
