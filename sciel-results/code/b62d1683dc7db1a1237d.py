import numpy as np
import time
from scipy.optimize import linprog

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # 1. LP Relaxation for initial solution
    # Variables: x_{i,j} for i in 0..n_cust-1, j in 0..n_fac-1, and y_j for j in 0..n_fac-1
    # Total vars: n_cust * n_fac + n_fac
    # Index mapping:
    # x_{i,j} index = i * n_fac + j
    # y_j index = n_cust * n_fac + j
    
    c = np.zeros(n_cust * n_fac + n_fac)
    # Objective: min sum(open_cost_j * y_j) + sum(assign_cost_{i,j} * x_{i,j})
    for j in range(n_fac):
        c[n_cust * n_fac + j] = open_cost[j]
    for i in range(n_cust):
        for j in range(n_fac):
            c[i * n_fac + j] = assign_cost[i, j]
            
    # Constraints:
    # 1. Sum_j x_{i,j} = 1 for each i
    # A_eq shape: (n_cust, n_cust * n_fac + n_fac)
    A_eq = np.zeros((n_cust, n_cust * n_fac + n_fac))
    b_eq = np.ones(n_cust)
    for i in range(n_cust):
        for j in range(n_fac):
            A_eq[i, i * n_fac + j] = 1.0
            
    # 2. Capacity: Sum_i demand_i * x_{i,j} <= capacity_j
    # A_ub shape: (n_fac, n_cust * n_fac + n_fac)
    A_ub = np.zeros((n_fac, n_cust * n_fac + n_fac))
    b_ub = capacities.copy()
    for j in range(n_fac):
        for i in range(n_cust):
            A_ub[j, i * n_fac + j] = demands[i]
            
    # 3. x_{i,j} <= y_j
    # This is n_cust * n_fac constraints.
    # x_{i,j} - y_j <= 0
    # We can combine these into A_ub by adding rows.
    # Current A_ub has n_fac rows. We add n_cust * n_fac rows.
    A_ub_extra = np.zeros((n_cust * n_fac, n_cust * n_fac + n_fac))
    b_ub_extra = np.zeros(n_cust * n_fac)
    for i in range(n_cust):
        for j in range(n_fac):
            idx = i * n_fac + j
            A_ub_extra[idx, i * n_fac + j] = 1.0
            A_ub_extra[idx, n_cust * n_fac + j] = -1.0
            
    A_ub_total = np.vstack([A_ub, A_ub_extra])
    b_ub_total = np.concatenate([b_ub, b_ub_extra])
    
    # Bounds: 0 <= x_{i,j} <= 1, 0 <= y_j <= 1
    bounds = [(0, 1)] * (n_cust * n_fac + n_fac)
    
    # Solve LP
    res = None
    try:
        res = linprog(c, A_ub=A_ub_total, b_ub=b_ub_total, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    except Exception:
        pass
        
    # 2. Initial Assignment based on LP or Greedy Fallback
    result = np.zeros(n_cust, dtype=int)
    load = np.zeros(n_fac)
    
    if res is not None and res.status == 0:
        x_vals = res.x[:n_cust * n_fac].reshape(n_cust, n_fac)
        # Sort customers by demand descending to place large ones first
        order = np.argsort(-demands)
        for i in order:
            # Get facilities sorted by LP value descending
            fac_order = np.argsort(-x_vals[i])
            assigned = False
            for j in fac_order:
                if load[j] + demands[i] <= capacities[j]:
                    load[j] += demands[i]
                    result[i] = j
                    assigned = True
                    break
            if not assigned:
                # Fallback: cheapest feasible
                for j in np.argsort(assign_cost[i]):
                    if load[j] + demands[i] <= capacities[j]:
                        load[j] += demands[i]
                        result[i] = j
                        break
                # If still not assigned (should not happen if capacities are sufficient),
                # we might need a more robust repair, but let's assume feasible instances.
                if not assigned:
                    # Force assign to facility with most remaining capacity
                    j = np.argmax(capacities - load)
                    load[j] += demands[i]
                    result[i] = j
    else:
        # Greedy Fallback
        order = np.argsort(-demands)
        for i in order:
            for j in np.argsort(assign_cost[i]):
                if load[j] + demands[i] <= capacities[j]:
                    load[j] += demands[i]
                    result[i] = j
                    break
            if result[i] == 0 and load[0] < demands[0]: # crude check if assigned
                 # Actually result is initialized to 0, so this is ambiguous.
                 pass

    # Helper to calculate cost
    def calc_cost(assign_vec, load_vec):
        cost = 0.0
        open_fac = set(assign_vec)
        for j in open_fac:
            cost += open_cost[j]
        for i in range(n_cust):
            cost += assign_cost[i, assign_vec[i]]
        return cost

    current_cost = calc_cost(result, load)
    
    # 3. Local Search
    # We will perform moves until no improvement or time limit.
    # Move types:
    # 1. Relocate: Move customer i from f1 to f2.
    # 2. Swap: Swap customers i and k between f1 and f2.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # --- Pass 1: Relocations ---
        # To speed up, we can just check all possible moves.
        # N=100, M=20. 100*19 = 1900 checks.
        for i in range(n_cust):
            f1 = result[i]
            d_i = demands[i]
            # If f1 becomes empty, we save open_cost[f1]
            # We need to know if f1 is used by others.
            # We can maintain a count of customers per facility.
            # Let's use a temporary array for counts.
            
            # Check all other facilities
            for f2 in range(n_fac):
                if f2 == f1:
                    continue
                if load[f2] + d_i <= capacities[f2]:
                    # Calculate cost change
                    # Remove i from f1: save assign_cost[i, f1]
                    # Add i to f2: pay assign_cost[i, f2]
                    delta_assign = assign_cost[i, f2] - assign_cost[i, f1]
                    
                    delta_open = 0.0
                    # If f1 becomes empty, we save open_cost[f1]
                    # If f2 was empty, we pay open_cost[f2]
                    
                    # We need counts to determine this.
                    # Let's maintain counts array.
                    # For efficiency, we can just check if load[f1] == d_i (means only i is there? No, load is sum of demands).
                    # We need customer counts.
                    
                    # Let's pre-compute counts outside the loop?
                    # Or just check:
                    # If no other customer is at f1, it closes.
                    # If no other customer is at f2, it opens.
                    
                    # To avoid O(N) check for "is f1 empty", we maintain `counts`.
                    pass
                    
        # Let's refactor to use counts for efficiency.
        
    # Restarting local search logic with counts
    
    counts = np.zeros(n_fac, dtype=int)
    for i in range(n_cust):
        counts[result[i]] += 1
        
    def get_cost_delta_relocate(i, f1, f2):
        # Returns cost delta
        delta = assign_cost[i, f2] - assign_cost[i, f1]
        if counts[f1] == 1:
            delta -= open_cost[f1]
        if counts[f2] == 0:
            delta += open_cost[f2]
        return delta

    def get_cost_delta_swap(i, k, f1, f2):
        # i at f1, k at f2. Swap to i at f2, k at f1.
        delta = (assign_cost[i, f2] + assign_cost[k, f1]) - (assign_cost[i, f1] + assign_cost[k, f2])
        # Open costs don't change unless a facility becomes empty or was empty.
        # f1 count: was counts[f1]. If counts[f1]==1, it becomes 0 (if i leaves and k doesn't come? No, k comes to f1).
        # If counts[f1] == 1, then only i was there. After swap, k is there. So f1 stays open (if k != i).
        # Wait, if counts[f1] == 1, it means only i is at f1. After swap, k is at f1. So f1 is still open.
        # If counts[f2] == 1, only k was there. After swap, i is at f2. So f2 is still open.
        # So open costs never change in a swap between two non-empty facilities?
        # What if f1 == f2? Not a swap.
        # What if one facility becomes empty?
        # If counts[f1] == 1, i leaves f1. k enters f1. So f1 still has 1 customer.
        # If counts[f2] == 1, k leaves f2. i enters f2. So f2 still has 1 customer.
        # So yes, open costs are invariant in swaps.
        return delta

    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Relocations
        # We iterate through customers. To avoid order bias, we can randomize order each pass.
        order = np.random.permutation(n_cust)
        for i in order:
            f1 = result[i]
            d_i = demands[i]
            best_f2 = -1
            best_delta = 0.0
            for f2 in range(n_fac):
                if f2 == f1:
                    continue
                if load[f2] + d_i <= capacities[f2]:
                    delta = get_cost_delta_relocate(i, f1, f2)
                    if delta < best_delta - 1e-9:
                        best_delta = delta
                        best_f2 = f2
            
            if best_f2 != -1:
                # Apply move
                load[f1] -= d_i
                counts[f1] -= 1
                load[best_f2] += d_i
                counts[best_f2] += 1
                result[i] = best_f2
                current_cost += best_delta
                improved = True
                # Continue scanning or break? First improvement strategy is usually faster per pass.
                # Let's try first improvement to allow more passes.
                # But checking all f2 for each i is O(M).
                # If we apply immediately, we change load/counts for subsequent i in the same pass.
                # This is fine.
                continue
        
        # Swaps
        # Only if time allows. Swaps are O(N^2).
        if time.time() < deadline - 10: # Reserve time
            # Randomize pairs? Or just iterate.
            # Iterate i, then k > i.
            for i in range(n_cust):
                f1 = result[i]
                d_i = demands[i]
                for k in range(i + 1, n_cust):
                    f2 = result[k]
                    if f1 == f2:
                        continue
                    d_k = demands[k]
                    # Check capacity for swap
                    # f1 gets d_k, loses d_i. New load: load[f1] - d_i + d_k
                    # f2 gets d_i, loses d_k. New load: load[f2] - d_k + d_i
                    if load[f1] - d_i + d_k <= capacities[f1] and load[f2] - d_k + d_i <= capacities[f2]:
                        delta = get_cost_delta_swap(i, k, f1, f2)
                        if delta < -1e-9:
                            # Apply swap
                            load[f1] -= d_i
                            load[f1] += d_k
                            load[f2] -= d_k
                            load[f2] += d_i
                            # Counts don't change
                            result[i] = f2
                            result[k] = f1
                            current_cost += delta
                            improved = True
                            # Break out of k loop? Or continue?
                            # Let's break to restart pass to re-evaluate other pairs with new state.
                            break
                if improved:
                    # Break out of i loop?
                    # If we break both, we restart the whole while loop (Relocations + Swaps).
                    # This is a "first improvement" strategy for the whole pass.
                    break
                if time.time() > deadline:
                    break

    return result

# The above code has some structural issues with the `improved` flag logic in nested loops.
# Let's clean it up.
