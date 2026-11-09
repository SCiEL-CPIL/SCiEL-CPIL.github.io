import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # 1. Initialization: Greedy Largest Demand First
    # This is a strong baseline.
    result = np.zeros(n_customers, dtype=int)
    load = np.zeros(n_facilities, dtype=int)
    
    # Sort customers by demand descending
    order = np.argsort(-demands)
    
    for i in order:
        # Find feasible facilities
        # We want the one with min assign_cost[i, f] among those with load[f] + demands[i] <= capacities[f]
        feasible_mask = (load + demands[i]) <= capacities
        if not np.any(feasible_mask):
            # This shouldn't happen if total capacity >= total demand, 
            # but if it does, we have a problem. 
            # For robustness, pick any facility with most space? Or just fail?
            # The problem statement implies a feasible solution exists.
            # If not, we might need to relax constraints, but let's assume feasibility.
            # To be safe, pick the facility with max remaining capacity.
            f = np.argmax(capacities - load)
        else:
            # Mask out infeasible costs to infinity
            costs = np.where(feasible_mask, assign_cost[i], np.inf)
            f = np.argmin(costs)
            
        result[i] = f
        load[f] += demands[i]
        
    # 2. Local Search Improvement
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    def get_cost():
        # Sum of assignment costs
        total_assign = np.sum(assign_cost[np.arange(n_customers), result])
        # Sum of opening costs for facilities with load > 0
        open_facilities = load > 0
        total_open = np.sum(open_cost[open_facilities])
        return total_assign + total_open
    
    current_cost = get_cost()
    
    # Pre-compute indices for faster access
    indices = np.arange(n_customers)
    
    while time.time() < deadline:
        improved = False
        
        # Single Move Local Search
        # We iterate through customers. To avoid bias, we can randomize order or just go 0..N.
        # Randomizing order helps escape local minima caused by specific ordering.
        # But for deterministic reproducibility and speed, let's try a few passes.
        
        for i in indices:
            if time.time() > deadline:
                break
                
            f_old = result[i]
            d_i = demands[i]
            
            # Potential new facilities
            # We only need to consider facilities where capacity allows
            # load_new = load - d_i (if we remove i from f_old)
            # But we are checking f_new != f_old.
            
            # Calculate delta for moving i from f_old to f_new
            # delta_assign = assign_cost[i, f_new] - assign_cost[i, f_old]
            # delta_open:
            #   - If f_old becomes empty: -open_cost[f_old]
            #   - If f_new was empty: +open_cost[f_new]
            
            # Vectorized check for all f_new != f_old
            # Let's just loop f_new for clarity, N=100, M=20 is small.
            
            for f_new in range(n_facilities):
                if f_new == f_old:
                    continue
                
                # Check capacity for f_new
                # If f_new == f_old, load is same. 
                # Here f_new != f_old.
                # Current load at f_new is load[f_new].
                # We add d_i to f_new.
                if load[f_new] + d_i > capacities[f_new]:
                    continue
                
                # Calculate Delta Cost
                delta_assign = assign_cost[i, f_new] - assign_cost[i, f_old]
                
                delta_open = 0.0
                # Check if f_old becomes empty
                if load[f_old] == d_i:
                    delta_open -= open_cost[f_old]
                
                # Check if f_new was empty
                if load[f_new] == 0:
                    delta_open += open_cost[f_new]
                
                delta_total = delta_assign + delta_open
                
                if delta_total < -1e-9:
                    # Perform Move
                    load[f_old] -= d_i
                    load[f_new] += d_i
                    result[i] = f_new
                    current_cost += delta_total
                    improved = True
                    # Do not break inner loop immediately? 
                    # If we move i, the state changes. 
                    # We can break inner loop to restart scan for i? 
                    # Or just continue checking other f_new for this i?
                    # Since we moved to a better facility, other moves for this i 
                    # are less likely to be good, but possible if opening costs interact.
                    # However, re-evaluating f_new against new state is tricky in vectorized way.
                    # Let's just break inner loop and let the outer loop re-evaluate i in next pass 
                    # if needed, or just continue. 
                    # Actually, if we break, we might miss a compound improvement? 
                    # No, single move is atomic. 
                    # Let's break to ensure we don't use stale `load` info for other f_new checks 
                    # for the same i (though load[f_old] changed, so subsequent checks for i 
                    # would be invalid if we didn't update). 
                    # Since we update load immediately, we *could* continue, but 
                    # the delta calculation for the *next* f_new would need to account 
                    # for the fact that i is now at f_new (which is the current f_new).
                    # It's safer to break and let the next iteration of the outer loop 
                    # (or a new pass) handle it.
                    break
        
        if not improved:
            # Try Pairwise Swaps?
            # This is O(N^2 * M) which is 100^2 * 20 = 200,000 operations. 
            # Might be worth it if single moves stall.
            
            # Simple Pairwise Swap Logic
            # For i in range(N):
            #   For j in range(i+1, N):
            #     if result[i] == result[j]: continue
            #     f_i = result[i], f_j = result[j]
            #     Check if i can go to f_j and j can go to f_i
            #     Calculate delta
            #     If delta < 0, swap
            
            # Given the time budget, let's implement a limited version of swaps 
            # only if single moves didn't improve in the last pass.
            
            swap_improved = False
            for i in indices:
                if time.time() > deadline:
                    break
                f_i = result[i]
                d_i = demands[i]
                for j in range(i+1, n_customers):
                    f_j = result[j]
                    if f_i == f_j:
                        continue
                    d_j = demands[j]
                    
                    # Check capacities
                    # If i moves to f_j: load[f_j] + d_i <= cap[f_j]
                    # If j moves to f_i: load[f_i] + d_j <= cap[f_i]
                    # Note: i leaves f_i, j leaves f_j.
                    # So effective load at f_j before check is load[f_j] - d_j + d_i
                    # Effective load at f_i before check is load[f_i] - d_i + d_j
                    
                    if load[f_j] - d_j + d_i > capacities[f_j]:
                        continue
                    if load[f_i] - d_i + d_j > capacities[f_i]:
                        continue
                    
                    # Calculate Delta
                    delta_assign = (assign_cost[i, f_j] + assign_cost[j, f_i]) - (assign_cost[i, f_i] + assign_cost[j, f_j])
                    
                    delta_open = 0.0
                    # f_i might become empty
                    if load[f_i] == d_i:
                        delta_open -= open_cost[f_i]
                    # f_j might become empty
                    if load[f_j] == d_j:
                        delta_open -= open_cost[f_j]
                    
                    # f_i might become non-empty (if it was empty? No, i was there)
                    # f_j might become non-empty (if it was empty? No, j was there)
                    
                    # Wait, if f_i had only i, and i moves, f_i becomes empty.
                    # If f_j had only j, and j moves, f_j becomes empty.
                    # Are there cases where a facility becomes *newly* open?
                    # No, because we are swapping between existing facilities 
                    # that already have customers (i at f_i, j at f_j).
                    # So no *new* openings. Only potential closures.
                    
                    delta_total = delta_assign + delta_open
                    
                    if delta_total < -1e-9:
                        # Perform Swap
                        load[f_i] -= d_i
                        load[f_j] -= d_j
                        load[f_j] += d_i
                        load[f_i] += d_j
                        result[i] = f_j
                        result[j] = f_i
                        current_cost += delta_total
                        swap_improved = True
                        improved = True
                        # Break inner j loop? 
                        # State changed. 
                        break
                
                if swap_improved:
                    break # Restart outer i loop? Or just break to end pass?
                    # Let's break out of both to restart the single move pass 
                    # with the new state.
            
            if not improved:
                # No single moves or swaps improved the solution.
                # We can stop.
                break
        # If improved, loop again to check for further improvements

    return result
