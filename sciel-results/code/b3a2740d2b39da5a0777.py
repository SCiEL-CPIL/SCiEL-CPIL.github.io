# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_cost(result, demands, open_cost, assign_cost):
    total_assign = np.sum(assign_cost[np.arange(len(demands)), result])
    open_mask = np.zeros(len(open_cost), dtype=bool)
    for f in result:
        open_mask[f] = True
    total_open = np.sum(open_cost[open_mask])
    return total_assign + total_open

def _repair(result, demands, capacities, assign_cost):
    """Fix capacity violations by moving customers from overloaded facilities to cheapest feasible ones."""
    n = len(demands)
    m = len(capacities)
    load = np.zeros(m)
    for i in range(n):
        load[result[i]] += demands[i]
    
    # Iteratively fix overloaded facilities
    for _ in range(200):
        overloaded = np.where(load > capacities + 1e-6)[0]
        if len(overloaded) == 0:
            break
        for f_over in overloaded:
            excess = load[f_over] - capacities[f_over]
            # Find customers at f_over, move those with most savings to other facilities
            custs_at_f = [i for i in range(n) if result[i] == f_over]
            # Sort by demand descending
            custs_at_f.sort(key=lambda i: -demands[i])
            for i in custs_at_f:
                if excess <= 1e-6:
                    break
                d = demands[i]
                # Find cheapest feasible facility (not f_over)
                best_f = -1
                best_cost = np.inf
                for f in range(m):
                    if f == f_over:
                        continue
                    if load[f] + d <= capacities[f] + 1e-6:
                        c = assign_cost[i, f]
                        if c < best_cost:
                            best_cost = c
                            best_f = f
                if best_f >= 0:
                    load[f_over] -= d
                    load[best_f] += d
                    result[i] = best_f
                    excess -= d
                else:
                    # Fallback
                    rem = capacities - load
                    best_f = int(np.argmax(rem))
                    load[f_over] -= d
                    load[best_f] += d
                    result[i] = best_f
                    excess -= d
    return result

def _local_search(result, demands, capacities, open_cost, assign_cost, max_iters=500, deadline=None, rng=None):
    """Local search with single moves and 2-opt swaps."""
    if rng is None:
        rng = np.random.default_rng()
    
    n = len(demands)
    m = len(capacities)
    
    load = np.zeros(m)
    for i in range(n):
        load[result[i]] += demands[i]
    
    current_cost = _compute_cost(result, demands, open_cost, assign_cost)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        order = rng.permutation(n)
        
        # Single moves
        for i in order:
            current_f = result[i]
            d = demands[i]
            
            best_f = -1
            best_delta = 0.0
            
            for f in range(m):
                if f == current_f:
                    continue
                if load[f] + d > capacities[f] + 1e-6:
                    continue
                
                delta = assign_cost[i, f] - assign_cost[i, current_f]
                
                if load[current_f] == d:
                    delta -= open_cost[current_f]
                if load[f] == 0:
                    delta += open_cost[f]
                
                if delta < best_delta - 1e-9:
                    best_delta = delta
                    best_f = f
            
            if best_f >= 0:
                load[current_f] -= d
                load[best_f] += d
                result[i] = best_f
                current_cost += best_delta
                improved = True
        
        # 2-opt swaps
        if deadline is not None and time.time() > deadline:
            break
        
        swap_order = rng.permutation(n)
        for i in swap_order:
            if deadline is not None and time.time() > deadline:
                break
            fi = result[i]
            di = demands[i]
            
            for j in range(i+1, n):
                fj = result[j]
                if fj == fi:
                    continue
                dj = demands[j]
                
                # Check capacity
                if load[fi] - di + dj > capacities[fi] + 1e-6:
                    continue
                if load[fj] - dj + di > capacities[fj] + 1e-6:
                    continue
                
                delta = (assign_cost[i, fj] + assign_cost[j, fi]) - (assign_cost[i, fi] + assign_cost[j, fj])
                
                # Open cost changes
                new_load_fi = load[fi] - di + dj
                new_load_fj = load[fj] - dj + di
                
                if load[fi] > 0 and new_load_fi == 0:
                    delta -= open_cost[fi]
                elif load[fi] == 0 and new_load_fi > 0:
                    delta += open_cost[fi]
                
                if load[fj] > 0 and new_load_fj == 0:
                    delta -= open_cost[fj]
                elif load[fj] == 0 and new_load_fj > 0:
                    delta += open_cost[fj]
                
                if delta < -1e-9:
                    load[fi] = new_load_fi
                    load[fj] = new_load_fj
                    result[i] = fj
                    result[j] = fi
                    current_cost += delta
                    improved = True
                    break
        
        if not improved:
            break
    
    return result, current_cost

def _lagrangian_heuristic(demands, capacities, open_cost, assign_cost, max_iters=1000, deadline=None):
    """
    Lagrangian relaxation heuristic.
    Relaxes capacity constraints with multipliers mu.
    """
    n = len(demands)
    m = len(capacities)
    
    # Initialize multipliers
    mu = np.zeros(m)
    
    best_result = None
    best_cost = np.inf
    
    step_size = 1.0
    best_step_size = 1.0
    
    # Upper bound for step size (based on duality gap)
    # We need an initial feasible solution for the upper bound
    # Use a simple greedy
    temp_result = np.argmin(assign_cost, axis=1).astype(int)
    temp_result = _repair(temp_result, demands, capacities, assign_cost)
    ub = _compute_cost(temp_result, demands, open_cost, assign_cost)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
            
        # Solve relaxed problem
        # For each customer i, pick j that minimizes assign_cost[i, j] + mu[j]
        # But we also need to decide y_j.
        # The relaxed objective is sum_j open_cost[j] y_j + sum_i,j (assign_cost[i,j] + mu[j]) x_ij + sum_j mu[j] capacities[j]
        # If we assign i to j, we pay assign_cost[i,j] + mu[j].
        # If any i is assigned to j, we must pay open_cost[j].
        
        # This is still a facility location problem, but without capacity constraints.
        # We can solve it by:
        # 1. For each customer, find best j based on reduced cost.
        # 2. But opening cost makes it coupled.
        
        # Actually, the Lagrangian relaxation of capacity constraints in CFLP
        # still leaves the coupling via opening costs.
        # A simpler Lagrangian is to relax the "x_ij <= y_j" or just use subgradient on capacities
        # and solve the resulting problem heuristically.
        
        # Let's use a simpler approach:
        # Adjusted cost c'_ij = assign_cost[i, j] + mu[j]
        # Assign each customer to argmin_j c'_ij.
        # Check capacities.
        # If feasible, update best.
        # Update mu using subgradient.
        
        adjusted_cost = assign_cost + mu[np.newaxis, :]
        result = np.argmin(adjusted_cost, axis=1).astype(int)
        
        # Check feasibility
        load = np.zeros(m)
        for i in range(n):
            load[result[i]] += demands[i]
        
        if np.all(load <= capacities + 1e-6):
            cost = _compute_cost(result, demands, open_cost, assign_cost)
            if cost < best_cost:
                best_cost = cost
                best_result = result.copy()
        
        # Update multipliers
        # Subgradient g_j = load[j] - capacities[j]
        g = load - capacities
        
        # Step size
        # alpha = gamma * (ub - lb) / (sum(g^2))
        # lb is the value of the Lagrangian function
        lagrangian_value = np.sum(open_cost) * 0 # This is not quite right, y_j is binary in original, but relaxed?
        # In the relaxation, y_j is not directly in the objective if we just add mu to assignment.
        # The standard Lagrangian for CFLP capacity constraints:
        # L(mu) = min { sum f_j y_j + sum c_ij x_ij + sum mu_j (sum d_i x_ij - U_j) : sum x_ij=1, x_ij<=y_j, x,y>=0 }
        # This is hard to solve directly.
        
        # Let's stick to the simpler heuristic:
        # If load[j] > cap[j], increase mu[j]. If load[j] < cap[j], decrease mu[j].
        
        norm_g = np.sum(g**2)
        if norm_g < 1e-6:
            break
            
        # Simple update
        alpha = 0.5
        mu = np.maximum(0, mu + alpha * g)
        
        # Decay step size if we are not improving?
        # Keep it simple for now.
        
    if best_result is None:
        # Fallback to greedy
        best_result = np.argmin(assign_cost, axis=1).astype(int)
        best_result = _repair(best_result, demands, capacities, assign_cost)
        
    return best_result

def assign(demands, capacities, open_cost, assign_cost):
    demands = np.asarray(demands, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.float64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)
    
    n = len(demands)
    m = len(capacities)
    
    start_time = time.time()
    deadline = start_time + 140.0
    
    rng = np.random.default_rng(42)
    
    best_result = None
    best_cost = np.inf
    
    # 1. Lagrangian Heuristic
    result = _lagrangian_heuristic(demands, capacities, open_cost, assign_cost, max_iters=2000, deadline=deadline)
    result, cost = _local_search(result, demands, capacities, open_cost, assign_cost, max_iters=300, deadline=deadline, rng=rng)
    if cost < best_cost:
        best_cost = cost
        best_result = result.copy()
        
    if time.time() > deadline - 20:
        return best_result
        
    # 2. Greedy Init + Local Search
    # Greedy: largest demand first
    order = np.argsort(-demands)
    result = np.full(n, -1, dtype=int)
    load = np.zeros(m)
    for i in order:
        d = demands[i]
        best_f = -1
        best_c = np.inf
        for f in range(m):
            if load[f] + d <= capacities[f] + 1e-6:
                c = assign_cost[i, f]
                if c < best_c:
                    best_c = c
                    best_f = f
        if best_f >= 0:
            result[i] = best_f
            load[best_f] += d
        else:
            rem = capacities - load
            best_f = int(np.argmax(rem))
            result[i] = best_f
            load[best_f] += d
            
    result = _repair(result, demands, capacities, assign_cost)
    result, cost = _local_search(result, demands, capacities, open_cost, assign_cost, max_iters=300, deadline=deadline, rng=rng)
    if cost < best_cost:
        best_cost = cost
        best_result = result.copy()
        
    if time.time() > deadline - 10:
        return best_result
        
    # 3. Cheapest Init + Local Search
    result = np.argmin(assign_cost, axis=1).astype(int)
    result = _repair(result, demands, capacities, assign_cost)
    result, cost = _local_search(result, demands, capacities, open_cost, assign_cost, max_iters=300, deadline=deadline, rng=rng)
    if cost < best_cost:
        best_cost = cost
        best_result = result.copy()
        
    # Final validity check
    load = np.zeros(m)
    valid = True
    for i in range(n):
        f = best_result[i]
        if f < 0 or f >= m:
            valid = False
            break
        load[f] += demands[i]
        
    if not valid or np.any(load > capacities + 1e-6):
        # Fallback
        best_result = _repair(best_result, demands, capacities, assign_cost)
        
    return best_result
# EVOLVE-BLOCK-END
