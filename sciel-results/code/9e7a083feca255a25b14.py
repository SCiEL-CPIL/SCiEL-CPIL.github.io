import time
import numpy as np
import random

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assigns each customer to a facility using Lagrangian Relaxation for initialization
    and Tabu Search with improved local search for refinement.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    
    demands = np.asarray(demands, dtype=np.int64)
    capacities = np.asarray(capacities, dtype=np.int64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)
    
    # Deadline with safety margin (150s limit, keep 15s buffer)
    deadline = time.time() + 135.0

    def calc_cost(res):
        used = np.unique(res)
        if len(used) == 0:
            return 0.0
        open_sum = float(np.sum(open_cost[used]))
        assign_sum = float(np.sum(assign_cost[np.arange(n_cust), res]))
        return open_sum + assign_sum

    def is_valid(res, load):
        if res is None or len(res) != n_cust:
            return False
        if np.any(res < 0) or np.any(res >= n_fac):
            return False
        if np.any(load > capacities) or np.any(load < 0):
            return False
        return True

    def loads_of(res):
        load = np.zeros(n_fac, dtype=np.int64)
        np.add.at(load, res, demands)
        return load

    def make_feasible(res, load):
        """
        If the assignment violates capacity, reassign excess customers to other facilities.
        Optimized repair heuristic using vectorized operations.
        """
        res = res.copy()
        load = load.copy()
        
        # Identify facilities with excess load
        excess_facs = np.where(load > capacities)[0]
        
        for f_excess in excess_facs:
            # Customers assigned to this facility, sorted by demand descending
            custs = np.where(res == f_excess)[0]
            if len(custs) == 0:
                continue
            # Sort by demand descending to move large customers first (fewer moves)
            custs = custs[np.argsort(-demands[custs])]
            
            while load[f_excess] > capacities[f_excess]:
                # Find customer to move with smallest cost increase
                min_delta = np.inf
                best_c = -1
                best_f_new = -1
                
                for c in custs:
                    if res[c] != f_excess:
                        continue
                    d_c = demands[c]
                    
                    # Find feasible targets (excluding current facility)
                    feasible = np.where((load + d_c <= capacities) & (np.arange(n_fac) != f_excess))[0]
                    if len(feasible) == 0:
                        break
                    
                    # Calculate deltas for all feasible targets at once
                    delta_assign = assign_cost[c, feasible] - assign_cost[c, f_excess]
                    
                    # Open cost adjustments
                    new_load_f = load[f_excess] - d_c
                    if new_load_f == 0:
                        delta_assign = delta_assign - open_cost[f_excess]
                    
                    for idx, f_new in enumerate(feasible):
                        if load[f_new] == 0:
                            delta_assign[idx] += open_cost[f_new]
                    
                    best_idx = int(np.argmin(delta_assign))
                    if delta_assign[best_idx] < min_delta:
                        min_delta = delta_assign[best_idx]
                        best_c = c
                        best_f_new = feasible[best_idx]
                
                if best_c == -1:
                    break
                
                # Perform move
                res[best_c] = best_f_new
                load[f_excess] -= demands[best_c]
                load[best_f_new] += demands[best_c]
                
        return res, load

    def lagrangian_init(max_time=5.0):
        """
        Use Lagrangian Relaxation to find a good initial solution.
        """
        start_time = time.time()
        best_res = None
        best_score = np.inf
        best_load = None
        
        # Initial multipliers
        lam = np.zeros(n_fac, dtype=np.float64)
        
        # Step size
        step_size = 1.0
        max_iter = 1000
        
        # Precompute costs
        # For each customer i, cost to facility f is assign_cost[i, f] + lam[f] * demands[i]
        
        for iter in range(max_iter):
            if time.time() > start_time + max_time:
                break
            
            # 1. Solve relaxed problem
            # For each customer i, choose f minimizing assign_cost[i, f] + lam[f] * demands[i]
            # This is a vectorized operation
            # costs_relaxed = assign_cost + np.outer(demands, lam)
            # res_relaxed = np.argmin(costs_relaxed, axis=1)
            
            # To save memory/time, we can do it in chunks or just use numpy broadcasting
            # 100x20 is small, so broadcasting is fine.
            costs_relaxed = assign_cost + np.outer(demands, lam)
            res_relaxed = np.argmin(costs_relaxed, axis=1)
            
            # 2. Calculate load
            load_relaxed = np.zeros(n_fac, dtype=np.int64)
            np.add.at(load_relaxed, res_relaxed, demands)
            
            # 3. Update multipliers (Subgradient)
            # Subgradient g_f = load_relaxed[f] - capacities[f]
            g = load_relaxed - capacities
            g = np.where(g > 0, g, 0) # Only positive violations contribute to subgradient for minimization?
            # Actually, the subgradient of the Lagrangian dual is (load - capacity).
            # If load > capacity, we need to increase lam to penalize this facility.
            # If load < capacity, we can decrease lam.
            # Standard subgradient: lam_new = max(0, lam + step * (load - capacity))
            
            # Normalize subgradient
            norm_g = np.linalg.norm(g)
            if norm_g < 1e-6:
                # Convergence
                break
                
            # Step size
            # Use a diminishing step size
            # step = alpha * (UB - LB) / ||g||^2
            # We don't have LB directly, but we can approximate.
            # For now, use a fixed step size that decays.
            step = 1.0 / (iter + 1)
            
            lam = np.maximum(0, lam + step * g)
            
            # 4. Check for feasible solution
            # The relaxed solution might be infeasible.
            # We can try to project it to a feasible solution.
            # If it's already feasible, it's a candidate.
            if np.all(load_relaxed <= capacities):
                score = calc_cost(res_relaxed)
                if score < best_score:
                    best_score = score
                    best_res = res_relaxed.copy()
                    best_load = load_relaxed.copy()
            else:
                # Try to repair
                res_feas, load_feas = make_feasible(res_relaxed, load_relaxed)
                if is_valid(res_feas, load_feas):
                    score = calc_cost(res_feas)
                    if score < best_score:
                        best_score = score
                        best_res = res_feas.copy()
                        best_load = load_feas.copy()
                        
        # If no feasible solution found, fall back to greedy
        if best_res is None:
            res = np.zeros(n_cust, dtype=int)
            load = np.zeros(n_fac, dtype=np.int64)
            order = np.argsort(-demands)
            for i in order:
                d_i = demands[i]
                feasible = np.where(load + d_i <= capacities)[0]
                if len(feasible) == 0:
                    f = int(np.argmax(capacities - load))
                else:
                    costs = assign_cost[i, feasible]
                    f = int(feasible[np.argmin(costs)])
                res[i] = f
                load[f] += d_i
            best_res = res
            best_load = load
            
        return best_res, best_load

    def tabu_search(res, load, max_time=10.0):
        """
        Tabu Search with proper tenure-based tabu management and 2-swap moves.
        """
        res = res.copy()
        load = load.copy()
        best_res = res.copy()
        best_load = load.copy()
        best_score = calc_cost(res)
        current_score = best_score
        
        start_time = time.time()
        tabu_tenure = 15
        # tabu[i][f] = iteration when it expires
        tabu = [np.full(n_fac, -1, dtype=np.int32) for _ in range(n_cust)]
        
        iteration = 0
        no_improve_count = 0
        max_no_improve = 50
        
        while time.time() < start_time + max_time and time.time() < deadline - 5.0:
            iteration += 1
            
            # Find best move (single move or 2-swap)
            best_move_score = current_score
            best_move_type = 0  # 0: single, 1: swap
            best_i = -1
            best_j = -1
            best_f_old = -1
            best_f_new = -1
            
            # 1. Single moves
            for i in range(n_cust):
                f_old = res[i]
                d_i = demands[i]
                
                feasible = np.where((load + d_i <= capacities) & (np.arange(n_fac) != f_old))[0]
                if len(feasible) == 0:
                    continue
                
                # Calculate deltas
                base_assign = assign_cost[i, f_old]
                deltas = assign_cost[i, feasible] - base_assign
                
                if load[f_old] == d_i:
                    deltas = deltas - open_cost[f_old]
                    
                for idx in range(len(feasible)):
                    if load[feasible[idx]] == 0:
                        deltas[idx] += open_cost[feasible[idx]]
                    
                    # Check tabu with aspiration
                    if tabu[i][feasible[idx]] >= iteration - tabu_tenure:
                        # Tabu move: only accept if it improves best_score
                        if current_score + deltas[idx] >= best_score:
                            deltas[idx] = np.inf
                        # else: aspiration, allow it
                
                best_idx = int(np.argmin(deltas))
                if deltas[best_idx] < np.inf:
                    move_score = current_score + deltas[best_idx]
                    if move_score < best_move_score:
                        best_move_score = move_score
                        best_move_type = 0
                        best_i = i
                        best_f_old = f_old
                        best_f_new = feasible[best_idx]
            
            # 2. 2-swap moves (sampled to control cost)
            n_swap_samples = 200
            for _ in range(n_swap_samples):
                if time.time() > start_time + max_time or time.time() > deadline - 5.0:
                    break
                i, j = random.sample(range(n_cust), 2)
                f_i = res[i]
                f_j = res[j]
                if f_i == f_j:
                    continue
                d_i = demands[i]
                d_j = demands[j]
                
                # Check capacity constraints for swap
                if load[f_j] - d_j + d_i > capacities[f_j]:
                    continue
                if load[f_i] - d_i + d_j > capacities[f_i]:
                    continue
                
                # Check tabu for both moves
                if tabu[i][f_j] >= iteration - tabu_tenure and current_score + (assign_cost[i, f_j] - assign_cost[i, f_i] + assign_cost[j, f_i] - assign_cost[j, f_j]) >= best_score:
                    continue
                if tabu[j][f_i] >= iteration - tabu_tenure and current_score + (assign_cost[i, f_j] - assign_cost[i, f_i] + assign_cost[j, f_i] - assign_cost[j, f_j]) >= best_score:
                    continue
                
                delta_assign = (assign_cost[i, f_j] - assign_cost[i, f_i]) + \
                               (assign_cost[j, f_i] - assign_cost[j, f_j])
                
                new_load_i = load[f_i] - d_i + d_j
                new_load_j = load[f_j] - d_j + d_i
                
                old_open_i = open_cost[f_i] if load[f_i] > 0 else 0.0
                old_open_j = open_cost[f_j] if load[f_j] > 0 else 0.0
                new_open_i = open_cost[f_i] if new_load_i > 0 else 0.0
                new_open_j = open_cost[f_j] if new_load_j > 0 else 0.0
                delta_open = (new_open_i + new_open_j) - (old_open_i + old_open_j)
                
                total_delta = delta_assign + delta_open
                move_score = current_score + total_delta
                
                if move_score < best_move_score:
                    best_move_score = move_score
                    best_move_type = 1
                    best_i = i
                    best_j = j
                    best_f_old = f_i
                    best_f_new = f_j
            
            if best_move_type == 0 and best_i != -1:
                # Perform single move
                i = best_i
                f_old = best_f_old
                f_new = best_f_new
                d_i = demands[i]
                
                res[i] = f_new
                load[f_old] -= d_i
                load[f_new] += d_i
                current_score = best_move_score
                
                # Update tabu
                tabu[i][f_new] = iteration + tabu_tenure
                
                if current_score < best_score:
                    best_score = current_score
                    best_res = res.copy()
                    best_load = load.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    
            elif best_move_type == 1 and best_i != -1:
                # Perform swap
                i, j = best_i, best_j
                f_i, f_j = res[i], res[j]
                d_i, d_j = demands[i], demands[j]
                
                res[i], res[j] = res[j], res[i]
                load[f_i] -= d_i
                load[f_i] += d_j
                load[f_j] -= d_j
                load[f_j] += d_i
                current_score = best_move_score
                
                # Update tabu
                tabu[i][f_j] = iteration + tabu_tenure
                tabu[j][f_i] = iteration + tabu_tenure
                
                if current_score < best_score:
                    best_score = current_score
                    best_res = res.copy()
                    best_load = load.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                # No improving move found
                if no_improve_count > max_no_improve:
                    break
                # Try a random perturbation
                i = random.randint(0, n_cust - 1)
                f_old = res[i]
                d_i = demands[i]
                feasible = np.where((load + d_i <= capacities) & (np.arange(n_fac) != f_old))[0]
                if len(feasible) > 0:
                    f_new = feasible[random.randint(0, len(feasible) - 1)]
                    res[i] = f_new
                    load[f_old] -= d_i
                    load[f_new] += d_i
                    current_score = calc_cost(res)
                    tabu[i][f_new] = iteration + tabu_tenure
                else:
                    break
            
            if no_improve_count > max_no_improve * 2:
                break
        
        # Final polish with greedy local search
        remaining_time = min(3.0, deadline - time.time() - 3.0)
        if remaining_time > 1.0:
            res, load = greedy_local_search(res, load, max_time=remaining_time)
            score = calc_cost(res)
            if score < best_score:
                best_score = score
                best_res = res.copy()
                best_load = load.copy()
            
        return best_res, best_load

    def greedy_local_search(res, load, max_time=5.0):
        """
        Greedy local search with single moves and systematic 2-swap exploration.
        """
        res = res.copy()
        load = load.copy()
        start_time = time.time()
        
        improved = True
        while improved and (time.time() - start_time) < max_time and time.time() < deadline - 3.0:
            improved = False
            
            # Single Move Pass
            any_improve = False
            for i in range(n_cust):
                if time.time() > deadline - 3.0:
                    break
                f_old = res[i]
                d_i = demands[i]
                
                feasible_mask = (load + d_i <= capacities)
                feasible_mask[f_old] = False
                feasible = np.where(feasible_mask)[0]
                
                if len(feasible) == 0:
                    continue
                
                base_assign = assign_cost[i, f_old]
                deltas = assign_cost[i, feasible] - base_assign
                
                if load[f_old] == d_i:
                    deltas = deltas - open_cost[f_old]
                
                for idx in range(len(feasible)):
                    if load[feasible[idx]] == 0:
                        deltas[idx] += open_cost[feasible[idx]]
                
                best_idx = int(np.argmin(deltas))
                best_delta = deltas[best_idx]
                
                if best_delta < -1e-9:
                    f_new = feasible[best_idx]
                    res[i] = f_new
                    load[f_old] -= d_i
                    load[f_new] += d_i
                    any_improve = True
                    improved = True
            
            if not any_improve:
                # Systematic 2-swap exploration
                # Try all pairs of customers assigned to different facilities
                # Use a more efficient approach: group by facility
                fac_to_custs = [[] for _ in range(n_fac)]
                for i in range(n_cust):
                    fac_to_custs[res[i]].append(i)
                
                swap_improved = False
                # Sample swaps to control time
                max_swap_checks = 1000
                checks = 0
                for f1 in range(n_fac):
                    if len(fac_to_custs[f1]) == 0:
                        continue
                    for f2 in range(f1 + 1, n_fac):
                        if len(fac_to_custs[f2]) == 0:
                            continue
                        for i in fac_to_custs[f1]:
                            if checks >= max_swap_checks:
                                break
                            for j in fac_to_custs[f2]:
                                if checks >= max_swap_checks:
                                    break
                                checks += 1
                                d_i = demands[i]
                                d_j = demands[j]
                                
                                if load[f2] - d_j + d_i > capacities[f2]:
                                    continue
                                if load[f1] - d_i + d_j > capacities[f1]:
                                    continue
                                
                                delta_assign = (assign_cost[i, f2] - assign_cost[i, f1]) + \
                                               (assign_cost[j, f1] - assign_cost[j, f2])
                                
                                new_load_f1 = load[f1] - d_i + d_j
                                new_load_f2 = load[f2] - d_j + d_i
                                
                                old_open_f1 = open_cost[f1] if load[f1] > 0 else 0.0
                                old_open_f2 = open_cost[f2] if load[f2] > 0 else 0.0
                                new_open_f1 = open_cost[f1] if new_load_f1 > 0 else 0.0
                                new_open_f2 = open_cost[f2] if new_load_f2 > 0 else 0.0
                                delta_open = (new_open_f1 + new_open_f2) - (old_open_f1 + old_open_f2)
                                
                                total_delta = delta_assign + delta_open
                                if total_delta < -1e-9:
                                    res[i], res[j] = res[j], res[i]
                                    load[f1] = new_load_f1
                                    load[f2] = new_load_f2
                                    swap_improved = True
                                    improved = True
                                    break
                            if swap_improved:
                                break
                        if swap_improved:
                            break
                    if swap_improved:
                        break
                
                if not swap_improved and checks < max_swap_checks:
                    # Try random swaps if systematic didn't find improvement
                    rng = np.random.default_rng(int(time.time() * 10000) % (2**32))
                    for _ in range(200):
                        if time.time() > deadline - 3.0:
                            break
                        i, j = rng.integers(0, n_cust, size=2)
                        if i == j:
                            continue
                        f_i = res[i]
                        f_j = res[j]
                        if f_i == f_j:
                            continue
                        d_i = demands[i]
                        d_j = demands[j]
                        
                        if load[f_j] - d_j + d_i > capacities[f_j]:
                            continue
                        if load[f_i] - d_i + d_j > capacities[f_i]:
                            continue
                        
                        delta_assign = (assign_cost[i, f_j] - assign_cost[i, f_i]) + \
                                       (assign_cost[j, f_i] - assign_cost[j, f_j])
                        
                        new_load_i = load[f_i] - d_i + d_j
                        new_load_j = load[f_j] - d_j + d_i
                        
                        old_open_i = open_cost[f_i] if load[f_i] > 0 else 0.0
                        old_open_j = open_cost[f_j] if load[f_j] > 0 else 0.0
                        new_open_i = open_cost[f_i] if new_load_i > 0 else 0.0
                        new_open_j = open_cost[f_j] if new_load_j > 0 else 0.0
                        delta_open = (new_open_i + new_open_j) - (old_open_i + old_open_j)
                        
                        total_delta = delta_assign + delta_open
                        if total_delta < -1e-9:
                            res[i], res[j] = res[j], res[i]
                            load[f_i] = new_load_i
                            load[f_j] = new_load_j
                            improved = True
                            break
                        
        return res, load

    # 1. Initialization
    # Use Lagrangian Relaxation for a strong start
    res_init, load_init = lagrangian_init(max_time=5.0)
    
    # Also generate a few greedy starts for diversity
    rng = np.random.default_rng(123)
    
    def greedy_construct(order, open_penalty_scale=0.0):
        res = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac, dtype=np.int64)
        for i in order:
            d_i = demands[i]
            feasible = np.where(load + d_i <= capacities)[0]
            if len(feasible) == 0:
                return None
            scores = assign_cost[i, feasible].copy()
            for idx in range(len(feasible)):
                if load[feasible[idx]] == 0:
                    scores[idx] += open_cost[feasible[idx]] * open_penalty_scale
            f = int(feasible[np.argmin(scores)])
            res[i] = f
            load[f] += d_i
        if not is_valid(res, load):
            return None
        return res, load

    candidates = []
    candidates.append((res_init, load_init))
    
    # Greedy 1: largest demand first, full open cost penalty
    order1 = np.argsort(-demands)
    cand = greedy_construct(order1, 1.0)
    if cand: candidates.append(cand)
    
    # Greedy 2: largest demand first, moderate open cost penalty
    cand = greedy_construct(order1, 0.5)
    if cand: candidates.append(cand)
    
    # Greedy 3: random order, moderate open cost penalty
    order3 = rng.permutation(n_cust)
    cand = greedy_construct(order3, 0.8)
    if cand: candidates.append(cand)
    
    # Greedy 4: second-largest demand first
    order4 = np.argsort(-demands)
    # Rotate to start with second largest
    order4_rotated = np.concatenate([order4[1:], order4[:1]])
    cand = greedy_construct(order4_rotated, 0.7)
    if cand: candidates.append(cand)
    
    # 2. Refine with Tabu Search
    best_result = None
    best_score = np.inf
    
    n_candidates = len(candidates)
    for idx, (res, load) in enumerate(candidates):
        if time.time() > deadline - 15.0:
            break
            
        # Allocate time for Tabu Search proportionally
        remaining_candidates = n_candidates - idx
        if idx == 0:
            ts_time = min(25.0, (deadline - time.time() - 10.0) * 0.4)
        elif idx == 1:
            ts_time = min(15.0, (deadline - time.time() - 10.0) * 0.3)
        else:
            ts_time = min(8.0, (deadline - time.time() - 10.0) / max(1, remaining_candidates))
            
        if ts_time < 1.0:
            ts_time = 1.0
            
        res_ts, load_ts = tabu_search(res, load, max_time=ts_time)
        score = calc_cost(res_ts)
        
        if is_valid(res_ts, load_ts) and score < best_score:
            best_score = score
            best_result = res_ts.copy()
            
    # Fallback
    if best_result is None:
        res = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac, dtype=np.int64)
        order = np.argsort(-demands)
        for i in order:
            d_i = demands[i]
            feasible = np.where(load + d_i <= capacities)[0]
            if len(feasible) == 0:
                f = int(np.argmax(capacities - load))
            else:
                costs = assign_cost[i, feasible]
                f = int(feasible[np.argmin(costs)])
            res[i] = f
            load[f] += d_i
        best_result = res

    # Final polish with extended local search
    if time.time() < deadline - 5.0:
        final_load = loads_of(best_result)
        polish_time = min(15.0, deadline - time.time() - 3.0)
        if polish_time > 1.0:
            best_result, final_load = greedy_local_search(best_result, final_load, max_time=polish_time)
            
    # Final validation
    final_load = loads_of(best_result)
    if not is_valid(best_result, final_load):
        # Emergency repair
        best_result, final_load = make_feasible(best_result, final_load)
        
    return best_result.astype(int)
