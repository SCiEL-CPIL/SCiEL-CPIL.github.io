# EVOLVE-BLOCK-START
import time
import numpy as np


def _feasible_objective(res, demands, capacities, open_cost, assign_cost):
    """Return objective if feasible, else inf."""
    load = np.bincount(res, minlength=len(capacities)).astype(float)
    if np.any(load > capacities):
        return float("inf")
    open_sum = float(open_cost[load > 0].sum())
    assign_sum = float(assign_cost[np.arange(len(res)), res].sum())
    return open_sum + assign_sum


def _greedy_assign(demands, capacities, assign_cost):
    """Largest-demand-first greedy assignment."""
    n_cust = len(demands)
    res = np.zeros(n_cust, dtype=int)
    load = np.zeros(len(capacities), dtype=float)
    order = np.argsort(-demands)
    for i in order:
        d = demands[i]
        best_f = -1
        # try cheapest facility first
        for f in np.argsort(assign_cost[i]):
            if load[f] + d <= capacities[f]:
                best_f = f
                break
        if best_f < 0:
            for f in range(len(capacities)):
                if load[f] + d <= capacities[f]:
                    best_f = f
                    break
        res[i] = best_f
        load[best_f] += d
    return res, load


def _vectorized_local_search(res, demands, capacities, open_cost, assign_cost, deadline, time_budget=0.85):
    """Vectorized 1-move local search with close-facility moves."""
    n_cust = len(demands)
    n_fac = len(capacities)
    load = np.bincount(res, minlength=n_fac).astype(float)
    obj = _feasible_objective(res, demands, capacities, open_cost, assign_cost)
    if obj == float("inf"):
        return res, obj
    
    sorted_facs = np.argsort(assign_cost, axis=1)
    
    improved = True
    while improved and time.time() < deadline * time_budget:
        improved = False
        
        # --- Single customer moves (vectorized inner) ---
        for c in range(n_cust):
            if time.time() > deadline * time_budget:
                break
            d = demands[c]
            f_old = res[c]
            
            # Find best feasible new facility for customer c
            best_delta = -1e-9
            best_f_new = -1
            for f_new in sorted_facs[c]:
                if f_new == f_old:
                    continue
                if load[f_new] + d > capacities[f_new]:
                    continue
                delta = assign_cost[c, f_new] - assign_cost[c, f_old]
                if load[f_old] == d:
                    delta -= open_cost[f_old]
                if load[f_new] == 0:
                    delta += open_cost[f_new]
                if delta < best_delta:
                    best_delta = delta
                    best_f_new = f_new
                if best_delta > 0:
                    break  # sorted by cost, can't get better (open costs are positive)
            
            if best_f_new >= 0:
                res[c] = best_f_new
                load[f_old] -= d
                load[best_f_new] += d
                obj += best_delta
                improved = True
        
        # --- Close-facility move: try closing each non-empty facility ---
        for f_close in range(n_fac):
            if time.time() > deadline * time_budget:
                break
            if load[f_close] <= 0:
                continue
            
            # Get customers assigned to f_close
            custs = np.where(res == f_close)[0]
            if len(custs) == 0:
                continue
            
            # Try to reassign all these customers to other facilities
            # Check if feasible
            delta = 0.0
            feasible = True
            temp_res = res.copy()
            temp_load = load.copy()
            temp_load[f_close] = 0.0
            
            for c in custs:
                d = demands[c]
                best_f = -1
                best_c = None
                for f_new in sorted_facs[c]:
                    if f_new == f_close:
                        continue
                    if temp_load[f_new] + d <= capacities[f_new]:
                        if best_c is None or assign_cost[c, f_new] < best_c:
                            best_f = f_new
                            best_c = assign_cost[c, f_new]
                        break
                if best_f < 0:
                    feasible = False
                    break
                delta += assign_cost[c, best_f] - assign_cost[c, f_close]
                temp_res[c] = best_f
                temp_load[best_f] += d
            
            if not feasible:
                continue
            
            # Add open cost savings
            delta -= open_cost[f_close]
            
            # Check if any new facility was opened
            for f_new in np.unique(temp_res[custs]):
                if f_new != f_close and load[f_new] == 0:
                    delta += open_cost[f_new]
            
            if delta < -1e-9:
                res = temp_res
                load = temp_load
                obj += delta
                improved = True
    
    return res, obj


def _multi_start_local_search(demands, capacities, open_cost, assign_cost, deadline):
    """Multi-start with perturbation and vectorized local search."""
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Phase 1: Greedy
    res, load = _greedy_assign(demands, capacities, assign_cost)
    best_res = res.copy()
    best_obj = _feasible_objective(best_res, demands, capacities, open_cost, assign_cost)
    
    # Phase 2: Local search on greedy solution
    polished, obj = _vectorized_local_search(best_res, demands, capacities, open_cost, assign_cost, deadline)
    if obj < best_obj:
        best_res = polished.copy()
        best_obj = obj
    
    # Phase 3: Multi-start with random restarts if time permits
    time_per_restart = 3.0
    elapsed = time.time() - (deadline - 110.0)
    remaining = (110.0 - elapsed) * 0.4  # use 40% of remaining time for multi-start
    
    if remaining > time_per_restart:
        np.random.seed(42)
        max_restarts = int(remaining / time_per_restart)
        
        for r in range(max_restarts):
            if time.time() > deadline - 10.0:
                break
            
            # Perturbation: randomly reassign 10-20% of customers
            n_perturb = max(1, int(0.15 * n_cust))
            pert_res = best_res.copy()
            pert_load = np.bincount(pert_res, minlength=n_fac).astype(float)
            
            # Remove some customers and reassign
            to_move = np.random.choice(n_cust, size=n_perturb, replace=False)
            for c in to_move:
                pert_load[pert_res[c]] -= demands[c]
            
            # Reassign moved customers using greedy
            for c in to_move:
                d = demands[c]
                best_f = -1
                for f in np.argsort(assign_cost[c]):
                    if pert_load[f] + d <= capacities[f]:
                        best_f = f
                        break
                if best_f >= 0:
                    pert_res[c] = best_f
                    pert_load[best_f] += d
            
            # Local search on perturbed solution
            polished_p, obj_p = _vectorized_local_search(
                pert_res, demands, capacities, open_cost, assign_cost,
                deadline - 5.0  # leave 5s buffer
            )
            
            if obj_p < best_obj - 1e-9:
                best_res = polished_p.copy()
                best_obj = obj_p
    
    # Final polish
    if time.time() < deadline - 5.0:
        polished_final, obj_final = _vectorized_local_search(
            best_res, demands, capacities, open_cost, assign_cost,
            deadline
        )
        if obj_final < best_obj:
            best_res = polished_final.copy()
            best_obj = obj_final
    
    return best_res


def assign(demands, capacities, open_cost, assign_cost):
    t0 = time.time()
    deadline = t0 + 105.0  # safety margin before 120s kill (with 150s wall-clock, ~30s buffer)
    
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    
    # Main optimization
    res = _multi_start_local_search(demands, capacities, open_cost, assign_cost, deadline)
    
    # Validate
    load = np.bincount(res, minlength=len(capacities)).astype(float)
    if np.any(load > capacities):
        # Fallback to greedy
        res, _ = _greedy_assign(demands, capacities, assign_cost)
    
    return res.astype(int)
# EVOLVE-BLOCK-END
