"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog

def _solve_lp_transport(demands, capacities, open_facilities, assign_cost):
    """
    Solve the transportation problem: assign customers to a given set of open facilities
    to minimize assignment cost, respecting capacities.
    Uses scipy's linear programming (HiGHS) for the LP relaxation.
    The transportation problem has integer optimal solutions when demands and capacities are integers.
    """
    open_list = list(open_facilities)
    k = len(open_list)
    if k == 0:
        return None
    
    n_customers = len(demands)
    n_vars = n_customers * k
    
    # Objective: minimize sum assign_cost[c, f] * x[c, f]
    c_obj = np.zeros(n_vars)
    for ci in range(n_customers):
        for fi_idx, f in enumerate(open_list):
            c_obj[ci * k + fi_idx] = assign_cost[ci, f]
    
    # Constraints:
    # 1. For each customer c: sum over open facilities f of x[c, f] = 1
    A_eq = np.zeros((n_customers, n_vars))
    b_eq = np.ones(n_customers)
    for ci in range(n_customers):
        for fi_idx, f in enumerate(open_list):
            A_eq[ci, ci * k + fi_idx] = 1.0
    
    # 2. For each open facility f: sum over customers c of demands[c] * x[c, f] <= capacities[f]
    A_ub = np.zeros((k, n_vars))
    b_ub = np.array([capacities[f] for f in open_list], dtype=float)
    for fi_idx, f in enumerate(open_list):
        for ci in range(n_customers):
            A_ub[fi_idx, ci * k + fi_idx] = demands[ci]
    
    bounds = [(0, 1)] * n_vars
    
    res = linprog(c_obj, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, 
                  bounds=bounds, method='highs', options={'presolve': True})
    
    if res.status == 0:
        x = res.x.reshape(n_customers, k)
        assignment = np.zeros(n_customers, dtype=int)
        for ci in range(n_customers):
            best_idx = np.argmax(x[ci])
            assignment[ci] = open_list[best_idx]
        # Verify feasibility
        loads = np.zeros(len(capacities))
        for ci in range(n_customers):
            loads[assignment[ci]] += demands[ci]
        if np.all(loads <= capacities + 1e-9):
            return assignment
    return None


def assign(demands, capacities, open_cost, assign_cost):
    """
    Capacitated facility location: assign customers to facilities minimizing
    open_cost + assign_cost, respecting capacities.
    
    Strategy:
    1. Generate diverse candidate facility subsets using cost-aware heuristics.
    2. For each subset, solve the transportation LP (exact for fixed facility set).
    3. Use local search (1-swap and 2-swap) to improve.
    4. Track best solution found within time budget.
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
    
    def local_search(assignment, max_iters=300):
        """Improve assignment via 1-swap local search with open-cost awareness."""
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
                best_delta = 0
                best_f = current_f
                for f in range(n_facilities):
                    if f == current_f:
                        continue
                    if loads[f] + demands[ci] <= capacities[f] + 1e-9:
                        old_cost = assign_cost[ci, current_f]
                        new_cost = assign_cost[ci, f]
                        open_delta = 0
                        if loads[current_f] == demands[ci]:
                            open_delta -= open_cost[current_f]
                        if loads[f] == 0:
                            open_delta += open_cost[f]
                        
                        delta = new_cost - old_cost + open_delta
                        if delta < best_delta - 1e-9:
                            best_delta = delta
                            best_f = f
                
                if best_f != current_f:
                    loads[current_f] -= demands[ci]
                    loads[best_f] += demands[ci]
                    assignment[ci] = best_f
                    current_cost += best_delta
                    improved = True
        return assignment, current_cost
    
    def two_swap_local_search(assignment, max_iters=100):
        """Improve via 2-swap: move two customers simultaneously."""
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
            # Try swapping assignments of pairs of customers
            for i in range(n_customers):
                for j in range(i+1, n_customers):
                    fi = assignment[i]
                    fj = assignment[j]
                    if fi == fj:
                        continue
                    # Try: i->fj, j->fi
                    if loads[fi] - demands[i] + demands[j] <= capacities[fi] + 1e-9 and \
                       loads[fj] - demands[j] + demands[i] <= capacities[fj] + 1e-9:
                        # Check if this is different from current
                        if fi != fj:
                            old_cost = assign_cost[i, fi] + assign_cost[j, fj]
                            new_cost = assign_cost[i, fj] + assign_cost[j, fi]
                            # Open cost changes
                            open_delta = 0
                            new_load_i = loads[fi] - demands[i] + demands[j]
                            new_load_j = loads[fj] - demands[j] + demands[i]
                            if loads[fi] > 0 and new_load_i == 0:
                                open_delta -= open_cost[fi]
                            if loads[fi] == 0 and new_load_i > 0:
                                open_delta += open_cost[fi]
                            if loads[fj] > 0 and new_load_j == 0:
                                open_delta -= open_cost[fj]
                            if loads[fj] == 0 and new_load_j > 0:
                                open_delta += open_cost[fj]
                            
                            delta = new_cost - old_cost + open_delta
                            if delta < -1e-9:
                                loads[fi] = new_load_i
                                loads[fj] = new_load_j
                                assignment[i] = fj
                                assignment[j] = fi
                                current_cost += delta
                                improved = True
                                break
                if improved:
                    break
        return assignment, current_cost
    
    def greedy_construct(demands, capacities, open_cost, assign_cost, mode='nearest', rng=None):
        """Greedy construction with different modes."""
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
                if mode == 'nearest':
                    f = min(feasible, key=lambda f: assign_cost[i, f])
                elif mode == 'open_aware':
                    eff_costs = []
                    for f in feasible:
                        if load[f] == 0:
                            eff_costs.append(assign_cost[i, f] + open_cost[f])
                        else:
                            eff_costs.append(assign_cost[i, f])
                    f = feasible[np.argmin(eff_costs)]
                elif mode == 'random' and rng is not None:
                    costs = np.array([assign_cost[i, f] for f in feasible])
                    max_c = np.max(costs) if len(costs) > 0 else 1
                    costs_noisy = costs + rng.uniform(0, 1) * max_c * 0.1
                    min_idx = np.argmin(costs_noisy)
                    f = feasible[min_idx]
                else:
                    f = min(feasible, key=lambda f: assign_cost[i, f])
                result[i] = f
                load[f] += demands[i]
        return result
    
    def generate_facility_subsets(demands, capacities, open_cost, assign_cost, n_subsets=15):
        """Generate diverse facility subsets using cost-aware heuristics."""
        subsets = []
        
        # 1. All facilities
        subsets.append(set(range(n_facilities)))
        
        # 2. Top-k facilities by lowest open cost
        sorted_by_open = np.argsort(open_cost)
        for k in range(3, min(n_facilities+1, 12)):
            subsets.append(set(sorted_by_open[:k].tolist()))
        
        # 3. Facilities with lowest average assignment cost
        avg_assign_cost = assign_cost.mean(axis=0)
        sorted_by_assign = np.argsort(avg_assign_cost)
        for k in range(3, min(n_facilities+1, 12)):
            subsets.append(set(sorted_by_assign[:k].tolist()))
        
        # 4. Combined score: open_cost + avg_assign_cost * (total_demand / capacity)
        total_demand = demands.sum()
        combined_score = open_cost + avg_assign_cost * (total_demand / np.maximum(capacities, 1))
        sorted_by_combined = np.argsort(combined_score)
        for k in range(3, min(n_facilities+1, 12)):
            subsets.append(set(sorted_by_combined[:k].tolist()))
        
        # 5. Random subsets
        rng = np.random.default_rng(42)
        for _ in range(5):
            k = rng.integers(4, min(n_facilities+1, 12))
            subset = set(rng.choice(n_facilities, size=k, replace=False).tolist())
            subsets.append(subset)
        
        # Deduplicate and ensure feasibility (total capacity >= total demand)
        unique_subsets = []
        seen = set()
        for s in subsets:
            key = frozenset(s)
            if key not in seen:
                seen.add(key)
                # Check if total capacity is sufficient
                total_cap = sum(capacities[f] for f in s)
                if total_cap >= total_demand:
                    unique_subsets.append(s)
        
        # If we don't have enough feasible subsets, add the full set
        if not unique_subsets:
            unique_subsets.append(set(range(n_facilities)))
        
        return unique_subsets[:n_subsets]
    
    # Generate and evaluate facility subsets
    facility_subsets = generate_facility_subsets(demands, capacities, open_cost, assign_cost, n_subsets=20)
    
    for subset in facility_subsets:
        if time.time() > deadline - 10:
            break
        
        # Check if total capacity is sufficient
        total_cap = sum(capacities[f] for f in subset)
        if total_cap < demands.sum():
            continue
        
        # Solve LP for this facility set
        assignment = _solve_lp_transport(demands, capacities, subset, assign_cost)
        if assignment is not None:
            cost = total_cost(assignment)
            if cost < best_cost:
                best_cost = cost
                best_result = assignment.copy()
    
    # Also try greedy constructions and local search
    if time.time() < deadline - 15:
        # Deterministic heuristics
        candidates = []
        candidates.append(greedy_construct(demands, capacities, open_cost, assign_cost, mode='nearest'))
        candidates.append(greedy_construct(demands, capacities, open_cost, assign_cost, mode='open_aware'))
        
        # Randomized heuristics
        for seed in range(8):
            rng_i = np.random.default_rng(seed)
            cand = greedy_construct(demands, capacities, open_cost, assign_cost, mode='random', rng=rng_i)
            candidates.append(cand)
        
        for cand in candidates:
            if time.time() > deadline - 10:
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
            
            # Local search
            if time.time() < deadline - 8:
                improved, improved_cost = local_search(cand.copy(), max_iters=150)
                if improved_cost < best_cost:
                    best_cost = improved_cost
                    best_result = improved.copy()
    
    # Final improvement on best solution
    if best_result is not None and time.time() < deadline - 5:
        # 1-swap local search
        while time.time() < deadline - 3:
            improved, improved_cost = local_search(best_result.copy(), max_iters=100)
            if improved_cost < best_cost - 1e-9:
                best_cost = improved_cost
                best_result = improved.copy()
            else:
                break
        
        # 2-swap local search
        if time.time() < deadline - 2:
            improved, improved_cost = two_swap_local_search(best_result.copy(), max_iters=50)
            if improved_cost < best_cost - 1e-9:
                best_cost = improved_cost
                best_result = improved.copy()
    
    if best_result is None:
        # Fallback: simple greedy
        best_result = np.zeros(n_customers, dtype=int)
        load = np.zeros(n_facilities)
        order = np.argsort(-demands)
        for i in order:
            feasible = [f for f in range(n_facilities) if load[f] + demands[i] <= capacities[f]]
            if not feasible:
                f = np.argmax(capacities - load)
            else:
                f = min(feasible, key=lambda f: assign_cost[i, f])
            best_result[i] = f
            load[f] += demands[i]
    
    return best_result.astype(int)

# EVOLVE-BLOCK-END
