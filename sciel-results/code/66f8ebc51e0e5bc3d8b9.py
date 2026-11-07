"""
Structural Change: Network Flow Formulation with LP Relaxation and Local Search.

This approach formulates the CFLP as a minimum-cost flow problem:
1. Create a network with source -> customers -> facilities -> sink
2. Customer nodes have unit supply, facility nodes have capacity constraints
3. Edge costs represent assignment costs
4. Solve the LP relaxation using scipy.optimize.linprog
5. Round the fractional solution using a greedy conflict-resolution strategy
6. Polish with aggressive local search (1-moves, 2-swaps, 3-opt)

The LP relaxation provides a lower bound and guides the rounding,
which can find assignment patterns unreachable by pure heuristics.
"""

import numpy as np
import time
from scipy.optimize import linprog

DEADLINE = 118.0  # Safety margin before 120s limit


def _total_cost(assignment, demands, capacities, open_cost, assign_cost):
    """Calculate the total cost of an assignment."""
    used = np.zeros(len(capacities), dtype=bool)
    for f in assignment:
        used[f] = True
    open_sum = float(open_cost[used].sum())
    cust = np.arange(len(assignment))
    assign_sum = float(assign_cost[cust, assignment].sum())
    return open_sum + assign_sum


def _solve_lp_relaxation(demands, capacities, assign_cost, deadline):
    """
    Solve the LP relaxation of the CFLP using scipy.optimize.linprog.
    
    Variables: x[i,j] = fraction of customer i assigned to facility j
    Objective: minimize sum_{i,j} assign_cost[i,j] * x[i,j]
    Constraints:
      - sum_j x[i,j] = 1 for each customer i
      - sum_i demands[i] * x[i,j] <= capacities[j] for each facility j
      - 0 <= x[i,j] <= 1
    """
    n_c = len(demands)
    n_f = len(capacities)
    t0 = time.time()
    
    # Number of variables
    n_vars = n_c * n_f
    
    # Objective: c[i,j] = assign_cost[i,j]
    c = assign_cost.flatten()
    
    # Equality constraints: sum_j x[i,j] = 1 for each i
    A_eq = np.zeros((n_c, n_vars))
    b_eq = np.ones(n_c)
    for i in range(n_c):
        for j in range(n_f):
            A_eq[i, i * n_f + j] = 1.0
    
    # Inequality constraints: sum_i demands[i] * x[i,j] <= capacities[j]
    A_ub = np.zeros((n_f, n_vars))
    b_ub = capacities.copy()
    for j in range(n_f):
        for i in range(n_c):
            A_ub[j, i * n_f + j] = demands[i]
    
    # Bounds: 0 <= x[i,j] <= 1
    bounds = [(0, 1)] * n_vars
    
    # Solve
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                        bounds=bounds, method='highs', options={'presolve': True})
        
        if result.success:
            # Reshape solution
            x = result.x.reshape(n_c, n_f)
            return x
        else:
            return None
    except Exception as e:
        return None


def _valid(assignment, demands, capacities):
    """Check if assignment is valid (integer indices, capacity constraints)."""
    if len(assignment) != len(demands):
        return False
    if np.any(assignment < 0) or np.any(assignment >= len(capacities)):
        return False
    if np.any(np.asarray(assignment, dtype=float) % 1 != 0):
        return False
    loads = np.zeros(len(capacities))
    np.add.at(loads, assignment, demands)
    return bool(np.all(loads <= capacities + 1e-9))


def _greedy_construct(demands, capacities, assign_cost, order):
    """
    Greedy construction: assign customers in the given order to the cheapest
    feasible facility.
    """
    n_c = len(demands)
    n_f = len(capacities)
    
    assignment = np.full(n_c, -1, dtype=int)
    loads = np.zeros(n_f)
    
    for i in order:
        d = demands[i]
        # Find feasible facilities (sorted by cost for this customer)
        costs = assign_cost[i, :]
        # Get facilities sorted by cost
        sorted_f = np.argsort(costs)
        
        assigned = False
        for f in sorted_f:
            if loads[f] + d <= capacities[f] + 1e-9:
                assignment[i] = f
                loads[f] += d
                assigned = True
                break
        
        if not assigned:
            # Fallback: assign to facility with most remaining capacity
            best_f = np.argmax(capacities - loads)
            assignment[i] = best_f
            loads[best_f] += d
    
    return assignment


def _lp_guided_rounding(lp_solution, demands, capacities, assign_cost, open_cost):
    """
    Convert LP solution to a feasible integer assignment using regret-based rounding.
    
    Strategy:
    1. For each customer, compute the "regret" of assigning to their best facility
       (difference between best and second-best fractional values)
    2. Process customers in order of decreasing regret (most confident first)
    3. Assign to the facility with highest LP value if feasible, else next best
    4. Resolve capacity conflicts by reassigning customers with smallest cost increase
    """
    n_c = len(demands)
    n_f = len(capacities)
    
    if lp_solution is None:
        # Fallback to greedy
        return _greedy_construct(demands, capacities, assign_cost, np.arange(n_c))
    
    # Compute regret for each customer
    sorted_lp = np.sort(lp_solution, axis=1)
    regrets = sorted_lp[:, -1] - sorted_lp[:, -2]  # Max - second max
    
    # Process in order of decreasing regret
    order = np.argsort(-regrets)
    
    assignment = np.full(n_c, -1, dtype=int)
    loads = np.zeros(n_f)
    
    # First pass: assign high-regret customers
    for i in order:
        d = demands[i]
        # Get facilities sorted by LP value (descending)
        sorted_f = np.argsort(-lp_solution[i])
        
        assigned = False
        for f in sorted_f:
            if loads[f] + d <= capacities[f] + 1e-9:
                assignment[i] = f
                loads[f] += d
                assigned = True
                break
        
        if not assigned:
            # Fallback: assign to facility with most remaining capacity
            best_f = np.argmax(capacities - loads)
            assignment[i] = best_f
            loads[best_f] += d
    
    # Check if any customers are unassigned (shouldn't happen with fallback)
    unassigned = np.where(assignment == -1)[0]
    for i in unassigned:
        d = demands[i]
        best_f = np.argmax(capacities - loads)
        assignment[i] = best_f
        loads[best_f] += d
    
    return assignment


def _local_search(assignment, demands, capacities, open_cost, assign_cost, deadline):
    """
    Aggressive local search with 1-moves, 2-swaps, and 3-opt exchanges.
    """
    n_c = len(assignment)
    n_f = len(capacities)
    
    if n_c == 0:
        return assignment
        
    assignment = assignment.copy()
    loads = np.zeros(n_f)
    np.add.at(loads, assignment, demands)
    
    in_use = np.zeros(n_f, dtype=bool)
    for f in assignment:
        in_use[f] = True
        
    t_start = time.time()
    
    # Precompute current costs for each customer
    current_costs = assign_cost[np.arange(n_c), assignment]
    
    max_rounds = 30
    for round_num in range(max_rounds):
        if time.time() >= deadline:
            break
            
        improved = False
        
        # --- 1-Move Search ---
        for i in range(n_c):
            if time.time() >= deadline:
                break
                
            cur_f = assignment[i]
            d = demands[i]
            cur_cost = current_costs[i]
            
            new_costs = assign_cost[i, :]
            
            # Open cost changes
            open_deltas = np.zeros(n_f)
            open_deltas[~in_use] = open_cost[~in_use]
            if loads[cur_f] == d:
                open_deltas -= open_cost[cur_f]
            
            total_deltas = new_costs - cur_cost + open_deltas
            
            feasible = loads + d <= capacities + 1e-9
            feasible = feasible & (np.arange(n_f) != cur_f)
            
            if not np.any(feasible):
                continue
                
            total_deltas[~feasible] = np.inf
            
            best_f = np.argmin(total_deltas)
            gain = -total_deltas[best_f]
            
            if gain > 1e-9:
                loads[cur_f] -= d
                loads[best_f] += d
                assignment[i] = best_f
                current_costs[i] = assign_cost[i, best_f]
                
                in_use[best_f] = True
                if loads[cur_f] == 0:
                    in_use[cur_f] = False
                    
                improved = True
        
        # --- 2-Swap Search ---
        if time.time() >= deadline:
            break
            
        for i in range(n_c):
            if time.time() >= deadline:
                break
            fi = assignment[i]
            di = demands[i]
            ci = current_costs[i]
            
            for k in range(i + 1, n_c):
                fk = assignment[k]
                if fi == fk:
                    continue
                dk = demands[k]
                ck = current_costs[k]
                
                if loads[fi] - di + dk > capacities[fi] + 1e-9:
                    continue
                if loads[fk] - dk + di > capacities[fk] + 1e-9:
                    continue
                
                new_ci = assign_cost[i, fk]
                new_ck = assign_cost[k, fi]
                
                delta_assign = (new_ci + new_ck) - (ci + ck)
                
                if delta_assign < -1e-9:
                    loads[fi] = loads[fi] - di + dk
                    loads[fk] = loads[fk] - dk + di
                    assignment[i] = fk
                    assignment[k] = fi
                    current_costs[i] = new_ci
                    current_costs[k] = new_ck
                    improved = True
                    break
        
        # --- 3-Opt Exchange ---
        if time.time() >= deadline:
            break
            
        for i in range(n_c):
            if time.time() >= deadline:
                break
            fi = assignment[i]
            di = demands[i]
            ci = current_costs[i]
            
            for k in range(i + 1, n_c):
                if time.time() >= deadline:
                    break
                fk = assignment[k]
                if fi == fk:
                    continue
                dk = demands[k]
                ck = current_costs[k]
                
                for m in range(k + 1, n_c):
                    if time.time() >= deadline:
                        break
                    fm = assignment[m]
                    if fi == fm or fk == fm:
                        continue
                    dm = demands[m]
                    cm = current_costs[m]
                    
                    # Check capacity after cyclic exchange: i->fk, k->fm, m->fi
                    if loads[fi] - di + dm > capacities[fi] + 1e-9:
                        continue
                    if loads[fk] - dk + di > capacities[fk] + 1e-9:
                        continue
                    if loads[fm] - dm + dk > capacities[fm] + 1e-9:
                        continue
                    
                    new_ci = assign_cost[i, fk]
                    new_ck = assign_cost[k, fm]
                    new_cm = assign_cost[m, fi]
                    
                    delta_assign = (new_ci + new_ck + new_cm) - (ci + ck + cm)
                    
                    if delta_assign < -1e-9:
                        loads[fi] = loads[fi] - di + dm
                        loads[fk] = loads[fk] - dk + di
                        loads[fm] = loads[fm] - dm + dk
                        assignment[i] = fk
                        assignment[k] = fm
                        assignment[m] = fi
                        current_costs[i] = new_ci
                        current_costs[k] = new_ck
                        current_costs[m] = new_cm
                        improved = True
                        break  # Restart after a 3-opt
        
        if not improved:
            break
    
    return assignment


def assign(demands, capacities, open_cost, assign_cost):
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    
    n_c = len(demands)
    n_f = len(capacities)
    t0 = time.time()
    
    best_assign = None
    best_cost = np.inf
    
    # Phase 1: LP relaxation guided rounding
    if time.time() - t0 < DEADLINE - 60:
        lp_solution = _solve_lp_relaxation(demands, capacities, assign_cost, t0 + DEADLINE - 50)
        
        if lp_solution is not None:
            candidate = _lp_guided_rounding(lp_solution, demands, capacities, assign_cost, open_cost)
            
            if _valid(candidate, demands, capacities):
                cost = _total_cost(candidate, demands, capacities, open_cost, assign_cost)
                if cost < best_cost:
                    best_cost = cost
                    best_assign = candidate.copy()
    
    # Phase 2: Multi-start greedy construction with diverse orderings
    if best_assign is None or time.time() - t0 < DEADLINE - 40:
        orderings = []
        
        # Natural order
        orderings.append(np.arange(n_c))
        
        # Random orders
        rng = np.random.default_rng(42)
        for _ in range(8):
            orderings.append(rng.permutation(n_c))
        
        # Demand-based orders
        orderings.append(np.argsort(-demands))
        orderings.append(np.argsort(demands))
        
        # Cost-based orders
        sum_costs = assign_cost.sum(axis=1)
        orderings.append(np.argsort(sum_costs))
        orderings.append(np.argsort(-sum_costs))
        
        # Regret-based orders
        sorted_costs = np.sort(assign_cost, axis=1)
        regrets = sorted_costs[:, 1] - sorted_costs[:, 0]
        orderings.append(np.argsort(-regrets))
        orderings.append(np.argsort(regrets))
        
        for order in orderings:
            if time.time() - t0 > DEADLINE - 30:
                break
                
            candidate = _greedy_construct(demands, capacities, assign_cost, order)
            
            if not _valid(candidate, demands, capacities):
                continue
            
            cost = _total_cost(candidate, demands, capacities, open_cost, assign_cost)
            
            if cost < best_cost:
                best_cost = cost
                best_assign = candidate.copy()
    
    # Fallback if nothing found
    if best_assign is None:
        best_assign = _greedy_construct(demands, capacities, assign_cost, np.arange(n_c))
        best_cost = _total_cost(best_assign, demands, capacities, open_cost, assign_cost)
    
    # Phase 3: Polish with aggressive local search
    remaining_time = DEADLINE - (time.time() - t0)
    if remaining_time > 10:
        deadline_for_ls = t0 + DEADLINE - 3
        polished = _local_search(best_assign, demands, capacities, open_cost, assign_cost, deadline_for_ls)
        if _valid(polished, demands, capacities):
            polished_cost = _total_cost(polished, demands, capacities, open_cost, assign_cost)
            if polished_cost < best_cost:
                best_assign = polished
                best_cost = polished_cost
    
    # Phase 4: If time remains, try more starts and polish
    if time.time() - t0 < DEADLINE - 25:
        rng2 = np.random.default_rng(123)
        for _ in range(15):
            if time.time() - t0 > DEADLINE - 15:
                break
                
            order = rng2.permutation(n_c)
            candidate = _greedy_construct(demands, capacities, assign_cost, order)
            
            if not _valid(candidate, demands, capacities):
                continue
            
            cost = _total_cost(candidate, demands, capacities, open_cost, assign_cost)
            
            if cost < best_cost:
                best_cost = cost
                best_assign = candidate.copy()
                
                if time.time() - t0 < DEADLINE - 8:
                    deadline_for_ls = t0 + DEADLINE - 3
                    polished = _local_search(best_assign, demands, capacities, open_cost, assign_cost, deadline_for_ls)
                    if _valid(polished, demands, capacities):
                        polished_cost = _total_cost(polished, demands, capacities, open_cost, assign_cost)
                        if polished_cost < best_cost:
                            best_assign = polished
                            best_cost = polished_cost
    
    return best_assign.astype(int)
