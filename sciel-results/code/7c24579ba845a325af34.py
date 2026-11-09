"""Capacitated facility location: improved assignment via greedy + local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog

DEADLINE_SECONDS = 115  # safety margin before 120s limit


def _compute_cost(demands, capacities, open_cost, assign_cost, assignment):
    """Compute total cost for a given assignment."""
    n_facs = len(capacities)
    # Check validity
    for i in range(len(demands)):
        f = assignment[i]
        if f < 0 or f >= n_facs:
            return float('inf')
    
    load = np.zeros(n_facs)
    for i in range(len(demands)):
        load[assignment[i]] += demands[i]
    
    for f in range(n_facs):
        if load[f] > capacities[f] + 1e-9:
            return float('inf')
    
    # Open cost
    open_total = 0.0
    for f in range(n_facs):
        if load[f] > 0:
            open_total += open_cost[f]
    
    # Assignment cost
    assign_total = 0.0
    for i in range(len(demands)):
        assign_total += assign_cost[i, assignment[i]]
    
    return open_total + assign_total


def _greedy_assign(demands, capacities, open_cost, assign_cost, order='largest_first'):
    """Greedy assignment: largest demand first, cheapest feasible facility."""
    n_cust = len(demands)
    n_facs = len(capacities)
    result = np.zeros(n_cust, dtype=int)
    load = np.zeros(n_facs)
    
    if order == 'largest_first':
        order_idx = np.argsort(-demands)
    else:
        order_idx = np.random.permutation(n_cust)
    
    for i in order_idx:
        # Find cheapest feasible facility
        best_f = -1
        best_cost = float('inf')
        for f in np.argsort(assign_cost[i]):
            if load[f] + demands[i] <= capacities[f]:
                best_f = f
                break
        
        if best_f == -1:
            # No feasible facility found, assign to least-loaded (shouldn't happen with valid instances)
            best_f = np.argmin(load)
        
        load[best_f] += demands[i]
        result[i] = best_f
    
    return result


def _local_search(demands, capacities, open_cost, assign_cost, assignment, max_iters=2000):
    """Local search: try moving each customer to a better facility."""
    n_cust = len(demands)
    n_facs = len(capacities)
    assignment = assignment.copy()
    load = np.zeros(n_facs)
    for i in range(n_cust):
        load[assignment[i]] += demands[i]
    
    current_cost = _compute_cost(demands, capacities, open_cost, assign_cost, assignment)
    
    for it in range(max_iters):
        improved = False
        for i in range(n_cust):
            old_f = assignment[i]
            # Try moving to each other facility
            for new_f in range(n_facs):
                if new_f == old_f:
                    continue
                # Check capacity
                if load[new_f] + demands[i] > capacities[new_f]:
                    continue
                
                # Compute cost change
                # Remove i from old_f, add to new_f
                old_open = 1.0 if (load[old_f] - demands[i]) > 0 else 0.0
                new_open = 1.0 if (load[new_f] + demands[i]) > 0 else 0.0
                
                delta_assign = assign_cost[i, new_f] - assign_cost[i, old_f]
                delta_open = (new_open - old_open) * open_cost[new_f] if new_f != old_f else 0.0
                
                # If old_f becomes empty, we save its open cost
                if (load[old_f] - demands[i]) <= 0:
                    delta_open -= open_cost[old_f]
                
                total_delta = delta_assign + delta_open
                
                if total_delta < -1e-9:
                    # Apply move
                    load[old_f] -= demands[i]
                    load[new_f] += demands[i]
                    assignment[i] = new_f
                    current_cost += total_delta
                    improved = True
                    break
        
        if not improved:
            break
    
    return assignment, current_cost


def _multi_start(demands, capacities, open_cost, assign_cost, n_starts=6, time_budget=100):
    """Multi-start greedy + local search."""
    best_assignment = None
    best_cost = float('inf')
    
    start_time = time.time()
    
    for s in range(n_starts):
        if time.time() - start_time > time_budget:
            break
        
        # Different greedy orders
        if s == 0:
            assignment = _greedy_assign(demands, capacities, open_cost, assign_cost, 'largest_first')
        elif s == 1:
            assignment = _greedy_assign(demands, capacities, open_cost, assign_cost, 'random')
        else:
            # Perturb the best so far
            if best_assignment is not None:
                assignment = best_assignment.copy()
                # Randomly reassign a few customers
                n_perturb = max(1, len(demands) // 10)
                idx = np.random.choice(len(demands), size=n_perturb, replace=False)
                load = np.zeros(len(capacities))
                for i in range(len(demands)):
                    load[assignment[i]] += demands[i]
                for i in idx:
                    old_f = assignment[i]
                    load[old_f] -= demands[i]
                    # Find feasible facility
                    feasible = [f for f in range(len(capacities)) if load[f] + demands[i] <= capacities[f]]
                    if feasible:
                        new_f = feasible[np.random.randint(len(feasible))]
                    else:
                        new_f = old_f
                    load[new_f] += demands[i]
                    assignment[i] = new_f
            else:
                assignment = _greedy_assign(demands, capacities, open_cost, assign_cost, 'random')
        
        # Local search
        assignment, cost = _local_search(demands, capacities, open_cost, assign_cost, assignment, max_iters=500)
        
        if cost < best_cost:
            best_cost = cost
            best_assignment = assignment.copy()
    
    return best_assignment, best_cost


def assign(demands, capacities, open_cost, assign_cost):
    """Assign each customer to a facility minimizing total cost."""
    start_time = time.time()
    
    demands = np.asarray(demands)
    capacities = np.asarray(capacities)
    open_cost = np.asarray(open_cost)
    assign_cost = np.asarray(assign_cost)
    
    n_cust = len(demands)
    n_facs = len(capacities)
    
    # Phase 1: Multi-start greedy + local search
    best_assignment, best_cost = _multi_start(demands, capacities, open_cost, assign_cost, 
                                               n_starts=8, time_budget=DEADLINE_SECONDS - 20)
    
    # Phase 2: Final polish with more local search iterations
    if best_assignment is not None and time.time() - start_time < DEADLINE_SECONDS - 5:
        best_assignment, best_cost = _local_search(demands, capacities, open_cost, assign_cost, 
                                                    best_assignment, max_iters=5000)
    
    # Ensure validity
    if best_assignment is None:
        # Fallback: simple greedy
        best_assignment = _greedy_assign(demands, capacities, open_cost, assign_cost, 'largest_first')
    
    # Final validity check
    load = np.zeros(n_facs)
    for i in range(n_cust):
        f = best_assignment[i]
        if f < 0 or f >= n_facs:
            # Invalid, use fallback
            best_assignment = _greedy_assign(demands, capacities, open_cost, assign_cost, 'largest_first')
            break
        load[f] += demands[i]
    
    for f in range(n_facs):
        if load[f] > capacities[f] + 1e-9:
            # Invalid, use fallback
            best_assignment = _greedy_assign(demands, capacities, open_cost, assign_cost, 'largest_first')
            break
    
    return best_assignment
# EVOLVE-BLOCK-END
