"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def assign(demands, capacities, open_cost, assign_cost):
    n_customers = len(demands)
    n_facilities = len(capacities)
    N = n_customers
    F = n_facilities
    
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Convert to numpy for speed
    demands = np.array(demands, dtype=np.float64)
    capacities = np.array(capacities, dtype=np.float64)
    open_cost = np.array(open_cost, dtype=np.float64)
    assign_cost = np.array(assign_cost, dtype=np.float64)
    
    def is_valid(assignment):
        for f in range(F):
            mask = assignment == f
            if mask.any():
                if demands[mask].sum() > capacities[f] + 1e-6:
                    return False
        return True
    
    def calc_cost(assignment):
        total_assign = np.sum(assign_cost[np.arange(N), assignment])
        used = set(assignment.tolist())
        total_open = np.sum(open_cost[list(used)])
        return total_assign + total_open
    
    # Greedy initialization with multiple strategies
    def greedy_init(seed):
        rng = random.Random(seed)
        result = np.zeros(N, dtype=int)
        load = np.zeros(F)
        
        # Multiple random orders weighted by demand
        for attempt in range(20):
            result_trial = np.zeros(N, dtype=int)
            load_trial = np.zeros(F)
            success = True
            
            # Create order: mix of demand-desc and random
            if attempt % 3 == 0:
                order = np.argsort(-demands)
            elif attempt % 3 == 1:
                order = np.arange(N)
                rng.shuffle(order.tolist())
            else:
                # Weighted: high demand first but with randomness
                weights = demands + 1
                order = np.argsort(-weights + rng.random() * 10)
            
            for i in order:
                # Get facilities sorted by cost
                sorted_f = np.argsort(assign_cost[i])
                assigned = False
                for f in sorted_f:
                    if load_trial[f] + demands[i] <= capacities[f]:
                        load_trial[f] += demands[i]
                        result_trial[i] = f
                        assigned = True
                        break
                if not assigned:
                    # Try any facility with space
                    for f in range(F):
                        if load_trial[f] + demands[i] <= capacities[f]:
                            load_trial[f] += demands[i]
                            result_trial[i] = f
                            assigned = True
                            break
                if not assigned:
                    success = False
                    break
            
            if success:
                cost = calc_cost(result_trial)
                if cost < calc_cost(result):
                    result = result_trial.copy()
                    load = load_trial.copy()
        
        return result, load
    
    # Lagrangian relaxation for better initialization
    def lagrangian_init():
        # Multipliers for capacity constraints
        lam = np.zeros(F)
        best_assign = None
        best_cost = float('inf')
        
        for iteration in range(1000):
            if time.time() > deadline - 30:
                break
            
            # For each customer, find best facility considering modified cost
            # Modified cost: assign_cost[i,f] + lam[f]*demands[i]
            # But we also need to handle open costs - approximate by adding small penalty
            modified_cost = assign_cost + lam[np.newaxis, :] * demands[:, np.newaxis]
            
            # Each customer picks their cheapest modified-cost facility
            tentative = np.argmin(modified_cost, axis=1)
            
            # Check capacity violations
            load = np.zeros(F)
            for i in range(N):
                load[tentative[i]] += demands[i]
            
            # Update multipliers (subgradient method)
            violation = load - capacities
            step_size = 0.5 / (iteration + 1)
            lam = np.maximum(0, lam + step_size * violation)
            
            # Project: if a facility is over capacity, try to fix
            # Get assignment from tentative, then fix infeasibilities
            assignment = tentative.copy()
            load = np.zeros(F)
            for i in range(N):
                load[assignment[i]] += demands[i]
            
            # Fix over-capacity facilities by moving customers
            for _ in range(50):  # max iterations to fix
                over = [f for f in range(F) if load[f] > capacities[f] + 1e-6]
                if not over:
                    break
                f_over = over[0]
                # Find customer at f_over to move
                customers_at_f = np.where(assignment == f_over)[0]
                if len(customers_at_f) == 0:
                    break
                # Move the one with most savings
                best_i = -1
                best_saving = 0
                for i in customers_at_f:
                    for f_new in range(F):
                        if f_new == f_over:
                            continue
                        if load[f_new] + demands[i] <= capacities[f_new]:
                            saving = assign_cost[i, f_over] - assign_cost[i, f_new]
                            if saving > best_saving:
                                best_saving = saving
                                best_i = i
                                best_f_new = f_new
                if best_i >= 0:
                    load[f_over] -= demands[best_i]
                    load[best_f_new] += demands[best_i]
                    assignment[best_i] = best_f_new
                else:
                    break
            
            # Calculate cost of this feasible solution
            if is_valid(assignment):
                c = calc_cost(assignment)
                if c < best_cost:
                    best_cost = c
                    best_assign = assignment.copy()
        
        return best_assign
    
    # Get initial solutions
    init1, load1 = greedy_init(42)
    init2, load2 = greedy_init(123)
    
    # Try Lagrangian if time permits
    lag_assign = None
    if time.time() < deadline - 60:
        lag_assign = lagrangian_init()
    
    # Pick best initial
    candidates = [init1, init2]
    if lag_assign is not None:
        candidates.append(lag_assign)
    
    best_init = min(candidates, key=lambda a: calc_cost(a) if is_valid(a) else float('inf'))
    
    # Simulated Annealing
    assignment = best_init.copy()
    load = np.zeros(F)
    for i in range(N):
        load[assignment[i]] += demands[i]
    
    current_cost = calc_cost(assignment)
    best_assignment = assignment.copy()
    best_cost = current_cost
    
    # Count customers at each facility for open cost tracking
    facility_count = np.zeros(F, dtype=int)
    for i in range(N):
        facility_count[assignment[i]] += 1
    
    # SA parameters
    T = 50.0  # Initial temperature
    T_min = 0.1
    cooling_rate = 0.995
    max_steps = 50000
    
    step = 0
    while time.time() < deadline - 5 and step < max_steps:
        step += 1
        T = max(T_min, T * cooling_rate)
        
        # Choose move type: 70% move, 30% swap
        if random.random() < 0.7:
            # Move customer i to facility f
            i = random.randint(0, N - 1)
            current_f = assignment[i]
            f = random.randint(0, F - 1)
            if f == current_f:
                continue
            
            # Check capacity
            if load[f] + demands[i] > capacities[f]:
                continue
            
            # Calculate delta
            delta = assign_cost[i, f] - assign_cost[i, current_f]
            
            # Open cost changes
            if facility_count[current_f] == 1:
                delta -= open_cost[current_f]
            if facility_count[f] == 0:
                delta += open_cost[f]
            
            # Acceptance
            if delta < 0 or random.random() < np.exp(-delta / T):
                # Make move
                load[current_f] -= demands[i]
                load[f] += demands[i]
                facility_count[current_f] -= 1
                facility_count[f] += 1
                assignment[i] = f
                current_cost += delta
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_assignment = assignment.copy()
        else:
            # Swap customers i and j
            i = random.randint(0, N - 1)
            j = random.randint(0, N - 1)
            if i == j:
                continue
            fi = assignment[i]
            fj = assignment[j]
            if fi == fj:
                continue
            
            # Check capacity constraints
            if load[fi] - demands[i] + demands[j] > capacities[fi]:
                continue
            if load[fj] - demands[j] + demands[i] > capacities[fj]:
                continue
            
            # Calculate delta
            delta = (assign_cost[i, fj] + assign_cost[j, fi]) - (assign_cost[i, fi] + assign_cost[j, fj])
            
            # Acceptance
            if delta < 0 or random.random() < np.exp(-delta / T):
                # Make swap
                load[fi] = load[fi] - demands[i] + demands[j]
                load[fj] = load[fj] - demands[j] + demands[i]
                assignment[i] = fj
                assignment[j] = fi
                current_cost += delta
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_assignment = assignment.copy()
    
    # Final local search polish
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        for i in range(N):
            if time.time() > deadline - 2:
                break
            current_f = best_assignment[i]
            # Recalculate load for best_assignment
            load_best = np.zeros(F)
            for k in range(N):
                load_best[best_assignment[k]] += demands[k]
            for f in range(F):
                if f == current_f:
                    continue
                if load_best[f] + demands[i] <= capacities[f]:
                    delta = assign_cost[i, f] - assign_cost[i, current_f]
                    # Open cost
                    count_current = np.sum(best_assignment == current_f)
                    count_new = np.sum(best_assignment == f)
                    if count_current == 1:
                        delta -= open_cost[current_f]
                    if count_new == 0:
                        delta += open_cost[f]
                    if delta < -1e-9:
                        best_assignment[i] = f
                        improved = True
                        break
    
    # Verify and return
    if is_valid(best_assignment):
        return best_assignment.astype(int)
    else:
        # Fallback
        result, _ = greedy_init(42)
        if is_valid(result):
            return result.astype(int)
        # Last resort: force valid by redistribution
        result = np.zeros(N, dtype=int)
        load = np.zeros(F)
        for i in np.argsort(-demands):
            for f in np.argsort(assign_cost[i]):
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    result[i] = f
                    break
            else:
                # Should not happen if capacities are sufficient
                result[i] = 0
        return result.astype(int)
# EVOLVE-BLOCK-END
