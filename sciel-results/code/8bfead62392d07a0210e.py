"""Capacitated facility location solver with local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def assign(demands, capacities, open_cost, assign_cost):
    """
    Assign each customer to a facility minimizing total cost
    (open_cost for used facilities + assign_cost for each customer-facility pair),
    subject to capacity constraints.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Convert to numpy arrays for efficiency
    demands = np.asarray(demands, dtype=np.int64)
    capacities = np.asarray(capacities, dtype=np.int64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)
    
    def compute_total_cost(assignment, load):
        """Compute total cost given an assignment and load vector."""
        used_facilities = np.unique(assignment[assignment >= 0])
        open_costs = np.sum(open_cost[used_facilities])
        assign_costs = np.sum(assign_cost[np.arange(n_customers), assignment])
        return open_costs + assign_costs
    
    def is_valid(assignment, load):
        """Check if assignment is valid (all customers assigned, no capacity exceeded)."""
        if np.any(assignment < 0):
            return False
        if np.any(load > capacities):
            return False
        return True
    
    def greedy_seed(open_weight=1.0):
        """
        Generate a greedy seed assignment.
        open_weight scales the effective open cost to balance assignment vs opening.
        """
        result = np.full(n_customers, -1, dtype=int)
        load = np.zeros(n_facilities, dtype=np.int64)
        
        # Sort customers by demand (largest first) for better packing
        order = np.argsort(-demands)
        
        # Precompute effective cost: assign_cost + amortized open cost
        # For each facility, the marginal cost of opening it (if not already open)
        # We approximate by adding open_cost / expected_customers_per_facility
        
        # Simpler approach: try facilities in order of assign_cost, 
        # but penalize opening a new facility
        for i in order:
            # Get facilities sorted by assign cost
            fac_order = np.argsort(assign_cost[i])
            best_f = -1
            best_cost = np.inf
            
            for f in fac_order:
                if load[f] + demands[i] > capacities[f]:
                    continue
                
                # Compute marginal cost: assign cost + potential open cost
                marginal = assign_cost[i, f]
                if load[f] == 0:
                    # This would open a new facility
                    # Estimate amortized open cost
                    marginal += open_cost[f] * open_weight / max(1, n_customers // n_facilities)
                
                if marginal < best_cost:
                    best_cost = marginal
                    best_f = f
            
            if best_f >= 0:
                result[i] = best_f
                load[best_f] += demands[i]
            else:
                # Fallback: find any feasible facility
                for f in range(n_facilities):
                    if load[f] + demands[i] <= capacities[f]:
                        result[i] = f
                        load[f] += demands[i]
                        break
        
        # Ensure all customers are assigned
        for i in range(n_customers):
            if result[i] < 0:
                for f in range(n_facilities):
                    if load[f] + demands[i] <= capacities[f]:
                        result[i] = f
                        load[f] += demands[i]
                        break
        
        return result, load
    
    def local_search(assignment, load, max_iters=500):
        """
        Local search with reassign and swap moves.
        Returns improved assignment and load.
        """
        assignment = assignment.copy()
        load = load.copy()
        current_cost = compute_total_cost(assignment, load)
        
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            
            # Check deadline
            if time.time() > deadline:
                break
            
            # Try reassign moves: move customer i to a different facility
            for i in range(n_customers):
                if time.time() > deadline:
                    break
                
                old_f = assignment[i]
                # Try all other facilities
                for new_f in range(n_facilities):
                    if new_f == old_f:
                        continue
                    if load[new_f] + demands[i] > capacities[new_f]:
                        continue
                    
                    # Compute cost change
                    # Remove i from old_f, add to new_f
                    old_load = load[old_f]
                    new_load = load[new_f]
                    
                    cost_change = assign_cost[i, new_f] - assign_cost[i, old_f]
                    
                    # Open cost changes
                    if old_load == demands[i] and new_load == 0:
                        # Closing old facility, opening new
                        cost_change += open_cost[new_f] - open_cost[old_f]
                    elif old_load == demands[i] and new_load > 0:
                        # Closing old facility
                        cost_change -= open_cost[old_f]
                    elif old_load > demands[i] and new_load == 0:
                        # Opening new facility
                        cost_change += open_cost[new_f]
                    
                    if cost_change < -1e-9:
                        # Accept move
                        load[old_f] -= demands[i]
                        load[new_f] += demands[i]
                        assignment[i] = new_f
                        current_cost += cost_change
                        improved = True
            
            # Try swap moves: swap two customers between facilities
            if not improved:
                for i in range(n_customers):
                    if time.time() > deadline:
                        break
                    for j in range(i + 1, n_customers):
                        if time.time() > deadline:
                            break
                        
                        fi = assignment[i]
                        fj = assignment[j]
                        if fi == fj:
                            continue
                        
                        # Check capacity after swap
                        if load[fi] - demands[i] + demands[j] > capacities[fi]:
                            continue
                        if load[fj] - demands[j] + demands[i] > capacities[fj]:
                            continue
                        
                        # Compute cost change
                        cost_change = (assign_cost[i, fj] + assign_cost[j, fi]) - \
                                      (assign_cost[i, fi] + assign_cost[j, fj])
                        
                        # Open cost changes
                        old_load_i = load[fi]
                        old_load_j = load[fj]
                        new_load_i = old_load_i - demands[i] + demands[j]
                        new_load_j = old_load_j - demands[j] + demands[i]
                        
                        # Track which facilities change open status
                        open_change = 0
                        if old_load_i == 0 and new_load_i > 0:
                            open_change += open_cost[fi]
                        elif old_load_i > 0 and new_load_i == 0:
                            open_change -= open_cost[fi]
                        
                        if old_load_j == 0 and new_load_j > 0:
                            open_change += open_cost[fj]
                        elif old_load_j > 0 and new_load_j == 0:
                            open_change -= open_cost[fj]
                        
                        cost_change += open_change
                        
                        if cost_change < -1e-9:
                            # Accept swap
                            load[fi] = new_load_i
                            load[fj] = new_load_j
                            assignment[i] = fj
                            assignment[j] = fi
                            current_cost += cost_change
                            improved = True
        
        return assignment, load, current_cost
    
    # Generate multiple seeds with different open weights
    best_assignment = None
    best_load = None
    best_cost = np.inf
    
    open_weights = [0.1, 0.5, 1.0, 2.0, 5.0]
    
    for w in open_weights:
        if time.time() > deadline:
            break
        
        seed_assignment, seed_load = greedy_seed(open_weight=w)
        
        if not is_valid(seed_assignment, seed_load):
            continue
        
        seed_cost = compute_total_cost(seed_assignment, seed_load)
        
        # Run local search
        ls_assignment, ls_load, ls_cost = local_search(seed_assignment, seed_load, max_iters=200)
        
        if is_valid(ls_assignment, ls_load) and ls_cost < best_cost:
            best_cost = ls_cost
            best_assignment = ls_assignment.copy()
            best_load = ls_load.copy()
    
    # If no valid solution found, use simple greedy
    if best_assignment is None:
        best_assignment, best_load = greedy_seed(open_weight=1.0)
    
    # Final validation
    if not is_valid(best_assignment, best_load):
        # Last resort: simple greedy
        best_assignment = np.zeros(n_customers, dtype=int)
        best_load = np.zeros(n_facilities, dtype=np.int64)
        for i in np.argsort(-demands):
            for f in range(n_facilities):
                if best_load[f] + demands[i] <= capacities[f]:
                    best_assignment[i] = f
                    best_load[f] += demands[i]
                    break
    
    return best_assignment.astype(int)
# EVOLVE-BLOCK-END
