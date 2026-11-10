import numpy as np
import time

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix for ant colony optimization
    of capacitated vehicle routing.
    
    Structural approach:
    1. Power-law base (1/d^1.5) for strong local preference.
    2. Clarke-Wright savings boost for distance-saving merges.
    3. Remaining-capacity potential field: precompute a greedy insertion tour
       to estimate residual vehicle capacity at each node's visit position,
       then modulate edge attractiveness by the ratio of destination demand
       to estimated residual capacity, penalizing edges that risk exceeding
       the vehicle limit and forcing early, compact route formation.
    """
    start_time = time.time()
    deadline = start_time + 250.0  # Safety margin before 300s limit
    
    n = distance_matrix.shape[0]
    demands = np.asarray(demands, dtype=np.float64)
    coords = np.asarray(coordinates, dtype=np.float64)
    cap = float(capacity)
    
    # Ensure finite and positive distance matrix
    dm = np.where(np.isfinite(distance_matrix), distance_matrix, 1e6)
    dm = np.maximum(dm, 1e-9)
    
    # --- Step 1: Power-law Base Heuristic ---
    base = 1.0 / (dm ** 1.5)
    
    # --- Step 2: Clarke-Wright Savings ---
    # Savings(i, j) = dist(0, i) + dist(0, j) - dist(i, j)
    savings = np.zeros((n, n), dtype=np.float64)
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                s = dm[0, i] + dm[0, j] - dm[i, j]
                savings[i, j] = max(s, 0.0)
    
    if savings.max() > 0:
        savings = savings / savings.max()
    
    # Combine base with savings boost
    heuristic = base + 0.8 * savings * base
    
    # --- Step 3: Remaining-Capacity Potential Field ---
    # Precompute a greedy insertion tour to estimate residual capacity
    # at each node's visit position in a typical route.
    
    # Greedy insertion: build a tour by repeatedly inserting the customer
    # with the smallest insertion cost into the current route.
    # This gives us an ordering of customers that approximates a good route.
    
    # Start with the closest customer to the depot
    remaining = set(range(1, n))
    if not remaining:
        # No customers, return base heuristic
        return np.maximum(heuristic, 1e-9)
    
    # Find the closest customer to the depot
    closest_to_depot = min(remaining, key=lambda c: dm[0, c])
    tour = [closest_to_depot]
    remaining.remove(closest_to_depot)
    
    # Greedily insert remaining customers
    while remaining:
        best_cost = float('inf')
        best_customer = None
        best_position = None
        
        for c in remaining:
            # Try inserting c at each position in the current tour
            for pos in range(len(tour)):
                # Cost of inserting c between tour[pos] and tour[pos+1]
                # (or at start/end)
                if pos == 0:
                    # Insert at beginning: depot -> c -> tour[0]
                    cost = dm[0, c] + dm[c, tour[0]] - dm[0, tour[0]]
                elif pos == len(tour) - 1:
                    # Insert at end: tour[-1] -> c -> depot
                    cost = dm[tour[-1], c] + dm[c, 0] - dm[tour[-1], 0]
                else:
                    # Insert in middle: tour[pos] -> c -> tour[pos+1]
                    cost = dm[tour[pos], c] + dm[c, tour[pos + 1]] - dm[tour[pos], tour[pos + 1]]
                
                if cost < best_cost:
                    best_cost = cost
                    best_customer = c
                    best_position = pos
        
        if best_customer is not None:
            tour.insert(best_position, best_customer)
            remaining.remove(best_customer)
    
    # Now we have a tour ordering. Estimate residual capacity at each position.
    # Assume the vehicle starts full (capacity = cap) and depletes as it visits customers.
    # For each customer in the tour, the residual capacity before visiting them
    # is cap minus the sum of demands of all previously visited customers in the tour.
    
    residual_capacity = np.zeros(n)
    residual_capacity[0] = cap  # Depot has full capacity available
    current_load = 0.0
    for node in tour:
        residual_capacity[node] = cap - current_load
        current_load += demands[node]
    
    # For nodes not in the tour (shouldn't happen, but handle gracefully)
    for i in range(1, n):
        if residual_capacity[i] == 0.0:
            residual_capacity[i] = cap * 0.5  # Default to half capacity
    
    # Ensure residual capacity is positive
    residual_capacity = np.maximum(residual_capacity, 1e-9)
    
    # Modulate edge attractiveness by the ratio of destination demand to
    # estimated residual capacity. Edges to high-demand nodes when residual
    # capacity is low should be penalized.
    
    # For edge (i, j), the penalty factor is:
    #   1 / (1 + demands[j] / residual_capacity[j])
    # This is 1 when demands[j] is small relative to residual capacity,
    # and decreases as demands[j] approaches or exceeds residual capacity.
    
    # Compute the penalty for each destination node j
    demand_to_residual_ratio = demands / residual_capacity
    penalty_factor = 1.0 / (1.0 + demand_to_residual_ratio)
    
    # Normalize penalty factor to [0.5, 1.0] range to avoid over-penalization
    pf_min = penalty_factor[1:].min() if n > 1 else 1.0
    pf_max = penalty_factor[1:].max() if n > 1 else 1.0
    if pf_max > pf_min:
        penalty_factor_norm = 0.5 + 0.5 * (penalty_factor - pf_min) / (pf_max - pf_min)
    else:
        penalty_factor_norm = np.ones(n) * 0.75
    
    # Depot gets a neutral penalty factor
    penalty_factor_norm[0] = 1.0
    
    # Apply the penalty factor to the heuristic matrix
    # Each column j is scaled by penalty_factor_norm[j]
    heuristic = heuristic * penalty_factor_norm[np.newaxis, :]
    
    # Ensure all entries are positive
    heuristic = np.maximum(heuristic, 1e-9)
    
    # Check deadline
    if time.time() > deadline:
        return heuristic
    
    return heuristic
