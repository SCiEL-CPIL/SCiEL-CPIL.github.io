"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    """
    Priority score for greedy maximum independent set.
    
    Strategy: Combine static topological features with a simulated
    parallel exclusion process to capture dynamic interference patterns.
    
    The score combines:
    1. Negative own degree (prefer low-degree vertices - cheaper to include)
    2. Average neighbor degree minus own degree (reward blocking hubs)
    3. Selection time from a simulated parallel exclusion process
       (earlier selection = higher priority, captures dynamic interference)
    """
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin before 120s limit
    
    adj = np.asarray(adjacency, dtype=np.float64)
    n = adj.shape[0]
    
    # Degree of each vertex
    degree = adj.sum(axis=1)
    
    # Average degree of neighbors (second-order feature)
    # For vertex i, avg_neighbor_degree[i] = sum_j(adj[i,j] * degree[j]) / degree[i]
    # Handle zero-degree vertices carefully
    neighbor_degree_sum = adj @ degree
    avg_neighbor_degree = np.where(degree > 0, neighbor_degree_sum / np.maximum(degree, 1), 0.0)
    
    # Base score: reward low-degree vertices that are adjacent to high-degree hubs
    # This is the measured effective strategy: avg_neighbor_degree - own_degree
    base_score = avg_neighbor_degree - degree
    
    # Simulate parallel discrete-time exclusion process
    # In each round, select vertices with the fewest active (unselected, non-neighbor-of-selected) neighbors
    # The round number at which a vertex becomes eligible serves as a dynamic priority signal
    
    # We'll simulate a simplified version:
    # Round 0: all vertices are "active"
    # Each round: among active vertices, those with minimum active-neighbor count are "selected" (marked)
    # Then their neighbors become "blocked" (removed from active set)
    # Record the round number for each vertex when it was selected or blocked
    
    selection_time = np.full(n, n, dtype=np.float64)  # Initialize to max (never selected)
    active = np.ones(n, dtype=bool)
    selected = np.zeros(n, dtype=bool)
    blocked = np.zeros(n, dtype=bool)
    
    current_round = 0
    max_rounds = min(n, 200)  # Cap to avoid infinite loops
    
    while active.any() and current_round < max_rounds:
        # Check deadline
        if time.time() > deadline:
            break
        
        # Among active vertices, count active neighbors
        # A vertex's active-neighbor count = number of its neighbors that are still active
        active_neighbor_count = np.zeros(n, dtype=np.float64)
        for i in range(n):
            if active[i]:
                # Count neighbors that are active
                active_neighbor_count[i] = adj[i] @ active.astype(np.float64)
        
        # Find active vertices with minimum active-neighbor count
        min_count = np.inf
        for i in range(n):
            if active[i]:
                if active_neighbor_count[i] < min_count:
                    min_count = active_neighbor_count[i]
        
        # Select all active vertices with this minimum count
        to_select = []
        for i in range(n):
            if active[i] and abs(active_neighbor_count[i] - min_count) < 1e-9:
                to_select.append(i)
        
        if not to_select:
            # Fallback: select the first active vertex
            for i in range(n):
                if active[i]:
                    to_select.append(i)
                    break
        
        # Mark selected vertices
        for i in to_select:
            selected[i] = True
            active[i] = False
            selection_time[i] = float(current_round)
        
        # Block neighbors of selected vertices
        for i in to_select:
            neighbors = np.where(adj[i] > 0)[0]
            for j in neighbors:
                if active[j]:
                    active[j] = False
                    blocked[j] = True
                    selection_time[j] = float(current_round)  # Blocked at this round
        
        current_round += 1
    
    # For vertices that were never selected or blocked (shouldn't happen, but safety)
    # Give them a high selection_time (low priority)
    never_processed = ~selected & ~blocked
    if never_processed.any():
        selection_time[never_processed] = float(max_rounds)
    
    # Combine scores:
    # - base_score rewards low-degree vertices adjacent to hubs
    # - -selection_time rewards vertices that were selected/blocked earlier in the exclusion process
    #   (earlier = lower time = higher priority)
    # 
    # We want higher score = higher priority
    # base_score: higher is better (already oriented correctly)
    # selection_time: lower is better, so we negate it
    
    # Normalize selection_time to be comparable in scale to base_score
    # selection_time ranges from 0 to max_rounds
    # base_score can be negative or positive
    
    # Weight the dynamic component
    dynamic_score = -selection_time
    
    # Combine: primary weight on the measured effective base score, secondary on dynamic
    # The base score (avg_neighbor_degree - degree) is the measured winner
    # The dynamic score captures interference patterns
    
    # Use a weighted combination
    # Scale dynamic_score to be in a reasonable range relative to base_score
    base_range = np.ptp(base_score) if n > 0 else 1.0
    dyn_range = np.ptp(dynamic_score) if n > 0 else 1.0
    
    if base_range > 1e-9:
        base_normalized = (base_score - base_score.min()) / base_range
    else:
        base_normalized = np.zeros(n)
    
    if dyn_range > 1e-9:
        dyn_normalized = (dynamic_score - dynamic_score.min()) / dyn_range
    else:
        dyn_normalized = np.zeros(n)
    
    # Weight: 70% base (measured effective), 30% dynamic (new insight)
    combined_score = 0.7 * base_normalized + 0.3 * dyn_normalized
    
    return combined_score
# EVOLVE-BLOCK-END
