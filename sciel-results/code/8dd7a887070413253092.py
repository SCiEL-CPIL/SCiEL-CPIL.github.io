import numpy as np
import time

def priority(adjacency):
    """
    Assigns priority scores to vertices for a greedy Maximum Independent Set decoder.
    Strategy: Prioritize vertices with low Core Number (peripheral/sparse parts of the graph)
    and, as a tie-breaker, low Degree.
    This ensures that vertices in dense cores (where independent sets are small) are picked
    last, allowing the greedy algorithm to maximize the set size in the sparse "skin" of the graph.
    """
    start_time = time.time()
    # Safety margin: 150s limit, keep 10s buffer
    deadline = start_time + 140.0
    
    # Convert to numpy array if not already
    if not isinstance(adjacency, np.ndarray):
        adjacency = np.array(adjacency, dtype=bool)
    
    n = adjacency.shape[0]
    
    # Calculate degrees
    degrees = adjacency.sum(axis=1).astype(int)
    
    # Calculate Core Numbers
    # Core number of a vertex v is the largest k such that v is in the k-core.
    # Algorithm:
    # 1. Create a list of vertices sorted by degree.
    # 2. Maintain a "current degree" for each vertex.
    # 3. Iterate k from 0 upwards? No, standard peeling.
    
    # Efficient O(m) or O(m log n) core number calculation
    # We can use a bucket sort approach.
    
    # Max degree
    max_deg = np.max(degrees) if n > 0 else 0
    
    # Initialize core numbers to 0
    core_numbers = np.zeros(n, dtype=int)
    
    # Current degrees (will be modified)
    cur_degrees = degrees.copy()
    
    # Bucket vertices by their current degree
    # buckets[d] contains list of vertices with current degree d
    # Since max_deg is small for N=200, this is fine.
    # Actually, degrees can be up to 199.
    
    # We need to process vertices in increasing order of degree.
    # We can use a simple list and sort, or buckets.
    # Given N=200, sorting is very fast.
    
    # Vertices to process
    # We will maintain a set of "active" vertices.
    # Initially all are active.
    
    # To find core numbers, we can peel layers.
    # Layer 0: Isolated nodes? No, 1-core is min degree >= 1.
    # Standard definition: k-core is maximal subgraph with min degree >= k.
    # Core number is the max k.
    
    # Peeling algorithm:
    # 1. Find all vertices with degree < 1. Assign core number 0. Remove them.
    # 2. Update degrees of neighbors.
    # 3. Find all vertices with degree < 2 (in remaining graph). Assign core number 1. Remove them.
    # 4. Update degrees.
    # 5. Find all vertices with degree < 3. Assign core number 2. Remove them.
    # ...
    
    # This is efficient enough for N=200.
    
    # Adjacency list for faster neighbor lookup
    adj_list = [np.where(adjacency[i])[0] for i in range(n)]
    
    # Active flags
    active = np.ones(n, dtype=bool)
    
    # We will iterate k from 1 to max_deg
    # For each k, we remove all active vertices with cur_degree < k
    # But wait, if we remove vertices with deg < k, their core number is k-1?
    # Yes. If a vertex is removed when we are looking for the k-core (min deg >= k),
    # it means it failed the condition for k, so it was in the (k-1)-core but not k-core.
    # So core number is k-1.
    
    # Initialize
    # Note: Vertices with initial degree 0 have core number 0.
    
    # We need to be careful with the loop.
    # Let's use a queue-based approach for efficiency, or just simple loops since N is small.
    
    # Simple loop approach:
    # current_k = 1
    # while there are active vertices:
    #   find active vertices with cur_degree < current_k
    #   if none, break? No, if none, then all remaining have deg >= current_k.
    #   if some, assign core number = current_k - 1 to them.
    #   remove them, update neighbors' cur_degree.
    #   if no vertices were removed, current_k += 1?
    #   Actually, if no vertices have degree < current_k, then the min degree of the remaining graph is >= current_k.
    #   So the remaining vertices have core number >= current_k.
    #   We can then increment current_k and check again.
    
    # However, if we remove vertices, degrees drop, so we might need to re-check for the same k?
    # No, if we remove vertices with deg < k, the neighbors' degrees drop.
    # They might now have deg < k. So we must repeat the "find deg < k" step until no such vertices exist.
    # Once no active vertex has deg < k, we know all remaining have core number >= k.
    # Then we increment k.
    
    k = 1
    while np.any(active):
        # Check time
        if time.time() > deadline:
            # Fallback to simple degree if core calc takes too long (unlikely for N=200)
            break
        
        # Find active vertices with cur_degree < k
        # Mask: active AND cur_degree < k
        mask = active & (cur_degrees < k)
        
        if not np.any(mask):
            # All remaining active vertices have degree >= k
            # So they are in the k-core (at least)
            # We can increment k
            k += 1
            continue
        
        # Assign core number k-1 to these vertices
        core_numbers[mask] = k - 1
        
        # Remove them
        active[mask] = False
        
        # Update degrees of neighbors
        # For each removed vertex, decrement degree of its active neighbors
        removed_indices = np.where(mask)[0]
        for v in removed_indices:
            neighbors = adj_list[v]
            # Only update active neighbors
            active_neighbors = neighbors[active[neighbors]]
            if len(active_neighbors) > 0:
                cur_degrees[active_neighbors] -= 1
        
        # Loop continues to catch any new vertices that dropped below k due to updates
    
    # Any remaining active vertices have core number >= k.
    # Actually, the loop terminates when active is empty or we hit deadline.
    # If active is not empty and we exited due to deadline or logic, 
    # the remaining vertices have core number at least k.
    # But usually the loop empties all active vertices.
    # If we break due to deadline, we assign remaining to current k-1? 
    # Or just leave as is? 
    # For N=200, it will finish instantly.
    
    # Construct score
    # Score = - (CoreNumber * LargeConst + Degree)
    # Higher score is picked first.
    # Low Core Number -> High Score.
    # Low Degree -> High Score.
    
    # Use 10000 as separator, max degree is 199.
    score = -(core_numbers * 10000 + degrees).astype(float)
    
    return score
