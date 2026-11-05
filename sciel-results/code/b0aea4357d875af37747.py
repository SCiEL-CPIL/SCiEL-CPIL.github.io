import numpy as np
import time

def priority(adjacency):
    """
    Assigns priority scores to vertices for a greedy Maximum Independent Set decoder.
    
    Strategy: Large Neighborhood Search (LNS) on Vertex Permutations
    1. Initialize with a heuristic ordering based on degree and neighbor density.
    2. Directly optimize the permutation (ordering) rather than scores.
    3. Use LNS: remove a random subset of vertices, re-insert them at optimal/random positions.
    4. Use Simulated Annealing acceptance criteria to escape local optima.
    5. Time-bounded search with early termination.
    """
    start_time = time.time()
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    # Precompute neighbor lists for fast simulation
    # Using lists of lists for O(1) access to neighbors in the simulation loop
    neighbors_list = [np.where(adj[i])[0].tolist() for i in range(n)]
    
    # 1. Initial Heuristic Ordering
    # Lower degree and lower neighbor density should come first (higher priority)
    deg = adj.sum(axis=1).astype(float)
    neighbor_deg_sum = adj @ deg
    safe_deg = deg + 1.0
    cluster_density = neighbor_deg_sum / safe_deg
    # We want vertices with low deg and low cluster_density to be picked early.
    # So score = -(deg + alpha * cluster_density)
    # The greedy decoder picks by descending score.
    # So initial order is argsort of -score, which is argsort of (deg + alpha * cluster_density)
    # Wait, the decoder picks descending score.
    # If score = -deg, high score = low deg.
    # So we want low deg vertices to have high scores.
    # Let's just define an initial order and convert to scores at the end.
    
    # Initial order: sort by (deg + 0.3 * cluster_density) ascending
    # This puts low complexity vertices first.
    init_key = deg + 0.3 * cluster_density
    current_order = np.argsort(init_key, kind='stable')
    
    def simulate_order(order):
        """
        Simulate the greedy decoder given a specific order of vertices.
        Returns the size of the independent set.
        """
        active = np.ones(n, dtype=bool)
        count = 0
        for v in order:
            if active[v]:
                count += 1
                active[v] = False
                nbrs = neighbors_list[v]
                if nbrs:
                    active[nbrs] = False
        return count
    
    best_val = simulate_order(current_order)
    best_order = current_order.copy()
    current_val = best_val
    
    # 2. LNS with Simulated Annealing
    # Temperature schedule
    T0 = 5.0
    T_end = 0.1
    cooling = 0.9995
    
    T = T0
    rng = np.random.default_rng(42)
    
    # Parameters for LNS
    # Remove k vertices, try to re-insert them
    # k should be small enough to be fast, large enough to escape local optima
    # Let's try a dynamic k or fixed k.
    # n=200. Simulation is O(n^2) worst case, but usually faster.
    # Let's estimate cost of one simulation.
    # 200 vertices. Inner loop checks active and updates neighbors.
    # Average degree in ER(200, p) is maybe 10-50.
    # 200 * 20 = 4000 ops. Very fast.
    # We can do many iterations.
    
    max_time = 115.0
    
    while time.time() - start_time < max_time:
        # LNS Move:
        # 1. Pick a random subset of vertices to "remove" from the current order
        # 2. Generate a new order by inserting them back
        # 3. Evaluate
        # 4. Accept if better or with probability exp(-delta/T)
        
        # Choose subset size k
        # Start with small k, maybe increase occasionally?
        # Or just random k between 5 and 20
        k = rng.integers(5, 21)
        
        # Pick k random indices in the current order
        # These are the vertices we will move
        move_indices = rng.choice(n, size=k, replace=False)
        
        # Create a list of vertices to move
        # We will remove them from current_order and insert them elsewhere
        # To make it efficient, let's work with a list of vertex IDs
        
        # Current order is an array of vertex IDs
        # Let's create a new order
        # 1. Filter out the move_indices vertices from current_order
        #    Note: move_indices are indices in the range [0, n), but they are vertex IDs?
        #    No, rng.choice(n) gives vertex IDs.
        #    But current_order is a permutation of vertex IDs.
        #    So we need to find where these vertices are in current_order?
        #    Actually, it's easier to just shuffle a subset of the list.
        
        # Alternative LNS:
        # 1. Copy current_order
        # 2. Pick k random positions in the list
        # 3. Extract those vertices
        # 4. Remove them
        # 5. Insert them at random positions
        
        new_order = current_order.copy()
        
        # Pick k random positions
        pos_indices = rng.choice(n, size=k, replace=False)
        # Sort positions to remove from end to start to keep indices valid?
        # Or just use a mask.
        
        # Let's use a mask approach for removal and then re-insertion
        # It's easier to just create a new list.
        
        # Extract vertices at pos_indices
        moved_vertices = new_order[pos_indices]
        
        # Create mask of positions to keep
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[pos_indices] = False
        remaining_vertices = new_order[keep_mask]
        
        # Now we have remaining_vertices (length n-k) and moved_vertices (length k)
        # We want to insert moved_vertices into remaining_vertices at random positions
        
        # Strategy: 
        # 1. Shuffle moved_vertices
        # 2. Insert each one at a random position in the remaining list
        #    This is O(n*k) which is fine for n=200, k=20 -> 4000 ops.
        
        # Actually, a better LNS move:
        # Just take the k moved vertices and shuffle their relative order,
        # and also shuffle their positions with respect to the rest?
        # A common LNS is "remove and re-insert at best position".
        # But finding best position is expensive (greedy search for each insertion).
        # Random re-insertion is cheaper and with SA can explore well.
        
        # Let's try: 
        # 1. Shuffle moved_vertices
        rng.shuffle(moved_vertices)
        
        # 2. Insert them one by one at random positions
        #    Start with remaining_vertices
        #    For each v in moved_vertices:
        #       pos = rng.integers(0, len(current_list) + 1)
        #       insert v at pos
        
        # This creates a new permutation.
        # Is it a good neighborhood?
        # It allows moving vertices far apart.
        
        # Implementation:
        # Using lists for insertion is O(n). Total O(n*k).
        # n=200, k=20 -> 4000 operations. Very fast.
        
        # Let's do it
        new_list = list(remaining_vertices)
        for v in moved_vertices:
            pos = rng.integers(0, len(new_list) + 1)
            new_list.insert(pos, v)
        
        new_order = np.array(new_list)
        
        # Evaluate
        new_val = simulate_order(new_order)
        
        # Acceptance
        delta = new_val - current_val
        if delta > 0:
            current_order = new_order
            current_val = new_val
            if new_val > best_val:
                best_val = new_val
                best_order = new_order.copy()
        else:
            # Accept with probability exp(-delta/T)
            # delta is negative, so -delta is positive
            # exp(positive/T) > 1, so prob > 1? No.
            # Standard SA: P = exp(-|delta|/T) = exp(delta/T) where delta < 0.
            if rng.random() < np.exp(delta / T):
                current_order = new_order
                current_val = new_val
        
        # Cool down
        T *= cooling
        if T < T_end:
            # Restart with higher T to escape deep local optima
            T = T0 * 0.5
            # Maybe shuffle current order slightly? Or just keep it.
            # Keeping it is fine, SA will explore.
            
    # 3. Convert best_order to scores
    # The greedy decoder sorts by descending score.
    # So the vertex at index 0 in best_order should have the highest score.
    # The vertex at index n-1 should have the lowest score.
    scores = np.zeros(n)
    # Assign scores such that argsort(-scores) == best_order
    # If best_order[0] is v0, v0 should have the highest score.
    # Let's assign scores based on rank.
    # Rank 0 (first in order) -> Score n
    # Rank n-1 (last in order) -> Score 1
    ranks = np.zeros(n, dtype=int)
    for i, v in enumerate(best_order):
        ranks[v] = i
    scores = n - ranks
    
    return scores
