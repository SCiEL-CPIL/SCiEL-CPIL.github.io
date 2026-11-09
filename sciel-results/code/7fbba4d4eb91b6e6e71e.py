import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s limit
    
    n = adjacency.shape[0]
    if n == 0:
        return np.array([])
        
    # Precompute neighbor lists and degrees
    adj_bool = adjacency.astype(bool)
    neighbors = [np.where(adj_bool[i])[0] for i in range(n)]
    degrees = adjacency.sum(axis=1).astype(float)
    
    def greedy_mis_size(order):
        """Fast greedy MIS size computation."""
        blocked = np.zeros(n, dtype=bool)
        count = 0
        for v in order:
            if not blocked[v]:
                count += 1
                blocked[v] = True
                nbrs = neighbors[v]
                if len(nbrs) > 0:
                    blocked[nbrs] = True
        return count

    # --- Initial Heuristic Orders ---
    
    # 1. Minimum Degree
    order_min_deg = list(np.argsort(degrees))
    
    # 2. Core-Number / Peeling Order
    # Vertices with lower core number are more peripheral.
    # We compute a proxy: iteratively remove min-degree vertices.
    # The layer in which a vertex is removed is its "peeling layer".
    # Lower layer = more peripheral = higher priority (selected earlier).
    # We assign score = -layer.
    
    def compute_peeling_order():
        remaining = np.ones(n, dtype=bool)
        deg = degrees.copy()
        layers = np.zeros(n, dtype=int)
        order = [None] * n
        current_layer = 0
        remaining_count = n
        
        while remaining_count > 0:
            # Find min degree among remaining
            min_deg = np.inf
            min_nodes = []
            for i in range(n):
                if remaining[i]:
                    if deg[i] < min_deg:
                        min_deg = deg[i]
                        min_nodes = [i]
                    elif deg[i] == min_deg:
                        min_nodes.append(i)
                        
            if not min_nodes:
                break
                
            # Remove all nodes with min_deg
            for i in min_nodes:
                if not remaining[i]:
                    continue
                remaining[i] = False
                layers[i] = current_layer
                order[i] = i # Placeholder, we'll sort later
                remaining_count -= 1
                # Update degrees of neighbors
                for u in neighbors[i]:
                    if remaining[u]:
                        deg[u] -= 1
                        
            current_layer += 1
            
        # Sort by layer (ascending), then by original degree (ascending) for stability
        sorted_indices = np.argsort(layers)
        # Ties in layer are broken by index, which is fine, or we can refine
        return list(sorted_indices)

    order_peel = compute_peeling_order()
    
    # 3. Average Neighbor Degree (higher avg neighbor degree -> more "central" -> lower priority? 
    #    Actually, for MIS, we often prefer low degree. 
    #    But if neighbors are high degree, they are "expensive" to block?
    #    Let's stick to the previous finding: avg_nd - deg was good.
    #    So score = avg_nd - deg.
    #    Let's compute avg_nd.
    sum_neighbor_degrees = np.zeros(n)
    for i in range(n):
        nbrs = neighbors[i]
        if len(nbrs) > 0:
            sum_neighbor_degrees[i] = degrees[nbrs].sum()
            
    avg_neighbor_degree = np.zeros(n)
    nonzero_mask = degrees > 0
    avg_neighbor_degree[nonzero_mask] = sum_neighbor_degrees[nonzero_mask] / degrees[nonzero_mask]
    
    # Score: avg_nd - deg
    score_avg_nd_minus_deg = avg_neighbor_degree - degrees
    order_avg_nd = list(np.argsort(-score_avg_nd_minus_deg)) # Descending score
    
    # 4. Random
    rng = np.random.default_rng(42)
    order_random = list(rng.permutation(n))
    
    # 5. Max Degree (usually bad, but maybe for some structures?)
    order_max_deg = list(np.argsort(-degrees))
    
    # 6. Core-Number based score: -layer
    # We already have order_peel.
    
    # --- Local Search ---
    
    best_order = order_min_deg[:]
    best_size = greedy_mis_size(best_order)
    
    # Evaluate initial heuristics
    for init_order in [order_peel, order_avg_nd, order_random, order_max_deg]:
        sz = greedy_mis_size(init_order)
        if sz > best_size:
            best_size = sz
            best_order = init_order[:]
            
    # Hill Climbing with Restarts
    # We will run multiple restarts.
    # Each restart starts from a slightly perturbed version of the current best or a random heuristic.
    
    num_restarts = 10
    steps_per_restart = 20000
    
    for r in range(num_restarts):
        if time.time() > deadline - 15:
            break
            
        # Choose starting order for this restart
        if r == 0:
            current_order = best_order[:]
        elif r == 1:
            current_order = order_peel[:]
        elif r == 2:
            current_order = order_avg_nd[:]
        elif r == 3:
            current_order = order_min_deg[:]
        elif r == 4:
            current_order = order_random[:]
        else:
            # Noisy version of best order
            # We can't easily "noisy" a permutation directly without indices.
            # Let's just take a random permutation or a mix.
            # For simplicity, use random.
            current_order = list(rng.permutation(n))
            
        current_size = greedy_mis_size(current_order)
        
        if current_size > best_size:
            best_size = current_size
            best_order = current_order[:]
            
        # Local Search
        for step in range(steps_per_restart):
            if time.time() > deadline - 5:
                break
                
            # Pick two random positions to swap
            i, j = rng.integers(0, n, 2)
            if i == j:
                continue
                
            current_order[i], current_order[j] = current_order[j], current_order[i]
            
            new_size = greedy_mis_size(current_order)
            
            if new_size >= current_size:
                current_size = new_size
                if current_size > best_size:
                    best_size = current_size
                    best_order = current_order[:]
            else:
                # Revert
                current_order[i], current_order[j] = current_order[j], current_order[i]
                
    # --- Final Polish: Try to improve by inserting vertices? ---
    # The greedy decoder is fixed. We just need to return the scores that induce the best_order.
    
    scores = np.zeros(n)
    for rank, v in enumerate(best_order):
        scores[v] = n - rank
        
    return scores
