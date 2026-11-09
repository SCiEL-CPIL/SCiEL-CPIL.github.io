import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n = adjacency.shape[0]
    if n == 0:
        return np.array([])
    
    # Precompute neighbor lists
    neighbors = [np.where(adjacency[i])[0] for i in range(n)]
    degrees = adjacency.sum(axis=1).astype(float)
    
    # Compute Core Numbers (k-core decomposition)
    def compute_core_numbers(adj, degs):
        n = len(degs)
        core = np.zeros(n, dtype=int)
        import heapq
        remaining_deg = degs.astype(int).copy()
        removed = np.zeros(n, dtype=bool)
        heap = [(int(remaining_deg[i]), i) for i in range(n)]
        heapq.heapify(heap)
        
        while heap:
            d, v = heapq.heappop(heap)
            if removed[v]:
                continue
            core[v] = d
            removed[v] = True
            for u in neighbors[v]:
                if not removed[u]:
                    remaining_deg[u] -= 1
                    heapq.heappush(heap, (int(remaining_deg[u]), u))
        
        return core

    core_numbers = compute_core_numbers(adjacency, degrees)
    
    # Fast greedy MIS evaluation
    def greedy_mis_size(order):
        selected = np.zeros(n, dtype=bool)
        count = 0
        for v in order:
            if not selected[v]:
                selected[v] = True
                count += 1
                if neighbors[v].size > 0:
                    selected[neighbors[v]] = True
        return count
    
    rng = np.random.default_rng(42)
    
    best_size = 0
    best_order = None
    
    def update_best(order, size):
        nonlocal best_size, best_order
        if size > best_size:
            best_size = size
            best_order = order[:]
    
    # Strategy: Model as Maximum Weight Independent Set with dynamic layer-based scores
    # Iteratively select vertex with max "independent potential" (remaining degree),
    # remove it and neighbors, assign scores proportional to removal layer.
    # This creates a priority order where vertices removed early (high potential) get high scores.
    
    def compute_layer_scores():
        """
        Compute scores based on iterative removal layers.
        Each layer removes the vertex with max remaining degree and its neighbors.
        Vertices removed in earlier layers get higher scores.
        """
        remaining = np.ones(n, dtype=bool)
        remaining_deg = degrees.copy()
        scores = np.zeros(n)
        layer = 0
        max_layer = n
        
        while np.sum(remaining) > 0 and layer < max_layer:
            # Find remaining vertices
            rem_indices = np.where(remaining)[0]
            if len(rem_indices) == 0:
                break
            
            # Select vertex with max remaining degree among remaining
            rem_deg = remaining_deg[rem_indices]
            best_local = np.argmax(rem_deg)
            best_v = rem_indices[best_local]
            
            # Remove best_v and its neighbors
            to_remove = set([best_v])
            for u in neighbors[best_v]:
                if remaining[u]:
                    to_remove.add(u)
            
            # Assign scores: earlier layers get higher scores
            # Score = (max_layer - layer) to make early removals have high priority
            for v in to_remove:
                if remaining[v]:
                    scores[v] = max_layer - layer
                    remaining[v] = False
                    # Update remaining degrees of neighbors
                    for u in neighbors[v]:
                        if remaining[u]:
                            remaining_deg[u] -= 1
            
            layer += 1
        
        return scores
    
    # Generate initial orderings
    initial_orders = []
    
    # 1. Layer-based scores (new strategy)
    layer_scores = compute_layer_scores()
    initial_orders.append(np.argsort(-layer_scores))
    
    # 2. Min Degree
    initial_orders.append(np.argsort(degrees))
    
    # 3. Min Core Number
    initial_orders.append(np.argsort(core_numbers))
    
    # 4. Min (Core Number, Degree)
    score_cd = core_numbers * n + degrees
    initial_orders.append(np.argsort(score_cd))
    
    # 5. Min (Degree, Core Number)
    score_dc = degrees * n + core_numbers
    initial_orders.append(np.argsort(score_dc))
    
    # 6. Random
    initial_orders.append(rng.permutation(n))
    
    # 7. Noisy Min Degree
    noisy_deg = degrees + rng.normal(0, 1, n)
    initial_orders.append(np.argsort(noisy_deg))
    
    # 8. Noisy Min Core
    noisy_core = core_numbers + rng.normal(0, 1, n)
    initial_orders.append(np.argsort(noisy_core))
    
    # 9. Max "Independence Potential" (Inverse Degree)
    potential = 1.0 / (degrees + 1.0)
    initial_orders.append(np.argsort(-potential))
    
    # 10. Combination: Layer + Core + Degree
    combo_score = layer_scores * 10 + core_numbers * 5 + (n - degrees)
    initial_orders.append(np.argsort(-combo_score))
    
    # Evaluate initial orders
    for order in initial_orders:
        size = greedy_mis_size(order)
        update_best(order, size)
    
    # Local Search: Simulated Annealing
    # Sort initial orders by their greedy size
    evaluated = [(greedy_mis_size(o), o) for o in initial_orders]
    evaluated.sort(key=lambda x: -x[0])
    
    # Pick top 4 distinct-ish initial orders for SA
    sa_candidates = []
    seen_hashes = set()
    for size, order in evaluated:
        h = tuple(order[:10])
        if h not in seen_hashes:
            sa_candidates.append(order)
            seen_hashes.add(h)
        if len(sa_candidates) >= 4:
            break
    
    if not sa_candidates:
        sa_candidates = [initial_orders[0]]
        
    steps_per_restart = 10000
    
    for r, start_order in enumerate(sa_candidates):
        if time.time() > deadline - 5:
            break
            
        order = start_order.copy()
        current_size = greedy_mis_size(order)
        update_best(order, current_size)
        
        # Simulated Annealing
        initial_temp = 2.0
        cooling_rate = 0.9995
        
        temperature = initial_temp
        for step in range(steps_per_restart):
            if time.time() > deadline - 2:
                break
                
            temperature *= cooling_rate
            
            # Pick two random indices to swap
            i, j = rng.integers(0, n, 2)
            if i == j:
                continue
                
            order[i], order[j] = order[j], order[i]
            
            new_size = greedy_mis_size(order)
            
            delta = new_size - current_size
            
            if delta >= 0:
                current_size = new_size
                update_best(order, current_size)
            else:
                if temperature > 1e-6:
                    prob = np.exp(delta / temperature)
                    if rng.random() < prob:
                        current_size = new_size
                    else:
                        order[i], order[j] = order[j], order[i]
                else:
                    order[i], order[j] = order[j], order[i]
                    
    # Convert best_order to scores
    if best_order is None:
        scores = np.zeros(n)
        order = np.argsort(degrees)
        for rank, v in enumerate(order):
            scores[v] = n - rank
        return scores
        
    scores = np.zeros(n)
    for rank, v in enumerate(best_order):
        scores[v] = n - rank
        
    return scores
