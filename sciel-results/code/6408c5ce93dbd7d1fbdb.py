import numpy as np
import time
import heapq

def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 142.0  # Safety margin before 150s limit
    
    n = adjacency.shape[0]
    if n == 0:
        return np.array([])
    
    # Precompute neighbor lists as integer arrays for fast access
    neighbors = [np.where(adjacency[i])[0] for i in range(n)]
    degrees = adjacency.sum(axis=1).astype(int)
    
    def greedy_mis_size(order):
        """
        order: list of vertices in the order they should be considered (first is highest priority)
        Returns the size of the independent set.
        """
        selected = np.zeros(n, dtype=bool)
        count = 0
        for v in order:
            if not selected[v]:
                selected[v] = True
                count += 1
                if len(neighbors[v]) > 0:
                    selected[neighbors[v]] = True
        return count
    
    # Compute k-core decomposition to get removal order
    remaining = np.ones(n, dtype=bool)
    current_degrees = degrees.copy()
    removal_time = np.zeros(n, dtype=int)
    t = 0
    
    # Use BFS-like peeling with a priority queue (min-heap by degree)
    heap = [(int(current_degrees[i]), i) for i in range(n)]
    heapq.heapify(heap)
    
    while heap:
        deg, v = heapq.heappop(heap)
        
        if not remaining[v]:
            continue
        
        if deg != current_degrees[v]:
            continue
        
        removal_time[v] = t
        t += 1
        
        # Update degrees of neighbors
        for u in neighbors[v]:
            if remaining[u]:
                current_degrees[u] -= 1
                heapq.heappush(heap, (int(current_degrees[u]), u))
        
        remaining[v] = False
    
    # Normalize degrees and removal times
    deg_norm = degrees / (degrees.max() + 1e-9)
    time_norm = removal_time / (removal_time.max() + 1e-9)
    
    rng = np.random.default_rng(42)
    
    # Generate multiple candidate orderings using different strategies
    def generate_ordering(strategy_idx, r):
        if strategy_idx == 0:
            # Core-based order: low removal time first (peripheral vertices)
            return list(np.lexsort((degrees, removal_time)))
        elif strategy_idx == 1:
            # Pure min degree
            return list(np.argsort(degrees))
        elif strategy_idx == 2:
            # Random
            return list(rng.permutation(n))
        elif strategy_idx == 3:
            # Max degree (control)
            return list(np.argsort(-degrees))
        elif strategy_idx == 4:
            # Noisy degree
            noisy_deg = degrees.astype(float) + rng.normal(0, 1, n)
            return list(np.argsort(noisy_deg))
        elif strategy_idx == 5:
            # Noisy core time
            noisy_time = removal_time.astype(float) + rng.normal(0, 1, n)
            return list(np.argsort(noisy_time))
        elif strategy_idx == 6:
            # Combined noisy
            combined = 0.5 * deg_norm + 0.5 * time_norm
            noisy_combined = combined + rng.normal(0, 0.1, n)
            return list(np.argsort(noisy_combined))
        elif strategy_idx == 7:
            # Weighted: emphasize low degree more
            weighted = 0.7 * deg_norm + 0.3 * time_norm
            noisy_weighted = weighted + rng.normal(0, 0.05, n)
            return list(np.argsort(noisy_weighted))
        elif strategy_idx == 8:
            # Weighted: emphasize core time more
            weighted = 0.3 * deg_norm + 0.7 * time_norm
            noisy_weighted = weighted + rng.normal(0, 0.05, n)
            return list(np.argsort(noisy_weighted))
        elif strategy_idx == 9:
            # Inverse: try higher degree first
            return list(np.lexsort((-degrees, -removal_time)))
        elif strategy_idx == 10:
            # DSATUR-inspired: color by degree, then order by color class size
            # Assign colors greedily based on degree
            colors = np.zeros(n, dtype=int)
            color_classes = {}
            for v in np.argsort(degrees):  # Process low degree first
                # Find the smallest color not used by neighbors
                used_colors = set()
                for u in neighbors[v]:
                    if colors[u] > 0:
                        used_colors.add(colors[u])
                c = 1
                while c in used_colors:
                    c += 1
                colors[v] = c
                if c not in color_classes:
                    color_classes[c] = []
                color_classes[c].append(v)
            
            # Order vertices by their color class size (larger classes first)
            # This approximates finding a large independent set
            class_sizes = {c: len(vs) for c, vs in color_classes.items()}
            # Sort vertices: first by class size descending, then by degree ascending
            v_class_size = np.array([class_sizes[colors[v]] for v in range(n)])
            return list(np.lexsort((degrees, -v_class_size)))
        elif strategy_idx == 11:
            # Distance-2 degree: consider neighbors of neighbors
            dist2_deg = np.zeros(n, dtype=int)
            for v in range(n):
                for u in neighbors[v]:
                    for w in neighbors[u]:
                        if w != v and not adjacency[v, w]:
                            dist2_deg[v] += 1
            combined = 0.5 * deg_norm + 0.5 * (dist2_deg / (dist2_deg.max() + 1e-9))
            return list(np.argsort(combined))
        elif strategy_idx == 12:
            # More aggressive noise
            combined = 0.5 * deg_norm + 0.5 * time_norm
            noisy_combined = combined + rng.normal(0, 0.3, n)
            return list(np.argsort(noisy_combined))
        elif strategy_idx == 13:
            # Shuffled best order (will be filled in later)
            return None
        else:
            # Fallback: random
            return list(rng.permutation(n))
    
    # Start with a few deterministic orderings
    best_order = None
    best_size = 0
    
    # Try initial strategies
    initial_strategies = [0, 1, 4, 6, 7, 8, 10, 11]
    
    for strat in initial_strategies:
        if time.time() > deadline - 20:
            break
        order = generate_ordering(strat, 0)
        if order is None:
            continue
        size = greedy_mis_size(order)
        if size > best_size:
            best_size = size
            best_order = order[:]
    
    # Multiple restarts with local search
    num_restarts = 15
    steps_per_restart = 3000
    
    for r in range(num_restarts):
        if time.time() > deadline - 15:
            break
        
        # Choose a strategy for this restart
        strat = r % 14
        if strat == 13:
            order = best_order[:]
            rng.shuffle(order)
        else:
            order = generate_ordering(strat, r)
            if order is None:
                continue
        
        current_size = greedy_mis_size(order)
        
        if current_size > best_size:
            best_size = current_size
            best_order = order[:]
        
        # Local Search Phase: Hill Climbing with random swaps
        for step in range(steps_per_restart):
            if time.time() > deadline - 10:
                break
                
            # Pick two random indices to swap
            i, j = rng.integers(0, n, 2)
            if i == j:
                continue
                
            order[i], order[j] = order[j], order[i]
            
            new_size = greedy_mis_size(order)
            
            if new_size >= current_size:
                current_size = new_size
                if current_size > best_size:
                    best_size = current_size
                    best_order = order[:]
            else:
                # Revert
                order[i], order[j] = order[j], order[i]
                
    # Final polish: try to improve best_order with targeted swaps
    if time.time() < deadline - 5:
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(n):
                if time.time() > deadline - 5:
                    break
                for j in range(i + 1, n):
                    if time.time() > deadline - 5:
                        break
                    best_order[i], best_order[j] = best_order[j], best_order[i]
                    new_size = greedy_mis_size(best_order)
                    if new_size > best_size:
                        best_size = new_size
                        improved = True
                    else:
                        best_order[i], best_order[j] = best_order[j], best_order[i]
                
    # Convert best_order to scores
    # The decoder sorts by descending score.
    # So the first element in best_order should have the highest score.
    scores = np.zeros(n)
    for rank, v in enumerate(best_order):
        scores[v] = n - rank
        
    return scores
