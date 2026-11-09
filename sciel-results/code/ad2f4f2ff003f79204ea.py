import numpy as np
import heapq
import time
import random

def priority(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([])
    
    # Precompute neighbors for fast evaluation
    # adjacency is a bool matrix
    neighbors = [np.where(adjacency[i, :])[0] for i in range(n)]
    
    # 1. Initial Order: Min-Degree Peeling
    degs = adjacency.sum(axis=1).astype(int)
    heap = [(degs[i], i) for i in range(n)]
    heapq.heapify(heap)
    
    removed = [False] * n
    order = []
    
    while heap:
        d, v = heapq.heappop(heap)
        if removed[v]:
            continue
        if degs[v] != d:
            continue
            
        removed[v] = True
        order.append(v)
        
        for u in neighbors[v]:
            if not removed[u]:
                degs[u] -= 1
                heapq.heappush(heap, (degs[u], u))
                
    # Convert order to a list of vertices
    # order is currently in the order they were removed (min degree first).
    # The greedy decoder picks in *descending* score.
    # So if we want the greedy decoder to pick the min-degree vertices first,
    # we need to assign HIGH scores to the vertices that appear EARLY in this peeling order.
    # Peeling order: [v1, v2, ..., vn] where v1 is min-degree initially.
    # We want v1 to be picked first. So score(v1) > score(v2) > ...
    # So priority[v1] = n, priority[v2] = n-1, ...
    
    initial_priority = np.zeros(n)
    for idx, v in enumerate(order):
        initial_priority[v] = n - idx
        
    # 2. Local Search on the Ordering
    # We represent the state as the priority vector, or the order list.
    # It's easier to work with the order list.
    # Current order: list of vertices such that priority[v] is high for early elements.
    # Actually, the evaluator sorts by score descending.
    # So the sequence of vertices considered is:
    # sorted_vertices = np.argsort(initial_priority)[::-1]
    
    current_order = np.argsort(initial_priority)[::-1]
    
    def evaluate(order_arr):
        # order_arr is the sequence of vertices to consider
        selected = 0
        blocked = np.zeros(n, dtype=bool)
        for v in order_arr:
            if not blocked[v]:
                selected += 1
                blocked[v] = True
                # Mark neighbors as blocked
                # neighbors[v] is an array of indices
                if len(neighbors[v]) > 0:
                    blocked[neighbors[v]] = True
        return selected

    best_score = evaluate(current_order)
    best_order = current_order.copy()
    
    # Time budget
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    no_improvement_count = 0
    max_no_improvement = 1000  # Restart if stuck
    
    random.seed(42) # Deterministic for reproducibility in eval? Or random?
    # The task doesn't specify determinism, but consistent results are good for comparison.
    # However, random restarts benefit from randomness.
    
    while time.time() < deadline:
        if no_improvement_count > max_no_improvement:
            # Restart with a perturbed version of the best order
            # Or just random shuffle?
            # Let's try a random perturbation of the best order
            current_order = best_order.copy()
            # Swap a few random pairs
            for _ in range(10):
                i, j = random.randint(0, n-1), random.randint(0, n-1)
                current_order[i], current_order[j] = current_order[j], current_order[i]
            no_improvement_count = 0
            continue
            
        # Pick two random positions to swap
        i, j = random.randint(0, n-1), random.randint(0, n-1)
        if i == j:
            continue
            
        current_order[i], current_order[j] = current_order[j], current_order[i]
        
        new_score = evaluate(current_order)
        
        if new_score >= best_score:
            best_score = new_score
            best_order = current_order.copy()
            no_improvement_count = 0
        else:
            # Revert
            current_order[i], current_order[j] = current_order[j], current_order[i]
            no_improvement_count += 1
            
    # Convert best_order back to priority scores
    # best_order[0] should have the highest priority
    final_priority = np.zeros(n)
    for idx, v in enumerate(best_order):
        final_priority[v] = n - idx
        
    return final_priority
