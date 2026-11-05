import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    adj = np.asarray(adjacency, dtype=float)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([])
    
    # Precompute neighbor masks for fast simulation
    # neighbors[i] is a boolean array of length n, True if j is neighbor of i
    # Storing as list of arrays or sparse? n=200, so 200x200 bool is fine.
    # But we need to simulate greedy selection.
    # Greedy: sort by score desc. Pick v. Remove v and N(v).
    
    # To simulate fast, we can represent the "available" set as a bitmask or bool array.
    # However, re-sorting every time is O(N log N).
    # Alternative: The score defines a permutation. 
    # Let's work with a permutation of indices: order[0] is highest priority.
    # Simulation:
    #   available = np.ones(n, dtype=bool)
    #   picked = []
    #   for v in order:
    #       if available[v]:
    #           picked.append(v)
    #           # Mark v and neighbors as unavailable
    #           available[v] = False
    #           # Need efficient neighbor update
    #           # adj[v] is a bool array. 
    #           available[adj[v] == 1] = False
    
    # Initial order: random permutation? Or based on a simple heuristic to get a good start?
    # A good start is crucial for local search.
    # Let's start with the current heuristic (mean_nb_deg - deg) or just random.
    # Random might be too slow to converge.
    # Let's start with a greedy construction:
    # Actually, the "priority" function returns scores. The decoder sorts by score.
    # So we are optimizing the scores (or equivalently, the rank order).
    
    # Let's optimize the permutation `order` directly.
    # Initial order: sort by a simple heuristic, e.g., low degree first? 
    # For Max Independent Set, picking low-degree nodes first is often good (Luby's heuristic).
    # Let's try Luby's heuristic as initial state.
    # Luby: repeatedly pick node with min degree in remaining graph.
    # This is O(N^2) or O(N^3) to simulate.
    # For N=200, O(N^3) = 8M ops, fast enough.
    
    def simulate_greedy(order, adj):
        available = np.ones(len(order), dtype=bool)
        picked = 0
        # We don't need the actual set, just the count
        for v in order:
            if available[v]:
                picked += 1
                # Mark v and neighbors as unavailable
                # adj[v] is row v of adjacency matrix
                # Note: adj is float, but boolean logic works
                available[v] = False
                # Vectorized update: available[adj[v] == 1] = False
                # This is O(N) per pick, total O(N^2)
                available[adj[v] == 1] = False
        return picked

    # Initialize order using Luby's heuristic (Min-Degree)
    # This requires dynamic degree calculation.
    # Simpler: Start with random permutation? 
    # Given the time budget, a good heuristic start is better.
    # Let's generate a few random initial states and pick the best one quickly.
    
    best_order = None
    best_score = -1
    
    # Try a few random initial permutations
    for _ in range(10):
        if time.time() > deadline:
            break
        order = np.random.permutation(n)
        score = simulate_greedy(order, adj)
        if score > best_score:
            best_score = score
            best_order = order.copy()
    
    # Also try the degree-based heuristic order (ascending degree)
    degrees = adj.sum(axis=1)
    order_deg = np.argsort(degrees)
    score_deg = simulate_greedy(order_deg, adj)
    if score_deg > best_score:
        best_score = score_deg
        best_order = order_deg.copy()
        
    # Also try descending mean neighbor degree (from current program)
    neighbor_degree_sums = adj @ degrees
    mask = degrees > 0
    mean_nb = np.zeros_like(degrees)
    mean_nb[mask] = neighbor_degree_sums[mask] / degrees[mask]
    scores_heur = mean_nb - degrees
    order_heur = np.argsort(-scores_heur)
    score_heur = simulate_greedy(order_heur, adj)
    if score_heur > best_score:
        best_score = score_heur
        best_order = order_heur.copy()

    if best_order is None:
        # Fallback
        best_order = np.arange(n)
        best_score = simulate_greedy(best_order, adj)

    # Local Search: Simulated Annealing or Hill Climbing with restarts
    # Move: Swap two positions in the order.
    # If we swap i and j in the order, the greedy result changes.
    # Recalculating the whole simulation is O(N^2).
    # With N=200, O(40000) ops per eval.
    # In 145s, we can do roughly 145 / (40000 / 1e9) ... wait, numpy is fast.
    # simulate_greedy is a python loop of 200 iterations, each doing numpy vector ops.
    # Each iteration: check bool, update bool array.
    # Let's benchmark mentally: 200 * (small overhead + vector op).
    # Vector op on 200 elements is ~microseconds.
    # So ~1ms per simulation?
    # If 1ms per sim, we can do ~100,000 simulations in 100s.
    # This is plenty for local search.
    
    current_order = best_order.copy()
    current_score = best_score
    
    # Hill Climbing with random restarts and perturbation
    # We will run until deadline.
    
    while time.time() < deadline:
        # Perform a batch of swaps
        # Pick two random indices to swap
        i = np.random.randint(0, n)
        j = np.random.randint(0, n)
        if i == j:
            continue
            
        # Swap
        current_order[i], current_order[j] = current_order[j], current_order[i]
        
        new_score = simulate_greedy(current_order, adj)
        
        # Accept if better or equal (to allow plateaus)
        # If worse, revert
        if new_score >= current_score:
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_order = current_order.copy()
        else:
            # Revert
            current_order[i], current_order[j] = current_order[j], current_order[i]
            
        # Optional: If stuck, do a random restart or larger perturbation
        # For simplicity, just keep swapping. If we don't improve in many steps,
        # we might get stuck in local optima.
        # Let's add a simple restart mechanism: if no improvement in 1000 steps,
        # revert to best_order and start fresh.
        # But we need a counter.
        
        # To save time, let's just do the swaps. The space is huge.
        # Actually, simple hill climbing on permutations can get stuck.
        # Let's add a "kick": if we haven't improved in X iterations,
        # we revert to best_order and do a random shuffle of a subset.
        
        # For this implementation, we'll stick to simple hill climbing but
        # occasionally revert to best to avoid drift into worse local minima
        # (though we revert immediately on worse moves, so we are always at 
        # a local optimum relative to single swaps from the current state).
        # Wait, if we only accept better/equal, we are doing steepest/hill climbing.
        # We might get stuck. 
        # Let's add: if time.time() > deadline - 5, break.
        
    # Convert best_order (permutation) back to scores
    # The decoder sorts by score descending.
    # So the vertex at position k in best_order should have rank k.
    # We assign scores such that argsort(-scores) == best_order.
    # Simplest: score[v] = n - 1 - rank(v)
    # Where rank(v) is the index of v in best_order.
    
    scores = np.zeros(n)
    for rank, v in enumerate(best_order):
        scores[v] = n - 1 - rank
        
    return scores
