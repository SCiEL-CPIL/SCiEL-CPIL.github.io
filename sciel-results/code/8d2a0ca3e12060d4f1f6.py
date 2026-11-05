import numpy as np
import time

def priority(adjacency):
    """
    Assigns priority scores via Monte Carlo estimation of each vertex's
    expected contribution to the independent set size.
    
    Strategy:
    - For each sample, simulate a random removal process: repeatedly pick a
      random unremoved vertex, add it to the "selected" set (if not already
      blocked), and mark its neighbors as blocked.
    - Track how many times each vertex is actually selected across samples.
    - The frequency of selection is a proxy for the vertex's "value" in a
      random greedy process. Vertices that are frequently selected in random
      orderings are likely to be "safe" and valuable in a well-optimized
      ordering too.
    - Additionally, estimate marginal contribution: for each vertex, measure
      how much the IS size decreases when that vertex is forced to be skipped.
    - Combine these signals into a final score.
    """
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill

    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]

    # Precompute neighbor lists for fast access
    neighbors_list = [np.where(adj[i])[0] for i in range(n)]
    deg = adj.sum(axis=1).astype(np.int32)

    def greedy_is_size_from_order(order):
        """Compute IS size given a fixed order (list of vertex indices)."""
        selected = np.zeros(n, dtype=bool)
        blocked = np.zeros(n, dtype=bool)
        count = 0
        for v in order:
            if not selected[v] and not blocked[v]:
                selected[v] = True
                count += 1
                for u in neighbors_list[v]:
                    if not selected[u]:
                        blocked[u] = True
        return count

    def greedy_is_size_with_skip(order, skip_vertex):
        """Compute IS size given order, but skip a specific vertex."""
        selected = np.zeros(n, dtype=bool)
        blocked = np.zeros(n, dtype=bool)
        count = 0
        for v in order:
            if v == skip_vertex:
                continue
            if not selected[v] and not blocked[v]:
                selected[v] = True
                count += 1
                for u in neighbors_list[v]:
                    if not selected[u]:
                        blocked[u] = True
        return count

    # Phase 1: Monte Carlo estimation of selection frequency and marginal value
    # Use a budget-aware number of samples
    # Each sample is O(n^2) worst case, but typically faster
    # Let's estimate: n=200, each sample ~ O(n * avg_degree) ~ O(n * 5) = O(1000)
    # We can afford maybe 5000-10000 samples in 140s? Let's be conservative.
    
    rng = np.random.default_rng(42)
    
    # We'll run multiple rounds of Monte Carlo.
    # In each round, we generate a random permutation, run greedy, and record selections.
    # We also estimate marginal contribution by checking how many times a vertex
    # was "available" but not selected because a neighbor was picked first.
    
    # Selection frequency: how often each vertex is picked
    sel_count = np.zeros(n, dtype=np.int32)
    # Availability count: how often each vertex is available (not blocked) when considered
    avail_count = np.zeros(n, dtype=np.int32)
    
    # Number of samples: adapt based on time
    # Start with a conservative estimate
    n_samples = 2000
    
    for s in range(n_samples):
        if time.time() - start_time > deadline - 20:
            break
        
        # Generate random permutation
        order = rng.permutation(n)
        
        selected = np.zeros(n, dtype=bool)
        blocked = np.zeros(n, dtype=bool)
        
        for v in order:
            if not blocked[v]:
                avail_count[v] += 1
                # In a random greedy process, we pick it if it's available
                # But for our scoring, we want to know: if we prioritize v,
                # does it help? So we track if v would be picked in this random order.
                if not selected[v]:
                    selected[v] = True
                    sel_count[v] += 1
                    for u in neighbors_list[v]:
                        if not selected[u]:
                            blocked[u] = True
    
    # Normalize to frequencies
    n_completed = s + 1 if s >= 0 else 1
    sel_freq = sel_count.astype(np.float64) / n_completed
    avail_freq = avail_count.astype(np.float64) / n_completed
    
    # Phase 2: Estimate marginal contribution via targeted sampling
    # For each vertex, estimate how much the IS size drops if we force it to be skipped.
    # We do this by running greedy on a random order, then re-running with that vertex skipped,
    # and measuring the difference. To save time, we do this for a subset of samples.
    
    # Actually, a more efficient approach: 
    # The "value" of a vertex is roughly (1 - sel_freq) * avail_freq
    # because if it's available but not selected, it means a neighbor was selected first.
    # If we boost its priority, it might get selected, gaining 1 but potentially blocking others.
    
    # Let's use a simpler heuristic based on the Monte Carlo results:
    # Score = sel_freq * (n / (deg + 1))  -- vertices that are often selected and have low degree are valuable
    # But let's also consider the "blocking potential": a vertex with high degree blocks many others.
    
    # Alternative: Use the concept of "independent set contribution"
    # A vertex contributes +1 if selected, but its selection may prevent neighbors from being selected.
    # Net contribution ≈ P(selected) - sum_{u in N(v)} P(u selected | v not selected) * P(v selected first)
    
    # This is getting complex. Let's use a simpler but effective heuristic:
    # Score = sel_freq / (1 + deg / n)
    # This rewards vertices that are frequently selected and penalizes high-degree vertices.
    
    # But wait: the greedy decoder picks in descending score order.
    # If we give high scores to frequently-selected low-degree vertices, the decoder
    # will pick them first, which should mimic the random greedy process that we sampled.
    
    # Let's also add a local search refinement on top of this initial scoring.
    
    # Initial scores
    scores = sel_freq / (1.0 + deg.astype(np.float64) / n)
    
    # Convert to an order: higher score = picked earlier
    order = np.argsort(-scores, kind='stable')
    current_size = greedy_is_size_from_order(order)
    
    # Phase 3: Local search refinement
    # Use the Monte Carlo-informed order as a starting point for hill climbing
    best_order = order.copy()
    best_size = current_size
    
    remaining_time = deadline - 20 - (time.time() - start_time)
    if remaining_time > 5:
        # Hill climbing with random swaps
        improved = True
        n_attempts_per_round = 200
        round_num = 0
        while improved and (time.time() - start_time) < deadline - 20:
            improved = False
            round_num += 1
            for _ in range(n_attempts_per_round):
                if time.time() - start_time > deadline - 20:
                    break
                
                i = rng.integers(0, n)
                j = rng.integers(0, n)
                if i == j:
                    continue
                
                v_i = best_order[i]
                v_j = best_order[j]
                
                # Perform swap
                best_order[i], best_order[j] = best_order[j], best_order[i]
                new_size = greedy_is_size_from_order(best_order)
                
                if new_size > best_size:
                    best_size = new_size
                    improved = True
                else:
                    # Swap back
                    best_order[i], best_order[j] = best_order[j], best_order[i]
    
    # Convert best_order to scores: vertex at position k gets score (n - k)
    final_scores = np.zeros(n)
    for k, v in enumerate(best_order):
        final_scores[v] = n - k
    
    return final_scores
