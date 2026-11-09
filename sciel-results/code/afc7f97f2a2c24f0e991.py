"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    adj = np.asarray(adjacency, dtype=np.float64)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([])
    
    adj_bool = adj > 0.5
    degrees = adj.sum(axis=1)
    
    # Precompute neighbor lists for fast access
    neighbor_indices = [np.nonzero(adj_bool[i])[0] for i in range(n)]
    
    # The core idea:
    # We want to produce a score vector `scores` such that `argsort(-scores)` 
    # yields an order that maximizes the greedy independent set size.
    # 
    # Standard greedy MIS:
    # 1. Pick vertex v with highest priority.
    # 2. Add v to MIS.
    # 3. Remove v and all its neighbors from the graph.
    # 4. Repeat.
    #
    # If we can determine the *optimal* order of removals, we can assign scores
    # such that the decoder reproduces this order.
    #
    # Strategy: "Greedy Construction with Lookahead"
    # We simulate the greedy process. At each step, we have a set of available vertices `active`.
    # We need to pick the next vertex `v` from `active`.
    # The "value" of picking `v` is 1 (it joins the MIS).
    # The "cost" is that all neighbors of `v` in `active` are removed (cannot join MIS).
    # So, a good heuristic for `v` is: 1 - (number of active neighbors of v) / (total active vertices)
    # Or better: We want to pick vertices that have few neighbors, so we don't waste potential MIS members.
    # This is equivalent to the "Low Degree First" heuristic, but dynamic.
    #
    # However, the current program already uses Low Degree First as a candidate.
    # To improve, we can use a more sophisticated weight:
    # Weight(v) = 1 / (1 + degree_active(v))
    # This favors low degree.
    #
    # Let's try a different angle: **Local Ratio / Weighted Greedy**
    # We assign weights to vertices. The standard greedy picks max weight.
    # If we set weight(v) = 1 for all, and pick min degree, that's standard.
    # 
    # Let's try **Iterative Local Search on the Order** directly, but more efficiently.
    # The previous program did random swaps. That's slow.
    # Instead, let's use a **Constructive Heuristic** that is stronger than simple degree.
    #
    # Heuristic: **Minimum Sum of Neighbor Degrees**
    # Picking a vertex v removes its neighbors. If v is connected to high-degree vertices,
    # we are "wasting" those high-degree vertices which might have been useful?
    # No, high-degree vertices are bad for MIS because they block many others.
    # So we WANT to remove high-degree vertices early?
    # Wait. If I pick a low-degree vertex, I remove few others.
    # If I pick a high-degree vertex, I remove many others.
    # So picking low-degree vertices is generally better for *size*.
    #
    # Let's look at **Core Number**.
    # Vertices with low core number are on the "periphery".
    # Picking periphery vertices first is good.
    #
    # Let's try a **Hybrid Dynamic Score**:
    # We simulate the greedy process.
    # active = set(all vertices)
    # scores = array of size n, initialized to 0.
    # current_rank = n
    #
    # while active is not empty:
    #    For each v in active:
    #       deg_active = number of neighbors of v that are in active
    #       # Heuristic: Prefer low deg_active
    #       # But also consider: if I pick v, how many *future* choices do I lose?
    #       # A simple proxy is deg_active.
    #       
    #       # Let's try: Score candidate v as -deg_active(v)
    #       # Pick v with min deg_active.
    #       # Assign score[v] = current_rank (high score = picked early)
    #       # Remove v and its active neighbors from active.
    #       # Decrement current_rank.
    #
    # This produces an order. The scores are just the inverse of the order.
    # When the decoder sorts by these scores, it will pick in the exact same order.
    #
    # Is this better than static low-degree?
    # Yes, because degrees change as neighbors are removed.
    # A vertex might start with high degree, but if its high-degree neighbors are removed,
    # its active degree drops, making it a good candidate later.
    #
    # Let's implement this "Dynamic Low Degree" greedy construction.
    # We can also try "Dynamic High Degree" (removing hubs first) to see if it helps,
    # though theoretically, low degree is usually better for MIS size.
    #
    # To be safe, we will generate TWO such orders:
    # 1. Dynamic Low Degree (pick min active degree)
    # 2. Dynamic "Independence Ratio" (pick min active degree / (1 + sum of active neighbor degrees))
    #    This penalizes picking vertices connected to other high-degree vertices.
    
    def construct_order_dynamic_low_degree():
        active = np.ones(n, dtype=bool)
        order = np.zeros(n, dtype=int)
        current_rank = n - 1
        # Copy of degrees that we will update
        active_deg = degrees.copy()
        
        for _ in range(n):
            if not active.any():
                break
            
            # Find vertex in active with minimum active_deg
            # To handle ties, we can use a secondary criterion, e.g., original degree
            # But for now, min active_deg.
            # We need to efficiently find min. N=200, so O(N^2) is fine.
            min_val = np.inf
            best_v = -1
            for v in range(n):
                if active[v]:
                    if active_deg[v] < min_val:
                        min_val = active_deg[v]
                        best_v = v
            
            if best_v == -1:
                break
            
            order[best_v] = current_rank
            active[best_v] = False
            current_rank -= 1
            
            # Update active_deg for neighbors of best_v
            for u in neighbor_indices[best_v]:
                if active[u]:
                    active_deg[u] -= 1
                    # If u becomes 0 degree, it's isolated, good candidate for next steps?
                    # The loop will handle it.
        
        # Assign scores based on order.
        # Higher score = earlier in order.
        scores = np.zeros(n)
        for v in range(n):
            scores[v] = order[v]
        return scores

    def construct_order_dynamic_ratio():
        """
        Heuristic: Pick vertex v minimizing active_deg(v) / (1 + sum_active_neighbor_degrees(v))
        This prefers vertices that are low-degree AND not connected to other high-degree vertices.
        """
        active = np.ones(n, dtype=bool)
        order = np.zeros(n, dtype=int)
        current_rank = n - 1
        active_deg = degrees.copy()
        
        for _ in range(n):
            if not active.any():
                break
            
            min_ratio = np.inf
            best_v = -1
            for v in range(n):
                if active[v]:
                    # Calculate sum of active neighbor degrees
                    sum_nb_deg = 0
                    for u in neighbor_indices[v]:
                        if active[u]:
                            sum_nb_deg += active_deg[u]
                    
                    denom = 1 + sum_nb_deg
                    if denom == 0:
                        ratio = 0
                    else:
                        ratio = active_deg[v] / denom
                    
                    if ratio < min_ratio:
                        min_ratio = ratio
                        best_v = v
            
            if best_v == -1:
                break
            
            order[best_v] = current_rank
            active[best_v] = False
            current_rank -= 1
            
            for u in neighbor_indices[best_v]:
                if active[u]:
                    active_deg[u] -= 1
        
        scores = np.zeros(n)
        for v in range(n):
            scores[v] = order[v]
        return scores

    # Generate candidates
    candidate_scores = []
    
    # 1. Static Low Degree (Baseline)
    candidate_scores.append(-degrees.copy())
    
    # 2. Dynamic Low Degree
    try:
        if time.time() < deadline:
            candidate_scores.append(construct_order_dynamic_low_degree())
    except Exception:
        pass
        
    # 3. Dynamic Ratio
    try:
        if time.time() < deadline:
            candidate_scores.append(construct_order_dynamic_ratio())
    except Exception:
        pass

    # 4. Static Core Number (Low Core First)
    try:
        deg_current = degrees.copy()
        core_numbers = np.zeros(n, dtype=int)
        remaining = set(range(n))
        deg_rem = degrees.copy()
        
        while remaining:
            min_deg = min(deg_rem[v] for v in remaining)
            to_remove = [v for v in remaining if deg_rem[v] == min_deg]
            for v in to_remove:
                core_numbers[v] = min_deg
                remaining.remove(v)
                for u in neighbor_indices[v]:
                    if u in remaining:
                        deg_rem[u] -= 1
        candidate_scores.append(-core_numbers.astype(float))
    except Exception:
        pass

    # Evaluate candidates
    def greedy_size_from_order(order):
        """
        order: array of vertex indices, where order[i] is the score/rank?
        No, let's make it consistent.
        Let's pass the SCORES. The decoder sorts by descending score.
        So we need to simulate that.
        """
        selected = np.zeros(n, dtype=bool)
        count = 0
        # Get the order of vertices by descending score
        # Since scores are unique in our constructed ones (ranks), argsort is stable.
        # But for static ones (degrees), ties are possible.
        # The evaluator uses "descending score order".
        # We assume stable sort or arbitrary tie-breaking.
        # To be safe, we replicate the likely behavior: argsort(-scores).
        sort_order = np.argsort(-order, kind='stable')
        
        for v in sort_order:
            if not selected[v]:
                count += 1
                selected[v] = True
                for u in neighbor_indices[v]:
                    selected[u] = True
        return count

    best_size = -1
    best_scores = candidate_scores[0]
    
    for scores in candidate_scores:
        if time.time() > deadline:
            break
        size = greedy_size_from_order(scores)
        if size > best_size:
            best_size = size
            best_scores = scores

    # If we have time, try to improve the best one with a few random swaps?
    # The dynamic construction is already quite strong.
    # Let's stick to the best constructed one to avoid noise.
    
    return best_scores.astype(float)

# EVOLVE-BLOCK-END
