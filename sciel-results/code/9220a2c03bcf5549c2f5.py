import numpy as np
import time

def priority(adjacency):
    """
    Assigns priority scores to vertices for a greedy Maximum Independent Set decoder.
    
    Strategy: Complement Graph Greedy Vertex Removal (Chung-Lu heuristic variant).
    1. Construct the complement graph (edges where original has no edge).
    2. Iteratively select the vertex with the minimum degree in the remaining complement graph.
       This vertex has the fewest non-edges, meaning it is "compatible" with the most 
       remaining vertices in the original graph, making it a strong candidate for the 
       independent set.
    3. Record the selection order.
    4. Convert the selection order into a priority score vector: 
       Vertices selected earlier get higher scores.
    5. The fixed greedy decoder will pick vertices in this order. Since we selected them 
       knowing they are non-adjacent to each other in the original graph (by construction 
       of the independent set), the decoder will successfully pick all of them, resulting 
       in an independent set of the same size as our heuristic.
    
    Why this is different:
    - Current: Optimizes continuous scores via local search, hoping the greedy decoder 
      behaves well.
    - New: Directly solves a related problem (finding a large independent set via 
      complement greedy) and encodes the solution order into the scores. This is a 
      constructive approach rather than an optimization approach on the score space.
    """
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([])
        
    # We want to find an ordering such that the greedy decoder picks a large independent set.
    # If we can find an independent set S, and we order vertices such that all v in S come 
    # before any vertex that is a neighbor of any v in S, then the decoder will pick all of S.
    # Actually, the decoder picks v if no neighbor was picked *before* it.
    # So if we have an independent set S, and we place all vertices in S at the top of the 
    # order, the decoder will pick the first one, then the second (since they are not neighbors), 
    # etc. It will pick all of S.
    # Then it will try to pick vertices outside S. Some might be neighbors of S (blocked), 
    # some might not. If not blocked, it might pick them, adding to the set.
    # So, finding a large independent set S and putting S first is a good strategy.
    
    # How to find a large independent set S?
    # Greedy on the complement graph:
    # In the complement graph, an edge exists between u and v if u and v are NOT connected in G.
    # An independent set in G is a clique in G_complement.
    # Finding a maximum clique is hard, but greedy heuristics work well.
    # Greedy heuristic for Max Clique (which is Max Independent Set in G):
    # 1. Start with all vertices.
    # 2. Repeatedly pick the vertex with the highest degree in the current subgraph of G_complement.
    #    (High degree in complement = many non-neighbors in G = compatible with many others).
    # 3. Add it to the set.
    # 4. Remove it and all its non-neighbors in G_complement (i.e., its neighbors in G) from the pool?
    #    No, that's for finding a clique.
    #    Let's stick to the standard greedy for Independent Set in G:
    #    Pick vertex with min degree in G? Or max degree in G_complement?
    #    Let's try: Pick vertex v with min degree in G. Add to S. Remove v and N(v) from G.
    #    This is the standard greedy for Max Independent Set.
    
    # However, we need the ORDER for the decoder.
    # If we produce an independent set S = {s1, s2, ..., sk} using the greedy method, 
    # we can assign scores such that s1 > s2 > ... > sk > others.
    # The decoder will pick s1, then s2 (not neighbor of s1), ..., sk.
    # Then it will pick any remaining vertices not adjacent to any s_i.
    # This might yield a set larger than S.
    
    # Let's implement the standard greedy:
    # 1. Compute degrees.
    # 2. Pick min-degree vertex v.
    # 3. Add v to order.
    # 4. Remove v and its neighbors from consideration.
    # 5. Repeat.
    # This gives an ordering of the "core" independent set.
    # What about the rest? We can just append them in any order (e.g., original index).
    
    # Optimization: Use a priority queue (heapq) for min-degree selection.
    import heapq
    
    # Current degrees
    deg = adj.sum(axis=1).astype(int).copy()
    # Active vertices
    active = np.ones(n, dtype=bool)
    
    # Initialize heap: (degree, vertex_index)
    # We need to handle lazy deletion because degrees change.
    heap = []
    for i in range(n):
        heapq.heappush(heap, (deg[i], i))
    
    order = []
    
    while active.sum() > 0:
        if time.time() - start_time > deadline:
            break
            
        # Find the active vertex with min degree
        found = False
        while heap:
            d, v = heapq.heappop(heap)
            if active[v] and deg[v] == d:
                # Valid vertex
                found = True
                break
            # else: stale entry, discard
        
        if not found:
            # Fallback: pick any active vertex
            active_indices = np.where(active)[0]
            if len(active_indices) == 0:
                break
            v = active_indices[0]
            d = deg[v]
        else:
            v = v
            
        # Select v
        order.append(v)
        
        # Remove v and its neighbors
        active[v] = False
        neighbors = np.where(adj[v])[0]
        for u in neighbors:
            if active[u]:
                active[u] = False
                # Update degree of remaining active vertices?
                # Strictly, we don't need to update degrees of remaining vertices 
                # if we are just doing a single pass greedy. 
                # However, the "min-degree" heuristic is usually applied on the 
                # induced subgraph. So degrees should decrease.
                # But updating degrees is O(n) per removal, making it O(n^2).
                # For n=200, O(n^2) is 40,000 ops, which is trivial.
                # Let's do it properly to get a better heuristic.
                pass
        
        # Actually, let's re-compute degrees for active vertices to be precise, 
        # or update them incrementally.
        # Incremental update:
        # When v is removed, for every active neighbor u of v, deg[u] decreases by 1.
        # But we just removed all neighbors of v from active. 
        # So we need to update degrees of active vertices that were neighbors of v?
        # No, v's neighbors are removed. 
        # The active vertices are those NOT adjacent to v and NOT v.
        # Their degrees in the induced subgraph decrease by 1 if they were adjacent to v.
        # But they were adjacent to v, so they are removed.
        # Wait.
        # If u is active, it is not a neighbor of v.
        # So u is not adjacent to v.
        # So the degree of u in the new induced subgraph (G \ {v, N(v)}) 
        # is the same as its degree in the previous induced subgraph (G \ previous_removals) 
        # MINUS 1 if u was adjacent to any vertex removed in this step?
        # The vertices removed are v and N(v).
        # u is not v. u is not in N(v).
        # So u is not adjacent to v.
        # Was u adjacent to any other vertex in N(v)?
        # Yes, possibly.
        # So the degree of u in the new subgraph is:
        # deg_old[u] - (number of neighbors of u that are in {v} U N(v))
        # Since u is not adjacent to v, it's:
        # deg_old[u] - (number of neighbors of u that are in N(v))
        
        # This is getting complex to update incrementally without checking all neighbors.
        # Given n=200, let's just recompute degrees for active vertices if we want strict min-degree.
        # Or, simpler: Just use the static degree of the original graph?
        # Static min-degree greedy is a valid heuristic (Birkhoff's theorem related?).
        # Let's try static min-degree first. It's O(n log n) to build order.
        
        # Let's restart the loop logic with static degrees for speed and simplicity.
        break 
        
    # The loop above was getting complicated. Let's implement a clean static min-degree greedy.
    
    # Static Min-Degree Greedy:
    # 1. Sort vertices by degree (ascending).
    # 2. Iterate through sorted list.
    # 3. If vertex is still "available" (not removed by previous selections), pick it.
    # 4. Mark it and its neighbors as removed.
    
    # This is O(n^2) worst case, but very fast for n=200.
    
    deg_static = adj.sum(axis=1).astype(int)
    sorted_vertices = np.argsort(deg_static, kind='stable')
    
    available = np.ones(n, dtype=bool)
    selected_order = []
    
    for v in sorted_vertices:
        if time.time() - start_time > deadline:
            break
        if available[v]:
            selected_order.append(v)
            available[v] = False
            neighbors = np.where(adj[v])[0]
            available[neighbors] = False
            
    # Now we have an independent set `selected_order`.
    # We want the decoder to pick these vertices first.
    # Assign scores: 
    # Vertices in selected_order get high scores, in the order they were selected.
    # Vertices not in selected_order get low scores.
    
    scores = np.zeros(n)
    # Give the selected vertices a base high score, decreasing as we go down the list?
    # Or just give them all high scores, and non-selected low?
    # If we give them all the same high score, the decoder will pick them in index order.
    # Does that matter? They are an independent set, so any order works for them.
    # But we want to ensure they are picked BEFORE any neighbor.
    # So, max score for selected, min score for non-selected.
    
    max_score = 1000.0
    min_score = 0.0
    
    for i, v in enumerate(selected_order):
        # To break ties in a way that might help if there are conflicts (there shouldn't be),
        # we can use the index in the order.
        scores[v] = max_score - i * 0.1
        
    # Non-selected vertices
    non_selected = np.where(~np.isin(np.arange(n), selected_order))[0]
    # Give them low scores. 
    # To potentially allow the decoder to pick some of them if they are not blocked,
    # we should give them scores lower than the selected ones, but maybe varying.
    # If a non-selected vertex is not adjacent to any selected vertex, it can be picked.
    # We want the decoder to pick as many of these as possible.
    # So we should order them well too.
    # We can apply the same greedy heuristic to the remaining graph?
    # Or just randomize / use original index.
    # Let's try to order them by their degree in the remaining graph (original degree - removed neighbors).
    
    # For simplicity and speed, let's just assign them low scores based on their original degree.
    # Low degree in original graph might be better?
    for v in non_selected:
        scores[v] = min_score + (deg_static[v] * 0.01) # Small variation
        
    # Ensure strict ordering: Selected > Non-selected
    # My current assignment: Selected ~ 1000, Non-selected ~ 0-2.
    # This is safe.
    
    return scores
