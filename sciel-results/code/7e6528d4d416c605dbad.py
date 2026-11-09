import numpy as np
import heapq

def priority(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([])
    
    # Compute core numbers
    # We simulate the peeling process
    # degrees is a list/array of current degrees
    # adjacency is a boolean matrix or sparse? 
    # The input is a bool matrix. For N=200, dense is fine.
    
    # To make peeling efficient, we can use a list of sets for neighbors
    # or just update the matrix. 
    # Since N is small (200), we can just recalculate degrees or use a simple heap.
    
    # Let's use a min-heap of (current_degree, vertex)
    # We need to track "alive" vertices.
    
    alive = np.ones(n, dtype=bool)
    current_degrees = adjacency.sum(axis=1).astype(int)
    
    # Heap contains (degree, vertex)
    # We need to handle stale entries in the heap if we don't decrease-key efficiently.
    # For N=200, we can just rebuild the heap or check if the degree matches.
    
    heap = [(current_degrees[i], i) for i in range(n)]
    heapq.heapify(heap)
    
    core_numbers = np.zeros(n, dtype=int)
    
    # The core number of a vertex v is the maximum k such that v is in the k-core.
    # In the peeling algorithm, when we remove a vertex v, its core number is the current "minimum degree threshold" we are peeling at?
    # Actually, the standard algorithm:
    # 1. Sort vertices by degree.
    # 2. Iterate.
    
    # Simpler approach for small N:
    # The core number k(v) is the max k such that the degree of v in the k-core subgraph is >= k.
    # We can compute this by repeatedly removing vertices with degree < k.
    
    # Let's just compute the "peeling order" and use that as a proxy, or compute actual core numbers.
    # Actual core number computation:
    # Initialize k(v) = 0 for all v.
    # This is tricky to do directly without the iterative peeling from high to low k?
    # Standard algorithm:
    # 1. Compute degrees.
    # 2. Put all vertices in a bucket based on degree.
    # 3. Let k = max_degree.
    # 4. While k > 0:
    #    a. If bucket[k] is empty, decrement k.
    #    b. Else, take a vertex v from bucket[k].
    #    c. Set core_number[v] = k.
    #    d. For each neighbor u of v:
    #       If core_number[u] == 0 (not processed yet? No, we process in descending degree? No.)
       
    # Let's use the simpler "min-degree peeling" which gives an order.
    # If we peel by min-degree, the vertex removed at step $t$ has a "peeling index" $t$.
    # Vertices removed early have low "coreness".
    # We can use the step index as the score component.
    
    # However, the "Core Number" is a specific graph invariant.
    # Let's compute Core Numbers properly.
    # Algorithm:
    # 1. Calculate degrees d[v].
    # 2. Create buckets B[d] containing vertices with degree d.
    # 3. Let k = max(d).
    # 4. While k > 0:
    #    a. If B[k] is empty, k -= 1, continue.
    #    b. Pop a vertex v from B[k].
    #    c. core_number[v] = k.
    #    d. For each neighbor u of v:
    #       if core_number[u] == 0: # u is not yet assigned?
    #          # We need to decrement u's degree
    #          # But we are iterating downwards.
    #          # This algorithm is for finding the core number.
    #          # Wait, the standard algorithm processes vertices in order of decreasing degree?
    #          # No, the algorithm by Batagelj and Zaversnik:
    #          # 1. Sort vertices by degree.
    #          # 2. Iterate through vertices in increasing order of degree?
    #          # No, let's look up "core number algorithm".
    #          # It involves maintaining a priority queue of (degree, vertex).
    #          # We pop the vertex with the smallest degree, say v.
    #          # The core number of v is the current degree of v? No.
    #          # The core number of v is the maximum k such that v is in the k-core.
    #          # When we pop v with degree d, we set core_number[v] = d?
    #          # Not exactly. If we have a triangle, degrees are 2.
    #          # Pop v (deg 2). core[v]=2.
    #          # Neighbors u, w have degree 2.
    #          # Remove v. u, w degree becomes 1.
    #          # Pop u (deg 1). core[u]=1?
    #          # But u is in the 2-core (triangle).
    #          # So the simple "pop min degree" assigns the degree at time of removal, which is NOT the core number.
    #          # The core number is the MAX degree it had while in the core?
    
    # Correct Algorithm for Core Number:
    # 1. Compute degrees.
    # 2. Put vertices in buckets by degree.
    # 3. Let k = max_degree.
    # 4. While k > 0:
    #    a. If bucket[k] is empty, k--.
    #    b. Else, take vertex v from bucket[k].
    #    c. core_number[v] = k.
    #    d. For each neighbor u of v:
    #       if core_number[u] == 0: # u hasn't been assigned yet
    #          # This implies u is still in the graph?
    #          # We need to decrement degree of u.
    #          # If degree[u] == k, move u from bucket[k] to bucket[k-1].
    #          # Else if degree[u] > k, just decrement.
    #          # Wait, u's degree can only decrease.
    #          # So if u was in bucket[deg_u], and deg_u == k, it moves to k-1.
    #          # If deg_u > k, it stays in its bucket but degree is decremented.
    #          # This requires updating buckets.
    
    # Given N=200, we can afford a simpler, slightly slower method or just use the "peeling order" as a heuristic score.
    # The "peeling order" (index in min-degree peeling) is a valid structural feature.
    # Vertices peeled early are "peripheral".
    # Let's use the peeling order.
    
    # Re-implementing min-degree peeling to get an order index.
    # We will assign a score based on the order they are removed.
    # Lower index = removed earlier = peripheral = Good to pick.
    # So Score should be high for low index.
    # Score = -index.
    
    # Let's do this.
    
    # 1. Current degrees
    degs = adjacency.sum(axis=1).copy()
    # 2. Heap of (degree, index)
    heap = [(degs[i], i) for i in range(n)]
    heapq.heapify(heap)
    
    removed = [False] * n
    order_index = [0] * n
    current_step = 0
    
    while heap:
        d, v = heapq.heappop(heap)
        if removed[v]:
            continue
        if degs[v] != d:
            # Stale entry, skip? Or push updated?
            # For N=200, we can just skip stale and rely on the fact that the correct entry will be there.
            # But if we skip, we might miss the vertex if its degree changed and the new entry is not yet popped?
            # Actually, if we update degrees, we should push the new (degree, vertex) into the heap.
            # We didn't do that in the loop above.
            # Let's restart with a cleaner loop that pushes updates.
            pass
        
    # Let's write the clean version.
    
    degs = adjacency.sum(axis=1).astype(int)
    heap = [(degs[i], i) for i in range(n)]
    heapq.heapify(heap)
    
    removed = [False] * n
    order_index = [0] * n
    step = 0
    
    while heap:
        d, v = heapq.heappop(heap)
        if removed[v]:
            continue
        if degs[v] != d:
            # This entry is stale. We need to have pushed the updated degree when it changed.
            # Since we didn't, this simple check fails if we don't push updates.
            # Let's assume we push updates.
            pass
            
    # Correct Implementation:
    degs = adjacency.sum(axis=1).astype(int)
    heap = [(degs[i], i) for i in range(n)]
    heapq.heapify(heap)
    
    removed = [False] * n
    order_index = [0] * n
    step = 0
    
    while heap:
        d, v = heapq.heappop(heap)
        if removed[v]:
            continue
        if degs[v] != d:
            # Stale entry. The correct entry for v is already in the heap (pushed when deg changed).
            # We just skip this one.
            continue
        
        # v is the min-degree vertex currently alive
        removed[v] = True
        order_index[v] = step
        step += 1
        
        # Update neighbors
        # Get neighbors of v
        neighbors = np.where(adjacency[v, :])[0]
        for u in neighbors:
            if not removed[u]:
                degs[u] -= 1
                heapq.heappush(heap, (degs[u], u))
                
    # Now order_index[v] is the step at which v was removed.
    # Low step = removed early = low coreness.
    # We want to pick low step vertices first.
    # The greedy decoder picks descending score.
    # So Score = -order_index.
    
    # To break ties (or add nuance), we can combine with original degree.
    # Original degree is a good tie-breaker: among vertices removed at the same step (or similar coreness),
    # lower original degree is generally better?
    # Actually, if they are removed at the same step, they have similar "current" degree.
    # Let's just use -order_index.
    
    scores = -np.array(order_index, dtype=float)
    
    return scores
