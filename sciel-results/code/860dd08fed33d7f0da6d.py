"""
Strategy:
1. Construct a base C4-free graph using a random linear 3-uniform hypergraph.
   The line graph of a linear hypergraph is C4-free (any two edges in a hypergraph
   intersect in at most one vertex, so any two vertices in the line graph share
   at most one common neighbor).
   We use a greedy random construction to get a high-degree, sparse C4-free graph.
2. To satisfy the "no independent set of 11" condition, we need to ensure the graph
   is "dense" enough in terms of connectivity or has a small independence number.
   However, C4-free graphs have bounded average degree.
   
   Actually, the standard approach for R(C4, K_k) constructions often uses
   polarity graphs of projective planes. The current program tries PG(2,6) (n=43).
   The issue with PG(2,6) is that it might have alpha >= 11.
   
   A more robust strategy for a "different" approach:
   Use a **random C4-free graph** construction (e.g., via the Alon-Spencer probabilistic method intuition,
   but implemented constructively via a greedy edge addition that strictly forbids C4s).
   Then, use a **local search** to add edges that reduce the independence number.
   
   Wait, the task says "Maximize n". The best known is 39.
   If I can construct a valid graph with n=40, 41, 42, or 43, I win.
   
   Let's try to construct a C4-free graph on n vertices by starting with n isolated vertices
   and adding edges one by one. An edge (u,v) can be added if it does not create a C4.
   This is equivalent to ensuring that u and v have at most 1 common neighbor.
   This greedy process usually yields a graph with alpha around sqrt(n log n) or similar.
   For n=43, sqrt(43*log(43)) is approx sqrt(43*3.7) = sqrt(160) = 12.6.
   So a random C4-free graph on 43 vertices likely has alpha >= 11.
   
   We need to *force* alpha < 11.
   One way to reduce alpha is to add edges. But we can't add edges that create C4s.
   So we are stuck with the structure of the C4-free graph.
   
   Alternative: Use a **blow-up** or **subgraph** of a known good construction?
   The best known construction for R(C4, K_11) > 39 is often related to PG(2, 5) or similar.
   PG(2, 5) has n = 31.
   PG(2, 6) has n = 43.
   
   If PG(2, 6) has alpha >= 11, can we remove a few vertices to break all independent sets of size 11?
   If we remove k vertices, n becomes 43-k.
   If we can find a small set of vertices whose removal reduces alpha to 10, we get a valid graph with n=43-k.
   If k=1, n=42. If k=2, n=41.
   
   Let's try to find a vertex cut for independent sets in PG(2,6).
   
   Plan:
   1. Construct PG(2, 6) (n=43).
   2. Check if it has an independent set of size 11.
      - Use a fast heuristic (greedy coloring or local search) to estimate alpha.
      - If alpha < 11, return it.
      - If alpha >= 11, try to remove vertices one by one.
        - For each vertex v, remove it, check if alpha < 11 (using a faster, approximate check or a limited exact check).
        - If removing v makes it valid, return n=42.
        - If not, try pairs of vertices? That's 900 pairs.
   
   To make this fast and robust:
   - We need a fast `alpha` estimator or a fast exact solver for n <= 43.
   - Exact solver for n=43 is too slow in Python.
   - Heuristic: Use a greedy algorithm to find a large independent set. If the greedy finds 11, it's definitely >= 11.
     If greedy finds < 11, it *might* be < 11, but not guaranteed.
     However, for these structured graphs, greedy is often close to optimal.
     
   Better: Use a **Branch and Bound** with a tight time limit (e.g., 5 seconds).
   If it proves alpha < 11, great.
   If it finds an independent set of 11, we know alpha >= 11.
   If it times out, we assume alpha >= 11 (conservative) and try to prune.
   
   Let's refine the pruning:
   If PG(2,6) is invalid, we try to remove 1 vertex. There are 43 options.
   For each option, we run a *very* fast check.
   What is a fast check?
   We can pre-compute the "criticality" of each vertex.
   Or, we can just run the B&B with a 1-second timeout for each candidate. 43 * 1s = 43s. Acceptable.
   
   If 1-vertex removal fails, try 2-vertex removal?
   43*42/2 = 903 pairs. 903 * 0.1s = 90s. Tight but possible if the check is fast.
   
   Let's implement a fast B&B for independent set.
   Optimizations:
   - Bitmasks.
   - Branching on the vertex with max degree in the remaining subgraph (to shrink the search space).
   - Pruning if current_size + remaining_count < target.
   
   Let's write the code.
"""

import time
import random

def _construct_polarity_graph(q):
    """
    Construct the polarity graph of PG(2, q) using the standard orthogonal polarity.
    """
    reps = []
    seen = set()
    
    for z in range(q):
        for y in range(q):
            for x in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                if x != 0:
                    norm = (1, (y * pow(x, -1, q)) % q, (z * pow(x, -1, q)) % q)
                elif y != 0:
                    norm = (0, 1, (z * pow(y, -1, q)) % q)
                else:
                    norm = (0, 0, 1)
                
                if norm not in seen:
                    seen.add(norm)
                    reps.append(norm)
    
    n = len(reps)
    if n != q*q + q + 1:
        raise ValueError(f"Expected {q*q+q+1} vertices, got {n}")
        
    adj = [set() for _ in range(n)]
    edges = []
    
    for i in range(n):
        for j in range(i+1, n):
            u = reps[i]
            v = reps[j]
            val = (u[0]*v[0] + u[1]*v[1] - u[2]*v[2]) % q
            if val == 0:
                adj[i].add(j)
                adj[j].add(i)
                edges.append([i, j])
                
    return n, edges, adj

def _fast_has_independent_set(n, adj, target_size, time_limit=1.0):
    """
    Returns True if an independent set of size target_size is found.
    Returns False if it is proven that no such set exists, OR if time limit is exceeded (conservative False? No, if timeout, we don't know. 
    For pruning logic: 
    - If we want to KEEP a graph, we need to be SURE it's valid (alpha < target). So if timeout, we should assume INVALID (return True for "found" or None for "unknown").
    - If we want to REMOVE vertices, we want to find a set to remove.
    
    Let's return:
    - 1 if found
    - 0 if proven not found
    - -1 if timeout
    """
    start = time.time()
    masks = [0] * n
    for i in range(n):
        mask = 0
        for j in adj[i]:
            mask |= (1 << j)
        masks[i] = mask
    
    all_mask = (1 << n) - 1
    
    # Precompute degrees for heuristic
    degrees = [masks[i].bit_count() for i in range(n)]
    
    def branch(remaining_mask, current_size):
        if time.time() - start > time_limit:
            return -1 # Timeout
        
        if current_size >= target_size:
            return 1 # Found
        
        if remaining_mask == 0:
            return 0 # Not found (exhausted)
        
        # Upper bound
        if current_size + remaining_mask.bit_count() < target_size:
            return 0
            
        # Pick vertex with max degree in remaining subgraph
        temp = remaining_mask
        v_idx = -1
        max_deg = -1
        # Optimization: If remaining count is small, just iterate
        while temp:
            lsb = temp & -temp
            v = (lsb.bit_length() - 1)
            neighbors_in_rem = masks[v] & remaining_mask
            deg = neighbors_in_rem.bit_count()
            if deg > max_deg:
                max_deg = deg
                v_idx = v
            temp -= lsb
            
        if v_idx == -1:
            return 0
            
        v_bit = 1 << v_idx
        
        # Option 1: Include v
        new_remaining = remaining_mask & ~(masks[v_idx] | v_bit)
        res = branch(new_remaining, current_size + 1)
        if res != 0:
            return res
            
        # Option 2: Exclude v
        new_remaining = remaining_mask & ~v_bit
        res = branch(new_remaining, current_size)
        if res != 0:
            return res
            
        return 0

    return branch(all_mask, 0)

def construct():
    """
    Strategy:
    1. Try PG(2, 6) (n=43).
    2. If valid (alpha < 11), return it.
    3. If invalid, try to remove 1 vertex to make it valid.
    4. If that fails, try to remove 2 vertices.
    5. If that fails, fallback to PG(2, 5) (n=31) or a known valid 39-vertex construction.
       (Note: The prompt says best known is 39. I should ensure I can at least get 39.
        PG(2,5) is 31. Is there a 39-vertex graph?
        Yes, subgraphs of PG(2,6) or specific constructions.
        If I can't find a 39-vertex graph quickly, I should return the best I have.
        But the seed scores 1.0333 which corresponds to n=31 (31/30 = 1.033).
        So I MUST beat 31 to improve.
        Getting 39 would be 39/30 = 1.3.
        Getting 43 would be 43/30 = 1.43.
        """
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # 1. Try PG(2, 6)
    try:
        n6, edges6, adj6 = _construct_polarity_graph(6)
        
        # Check validity
        status = _fast_has_independent_set(n6, adj6, 11, time_limit=5.0)
        if status == 0: # Proven alpha < 11
            return {'n': n6, 'edges': edges6}
        
        # If status is 1 (found) or -1 (timeout), assume invalid.
        # Try removing 1 vertex
        if time.time() < deadline:
            for v in range(n6):
                if time.time() > deadline: break
                
                # Create subgraph without v
                new_n = n6 - 1
                # Map old indices to new indices
                # This is expensive to do every time. 
                # Optimization: Just check if removing v breaks all IS of size 11.
                # We can run the B&B on the subgraph.
                
                # Efficient subgraph creation for B&B:
                # We can just pass a "forbidden" mask to the B&B?
                # Or just build the new masks.
                
                new_adj = [set() for _ in range(new_n)]
                for u in range(n6):
                    if u == v: continue
                    new_u = u if u < v else u - 1
                    for w in adj6[u]:
                        if w == v: continue
                        new_w = w if w < v else w - 1
                        new_adj[new_u].add(new_w)
                        
                # Check this subgraph
                status = _fast_has_independent_set(new_n, new_adj, 11, time_limit=0.5)
                if status == 0:
                    # Valid! Construct edges for this subgraph
                    final_edges = []
                    for u in range(new_n):
                        for w in new_adj[u]:
                            if u < w:
                                final_edges.append([u, w])
                    return {'n': new_n, 'edges': final_edges}
                
        # Try removing 2 vertices?
        # Only if we have time
        if time.time() < deadline - 30:
            # Sample a few pairs to save time
            pairs = []
            for i in range(n6):
                for j in range(i+1, n6):
                    pairs.append((i, j))
            random.shuffle(pairs)
            
            for v1, v2 in pairs:
                if time.time() > deadline: break
                
                new_n = n6 - 2
                new_adj = [set() for _ in range(new_n)]
                
                # Map
                def map_idx(x):
                    if x == v1: return -1
                    if x == v2: return -1
                    count = 0
                    for rem in [v1, v2]:
                        if x > rem:
                            count += 1
                    return x - count
                
                for u in range(n6):
                    mu = map_idx(u)
                    if mu < 0: continue
                    for w in adj6[u]:
                        mw = map_idx(w)
                        if mw < 0: continue
                        new_adj[mu].add(mw)
                        
                status = _fast_has_independent_set(new_n, new_adj, 11, time_limit=0.2)
                if status == 0:
                    final_edges = []
                    for u in range(new_n):
                        for w in new_adj[u]:
                            if u < w:
                                final_edges.append([u, w])
                    return {'n': new_n, 'edges': final_edges}
                    
    except Exception as e:
        pass

    # Fallback: PG(2, 5) n=31
    try:
        n5, edges5, adj5 = _construct_polarity_graph(5)
        # Verify PG(2,5) is valid (it should be, alpha is small)
        # Just return it to be safe
        return {'n': n5, 'edges': edges5}
    except Exception as e:
        pass

    # Final Fallback: 10 disjoint triangles (n=30, alpha=10)
    edges_30 = []
    for t in range(10):
        a, b, c = 3*t, 3*t+1, 3*t+2
        edges_30 += [[a,b], [a,c], [b,c]]
        
    return {'n': 30, 'edges': edges_30}
