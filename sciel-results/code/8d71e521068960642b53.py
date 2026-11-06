import time
import random
import sys

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill

    best_n = 1
    best_edges = []

    # Try to build graphs of increasing size from 39 down, or random sizes
    # Strategy: random-greedy edge construction for target n, verify with B&B MIS
    # We'll try multiple random seeds and target sizes

    for target_n in [39, 40, 41, 42, 43, 38, 37, 36]:
        if time.time() > deadline - 5:
            break

        for attempt in range(50):
            if time.time() > deadline - 5:
                break

            n = target_n
            # Bitset adjacency: adj[i] is a Python int where bit j is 1 if edge (i,j) exists
            adj = [0] * n

            # Random-greedy: try to add edges in random order, keep if C4-free
            # C4-free condition: for any edge (u,v), u and v must have no common neighbor
            # When adding edge (u,v): check that adj[u] & adj[v] == 0 (no common neighbor)
            # But also, after adding, we must ensure no C4 is formed.
            # Actually, the condition for C4-free is: for every pair (u,v), |N(u) ∩ N(v)| ≤ 1
            # When we add edge (u,v), we need to check that before adding, |N(u) ∩ N(v)| == 0
            # because after adding, v becomes a neighbor of u and u of v, so if they had a common neighbor w,
            # then u-w-v-... wait. Let's think carefully.
            
            # C4 is a cycle of length 4: u-w-v-x-u. 
            # The condition "no two vertices have two common neighbors" is equivalent to C4-free.
            # If we add edge (u,v), we need to ensure that u and v do NOT have a common neighbor in the current graph.
            # Because if w is a common neighbor of u and v in the current graph, then adding (u,v) creates the triangle u-w-v, 
            # but that's a triangle, not a C4. Wait, no.
            # Let me reconsider. A C4 is u-a-v-b-u. The vertices u and v have common neighbors a and b.
            # So if u and v already have two common neighbors, there's a C4.
            # When we add edge (u,v), does it create a new C4?
            # The new edge (u,v) could be part of a C4 if there exists a path of length 3 from u to v not using the edge.
            # That is, u-x-y-v where x,y are distinct and not u,v. Then u-x-y-v-u is a C4.
            # This means x and y are both neighbors of... wait. u is adjacent to x, x to y, y to v.
            # So x is a neighbor of u, y is a neighbor of v. And x-y is an edge.
            # This doesn't directly translate to common neighbors of u and v.
            
            # Let's use the standard criterion: A graph is C4-free iff for every pair of vertices (u,v), |N(u) ∩ N(v)| ≤ 1.
            # When we add edge (u,v), we change N(u) to N(u)∪{v} and N(v) to N(v)∪{u}.
            # We need to check that for all pairs, the common neighbor count is still ≤ 1.
            # The only pairs whose common neighbor count can increase are those involving u or v.
            # For pair (u,v): new common neighbors = (N_old(u) ∩ N_old(v)) ∪ {w: w adjacent to both u and v in new graph}
            # Actually, N_new(u) ∩ N_new(v) = (N_old(u) ∪ {v}) ∩ (N_old(v) ∪ {u})
            # = (N_old(u) ∩ N_old(v)) ∪ (N_old(u) ∩ {u}) ∪ ({v} ∩ N_old(v)) ∪ ({v} ∩ {u})
            # N_old(u) ∩ {u} = ∅ (no self-loops), {v} ∩ N_old(v) = ∅ (no self-loops), {v}∩{u}=∅ (u≠v)
            # So N_new(u) ∩ N_new(v) = N_old(u) ∩ N_old(v)
            # So adding edge (u,v) doesn't change the common neighbors of u and v!
            
            # What about pair (u, w) for w ≠ v?
            # N_new(u) ∩ N_new(w) = (N_old(u) ∪ {v}) ∩ N_old(w) = (N_old(u) ∩ N_old(w)) ∪ ({v} ∩ N_old(w))
            # So we gain v as a common neighbor of u and w IF v is in N_old(w), i.e., w-v is an edge.
            # So if w is a neighbor of v in the old graph, then after adding (u,v), u and w share common neighbor v.
            # We need |N_new(u) ∩ N_new(w)| ≤ 1.
            # Old count was |N_old(u) ∩ N_old(w)|. New count is old count + (1 if w~v else 0).
            # So we need: if w is adjacent to v, then |N_old(u) ∩ N_old(w)| must be 0.
            # Similarly for pair (v, w): if w is adjacent to u, then |N_old(v) ∩ N_old(w)| must be 0.
            
            # So the condition to add edge (u,v) is:
            # 1. (u,v) is not already an edge
            # 2. For all w adjacent to v in old graph: |N_old(u) ∩ N_old(w)| == 0
            # 3. For all w adjacent to u in old graph: |N_old(v) ∩ N_old(w)| == 0
            
            # This is expensive to check. Let's use a different approach:
            # Just build the graph, then verify C4-freeness at the end.
            # For random-greedy, we can use a simpler heuristic: 
            # Try to add edge (u,v) if adj[u] & adj[v] == 0 (no common neighbor currently).
            # This is a sufficient condition but not necessary? Actually, it's necessary and sufficient 
            # for the pair (u,v) itself, but we also need to check the conditions above.
            # Let's just do: add edge if adj[u] & adj[v] == 0. This ensures no C4 through u,v directly.
            # It might miss some C4s but for random construction, it's a good heuristic.
            # Actually, I think the condition adj[u] & adj[v] == 0 is necessary and sufficient for adding (u,v) 
            # to maintain C4-freeness, given that the graph was C4-free before. Let me verify.
            # Suppose G is C4-free. Add edge (u,v) with N(u)∩N(v)=∅.
            # Any new C4 must use the new edge (u,v). So it's of form u-v-x-y-u or u-v-y-x-u.
            # u-v-x-y-u: v-x edge, x-y edge, y-u edge. So x ∈ N(v), y ∈ N(u), x-y edge.
            # Also, x and y are common neighbors of... no. x is neighbor of v, y is neighbor of u.
            # For this to be a C4, we need x ≠ y, and the vertices are distinct.
            # Does this imply u and v have a common neighbor? Not directly.
            # Hmm, so my condition might not be sufficient.
            
            # Let's use a safer approach: build the graph, then verify C4-freeness separately.
            # For the random-greedy, use the condition adj[u] & adj[v] == 0 as a heuristic.
            
            random.shuffle(range(n * (n - 1) // 2))
            edge_attempts = []
            for i in range(n):
                for j in range(i + 1, n):
                    edge_attempts.append((i, j))
            random.shuffle(edge_attempts)

            edge_count = 0
            for u, v in edge_attempts:
                if time.time() > deadline - 5:
                    break
                # Heuristic: add edge if no common neighbor
                if adj[u] & adj[v] == 0:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edge_count += 1

            # Verify C4-freeness
            c4_free = True
            for i in range(n):
                for j in range(i + 1, n):
                    if adj[i] & (1 << j):  # edge exists
                        continue
                    common = adj[i] & adj[j]
                    if common & (common - 1):  # more than one bit set
                        c4_free = False
                        break
                if not c4_free:
                    break

            if not c4_free:
                continue

            # Check independent set of size 11 using B&B
            if not has_independent_set_11(adj, n):
                # Valid!
                if n > best_n:
                    best_n = n
                    # Extract edges
                    edges = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if adj[i] & (1 << j):
                                edges.append([i, j])
                    best_edges = edges

            if best_n >= 43:
                break
        if best_n >= 43:
            break

    return {'n': best_n, 'edges': best_edges}


def has_independent_set_11(adj, n):
    """Check if there's an independent set of size 11 using branch and bound with bitsets."""
    
    # Precompute: for each vertex, its non-neighbors (excluding itself)
    # non_neigh[i] = bitmask of vertices not adjacent to i, excluding i
    all_mask = (1 << n) - 1
    non_neigh = [0] * n
    for i in range(n):
        # Non-neighbors are all vertices except i and its neighbors
        non_neigh[i] = all_mask & ~(adj[i] | (1 << i))
    
    # Degree for ordering
    deg = [bin(adj[i]).count('1') for i in range(n)]
    
    # Sort vertices by degree descending for better pruning
    order = sorted(range(n), key=lambda x: -deg[x])
    
    # Use a recursive B&B with bitset representation of candidates
    # State: candidates (bitset), current_size
    # We want to find if we can pick 11 independent vertices from candidates
    
    import sys
    sys.setrecursionlimit(10000)
    
    found = [False]
    
    def bb(candidates, current_size):
        if found[0]:
            return
        if current_size == 11:
            found[0] = True
            return
        
        # Upper bound: current_size + popcount(candidates)
        if current_size + bin(candidates).count('1') < 11:
            return
        
        # Pick a vertex from candidates
        # Use the first set bit
        if candidates == 0:
            return
        
        # Get the first vertex in candidates
        v = (candidates & -candidates).bit_length() - 1
        
        # Option 1: include v
        new_candidates = candidates & non_neigh[v]
        if bb(new_candidates, current_size + 1):
            return
        
        # Option 2: exclude v
        candidates_without_v = candidates & ~(1 << v)
        if bb(candidates_without_v, current_size):
            return
    
    bb(all_mask, 0)
    return found[0]


# EVOLVE-BLOCK-END
