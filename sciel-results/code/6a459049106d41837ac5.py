import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n = 31
    m = 91
    
    # 1. Construct PG(2,5) points
    points = []
    for a in range(5):
        for b in range(5):
            points.append((1, a, b))
    for b in range(5):
        points.append((0, 1, b))
    points.append((0, 0, 1))
    
    # Map points to indices 0..30
    point_index = {p: i for i, p in enumerate(points)}
    
    # 2. Construct adjacency matrix for polarity graph
    adj = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i+1, n):
            p1 = points[i]
            p2 = points[j]
            # Dot product mod 5
            dot = (p1[0]*p2[0] + p1[1]*p2[1] + p1[2]*p2[2]) % 5
            if dot == 0:
                adj[i, j] = 1
                adj[j, i] = 1
                
    # Verify edge count
    edge_count = np.sum(adj) // 2
    # Should be 90
    
    # Helper: compute neighbors from adj
    def get_neighbors(adj):
        return [list(np.where(adj[i] == 1)[0]) for i in range(n)]
    
    # Helper: count C4s using common neighbor counts
    def count_c4s(adj, neighbors):
        total = 0
        for u in range(n):
            for v in range(u+1, n):
                # Count common neighbors
                common_count = 0
                for w in neighbors[u]:
                    if adj[w, v] == 1:
                        common_count += 1
                if common_count >= 2:
                    total += common_count * (common_count - 1) // 2
        return total // 2
    
    # Helper: compute delta in C4 count when adding edge (i,j)
    def delta_add_edge(adj, i, j):
        # Adding edge (i,j) creates C4s for each pair of common neighbors
        # Also, it increases common neighbor count for pairs (i,v) and (j,v)
        # where v is a common neighbor of i and j
        # New C4s = C(common_neighbors, 2)
        # But we also need to account for the fact that adding the edge changes
        # the common neighbor counts for other pairs.
        # Actually, the standard formula: when adding edge (i,j), the number of new C4s
        # is the number of paths of length 3 between i and j, which is
        # sum_{w in N(i)} |N(w) ∩ N(j)| - |N(i) ∩ N(j)|
        # Wait, let me think more carefully.
        # A new C4 containing edge (i,j) must be of the form i-j-w-x-i where
        # w is a neighbor of j, x is a neighbor of i, and w-x is an edge.
        # So new C4s = number of edges between N(i)\{j} and N(j)\{i}
        # But we need to be careful not to double count.
        # Actually, each new C4 containing edge (i,j) corresponds to a path i-x-w-j
        # where x in N(i), w in N(j), and (x,w) is an edge.
        # So new C4s = |{(x,w) : x in N(i), w in N(j), x≠w, adj[x,w]=1, x≠j, w≠i}|
        # But since i and j are not adjacent yet, N(i) doesn't contain j and N(j) doesn't contain i.
        # So new C4s = |{(x,w) : x in N(i), w in N(j), adj[x,w]=1}|
        # But this counts each C4 twice? No, each C4 i-j-w-x-i is counted once
        # with (x,w) = (x,w).
        # Wait, let's use the common neighbor approach.
        # When we add edge (i,j), the new C4s are those that use this edge.
        # A C4 using edge (i,j) has the form i-j-w-x-i where w∈N(j), x∈N(i), (w,x)∈E.
        # So new C4s = number of edges between N(i) and N(j).
        count = 0
        for x in np.where(adj[i] == 1)[0]:
            for w in np.where(adj[j] == 1)[0]:
                if adj[x, w] == 1:
                    count += 1
        return count // 2  # Each C4 is counted once, but let me verify
    
    # Actually, let me use a simpler approach: just recount after each move.
    # With n=31, this should be fast enough.
    
    # 3. Find best edge to add
    best_delta = float('inf')
    best_edge = None
    
    neighbors = get_neighbors(adj)
    
    for i in range(n):
        for j in range(i+1, n):
            if adj[i, j] == 0:
                # Count new C4s if we add edge (i,j)
                # New C4s = number of paths of length 3 from i to j
                count = 0
                for w in neighbors[i]:
                    for x in neighbors[w]:
                        if adj[x, j] == 1 and x != i and x != j:
                            count += 1
                if count < best_delta:
                    best_delta = count
                    best_edge = (i, j)
                    
    # Add the best edge
    if best_edge:
        i, j = best_edge
        adj[i, j] = 1
        adj[j, i] = 1
        neighbors = get_neighbors(adj)
        
    current_c4s = count_c4s(adj, neighbors)
    
    # Try to improve with local search
    edges = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i, j] == 1]
    non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i, j] == 0]
    
    improved = True
    max_iterations = 5000
    iteration = 0
    
    while improved and iteration < max_iterations and time.time() < deadline - 5:
        improved = False
        iteration += 1
        
        if current_c4s <= 1:
            break
            
        # Try all possible swaps (remove one edge, add one non-edge)
        # But that's O(91 * 374) = ~34k swaps, each requiring a C4 count
        # Let's try a subset: random swaps
        for _ in range(min(200, max(50, (deadline - 5 - time.time()) * 10))):
            if time.time() > deadline - 5:
                break
                
            # Pick a random edge to remove
            e_idx = random.randint(0, len(edges)-1)
            u, v = edges[e_idx]
            
            # Pick a random non-edge to add
            ne_idx = random.randint(0, len(non_edges)-1)
            i, j = non_edges[ne_idx]
            
            # Apply swap
            adj[u, v] = 0
            adj[v, u] = 0
            adj[i, j] = 1
            adj[j, i] = 1
            
            neighbors = get_neighbors(adj)
            new_c4s = count_c4s(adj, neighbors)
            
            if new_c4s < current_c4s:
                current_c4s = new_c4s
                edges[e_idx] = (i, j)
                non_edges[ne_idx] = (u, v)
                improved = True
            else:
                # Revert
                adj[i, j] = 0
                adj[j, i] = 0
                adj[u, v] = 1
                adj[v, u] = 1
                neighbors = get_neighbors(adj)
                
    # Return edges
    result = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i, j] == 1:
                result.append([i, j])
                
    return result
