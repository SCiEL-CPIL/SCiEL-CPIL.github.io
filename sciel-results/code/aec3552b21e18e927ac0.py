"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    n = 40
    deadline = time.time() + 110  # safety margin before 120s limit
    
    def make_adj_from_edges(edges):
        adj = np.zeros((n, n), dtype=np.uint8)
        for u, v in edges:
            adj[u, v] = 1
            adj[v, u] = 1
        return adj
    
    def has_c6(adj):
        # Check for C6 using matrix multiplication approach
        # A C6 exists if there's a 6-cycle
        # For small n=40, we can check paths of length 5 that connect back
        # adj^5[i][j] > 0 and adj[i][j] == 0 and i != j means a path of length 5
        # But we need to be careful about counting simple cycles
        # Simpler: check if (adj^2)[i][j] > 0 and (adj^2)[j][k] > 0 etc.
        # Actually for n=40, let's use a direct check
        
        # A C6: v0-v1-v2-v3-v4-v5-v0
        # We need to check all 6-cycles
        # For efficiency, use the fact that C6 exists iff there's a pair of 
        # non-adjacent vertices with a path of length 5 between them
        # that doesn't repeat vertices
        
        # For n=40, brute force with bitsets
        adj_bool = adj.astype(bool)
        
        # For each pair of non-adjacent vertices (i, j), check if there's a path of length 5
        # This is expensive. Let's use a different approach.
        
        # Use the trace of A^6 to count closed walks of length 6
        # But this counts all closed walks, not just simple cycles
        # For C6 detection, we need a more careful approach
        
        # Given n=40, let's do a BFS-based check
        # For each starting vertex, do BFS up to depth 5 and check if we can return
        # to start via a non-adjacent vertex
        
        # Actually, the most efficient for n=40 is to check all possible 6-cycles
        # There are C(40,6) * (6-1)!/2 = 3830400 * 60 = ~230M which is too many
        
        # Better approach: for each edge (u,v), check if there's a path of length 4
        # from v to u that doesn't use edge (u,v) and doesn't repeat vertices
        # This is still expensive
        
        # For n=40, let's use the adjacency matrix power approach
        # A C6 exists iff tr(A^6) > 0 after removing contributions from shorter cycles
        # This is complex. Let's use a simpler but slower check for now.
        
        # Direct check: for each 6-tuple of distinct vertices forming a cycle
        # Too slow for 40 vertices
        
        # Use scipy or numpy for matrix operations
        # A^2[i,j] = number of common neighbors of i and j
        # A C6: v0-v1-v2-v3-v4-v5-v0
        # v0 and v3 are at distance 3
        # v1 and v4 are at distance 3
        # v2 and v5 are at distance 3
        
        # Let's use a randomized approach or a more clever check
        # For n=40, we can check: does there exist vertices a,b,c,d,e,f all distinct
        # such that a-b, b-c, c-d, d-e, e-f, f-a are all edges?
        
        # This is equivalent to checking if the 6th power of the adjacency matrix
        # has non-zero diagonal entries that correspond to simple cycles
        
        # Given the constraints, let's use a practical approach:
        # Check if there's a C6 by looking for two vertices at distance exactly 3
        # that have a common neighbor at distance 2 from both
        
        # For now, use a simple but correct check
        # Since we're doing greedy addition, we can check if adding an edge creates a C6
        # by checking if there's a path of length 5 between the endpoints
        
        pass  # This function will be replaced
    
    def would_create_c6(adj, u, v):
        """Check if adding edge (u,v) creates a C6."""
        # A C6 is created if there's a path of length 5 from u to v
        # that doesn't use the edge (u,v)
        # Path: u - x1 - x2 - x3 - x4 - v
        # where all vertices are distinct and none is u or v (except endpoints)
        
        # BFS from u up to depth 5, track paths
        # For n=40, this is feasible
        
        # Use bitset representation for efficiency
        # neighbors[w] is a bitset of neighbors of w
        
        # Build neighbor bitsets
        nbrs = [0] * n
        for i in range(n):
            for j in range(n):
                if adj[i, j]:
                    nbrs[i] |= (1 << j)
        
        # BFS from u, tracking paths of length up to 5
        # We want to find if v is reachable in exactly 5 steps
        # without repeating vertices
        
        # This is expensive. Let's use a different approach.
        # A C6 exists if there are vertices a,b,c,d,e,f distinct with
        # a-b-c-d-e-f-a all edges
        # If we add edge (u,v), then we need a path of length 5 from u to v
        
        # For efficiency, use meet-in-the-middle:
        # Find vertices at distance 2 from u and distance 3 from v
        # and check if they're connected
        
        # dist2_from_u: vertices at distance exactly 2 from u
        # dist3_from_v: vertices at distance exactly 3 from v
        
        # dist2_from_u = {w : exists x such that u-x and x-w, x != w, x != u}
        # dist3_from_v = {w : exists x,y such that v-x, x-y, y-w, all distinct, w != v}
        
        # Actually, let's just do a direct check for small n
        # Check all possible C6s that include edge (u,v)
        
        # A C6 containing edge (u,v) has form:
        # u - v - a - b - c - d - u
        # or
        # u - d - c - b - a - v - u (same cycle)
        
        # So we need to find a,b,c,d distinct, not u or v, such that:
        # v-a, a-b, b-c, c-d, d-u are all edges
        
        # This is checking if there's a path of length 4 from v to u
        # avoiding u and v in the interior
        
        # Let's enumerate: for each a in nbrs[v] \ {u}, 
        # for each b in nbrs[a] \ {u, v},
        # for each c in nbrs[b] \ {u, v, a},
        # for each d in nbrs[c] \ {u, v, a, b},
        # check if d is in nbrs[u]
        
        # This is O(deg(v) * deg(a) * deg(b) * deg(c)) which could be large
        # but for sparse graphs it's manageable
        
        # For better performance, use bitset operations
        # nbrs_bit[w] is a bitset
        
        # Path v-a-b-c-d-u
        # a in nbrs[v] \ {u}
        # b in nbrs[a] \ {u, v}
        # c in nbrs[b] \ {u, v, a}
        # d in nbrs[c] \ {u, v, a, b} and d in nbrs[u]
        
        # Use bitsets for speed
        mask_u = ~(1 << u) & ((1 << n) - 1)
        mask_v = ~(1 << v) & ((1 << n) - 1)
        mask_uv = mask_u & mask_v
        
        for a in range(n):
            if not (nbrs[v] >> a) & 1 or a == u:
                continue
            mask_a = mask_uv & ~(1 << a)
            for b in range(n):
                if not (nbrs[a] >> b) & 1 or not (mask_a >> b) & 1:
                    continue
                mask_b = mask_a & ~(1 << b)
                for c in range(n):
                    if not (nbrs[b] >> c) & 1 or not (mask_b >> c) & 1:
                        continue
                    # Check if any neighbor of c (excluding u,v,a,b) is a neighbor of u
                    candidates = nbrs[c] & nbrs[u] & mask_b & ~(1 << c)
                    if candidates:
                        return True
        return False
    
    # Start with a dense base: 8 disjoint K5s
    edges = []
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                edges.append([5 * k + a, 5 * k + b])
    
    adj = make_adj_from_edges(edges)
    
    # Precompute all non-edges
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    existing = set()
    for u, v in edges:
        existing.add((min(u, v), max(u, v)))
    
    non_edges = [(u, v) for u, v in all_pairs if (u, v) not in existing]
    
    # Greedy edge addition
    while time.time() < deadline:
        random.shuffle(non_edges)
        added = False
        for u, v in non_edges:
            if time.time() > deadline - 5:
                break
            if (u, v) in existing:
                continue
            if not would_create_c6(adj, u, v):
                adj[u, v] = 1
                adj[v, u] = 1
                edges.append([u, v])
                existing.add((u, v))
                non_edges.remove((u, v))
                added = True
        if not added:
            break
    
    # Convert to list of [u, v] pairs
    result = [[u, v] for u, v in edges]
    return result
# EVOLVE-BLOCK-END
