"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
from collections import deque

def construct():
    n = 15
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    # We need to find a graph on 15 vertices maximizing edges such that:
    # No simple 3-edge path s-u-v-t (s != t) has deg(s) == deg(t) in the full graph.
    
    # Key insight: K_{7,8} is valid with 56 edges.
    # Part A = {0..6}, Part B = {7..14}
    # In K_{7,8}: deg(A vertices) = 8, deg(B vertices) = 7
    # Any 3-path s-u-v-t: 
    #   If s in A, t in A: need u in B (adj to s), v in A (adj to u), t in A (adj to v)
    #     deg(s)=8, deg(t)=8 -> VIOLATION!
    # Wait, let me re-check. In K_{7,8}:
    #   s in A, u in B, v in A, t in A: s-u is edge (A-B), u-v is edge (B-A), v-t is edge (A-A)? No!
    #   In K_{7,8}, there are NO edges within A or within B.
    #   So a 3-path s-u-v-t must alternate: s in A, u in B, v in A, t in B (or vice versa)
    #   deg(s) = 8 (if s in A), deg(t) = 7 (if t in B) -> not equal. OK.
    #   Or s in B, u in A, v in B, t in A: deg(s)=7, deg(t)=8 -> not equal. OK.
    # So K_{7,8} is valid.
    
    # Now we want to add edges within A or within B.
    # If we add edge (i,j) within A where i,j in {0..6}:
    #   Then deg(i) becomes 9, deg(j) becomes 9.
    #   Consider path s-u-v-t where s is in A\{i,j} with deg 8, and t is i or j with deg 9.
    #   We need: s-u (u in B), u-v (v in A), v-t.
    #   If t = i, v is adjacent to i. v could be j (if edge (i,j) added) or any other A vertex.
    #   Actually v must be in A (since u in B, u-v edge means v in A for K_{7,8} structure).
    #   s in A, u in B, v in A, t in A: but wait, v-t must be an edge. In K_{7,8}, A has no internal edges.
    #   If we add edge (i,j) within A, then v-t edge requires v,t both in A and edge between them.
    #   So if t=i and v=j (since (i,j) is an edge), then path s-u-j-i where s in A\{i,j}, u in B.
    #   deg(s)=8, deg(i)=9 -> not equal. OK so far.
    #   But what about s in A with deg 8 and t in A with deg 8 (a vertex not i or j)?
    #   Path s-u-v-t: s in A, u in B, v in A, t in A. Need edge v-t in A.
    #   If only edge in A is (i,j), then v-t must be (i,j) or (j,i).
    #   So t must be i or j. deg(t) = 9, deg(s) = 8. Not equal. OK.
    #   What about s in A\{i,j} (deg 8), u in B, v = i, t = j? deg(s)=8, deg(j)=9. Not equal.
    #   What about s = i (deg 9), u in B, v = j, t in A\{i,j} (deg 8)? deg(s)=9, deg(t)=8. Not equal.
    #   
    #   Hmm, but what about paths involving B vertices as endpoints?
    #   s in B (deg 7), u in A, v in B, t in B (deg 7). Need edge v-t in B. But B has no internal edges.
    #   So no such path exists.
    #   
    #   What about s in B (deg 7), u in A, v in A (edge in A?), t in B?
    #   Wait, u in A, v in A requires edge u-v in A. If only (i,j) in A:
    #   s in B, u=i, v=j, t in B: edge s-u (B-A) OK, u-v (i-j) OK, v-t (j-B) OK.
    #   deg(s)=7, deg(t)=7 -> VIOLATION!
    #   
    # So adding any edge within A creates a violation if there are at least 2 vertices in B.
    # Similarly for B.
    
    # So we CANNOT simply add intra-part edges to K_{7,8}. We need a different approach.
    
    # Let's think about what degree sequences are possible.
    # The constraint is: for every 3-edge path s-u-v-t with s!=t, deg(s) != deg(t).
    # This means: if deg(s) = deg(t) = d, then there is NO 3-edge path from s to t.
    # Equivalently: for each degree value d, the set of vertices with degree d must have
    # no pair at distance exactly 3 (via a simple path of length 3).
    
    # Strategy: Try to construct a graph with high edge count by using a degree sequence
    # where vertices of the same degree are far apart (distance > 3 or not connected by 3-path).
    
    # Alternative: Use a complete bipartite graph with modified structure.
    
    # Let's try a different approach: start with a random graph and use local search.
    
    def get_degrees(adj):
        return [sum(adj[i]) for i in range(n)]
    
    def is_valid_fast(adj, degs):
        """Check if graph satisfies the constraint. Returns True if valid."""
        # For each edge (u,v), for each s adjacent to u (s!=v), for each t adjacent to v (t!=u):
        # if s != t and degs[s] == degs[t], invalid.
        for u in range(n):
            for v in range(u+1, n):
                if not adj[u][v]:
                    continue
                # s adjacent to u, s != v
                for s in range(n):
                    if s == v or not adj[s][u]:
                        continue
                    # t adjacent to v, t != u
                    for t in range(n):
                        if t == u or not adj[v][t]:
                            continue
                        if s == t:
                            continue
                        if degs[s] == degs[t]:
                            return False
        return True
    
    def count_edges(adj):
        return sum(sum(row) for row in adj) // 2
    
    # Local search: start from K_{7,8}, try edge additions/removals
    
    # Initialize with K_{7,8}
    adj = [[0]*n for _ in range(n)]
    for i in range(7):
        for j in range(7, 15):
            adj[i][j] = 1
            adj[j][i] = 1
    
    degs = get_degrees(adj)
    assert is_valid_fast(adj, degs), "K_{7,8} should be valid"
    
    best_adj = [row[:] for row in adj]
    best_edges = count_edges(adj)
    
    # Local search: try adding or removing edges
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Try single edge additions first
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for (i, j) in all_pairs:
            if time.time() > deadline - 5:
                break
            if adj[i][j]:
                continue
            # Try adding edge (i,j)
            adj[i][j] = 1
            adj[j][i] = 1
            new_degs = get_degrees(adj)
            if is_valid_fast(adj, new_degs):
                new_edges = count_edges(adj)
                if new_edges > best_edges:
                    best_edges = new_edges
                    best_adj = [row[:] for row in adj]
                    improved = True
            else:
                adj[i][j] = 0
                adj[j][i] = 0
    
    # Now try edge swaps: remove one edge, add another
    if time.time() < deadline - 10:
        # Reset to best
        adj = [row[:] for row in best_adj]
        degs = get_degrees(adj)
        
        for _ in range(50):
            if time.time() > deadline - 10:
                break
            improved = False
            for (i, j) in all_pairs:
                if time.time() > deadline - 10:
                    break
                if not adj[i][j]:
                    continue
                # Try removing edge (i,j)
                adj[i][j] = 0
                adj[j][i] = 0
                for (k, l) in all_pairs:
                    if time.time() > deadline - 10:
                        break
                    if k == i or k == j or l == i or l == j:
                        continue
                    if adj[k][l]:
                        continue
                    # Try adding edge (k,l)
                    adj[k][l] = 1
                    adj[l][k] = 1
                    new_degs = get_degrees(adj)
                    if is_valid_fast(adj, new_degs):
                        new_edges = count_edges(adj)
                        if new_edges > best_edges:
                            best_edges = new_edges
                            best_adj = [row[:] for row in adj]
                            improved = True
                        # Keep the swap if valid (even if not better) to explore
                        adj[k][l] = 0
                        adj[l][k] = 0
                    else:
                        adj[k][l] = 0
                        adj[l][k] = 0
                # Restore edge (i,j)
                adj[i][j] = 1
                adj[j][i] = 1
            if not improved:
                break
    
    # Try 2-edge additions
    if time.time() < deadline - 15:
        adj = [row[:] for row in best_adj]
        for (i1, j1) in all_pairs:
            if time.time() > deadline - 15:
                break
            if adj[i1][j1]:
                continue
            adj[i1][j1] = 1
            adj[j1][i1] = 1
            for (i2, j2) in all_pairs:
                if time.time() > deadline - 15:
                    break
                if i2 == i1 or i2 == j1 or j2 == i1 or j2 == j1:
                    continue
                if adj[i2][j2]:
                    continue
                adj[i2][j2] = 1
                adj[j2][i2] = 1
                new_degs = get_degrees(adj)
                if is_valid_fast(adj, new_degs):
                    new_edges = count_edges(adj)
                    if new_edges > best_edges:
                        best_edges = new_edges
                        best_adj = [row[:] for row in adj]
                adj[i2][j2] = 0
                adj[j2][i2] = 0
            adj[i1][j1] = 0
            adj[j1][i1] = 0
    
    # Build edge list from best_adj
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i][j]:
                edges.append([int(i), int(j)])
    
    return edges
# EVOLVE-BLOCK-END
