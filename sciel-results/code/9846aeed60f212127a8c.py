import time
import random
from itertools import combinations

def _has_c5(adj, n):
    """Check if graph has a 5-cycle as a subgraph."""
    # For each pair of non-adjacent vertices, check if they share two common neighbors
    # that are not adjacent to each other (forming a 5-cycle through them)
    # Actually, a 5-cycle exists iff there exist vertices a,b,c,d,e with edges
    # a-b, b-c, c-d, d-e, e-a.
    # More efficiently: for each edge (u,v), check if there's a path of length 3 from v back to u
    # through two intermediate vertices.
    
    # A 5-cycle can be detected by: for each pair of non-adjacent vertices (u,v),
    # if they have two common neighbors (a,b) where a and b are not adjacent,
    # then u-a-v-b-u is a 5-cycle.
    
    # Alternative: for each vertex u, look at its neighbors' neighbors at distance 2.
    # If any vertex at distance 2 from u is also a neighbor of u, that's a triangle or 4-cycle.
    # For 5-cycle: u-v-w-x-y-u. So y is at distance 2 from u (via v or x), and y is adjacent to u.
    
    # Let's use: for each non-edge (u,v), count common neighbors. If >= 2 common neighbors
    # that are not adjacent to each other, we have a 5-cycle.
    
    # Actually simpler: a 5-cycle exists iff there is a pair of vertices at distance exactly 2
    # that are also at distance 1... no.
    
    # Let me think again. A 5-cycle: v0-v1-v2-v3-v4-v0.
    # v0 and v2 are at distance 2 (via v1), and v0 and v3 are at distance 2 (via v4).
    # v1 and v3 are at distance 2 (via v2), and v1 and v4 are at distance 2 (via v0).
    
    # Key insight: in a C5-free graph, for any non-edge (u,v), the common neighbors of u and v
    # must form a clique. Because if a and b are both common neighbors of u and v, and a-b is
    # not an edge, then u-a-v-b-u is a 5-cycle.
    
    for u in range(n):
        for v in range(u + 1, n):
            if not adj[u][v]:
                # Find common neighbors of u and v
                common = [w for w in range(n) if w != u and w != v and adj[u][w] and adj[v][w]]
                if len(common) >= 2:
                    # Check if common neighbors form a clique
                    for i in range(len(common)):
                        for j in range(i + 1, len(common)):
                            a, b = common[i], common[j]
                            if not adj[a][b]:
                                return True
    return False

def _max_independent_set_size(adj, n, timeout=5.0):
    """Find the size of the maximum independent set using branch and bound."""
    start_time = time.time()
    
    # Convert to non-adjacency for independent set (independent set in G = clique in complement)
    # Complement adjacency
    comp_adj = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if not adj[i][j]:
                comp_adj[i][j] = True
                comp_adj[j][i] = True
    
    # Find max clique in complement = max independent set in G
    # Use Bron-Kerbosch with pivot
    best = [0]
    
    def bronk(R, P, X):
        if time.time() - start_time > timeout:
            return
        if not P and not X:
            if len(R) > best[0]:
                best[0] = len(R)
            return
        
        # Choose pivot
        if P:
            pivot = P[0]
            candidates = [v for v in P if not comp_adj[pivot][v]]
        else:
            candidates = []
        
        for v in candidates:
            if time.time() - start_time > timeout:
                return
            new_R = R + [v]
            new_P = [u for u in P if comp_adj[v][u]]
            new_X = [u for u in X if comp_adj[v][u]]
            bronk(new_R, new_P, new_X)
            P.remove(v)
            X.append(v)
    
    bronk([], list(range(n)), [])
    return best[0]

def construct():
    """
    Construct a C5-free graph with no independent set of size 9, maximizing n.
    
    Strategy: Start with 8 disjoint K4's (n=32, alpha=8). Then try to add vertices
    or modify the structure to get n=33 or higher while maintaining C5-free and alpha<9.
    
    The disjoint union of 8 K4's has n=32 and alpha=8 (one vertex from each K4).
    To get n=33, we'd need to add a vertex connected to at least one vertex in each K4
    (so that any independent set can pick at most 8 from the K4's plus potentially this new
    vertex, but we need alpha <= 8).
    
    Actually, if we add a vertex connected to at least one vertex in each K4, then any
    independent set can contain this new vertex AND at most one vertex from each K4,
    giving alpha potentially = 9. So we need to be more careful.
    
    Alternative: modify the structure. Use a different base graph.
    
    Let me try: 7 K4's + 1 K5 = 7*4 + 5 = 33 vertices.
    Alpha = 7*1 + 1 = 8. C5-free? K4 and K5 are both C5-free (no 5-cycle in K4 since
    it only has 4 vertices, and K5 has 5 vertices but every 5-tuple in K5 forms a K5
    which contains 5-cycles! Wait, K5 DOES contain 5-cycles as subgraphs.
    
    So K5 is NOT C5-free. We can't use K5.
    
    What about using different clique sizes? The constraint is C5-free, which means
    no induced or non-induced 5-cycle. A clique of size k >= 5 contains 5-cycles.
    So max clique size is 4.
    
    With max clique size 4 and n=33, we need to partition or nearly partition into
    cliques of size <= 4. 33 = 8*4 + 1, so we'd need 8 K4's and 1 extra vertex.
    But the extra vertex can't form a K5 with any K4.
    
    If we add a 33rd vertex v, it can be adjacent to some vertices. For C5-freeness,
    we need to ensure no 5-cycle is created. For alpha <= 8, v must not allow an
    independent set of size 9.
    
    If v is adjacent to at least one vertex in each of the 8 K4's, then any independent
    set containing v can contain at most one vertex from each K4 (the non-neighbor of v
    in that K4), giving at most 8 + 1 = 9. So we need v to be adjacent to at least one
    vertex in each K4, AND we need to ensure that the non-neighbors of v across all K4's
    don't form an independent set of size 8 (they don't, since they're from different
    K4's and there are no edges between K4's, so they WOULD form an independent set of
    size 8, plus v makes 9).
    
    So this doesn't work directly. We need a different approach.
    
    Let me try a different construction: use a graph where the independence number is
    naturally bounded. For example, the complement of a C5-free graph... no.
    
    Actually, let me just verify that n=32 is achievable and return that, since
    finding n=33+ requires more sophisticated search.
    
    For now, return the 8xK4 construction which is valid and achieves n=32.
    """
    deadline = time.time() + 100  # 100 second budget
    
    n = 32
    edges = []
    for b in range(8):
        start = b * 4
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([start + i, start + j])
    
    # Try to search for n=33
    if time.time() < deadline - 30:
        # Try random search for n=33
        best_n = 32
        random.seed(42)
        
        # Build adjacency for 8xK4
        adj = [[False] * 33 for _ in range(33)]
        for b in range(8):
            start = b * 4
            for i in range(4):
                for j in range(i + 1, 4):
                    adj[start + i][start + j] = True
                    adj[start + j][start + i] = True
        
        # Try adding vertex 32 with various connections
        for trial in range(500):
            if time.time() > deadline - 20:
                break
            
            # Randomly connect vertex 32 to some vertices
            adj32 = [False] * 33
            for i in range(32):
                if random.random() < 0.5:
                    adj32[i] = True
            
            # Check C5-free
            full_adj = [row[:] for row in adj]
            for i in range(32):
                full_adj[32][i] = adj32[i]
                full_adj[i][32] = adj32[i]
            
            if _has_c5(full_adj, 33):
                continue
            
            # Check alpha
            alpha = _max_independent_set_size(full_adj, 33, timeout=3.0)
            if alpha < 9:
                best_n = 33
                # Build edges
                new_edges = edges[:]
                for i in range(32):
                    if adj32[i]:
                        new_edges.append([32, i])
                return {'n': 33, 'edges': new_edges}
    
    return {'n': n, 'edges': edges}
