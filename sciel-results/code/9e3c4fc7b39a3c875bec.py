import time
import random
import itertools

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a K3,3-free graph on 20 vertices maximizing the number of edges.
    Strategy: Greedy edge addition with local repair or careful ordering.
    We start with an empty graph and try to add all possible edges in a randomized order,
    checking if adding an edge creates a K3,3. If it does, we skip it.
    To improve density, we can also try removing edges that block many other additions,
    but a simple greedy approach with a good ordering often works well for n=20.
    
    A K3,3 exists if there are two disjoint sets of 3 vertices {u1,u2,u3} and {v1,v2,v3}
    such that all 9 edges between them exist.
    Equivalently, for any pair of vertices (u, v), let N(u) and N(v) be their neighbors.
    If |N(u) ∩ N(v)| >= 3, say {w1, w2, w3}, then we must ensure that no two of w1, w2, w3
    are connected to both u and v in a way that forms K3,3? No, that's not the right check.
    
    Correct Check for K3,3:
    A graph contains K3,3 if there exist vertices u, v, w such that they share 3 common neighbors?
    No, K3,3 has bipartition (A, B) with |A|=3, |B|=3.
    If we pick 3 vertices A = {a1, a2, a3}, we need 3 vertices B = {b1, b2, b3} such that
    every ai is connected to every bj.
    
    Efficient Check:
    For every pair of vertices (u, v), let C = N(u) ∩ N(v).
    If |C| >= 3, then we have candidates for the other side of the K3,3?
    Not exactly. If u and v are in the same part of the K3,3, they don't share neighbors in the other part.
    If u and v are in different parts, they share the other 2 vertices in their respective parts?
    
    Let's use the property:
    A graph is K3,3-free if and only if for every set of 3 vertices S, the number of common neighbors of all vertices in S is less than 3?
    No.
    
    Let's look at the condition:
    If we have a K3,3 with parts A={a1,a2,a3} and B={b1,b2,b3}.
    Then for any pair a1, a2 in A, their common neighbors include b1, b2, b3. So |N(a1) ∩ N(a2)| >= 3.
    Conversely, if there exists a pair (u, v) such that |N(u) ∩ N(v)| >= 3, let C = {w1, w2, w3} be 3 such common neighbors.
    This does NOT necessarily imply a K3,3. It implies a K2,3 subgraph.
    To get K3,3, we need a third vertex u' such that u' is also connected to w1, w2, w3?
    If we have u, v, u' all connected to w1, w2, w3, then we have a K3,3 with parts {u, v, u'} and {w1, w2, w3}.
    
    So, the condition for adding an edge (u, v) is:
    Let C = N(u) ∩ N(v) after adding the edge? No, before adding.
    If we add edge (u, v), we might create a new common neighbor for some other pair?
    
    Actually, a simpler sufficient condition for a graph to be K3,3-free is:
    For every pair of vertices (u, v), |N(u) ∩ N(v)| < 3.
    If this holds, can we have a K3,3?
    If we have a K3,3 with parts A and B, then for any u, v in A, N(u) ∩ N(v) contains all of B, so size 3.
    So if we enforce |N(u) ∩ N(v)| <= 2 for all pairs (u, v), the graph is K3,3-free.
    This is a sufficient condition, but not necessary. It might restrict us too much.
    However, for n=20, this constraint allows for a fairly dense graph.
    Max edges with max common neighborhood 2?
    This is related to the Zarankiewicz problem.
    
    Let's try a greedy approach:
    1. Initialize empty graph.
    2. Generate all possible edges.
    3. Shuffle them.
    4. For each edge (u, v):
       - Tentatively add (u, v).
       - Check if any pair (x, y) now has >= 3 common neighbors.
         Specifically, adding (u, v) increases the common neighborhood count for:
         - Pairs involving u or v?
         - If x, y are both neighbors of u and both neighbors of v, then u and v become common neighbors of x and y? No.
         - If u and v are both neighbors of x and y, then x and y's common neighbors include u and v.
         - The critical pairs are (u, w) for w in N(v), (v, w) for w in N(u), and (u, v) itself?
         - Actually, adding (u, v) affects the common neighbor count of:
           1. The pair (u, v): Their common neighbors are N(u) ∩ N(v).
           2. Pairs (u, w) where w is a neighbor of v (and w != u): u and w now share v as a neighbor? No, v is a neighbor of u and w. So v is in N(u) ∩ N(w).
           3. Pairs (v, w) where w is a neighbor of u (and w != v): u is in N(v) ∩ N(w).
           4. Pairs (x, y) where x, y are both neighbors of u and both neighbors of v? No, that's the intersection.
           
    Let's refine the check.
    When adding edge (u, v):
    - New common neighbors of (u, v) are N(u) ∩ N(v).
    - For any w in N(v) (w != u), the pair (u, w) gains v as a common neighbor (if v is not already in N(u) ∩ N(w)? v is in N(u) now, and v is in N(w) by definition). So count(u, w) increases by 1.
    - For any w in N(u) (w != v), the pair (v, w) gains u as a common neighbor.
    
    We need to ensure that for all these affected pairs, the new count < 3.
    i.e., count(u, v) < 3, count(u, w) < 3 for all w in N(v)\{u}, count(v, w) < 3 for all w in N(u)\{v}.
    
    This check is O(n) per edge addition. Total O(n^3) for the whole graph. Very fast.
    """
    n = 20
    deadline = time.time() + 145  # Safety margin
    
    # We will use a greedy algorithm with a randomized edge ordering.
    # To maximize edges, we might want to start with a dense structure or just try all edges.
    # Since the condition |N(u) ∩ N(v)| < 3 is sufficient, let's stick to it.
    # This ensures no K3,3.
    
    # Data structures
    adj = [set() for _ in range(n)]
    edge_count = 0
    edges = []
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Randomize to avoid bias
    random.shuffle(all_edges)
    
    # Helper to check if adding (u, v) violates the condition
    def can_add(u, v):
        # Check pair (u, v)
        # Common neighbors of u and v
        common_uv = adj[u] & adj[v]
        if len(common_uv) >= 3:
            return False
        
        # Check pairs (u, w) for w in adj[v]
        for w in adj[v]:
            if w == u: continue
            # Common neighbors of u and w
            # Note: v is a common neighbor of u and w (since v in adj[u] now, v in adj[w])
            # We need to check if the count becomes >= 3
            # Current common neighbors of u and w: adj[u] & adj[w]
            # But adj[u] doesn't contain v yet.
            # So current common = (adj[u] & adj[w])
            # New common = (adj[u] & adj[w]) | {v} (if v not already in common)
            # Since v in adj[w] and we are adding v to adj[u], v will be in new common.
            # Is v already in adj[u] & adj[w]? No, because v is not in adj[u] yet.
            # So new count = len(adj[u] & adj[w]) + 1
            if len(adj[u] & adj[w]) + 1 >= 3:
                return False
        
        # Check pairs (v, w) for w in adj[u]
        for w in adj[u]:
            if w == v: continue
            # Symmetric to above
            if len(adj[v] & adj[w]) + 1 >= 3:
                return False
        
        return True

    for u, v in all_edges:
        if time.time() > deadline:
            break
        
        if can_add(u, v):
            adj[u].add(v)
            adj[v].add(u)
            edges.append([u, v])
            edge_count += 1
            
    return edges
# EVOLVE-BLOCK-END
