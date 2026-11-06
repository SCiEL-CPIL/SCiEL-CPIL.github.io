import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin from 150s limit

    # Base graph: 8 disjoint K4s (n=32, alpha=8, C5-free)
    n_base = 32
    edges_base = []
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                edges_base.append([base + i, base + j])

    best_n = n_base
    best_edges = edges_base

    def build_adj(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def has_c5(n, adj):
        """Check if graph has a 5-cycle as a subgraph."""
        for v in range(n):
            neighbors = list(adj[v])
            for i in range(len(neighbors)):
                u = neighbors[i]
                for j in range(i + 1, len(neighbors)):
                    w = neighbors[j]
                    for x in adj[u]:
                        if x == v or x == w:
                            continue
                        for y in adj[w]:
                            if y == v or y == u or y == x:
                                continue
                            if y in adj[x]:
                                return True
        return False

    def alpha_exact(n, adj):
        """Exact maximum independent set size via branch and bound."""
        best = [0]
        adj_set = [set(adj[i]) for i in range(n)]
        
        def dfs(cand, size):
            if size + len(cand) <= best[0]:
                return
            if not cand:
                if size > best[0]:
                    best[0] = size
                return
            # Pick vertex with max degree in candidate subgraph for better pruning
            v = max(cand, key=lambda x: len(adj_set[x] & set(cand)))
            rest = [x for x in cand if x != v]
            
            # Branch 1: Include v
            new_cand = [x for x in rest if x not in adj_set[v]]
            dfs(new_cand, size + 1)
            
            # Branch 2: Exclude v
            dfs(rest, size)
        
        dfs(list(range(n)), 0)
        return best[0]

    def is_valid(n, edges):
        adj = build_adj(n, edges)
        if has_c5(n, adj):
            return False
        if alpha_exact(n, adj) >= 9:
            return False
        return True

    # Try to find a 33-vertex valid graph
    # Strategy: Add vertex 32, try different connection patterns to the 8 K4s
    # For C5-freeness, we need to be careful. Let's try systematic patterns.
    
    # Pattern: connect vertex 32 to one vertex in each K4.
    # There are 4^8 = 65536 patterns. We can check C5-freeness quickly.
    
    # Optimization: Precompute which patterns are C5-free.
    # A C5 involving vertex 32 must look like 32-a-b-c-d-32 where a,b,c,d are in the K4s.
    # Since K4s are cliques, if a and b are in the same K4, they are adjacent.
    # If a and c are in the same K4, they are adjacent.
    # If a and d are in the same K4, they are adjacent.
    # The cycle 32-a-b-c-d-32 requires edges (32,a), (a,b), (b,c), (c,d), (d,32).
    # If a,b are in same K4, (a,b) exists. If b,c are in same K4, (b,c) exists.
    # If c,d are in same K4, (c,d) exists.
    # So a C5 exists if there are K4s B1, B2, B3 such that:
    # a in B1, b in B2, c in B3, d in B1 (or similar cyclic arrangement)
    # Actually, since 32 connects to one vertex per K4, let's denote the chosen vertex in K4_i as v_i.
    # A cycle 32-v_i-v_j-v_k-v_l-32 requires:
    # (v_i, v_j) in E, (v_j, v_k) in E, (v_k, v_l) in E.
    # If i != j, v_i and v_j are in different K4s, so (v_i, v_j) is NOT an edge.
    # Therefore, for (v_i, v_j) to be an edge, we MUST have i == j. But v_i != v_i is false, they are the same vertex? No, distinct vertices in cycle.
    # Wait, the cycle vertices are distinct. So v_i, v_j, v_k, v_l must be distinct vertices.
    # If i == j, then v_i == v_j, which is not allowed.
    # Thus, any edge between two distinct vertices from different K4s does not exist.
    # So (v_i, v_j) is an edge ONLY if i == j? No, if i != j, they are in different components, no edge.
    # If i == j, they are the same vertex, not distinct.
    # Therefore, NO 5-cycle can involve vertex 32 and vertices from the base K4s if 32 connects to at most one vertex per K4?
    # Let's re-verify.
    # Cycle: 32 - a - b - c - d - 32.
    # Edges: (32,a), (a,b), (b,c), (c,d), (d,32).
    # a, b, c, d are in K4s.
    # If a and b are in different K4s, (a,b) is not an edge.
    # So a and b MUST be in the same K4.
    # Similarly, b and c MUST be in the same K4.
    # So a, b, c are in the same K4.
    # Similarly, c and d MUST be in the same K4.
    # So a, b, c, d are all in the same K4.
    # But 32 connects to only ONE vertex in that K4. Let it be v.
    # Then (32,a) requires a=v. (32,d) requires d=v.
    # So a=d=v. But a and d must be distinct in a simple cycle.
    # Contradiction.
    # Therefore, adding a vertex connected to at most one vertex per K4 NEVER creates a C5!
    
    # So we just need to check alpha < 9.
    # Base alpha is 8 (one from each K4).
    # Adding vertex 32. If 32 is independent from the chosen max independent set, alpha becomes 9.
    # The max independent set of the base graph picks one vertex from each K4.
    # Let S be a max independent set of size 8.
    # If 32 is not adjacent to any vertex in S, then S U {32} is an independent set of size 9.
    # To avoid this, 32 must be adjacent to at least one vertex in EVERY max independent set of size 8.
    # This is equivalent to: The set of vertices NOT adjacent to 32 must NOT contain a max independent set of size 8.
    # Let N(32) be the neighbors of 32. Let U = V \ N(32).
    # We need alpha(G[U]) <= 7.
    # Since G is a union of K4s, G[U] is a union of induced subgraphs on each K4.
    # In each K4, if we pick 1 vertex for N(32), then U has 3 vertices in that K4, which form a K3 (clique), so alpha is 1.
    # If we pick 0 vertices for N(32) in a K4, U has 4 vertices, alpha is 1.
    # If we pick 2 vertices for N(32) in a K4, U has 2 vertices, alpha is 1 (since they are adjacent in K4).
    # If we pick 3 vertices, U has 1, alpha 1.
    # If we pick 4, U has 0, alpha 0.
    # So in each K4, alpha(G[U] ∩ K4) is always 1 (unless U is empty, then 0).
    # So alpha(G[U]) = sum over K4s of alpha(G[U] ∩ K4).
    # If 32 connects to at least one vertex in EVERY K4, then each K4 contributes 1 to alpha(G[U]), total 8.
    # Wait, if each K4 contributes 1, total is 8. Then alpha(G[U]) = 8.
    # Then S (size 8) is independent in G[U]?
    # If S is a max independent set of G, it picks one vertex from each K4.
    # If 32 is adjacent to v_i in K4_i, then v_i is NOT in U.
    # So S cannot pick v_i. It must pick another vertex from K4_i.
    # Since K4_i has 4 vertices, and 1 is removed (v_i), there are 3 left. They form a K3.
    # So we can pick 1 from the remaining 3.
    # So there EXISTS an independent set of size 8 in G[U].
    # Thus alpha(G[U]) = 8.
    # Then S' (size 8) in G[U] plus 32 is an independent set of size 9 in G.
    # So connecting to exactly one vertex per K4 FAILS.
    
    # We need alpha(G[U]) <= 7.
    # This requires that for SOME K4, alpha(G[U] ∩ K4) = 0.
    # This means U ∩ K4 is empty, i.e., 32 is connected to ALL 4 vertices of some K4.
    # If 32 connects to all 4 vertices of K4_0, then U ∩ K4_0 is empty.
    # For other K4s, if 32 connects to at least one, alpha is 1.
    # Total alpha(G[U]) = 0 + 1 + ... + 1 = 7.
    # Then alpha(G) <= 8.
    
    # So the strategy is: Connect vertex 32 to ALL 4 vertices of one K4, and at least one vertex of each other K4.
    # Let's test this.
    
    # Pick K4_0 (vertices 0,1,2,3). Connect 32 to 0,1,2,3.
    # Connect 32 to one vertex from each of K4_1..K4_7.
    # To maximize chance of alpha < 9, we just need the condition above.
    # Let's construct one such graph.
    
    edges_try = list(edges_base)
    # Connect 32 to 0,1,2,3
    for v in [0, 1, 2, 3]:
        edges_try.append([32, v])
    # Connect 32 to one vertex from each other K4 (e.g., the first one)
    for b in range(1, 8):
        v = 4 * b
        edges_try.append([32, v])
        
    if time.time() < deadline - 5:
        if is_valid(33, edges_try):
            best_n = 33
            best_edges = edges_try
            # Try to find n=34
            # Add vertex 33.
            # Connect 33 to all 4 vertices of another K4, say K4_1.
            # And at least one from each other K4 (including K4_0? No, K4_0 is full for 32).
            # Wait, 33 is a new vertex.
            # Connect 33 to 4,5,6,7 (K4_1).
            # Connect 33 to one vertex from K4_0 (say 0), and one from K4_2..K4_7.
            edges_try2 = list(edges_try)
            for v in [4, 5, 6, 7]:
                edges_try2.append([33, v])
            edges_try2.append([33, 0])
            for b in range(2, 8):
                v = 4 * b
                edges_try2.append([33, v])
            
            if time.time() < deadline - 5:
                if is_valid(34, edges_try2):
                    best_n = 34
                    best_edges = edges_try2
                    # Try n=35
                    # Connect 34 to all 4 of K4_2, and one from others.
                    edges_try3 = list(edges_try2)
                    for v in [8, 9, 10, 11]:
                        edges_try3.append([34, v])
                    edges_try3.append([34, 0])
                    edges_try3.append([34, 4])
                    for b in range(3, 8):
                        v = 4 * b
                        edges_try3.append([34, v])
                    
                    if time.time() < deadline - 5:
                        if is_valid(35, edges_try3):
                            best_n = 35
                            best_edges = edges_try3

    return {'n': best_n, 'edges': best_edges}
