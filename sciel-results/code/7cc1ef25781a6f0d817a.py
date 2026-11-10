import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin (150s limit)

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj

    def bron_kerbosch(adj):
        n = len(adj)
        cliques = []
        
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
            
            P_X = P.union(X)
            if not P_X:
                return
            
            max_deg = -1
            pivot = None
            for u in P_X:
                deg = len(P.intersection(adj[u]))
                if deg > max_deg:
                    max_deg = deg
                    pivot = u
            
            for v in list(P - adj[pivot]):
                bk(R.union({v}), P.intersection(adj[v]), X.intersection(adj[v]))
                P.remove(v)
                X.add(v)

        bk(set(), set(range(n)), set())
        return cliques

    def solve_instance(n, p, seed):
        adj = random_graph(n, p, seed)
        cliques = bron_kerbosch(adj)
        
        cliques = [c for c in cliques if len(c) >= 2]
        
        if not cliques:
            return []
        
        m = len(cliques)
        
        # Precompute vertex to cliques mapping
        vertex_to_cliques = [[] for _ in range(n)]
        for i, c in enumerate(cliques):
            for v in c:
                vertex_to_cliques[v].append(i)
        
        # Build the clique-edge graph: edges connect pairs of vertices that co-occur
        # in at least one maximal clique. Hitting all maximal cliques of size >= 2
        # is equivalent to finding a minimum vertex cover in this derived graph.
        # A vertex cover S covers edge (u,v) if u in S or v in S.
        # Every edge (u,v) in the clique-edge graph corresponds to a pair in some
        # maximal clique C. If S is a vertex cover, then for every such pair, at
        # least one of u,v is in S, so S intersects C. Conversely, if S intersects
        # every maximal clique of size >= 2, then for every edge (u,v) (which is in
        # some clique C), at least one of u,v is in S (since S intersects C and
        # C is a clique so all pairs in C are edges... wait, no.
        
        # Actually, the correct equivalence is:
        # - A set S hits every maximal clique of size >= 2 iff S is a vertex cover
        #   of the graph G' where G' has the same vertices and an edge (u,v) exists
        #   iff u and v are in the same maximal clique of size >= 2.
        # 
        # Proof: If S is a vertex cover of G', then for any maximal clique C of size
        # >= 2, C induces a clique in G' (all pairs in C are edges in G'). Since S
        # covers all edges, for any edge (u,v) in C, at least one of u,v is in S.
        # In particular, pick any pair in C, so S intersects C.
        # Conversely, if S intersects every maximal clique of size >= 2, then for
        # any edge (u,v) in G', there exists a maximal clique C containing both u
        # and v (since the edge was created because u,v co-occur in some maximal
        # clique). Since |C| >= 2 and S intersects C, at least one of u,v is in S.
        # So S is a vertex cover.
        
        # Build G' adjacency
        g_prime_adj = [set() for _ in range(n)]
        for c in cliques:
            verts = list(c)
            for i in range(len(verts)):
                for j in range(i + 1, len(verts)):
                    u, v = verts[i], verts[j]
                    g_prime_adj[u].add(v)
                    g_prime_adj[v].add(u)
        
        # Count edges in G'
        edge_count = sum(len(s) for s in g_prime_adj) // 2
        
        # Lower bound: maximum matching in G' is a lower bound on min vertex cover
        # (by Konig's theorem for bipartite, but in general it's still a valid LB)
        # Use greedy matching as a quick LB
        def greedy_matching_lower_bound():
            matched = [False] * n
            matching_size = 0
            for u in range(n):
                if not matched[u]:
                    for v in g_prime_adj[u]:
                        if not matched[v]:
                            matched[u] = True
                            matched[v] = True
                            matching_size += 1
                            break
            return matching_size
        
        lb_matching = greedy_matching_lower_bound()
        
        # Also compute LP relaxation lower bound using a simple approach
        # For now, use the matching bound
        
        best_solution = None
        best_size = float('inf')
        
        # Strategy 1: Greedy vertex cover approximation (2-approx)
        def greedy_vertex_cover():
            """Standard greedy 2-approx for minimum vertex cover."""
            S = set()
            remaining_edges = set()
            for u in range(n):
                for v in g_prime_adj[u]:
                    if u < v:
                        remaining_edges.add((u, v))
            
            while remaining_edges:
                if time.time() > deadline - 15.0:
                    break
                # Pick a random edge
                u, v = random.choice(list(remaining_edges))
                # Add the vertex with higher degree in remaining edges
                deg_u = sum(1 for e in remaining_edges if u in e)
                deg_v = sum(1 for e in remaining_edges if v in e)
                chosen = u if deg_u >= deg_v else v
                S.add(chosen)
                # Remove all edges incident to chosen
                remaining_edges = {(a, b) for (a, b) in remaining_edges if chosen not in (a, b)}
            
            return S
        
        # Strategy 2: Local search on vertex cover
        def is_valid_vertex_cover(S):
            for u in range(n):
                if u in S:
                    continue
                for v in g_prime_adj[u]:
                    if v not in S:
                        return False
            return True
        
        def local_search_vc(S, max_iters=200):
            """Local search for minimum vertex cover: try removing and swapping."""
            S = set(S)
            improved = True
            iterations = 0
            while improved and time.time() < deadline - 15.0 and iterations < max_iters:
                improved = False
                iterations += 1
                
                # Try removing each vertex
                verts_to_try = list(S)
                random.shuffle(verts_to_try)
                for v in verts_to_try:
                    if v not in S:
                        continue
                    S.remove(v)
                    if is_valid_vertex_cover(S):
                        improved = True
                        # Keep it removed
                    else:
                        S.add(v)
                
                # Try 1-for-1 swap: remove v, add w
                if not improved and time.time() < deadline - 15.0:
                    verts_to_try = list(S)
                    random.shuffle(verts_to_try)
                    for v in verts_to_try:
                        if v not in S:
                            continue
                        S.remove(v)
                        # Find a vertex w not in S that, when added, makes it valid
                        # and doesn't increase size (we're swapping, so size stays same)
                        # Actually, for improvement we want to reduce size, so just
                        # check if removal is valid (handled above). For swaps, we
                        # might find a different vertex of same size.
                        # Let's try: for each v in S, remove it, then find minimal
                        # set of vertices to add back to make it valid.
                        S.add(v)  # restore for now
                    
                    # Better approach: try all pairs (v in S, w not in S)
                    for v in list(S):
                        if time.time() > deadline - 15.0:
                            break
                        S.remove(v)
                        # Check if valid without v
                        if is_valid_vertex_cover(S):
                            improved = True
                            continue
                        # Try adding one vertex w to make it valid
                        for w in range(n):
                            if w in S:
                                continue
                            S.add(w)
                            if is_valid_vertex_cover(S):
                                # Same size, different composition - might help future removals
                                improved = True
                                break
                            S.remove(w)
                        S.add(v)  # restore
                        if improved:
                            break
                        if time.time() > deadline - 15.0:
                            break
                
                # Try 2-for-1: remove two vertices, add one
                if not improved and time.time() < deadline - 15.0 and len(S) >= 2:
                    S_copy = set(S)
                    verts = list(S)
                    random.shuffle(verts)
                    for i in range(len(verts)):
                        for j in range(i + 1, len(verts)):
                            if time.time() > deadline - 15.0:
                                break
                            v1, v2 = verts[i], verts[j]
                            S.remove(v1)
                            S.remove(v2)
                            # Try adding one vertex w
                            for w in range(n):
                                if w in S:
                                    continue
                                S.add(w)
                                if is_valid_vertex_cover(S):
                                    improved = True
                                    break
                                S.remove(w)
                            S.add(v1)
                            S.add(v2)
                            if improved:
                                break
                        if improved:
                            break
                    if not improved:
                        S = S_copy
            
            return S
        
        # Run multiple restarts
        restarts = 50
        for r in range(restarts):
            if time.time() > deadline - 10.0:
                break
            
            # Get initial solution via greedy vertex cover
            S = greedy_vertex_cover()
            
            if not is_valid_vertex_cover(S):
                # Fallback: add all vertices
                S = set(range(n))
            
            # Local search to improve
            S = local_search_vc(S, max_iters=100)
            
            if len(S) < best_size:
                best_size = len(S)
                best_solution = set(S)
            
            # If we hit the lower bound, we're optimal
            if best_size <= lb_matching:
                break
        
        # Additional: try starting from a different initial solution
        # Use a degree-based heuristic
        for r in range(20):
            if time.time() > deadline - 10.0:
                break
            
            S = set()
            # Greedy: always pick the vertex covering most uncovered edges
            remaining_edges = set()
            for u in range(n):
                for v in g_prime_adj[u]:
                    if u < v:
                        remaining_edges.add((u, v))
            
            while remaining_edges and time.time() < deadline - 10.0:
                best_v = -1
                best_count = 0
                for v in range(n):
                    if v in S:
                        continue
                    count = sum(1 for e in remaining_edges if v in e)
                    if count > best_count:
                        best_count = count
                        best_v = v
                if best_v == -1 or best_count == 0:
                    break
                S.add(best_v)
                remaining_edges = {(a, b) for (a, b) in remaining_edges if best_v not in (a, b)}
            
            if not is_valid_vertex_cover(S):
                S = set(range(n))
            
            S = local_search_vc(S, max_iters=100)
            
            if len(S) < best_size:
                best_size = len(S)
                best_solution = set(S)
            
            if best_size <= lb_matching:
                break
        
        return sorted(best_solution) if best_solution else []

    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline - 15.0:
            # Fallback: return all touched vertices
            adj = random_graph(n, p, seed)
            touched = {x for u in range(n) for v in adj[u] for x in (u,v)}
            out.append(sorted(touched))
        else:
            out.append(solve_instance(n, p, seed))
            
    return out
