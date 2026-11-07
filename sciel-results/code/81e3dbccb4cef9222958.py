"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        u, v = e
        base_edges.add((min(u, v), max(u, v)))
    
    # Precompute adjacency for base path
    adj = [set() for _ in range(n)]
    for u, v in base_edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # We want to add edges to reduce diameter to 4.
    # The base path has diameter 23.
    # We need to find a set of edges S such that:
    # 1. S contains no edge in base_edges
    # 2. The graph (V, base_edges ∪ S) is triangle-free
    # 3. The graph is connected (it is, since base is a path)
    # 4. The diameter is at most 4
    # 5. |S| is minimized
    
    # Strategy:
    # A triangle-free graph with diameter 4 on 24 vertices.
    # The seed program adds all edges (u,v) where (u-v)%2 != 0 and (u,v) not in base.
    # That creates a bipartite graph. The path is bipartite with partitions:
    # Even: {0, 2, 4, ..., 22} (12 vertices)
    # Odd: {1, 3, 5, ..., 23} (12 vertices)
    # The seed adds all cross edges between even and odd that are not in the path.
    # This makes the graph a complete bipartite graph K_{12,12} minus the path edges.
    # Actually, the path edges are also cross edges. So the seed adds all remaining cross edges.
    # The resulting graph is K_{12,12} (since path edges are already there, and we add the rest).
    # Wait, the path edges are a subset of cross edges. So base ∪ added = all cross edges = K_{12,12}.
    # K_{12,12} has diameter 2. But it has many edges.
    # The number of cross edges in K_{12,12} is 12*12 = 144.
    # The path has 23 edges, all cross. So added = 144 - 23 = 121 edges.
    # That's a lot. We can do much better.
    
    # We need diameter ≤ 4. 
    # Consider a "hub" structure. Pick a central vertex, say 11 or 12.
    # Actually, let's think about what graph on 24 vertices with diameter 4 is triangle-free and sparse.
    
    # A key insight: In a triangle-free graph, if two vertices u and v are at distance 2,
    # they share a common neighbor. If they are at distance 4, there's a path u-a-b-c-v.
    
    # Let's try to construct a graph with a small number of edges.
    # The minimum number of edges in a triangle-free graph with diameter d on n vertices is related to
    # the Moore bound. For diameter 4 and triangle-free, the Moore bound is:
    # n ≤ 1 + Δ + Δ(Δ-1) + Δ(Δ-1)^2 + Δ(Δ-1)^3? No, that's for general graphs.
    # For triangle-free, the structure is more constrained.
    
    # Let's try a heuristic search. We'll use a greedy approach with backtracking.
    # Or better, let's think about the structure.
    
    # One good structure: a "double star" or "broom" but triangle-free.
    # Actually, consider the following: pick a center c. Connect c to all vertices at distance 1.
    # Then connect those to vertices at distance 2, etc. But we need to avoid triangles.
    
    # Alternative: Use the fact that the graph must be bipartite (since triangle-free and we start with a path).
    # Wait, triangle-free doesn't imply bipartite, but our base is bipartite and we add edges.
    # If we only add cross edges (between even and odd), the graph remains bipartite and hence triangle-free.
    # So we can restrict to adding cross edges only.
    
    # So we need to find a set of cross edges S such that the bipartite graph (Even, Odd, base_edges ∪ S)
    # has diameter ≤ 4.
    
    # Let E = {0, 2, 4, ..., 22}, O = {1, 3, 5, ..., 23}.
    # Base edges: (0,1), (1,2), (2,3), ..., (22,23).
    # In terms of E and O: (E_i, O_i) and (O_i, E_{i+1}) for i=0..11, where E_i = 2i, O_i = 2i+1.
    
    # We want to add cross edges to make the diameter ≤ 4.
    # The diameter of a bipartite graph is at most 4 iff for every pair of vertices, their distance is ≤ 4.
    # Since it's bipartite, distances are even or odd. Max distance 4 means:
    # - Vertices in same part (both E or both O) must be at distance ≤ 4.
    # - Vertices in different parts must be at distance ≤ 3 (since 5 > 4).
    
    # Let's think about which edges to add. A good strategy is to add edges that create short paths.
    # For example, if we add (E_i, O_j) for j ≠ i, i±1, we create shortcuts.
    
    # Let's try a specific construction:
    # Add edges from each E_i to O_{i+2} and O_{i-2} (mod 12, but carefully).
    # Actually, let's use a systematic approach.
    
    # We'll use a BFS-based check and a greedy addition.
    
    # First, let's define the bipartite graph.
    # E[i] = 2i, O[i] = 2i+1 for i in 0..11.
    
    # Base adjacency in bipartite form:
    # E[i] connected to O[i] and O[i-1] (if i>0)
    # O[i] connected to E[i] and E[i+1] (if i<11)
    
    # Let's represent the graph with adjacency sets.
    adj_bip = [set() for _ in range(n)]
    for u, v in base_edges:
        adj_bip[u].add(v)
        adj_bip[v].add(u)
    
    # We'll try to add edges greedily, checking diameter after each addition.
    # But greedy might not be optimal. Let's try a different approach.
    
    # Key observation: In a bipartite graph with parts E and O, if we add edges to make
    # the "even" part (E) have diameter ≤ 4 within the bipartite graph, and similarly for O,
    # and cross distances ≤ 3, we're good.
    
    # Let's try a specific small set of edges.
    # Idea: Add edges to create a "grid" structure.
    # For example, add (E_i, O_{i+1}) for all i (mod 12). But (E_i, O_{i+1}) is already in base for i=0..10.
    # Wait, base has (E_i, O_i) and (E_{i+1}, O_i). So (E_i, O_{i+1}) is NOT in base (except when i+1 = i, which is false).
    # Actually, base has (2i, 2i+1) and (2i+1, 2i+2). So (E_i, O_i) and (O_i, E_{i+1}).
    # So (E_i, O_{i+1}) is not in base. Let's add all (E_i, O_{i+1}) for i=0..10.
    # That's 11 edges. Plus we already have 23 base edges. Total 34 edges.
    # What's the diameter? Let's check.
    
    # Actually, let's just try a few candidate sets and check.
    
    def check_diameter(adj, n):
        """Check if diameter is at most 4."""
        from collections import deque
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            q = deque([start])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        if dist[v] > 4:
                            return False
                        q.append(v)
        return True
    
    def is_triangle_free(adj, n, added_edges):
        """Check if the graph is triangle-free."""
        # Since we only add cross edges and base is bipartite, it's always triangle-free.
        # But let's verify for safety.
        for u in range(n):
            for v in adj[u]:
                if v > u:
                    for w in adj[v]:
                        if w > v and w in adj[u]:
                            return False
        return True
    
    # Let's try a few candidate edge sets.
    
    # Candidate 1: Add (E_i, O_{i+1}) for i=0..10
    cand1 = set()
    for i in range(11):
        u = 2*i
        v = 2*(i+1)+1
        if (min(u,v), max(u,v)) not in base_edges:
            cand1.add((min(u,v), max(u,v)))
    
    # Candidate 2: Add (E_i, O_{i+2}) for i=0..9
    cand2 = set()
    for i in range(10):
        u = 2*i
        v = 2*(i+2)+1
        if (min(u,v), max(u,v)) not in base_edges:
            cand2.add((min(u,v), max(u,v)))
    
    # Candidate 3: Add (E_i, O_{i+1}) and (E_i, O_{i-1}) for appropriate i
    cand3 = set()
    for i in range(12):
        for j in [i+1, i-1]:
            if 0 <= j <= 11:
                u = 2*i
                v = 2*j+1
                if (min(u,v), max(u,v)) not in base_edges:
                    cand3.add((min(u,v), max(u,v)))
    
    # Let's check each candidate.
    best = None
    best_count = float('inf')
    
    candidates = [cand1, cand2, cand3]
    
    for cand in candidates:
        adj_test = [set(adj) for adj in adj_bip]
        for u, v in cand:
            adj_test[u].add(v)
            adj_test[v].add(u)
        
        if check_diameter(adj_test, n):
            count = len(cand)
            if count < best_count:
                best_count = count
                best = set(cand)
    
    # If none of the simple candidates work, try a more systematic search.
    if best is None:
        # Try adding edges one by one, checking diameter.
        # This is expensive, so let's try a smarter approach.
        
        # Let's think: what's the minimum number of edges needed?
        # The base path has 23 edges. We need diameter ≤ 4.
        # In a path, the diameter is n-1 = 23. To reduce to 4, we need shortcuts.
        
        # A known result: the minimum number of edges to add to a path to get diameter d is related to
        # the number of "jumps" needed.
        
        # Let's try a greedy approach: repeatedly add the edge that most reduces the diameter.
        
        adj_greedy = [set(adj) for adj in adj_bip]
        added_greedy = set()
        
        def get_diameter(adj, n):
            from collections import deque
            max_d = 0
            for start in range(n):
                dist = [-1] * n
                dist[start] = 0
                q = deque([start])
                while q:
                    u = q.popleft()
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            max_d = max(max_d, dist[v])
                            q.append(v)
            return max_d
        
        # Generate all possible cross edges not in base.
        all_cross = set()
        for i in range(12):
            for j in range(12):
                u = 2*i
                v = 2*j+1
                e = (min(u,v), max(u,v))
                if e not in base_edges:
                    all_cross.add(e)
        
        # Greedy: add edges that most reduce the diameter.
        deadline = time.time() + 60
        current_diam = get_diameter(adj_greedy, n)
        
        while current_diam > 4 and time.time() < deadline:
            best_edge = None
            best_diam = current_diam
            for e in all_cross:
                if e in added_greedy:
                    continue
                u, v = e
                adj_test = [set(adj) for adj in adj_greedy]
                adj_test[u].add(v)
                adj_test[v].add(u)
                d = get_diameter(adj_test, n)
                if d < best_diam:
                    best_diam = d
                    best_edge = e
            if best_edge is None:
                break
            u, v = best_edge
            adj_greedy[u].add(v)
            adj_greedy[v].add(u)
            added_greedy.add(best_edge)
            current_diam = best_diam
        
        if current_diam <= 4:
            count = len(added_greedy)
            if count < best_count:
                best_count = count
                best = added_greedy
    
    # Convert best to list of lists
    if best is not None:
        result = [[u, v] for u, v in best]
        # Ensure types are Python int
        result = [[int(u), int(v)] for u, v in result]
        return result
    else:
        # Fallback: return the seed solution
        return [[u, v] for u in range(n) for v in range(u+1, n) if (u-v)%2 and (min(u,v), max(u,v)) not in base_edges]
# EVOLVE-BLOCK-END
