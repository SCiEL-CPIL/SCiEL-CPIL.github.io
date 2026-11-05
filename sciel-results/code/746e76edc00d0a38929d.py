"""Construct a C4-free graph with high minimum degree for R(C4, K1,39)."""

def construct():
    """
    Construct a C4-free graph on n vertices with minimum degree >= n - 39.
    
    Approach: Use the incidence graph of the projective plane of order 7 (Hadamard graph).
    The projective plane of order q has q^2+q+1 points and q^2+q+1 lines.
    The incidence graph (bipartite) has 2*(q^2+q+1) vertices, each of degree q+1.
    For q=7: n = 2*(49+7+1) = 114, which is too large (>47).
    
    For q=3: n = 2*(9+3+1) = 26, min degree = 4. Need min degree >= 26-39 = -13. OK but n is small.
    
    Let me think differently. We want to maximize n <= 47 with min degree >= n-39.
    
    For n=47: min degree >= 8
    For n=46: min degree >= 7
    For n=45: min degree >= 6
    
    Key insight: A C4-free graph cannot have too many edges. The maximum number of edges in a
    C4-free graph on n vertices is bounded by (n/4)(1+sqrt(4n-3)) approximately.
    
    Actually, let's use a different approach. The Erdos-Renyi graph or random approach won't work
    for deterministic construction.
    
    Let me use the polarity graph construction. For a projective plane of order q, the polarity graph
    (also called the C4-free graph from a symmetric bilinear form) has q^2+q+1 vertices and is C4-free.
    
    For q=6: 37 vertices, degree ~7. Need degree >= 37-39 = -2. OK.
    For q=7: 57 vertices - too large.
    
    Wait, I need to reconsider. Let me use a construction that gives more vertices.
    
    Actually, the best known C4-free graphs with high minimum degree come from finite geometries.
    
    Let me try: Take the complete graph K_n and remove edges to destroy all C4s while keeping high minimum degree.
    
    Alternative: Use a strongly regular graph that is C4-free.
    
    Actually, let me use the following construction:
    - Vertices are pairs (i, j) where i in Z_m, j in Z_k
    - Edges based on some rule that avoids C4
    
    Let me try a simpler approach: the bipartite incidence graph of a symmetric design.
    
    For a (v, k, lambda) symmetric design, the incidence graph has 2v vertices, each of degree k,
    and is C4-free iff lambda = 1 (i.e., it's a projective plane).
    
    Projective plane of order 3: v=13, 2v=26 vertices, degree 4.
    Projective plane of order 4: v=21, 2v=42 vertices, degree 5.
    
    For n=42, we need min degree >= 42-39 = 3. Degree 5 >= 3. Good!
    
    But can we do better? What about n=43, 44, ..., 47?
    
    For these, we need a C4-free graph with min degree >= n-39.
    n=47: need degree >= 8
    n=46: need degree >= 7
    n=45: need degree >= 6
    n=44: need degree >= 5
    n=43: need degree >= 4
    
    The projective plane of order 4 gives n=42, degree 5. That's already good.
    
    Can I add vertices to the projective plane graph while maintaining C4-free and sufficient min degree?
    
    Actually, let me just use the projective plane of order 4 construction, which gives n=42.
    But the task says "known to be 46 or 47, so 46 vertices would settle it."
    
    So I should aim for n=46 or 47 if possible.
    
    Let me think about this more carefully. The constraint is:
    1. C4-free
    2. min degree >= n - 39
    
    For n=47, min degree >= 8. Sum of degrees >= 47*8 = 376, so at least 188 edges.
    For a C4-free graph, the number of edges is bounded. By the extremal result, ex(n, C4) <= (n/4)(1+sqrt(4n-3)).
    For n=47: (47/4)(1+sqrt(185)) ≈ 11.75*(1+13.6) ≈ 11.75*14.6 ≈ 171.5
    
    So we can't have 188 edges in a C4-free graph on 47 vertices! n=47 is impossible.
    
    For n=46: min degree >= 7. Sum of degrees >= 322, so at least 161 edges.
    ex(46, C4) <= (46/4)(1+sqrt(181)) ≈ 11.5*(1+13.45) ≈ 11.5*14.45 ≈ 166.2
    So 161 <= 166, it might be possible.
    
    For n=45: min degree >= 6. Sum >= 270, edges >= 135.
    ex(45, C4) <= (45/4)(1+sqrt(177)) ≈ 11.25*(1+13.3) ≈ 11.25*14.3 ≈ 160.9
    135 <= 160.9, possible.
    
    So n=46 might be achievable. Let me try to construct one.
    
    I'll use a greedy/randomized approach: start with a high-degree graph and remove edges to destroy C4s.
    
    Actually, let me use a known construction. The polarity graph of a hyperoval in PG(2,q) for q even.
    For q=8: 65 vertices - too large.
    For q=6: not a prime power, no projective plane.
    
    Let me try a different approach: use a random C4-free graph with high minimum degree.
    """
    import random
    
    random.seed(42)
    
    # Try to construct a C4-free graph on n=46 vertices with min degree >= 7
    n = 46
    
    # Start with a random regular-ish graph and remove C4s
    # Use a greedy approach: build graph edge by edge, checking for C4s
    
    def has_c4_with_edge(adj, u, v):
        """Check if adding edge (u,v) creates a C4."""
        # A C4 through (u,v) exists if there's a common neighbor w of u and v
        # such that w has another common neighbor x with u or v, forming u-w-v-x-u
        # Actually, adding (u,v) creates a C4 iff there exist w, x such that:
        # u-w, w-v, v-x, x-u are all edges (or the reverse)
        # i.e., w is common neighbor of u,v and x is common neighbor of u,v with w != x
        # Wait, that's not right either.
        # A C4 containing edge (u,v) has form u-v-w-x-u where all four edges exist.
        # So we need: v-w, w-x, x-u all exist (and u-v is the new edge)
        # Equivalently: there exist w, x such that:
        #   v-w exists, w-x exists, x-u exists, and w != x, w != u, w != v, x != u, x != v
        # More simply: let N_u = neighbors of u (excluding v), N_v = neighbors of v (excluding u)
        # A C4 u-v-w-x-u exists iff there's w in N_v and x in N_u such that w-x is an edge,
        # with w != x.
        # Even simpler: adding (u,v) creates a C4 iff there exist w, x with:
        #   w adjacent to v, x adjacent to u, w-x adjacent, w != x
        # This is equivalent to: in the subgraph induced by N_u ∪ N_v (excluding u,v),
        # there's an edge between some w in N_v and x in N_u (w != x).
        
        N_u = set(adj[u]) - {v}
        N_v = set(adj[v]) - {u}
        
        for w in N_v:
            for x in N_u:
                if w == x:
                    continue
                if x in adj[w]:
                    return True
        return False
    
    # Build adjacency list
    adj = [set() for _ in range(n)]
    
    # Try to add edges greedily, prioritizing pairs that keep degrees balanced
    # First, create a candidate edge list in random order
    candidates = []
    for i in range(n):
        for j in range(i+1, n):
            candidates.append((i, j))
    
    random.shuffle(candidates)
    
    for u, v in candidates:
        if len(adj[u]) >= n - 39 and len(adj[v]) >= n - 39:
            # Both already have sufficient degree, skip unless we need more
            continue
        # Check if adding this edge creates a C4
        if not has_c4_with_edge(adj, u, v):
            adj[u].add(v)
            adj[v].add(u)
    
    # Check minimum degree
    min_deg = min(len(adj[i]) for i in range(n))
    print(f"Initial construction: min degree = {min_deg}")
    
    if min_deg < n - 39:
        # Try to add more edges to increase minimum degree
        for _ in range(100):  # Multiple passes
            improved = False
            for i in range(n):
                if len(adj[i]) < n - 39:
                    # Try to add an edge involving vertex i
                    for j in range(n):
                        if i == j or j in adj[i]:
                            continue
                        if not has_c4_with_edge(adj, i, j):
                            adj[i].add(j)
                            adj[j].add(i)
                            improved = True
                            break
            if not improved:
                break
    
    min_deg = min(len(adj[i]) for i in range(n))
    print(f"After improvement: min degree = {min_deg}")
    
    if min_deg < n - 39:
        # Fall back to projective plane of order 4 (n=42)
        q = 4
        n_points = q*q + q + 1
        n_lines = q*q + q + 1
        n = n_points + n_lines
        
        add = [
            [0,1,2,3],
            [1,0,3,2],
            [2,3,0,1],
            [3,2,1,0]
        ]
        mul = [
            [0,0,0,0],
            [0,1,2,3],
            [0,2,3,1],
            [0,3,1,2]
        ]
        
        def mul_vec(v, c):
            return [mul[c][v[i]] for i in range(3)]
            
        def is_zero(v):
            return all(x == 0 for x in v)
            
        inv = [0,1,3,2]
        
        points = []
        seen_points = set()
        for x in range(4):
            for y in range(4):
                for z in range(4):
                    v = [x,y,z]
                    if is_zero(v):
                        continue
                    idx = -1
                    for i in range(3):
                        if v[i] != 0:
                            idx = i
                            break
                    if idx == -1:
                        continue
                    c = inv[v[idx]]
                    canon = mul_vec(v, c)
                    key = tuple(canon)
                    if key not in seen_points:
                        seen_points.add(key)
                        points.append(canon)
                    
        lines = points
        
        edges = []
        for i, p in enumerate(points):
            for j, l in enumerate(lines):
                dot = 0
                for k in range(3):
                    dot = add[dot][mul[p[k]][l[k]]]
                if dot == 0:
                    edges.append([i, n_points + j])
                    
        return {'n': n, 'edges': edges}
    
    edges = []
    for i in range(n):
        for j in adj[i]:
            if i < j:
                edges.append([i, j])
    
    return {'n': n, 'edges': edges}
