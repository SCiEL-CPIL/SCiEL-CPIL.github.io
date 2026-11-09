import time
from itertools import combinations, permutations

def construct():
    """
    Constructs a C5-free graph with no independent set of size 9.
    
    Strategy: 8 disjoint K4s (vertices 0-31) plus vertex 32 connected to
    all 4 vertices of K4_0 (vertices 0-3). This gives n=33.
    
    C5-free proof: The only edges between different K4-blocks go through
    vertex 32. A C5 through 32 would need 32-a-b-c-d-32 with a,d in K4_0
    (since 32 only connects to K4_0). But then b must be adjacent to a
    (in K4_0) and c must be adjacent to b and d must be adjacent to c and
    32. If b is in K4_0, we need 5 distinct vertices in K4_0 (impossible,
    |K4_0|=4). If b is in K4_j (j>=1), c must be in K4_j, and d must be
    adjacent to c (in K4_j) AND to 32 (in K4_0) - impossible since no
    edges between K4_j and K4_0. C5s not through 32 lie entirely within
    one K4 (no C5 in K4) or span multiple K4s (no edges between them).
    
    Alpha proof: Any independent set can contain at most 1 vertex from each
    K4_j. Vertex 32 is adjacent to all of K4_0, so an independent set
    containing 32 can have at most 7 other vertices (one from each of
    K4_1..K4_7), giving alpha <= 8. An independent set not containing 32
    has at most 8 vertices (one from each K4_j). So alpha = 8 < 9.
    
    We verify computationally as a safety check.
    """
    start_time = time.time()
    deadline = 130.0  # Safety margin

    def fast_verify_c5_free(adj, n):
        """Check if graph has no C5 subgraph using BFS-based approach."""
        # For each pair of non-adjacent vertices, check if there's a path of length 3
        # between them (which would form a C5 with the edge... no, C5 needs 5 vertices)
        # Better: for each vertex v, check if there's a path v-a-b-c-d-v where all 5 are distinct
        # and edges (v,a),(a,b),(b,c),(c,d),(d,v) exist.
        # Use: for each ordered pair (v, a) with v-a edge, find common neighbors of v and a
        # ... this gets complex. Let's use a smarter structural check.
        
        # Given the specific structure (8 K4s + one vertex connected to one K4),
        # we can verify structurally. But let's do a computational check that's fast.
        
        # For each 5-set, check if it contains a C5. C(33,5) = 237336, times 12 cycles = ~2.8M checks.
        # Each check is 5 set lookups. Should be fast enough.
        for verts in combinations(range(n), 5):
            v0, v1, v2, v3, v4 = verts
            # Check all 12 cycle orderings (fix v0, permute remaining 4, but only 12 distinct cycles)
            # Actually, for 5 labeled vertices, there are (5-1)!/2 = 12 distinct cycles.
            # Fix v0 as start. The remaining 4 can be arranged in 4!/2 = 12 ways (reversal equivalent).
            # But since we're checking directed cycle v0-v1-v2-v3-v4-v0, we need all 4! = 24 permutations
            # of {v1,v2,v3,v4}, but each cycle is counted twice (forward/reverse), so 12 unique.
            # For simplicity, just check all 24 and it's fine (just redundant).
            rest = [v1, v2, v3, v4]
            for p in permutations(rest):
                a, b, c, d = p
                if a in adj[v0] and b in adj[a] and c in adj[b] and d in adj[c] and v0 in adj[d]:
                    return False
        return True

    def fast_verify_alpha_lt_9(adj, n):
        """Check if independence number is < 9."""
        # For this specific graph structure, we can compute alpha directly.
        # But let's verify computationally.
        # C(33,9) = 129184560 - too many!
        # Use a smarter approach: backtracking with pruning.
        
        # Since the graph is a union of cliques (plus one extra vertex),
        # we can exploit the structure. But for safety, let's do a
        # branch-and-bound that should be fast for this structure.
        
        # Actually, for this specific graph, alpha is provably 8.
        # Let's just do a quick check: try to find an independent set of size 9.
        # Use greedy + backtracking.
        
        # Vertices grouped: K4_0 = {0,1,2,3}, K4_1 = {4,5,6,7}, ..., K4_7 = {28,29,30,31}, extra = {32}
        # 32 is connected to all of K4_0.
        
        # Any independent set: at most 1 from each K4_j, and if 32 is included, 0 from K4_0.
        # Max size: 8 (without 32: one from each K4) or 1+7=8 (with 32).
        # So alpha = 8 < 9. This is a structural proof.
        
        # Computational verification via sampling + targeted check:
        # Check that no independent set of size 9 exists by verifying the structure.
        
        # Verify: 32 is adjacent to all of {0,1,2,3}
        for v in range(4):
            if v not in adj[32]:
                return False  # Structure broken
        
        # Verify: each K4_j is a clique
        for b in range(8):
            base = 4 * b
            for i in range(4):
                for j in range(i + 1, 4):
                    if (base + j) not in adj[base + i]:
                        return False
        
        # Verify: no edges between different K4s (except through 32)
        for b in range(8):
            for c in range(b + 1, 8):
                base_b = 4 * b
                base_c = 4 * c
                for i in range(4):
                    for j in range(4):
                        u = base_b + i
                        v = base_c + j
                        if v in adj[u]:
                            return False
        
        # Verify: 32 only connects to K4_0
        for v in range(4, 32):
            if v in adj[32]:
                return False
        
        # Structure verified. Alpha is provably 8.
        return True

    # Construct n=33: 8 K4s + vertex 32 connected to all of K4_0
    n = 33
    adj = [set() for _ in range(n)]
    edges = []
    
    # Construct 8 disjoint K4s
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                u = base + i
                v = base + j
                adj[u].add(v)
                adj[v].add(u)
                edges.append([u, v])
    
    # Connect vertex 32 to all vertices in K4_0 (0,1,2,3)
    for i in range(4):
        adj[32].add(i)
        adj[i].add(32)
        edges.append([32, i])
    
    # Verify
    if time.time() - start_time > deadline:
        # Fall back to n=32
        n = 32
        adj = [set() for _ in range(n)]
        edges = []
        for b in range(8):
            base = 4 * b
            for i in range(4):
                for j in range(i + 1, 4):
                    u = base + i
                    v = base + j
                    adj[u].add(v)
                    adj[v].add(u)
                    edges.append([u, v])
        return {'n': n, 'edges': edges}
    
    alpha_ok = fast_verify_alpha_lt_9(adj, n)
    
    if alpha_ok:
        c5_ok = fast_verify_c5_free(adj, n)
        if c5_ok:
            return {'n': n, 'edges': edges}
    
    # Fall back to n=32
    n = 32
    adj = [set() for _ in range(n)]
    edges = []
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                u = base + i
                v = base + j
                adj[u].add(v)
                adj[v].add(u)
                edges.append([u, v])
    
    return {'n': n, 'edges': edges}
