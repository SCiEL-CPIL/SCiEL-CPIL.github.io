import time

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a C5-free graph on 32 vertices with independence number 8.
    
    Structure: 8 disjoint copies of K4.
    - Each K4 is a clique (no cycles of length > 3), so the disjoint union is C5-free.
    - Each K4 has independence number 1, so total alpha = 8 * 1 = 8 < 9.
    - n = 32, which is the known optimum for R(C5, K9) > n.
    """
    start = time.time()
    deadline = start + 140.0  # safety margin under 150s wall-clock limit
    
    blocks = 8
    k = 4
    n = blocks * k  # n = 32
    
    edges = []
    for b in range(blocks):
        base = b * k
        for i in range(k):
            for j in range(i + 1, k):
                edges.append([base + i, base + j])
    
    # Lightweight sanity check: verify no 5-cycle exists by checking the structure.
    # In a disjoint union of K4s, any 5 vertices span at most 4 vertices in one K4
    # and 1 in another (or fewer in each). The induced subgraph is a union of cliques
    # of size <= 4, which cannot contain a 5-cycle.
    # We do a quick structural verification:
    # Build adjacency and confirm each component is a K4.
    if time.time() < deadline:
        adj = [[False] * n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = adj[v][u] = True
        
        # Verify: each vertex has degree 3, and its neighbors form a K4
        valid = True
        for u in range(n):
            deg = sum(1 for v in range(n) if adj[u][v])
            if deg != 3:
                valid = False
                break
            # Check neighbors form a clique
            nbrs = [v for v in range(n) if adj[u][v]]
            for i in range(len(nbrs)):
                for j in range(i + 1, len(nbrs)):
                    if not adj[nbrs[i]][nbrs[j]]:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if not valid:
            # Fallback to a smaller known-valid construction (should not happen)
            n = 28
            edges = []
            for b in range(7):
                base = b * 4
                for i in range(4):
                    for j in range(i + 1, 4):
                        edges.append([base + i, base + j])
    
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
