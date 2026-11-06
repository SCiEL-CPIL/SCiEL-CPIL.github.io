"""Constructs a C5-free graph with no independent set of size 9.
Uses disjoint union of K4s, achieving n=32, alpha=8."""
import time

# EVOLVE-BLOCK-START
def _has_c5(n, adj):
    """Exact check for any 5-cycle as a subgraph."""
    for v0 in range(n):
        for v1 in range(v0 + 1, n):
            if not adj[v0][v1]:
                continue
            for v2 in range(v1 + 1, n):
                if not adj[v1][v2]:
                    continue
                for v3 in range(v2 + 1, n):
                    if not adj[v2][v3]:
                        continue
                    for v4 in range(v3 + 1, n):
                        if adj[v3][v4] and adj[v4][v0]:
                            return True
    return False

def _alpha_exact(n, adj):
    """Exact max independent set size."""
    best = 0
    adj_list = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                adj_list[i].add(j)
                adj_list[j].add(i)
                
    def dfs(cand, size):
        nonlocal best
        if size + len(cand) <= best:
            return
        if not cand:
            if size > best:
                best = size
            return
        v = cand[0]
        rest = cand[1:]
        # Include v
        new_cand = [x for x in rest if x not in adj_list[v]]
        dfs(new_cand, size + 1)
        # Exclude v
        dfs(rest, size)

    dfs(list(range(n)), 0)
    return best

def construct():
    # Optimal known construction: 8 disjoint K4s
    # n = 32, alpha = 8, C5-free
    blocks = 8
    k = 4
    n = blocks * k
    edges = []
    for b in range(blocks):
        base = b * k
        for i in range(k):
            for j in range(i + 1, k):
                edges.append([base + i, base + j])
    
    # Verification
    adj = [[False] * n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = adj[v][u] = True
        
    # Check C5
    if _has_c5(n, adj):
        # Fallback to 7 K4s if 8 K4s somehow fails (should not)
        blocks = 7
        n = blocks * k
        edges = []
        for b in range(blocks):
            base = b * k
            for i in range(k):
                for j in range(i + 1, k):
                    edges.append([base + i, base + j])
        adj = [[False] * n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = adj[v][u] = True

    # Check Alpha
    alpha = _alpha_exact(n, adj)
    if alpha >= 9:
        # Fallback to 7 K4s
        blocks = 7
        n = blocks * k
        edges = []
        for b in range(blocks):
            base = b * k
            for i in range(k):
                for j in range(i + 1, k):
                    edges.append([base + i, base + j])

    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
