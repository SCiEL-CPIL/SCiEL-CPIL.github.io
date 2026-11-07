"""Dense odd-order graphs avoiding equal-degree path endpoints."""
import time

# EVOLVE-BLOCK-START
_N = 15

def _is_valid(adj, n):
    """Check: no simple 3-edge path s-u-v-t with deg(s)==deg(t)."""
    deg = [len(adj[i]) for i in range(n)]
    for u in range(n):
        for v in adj[u]:
            for s in adj[u]:
                if s == v:
                    continue
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    if deg[s] == deg[t]:
                        return False
    return True

def _build_k78():
    """Build K_{7,8} bipartite graph on 15 vertices."""
    edges = []
    for i in range(7):
        for j in range(7, 15):
            edges.append([i, j])
    return edges

def _edges_to_adj(edges, n):
    """Convert edge list to adjacency list."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _try_add_edge(adj, edges, u, v, n):
    """Try adding edge (u,v). Returns new adj, edges if valid, else None."""
    if u in adj[v]:
        return None
    # Tentatively add
    adj[u].add(v)
    adj[v].add(u)
    if _is_valid(adj, n):
        edges.append([u, v])
        return adj, edges
    # Revert
    adj[u].remove(v)
    adj[v].remove(u)
    return None

def construct():
    start = time.time()
    deadline = start + 50.0
    
    # Start with K_{7,8}: 7*8 = 56 edges
    edges = _build_k78()
    adj = _edges_to_adj(edges, _N)
    
    # Verify initial validity
    if not _is_valid(adj, _N):
        # Fallback: star graph
        edges = [[0, v] for v in range(1, _N)]
        adj = _edges_to_adj(edges, _N)
        return [[int(u), int(v)] for u, v in edges]
    
    best_edges = len(edges)
    best_edges_copy = list(edges)
    
    # Try to add all possible missing edges
    added_any = True
    while added_any and time.time() < deadline:
        added_any = False
        for u in range(_N):
            for v in range(u + 1, _N):
                if v in adj[u]:
                    continue
                result = _try_add_edge(adj, edges, u, v, _N)
                if result is not None:
                    adj, edges = result
                    added_any = True
                    if len(edges) > best_edges:
                        best_edges = len(edges)
                        best_edges_copy = list(edges)
    
    # If we didn't improve, return the K_{7,8} base
    if len(edges) <= best_edges:
        edges = best_edges_copy
    
    # Final safety check
    adj_final = _edges_to_adj(edges, _N)
    if not _is_valid(adj_final, _N):
        # Fallback to K_{7,8}
        edges = _build_k78()
    
    # Return with correct types
    result = [[int(u), int(v)] for u, v in edges]
    return result
# EVOLVE-BLOCK-END
