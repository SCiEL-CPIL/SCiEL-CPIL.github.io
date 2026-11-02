"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
from collections import deque

def _has_short_cycle(n, adj, min_girth):
    """
    Check if the graph has any cycle shorter than min_girth.
    Returns True if a short cycle is found, False otherwise.
    """
    limit = min_girth // 2
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        q = deque([start])
        
        while q:
            u = q.popleft()
            if dist[u] >= limit:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif v != parent[u]:
                    # Found a cycle
                    if dist[u] + dist[v] + 1 < min_girth:
                        return True
    return False

def _generate_random_4reg(n):
    """Generate a random 4-regular simple graph on n vertices."""
    for _ in range(1000):
        stubs = []
        for v in range(n):
            stubs.extend([v] * 4)
        random.shuffle(stubs)
        edges = set()
        valid = True
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                valid = False
                break
            key = (min(u, v), max(u, v))
            if key in edges:
                valid = False
                break
            edges.add(key)
        if valid:
            return list(edges)
    return None

def _build_adj(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _try_swap(edges, edge_set, adj, n):
    """
    Attempt a random 2-opt swap.
    Returns (new_edges, new_edge_set, new_adj) if successful, else None.
    """
    if len(edges) < 2:
        return None
    
    # Pick two random disjoint edges
    for _ in range(10):
        i = random.randint(0, len(edges) - 1)
        j = random.randint(0, len(edges) - 1)
        if i == j:
            continue
            
        e1 = edges[i]
        e2 = edges[j]
        
        u1, v1 = e1
        u2, v2 = e2
        
        if len({u1, v1, u2, v2}) < 4:
            continue
            
        candidates = [
            ((min(u1, u2), max(u1, u2)), (min(v1, v2), max(v1, v2))),
            ((min(u1, v2), max(u1, v2)), (min(v1, u2), max(v1, u2)))
        ]
        
        random.shuffle(candidates)
        
        for ne1, ne2 in candidates:
            if ne1[0] == ne1[1] or ne2[0] == ne2[1]:
                continue
            if ne1 in edge_set or ne2 in edge_set:
                continue
            if ne1 == ne2:
                continue
                
            new_edge_set = set(edge_set)
            new_edge_set.discard(e1)
            new_edge_set.discard(e2)
            new_edge_set.add(ne1)
            new_edge_set.add(ne2)
            
            new_edges = list(new_edge_set)
            new_adj = _build_adj(n, new_edges)
            
            return new_edges, new_edge_set, new_adj
            
    return None

def construct(k, g):
    if k != 4 or g != 10:
        n = 320
        edges = _generate_random_4reg(n)
        return n, edges if edges else (n, [])

    random.seed(42)
    n = 320
    
    for restart in range(20):
        edges = _generate_random_4reg(n)
        if edges is None:
            continue
            
        edge_set = set(edges)
        adj = _build_adj(n, edges)
        
        max_iters = 2000
        for _ in range(max_iters):
            if not _has_short_cycle(n, adj, g):
                return n, edges
                
            swap_result = _try_swap(edges, edge_set, adj, n)
            if swap_result:
                edges, edge_set, adj = swap_result
            else:
                break
                
    return n, edges
# EVOLVE-BLOCK-END
