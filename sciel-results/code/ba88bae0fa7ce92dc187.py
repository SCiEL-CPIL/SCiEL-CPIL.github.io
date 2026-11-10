"""Program for dense triangle-free 4-chromatic graphs on 16 vertices."""
import random
import time

# EVOLVE-BLOCK-START

def construct():
    n = 16
    start_time = time.time()
    time_limit = 115
    
    def is_triangle_free(adj):
        for u in range(n):
            for v in adj[u]:
                if v <= u:
                    continue
                for w in adj[v]:
                    if w > v and w in adj[u]:
                        return False
        return True
    
    def is_3_colorable(adj):
        color = [-1] * n
        order = sorted(range(n), key=lambda x: -len(adj[x]))
        
        def backtrack(idx):
            if idx == n:
                return True
            v = order[idx]
            used = set()
            for u in adj[v]:
                if color[u] != -1:
                    used.add(color[u])
            for c in range(3):
                if c not in used:
                    color[v] = c
                    if backtrack(idx + 1):
                        return True
                    color[v] = -1
            return False
        
        return backtrack(0)
    
    def edge_count(adj):
        return sum(len(adj[u]) for u in range(n)) // 2
    
    def edges_from_adj(adj):
        edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.append([u, v])
        return edges
    
    def build_grotzsch():
        adj = [set() for _ in range(n)]
        for i in range(5):
            adj[i].add((i + 1) % 5)
            adj[(i + 1) % 5].add(i)
        for i in range(5):
            adj[5 + i].add((i + 1) % 5)
            adj[(i + 1) % 5].add(5 + i)
            adj[5 + i].add((i - 1) % 5)
            adj[(i - 1) % 5].add(5 + i)
        for i in range(5):
            adj[10].add(5 + i)
            adj[5 + i].add(10)
        return adj
    
    adj = build_grotzsch()
    best_adj = [set(s) for s in adj]
    best_edges = edge_count(adj)
    
    def try_add_edge(adj, u, v):
        if u == v or v in adj[u]:
            return False
        if adj[u] & adj[v]:
            return False
        adj[u].add(v)
        adj[v].add(u)
        return True
    
    def remove_edge(adj, u, v):
        adj[u].discard(v)
        adj[v].discard(u)
    
    # Phase 1: Greedily add edges
    for u in range(n):
        for v in range(u + 1, n):
            if v not in adj[u]:
                try_add_edge(adj, u, v)
    
    # Check if still 4-chromatic
    if is_3_colorable(adj):
        # Need to undo some edges
        adj = [set(s) for s in best_adj]
    
    current_edges = edge_count(adj)
    if current_edges > best_edges and not is_3_colorable(adj):
        best_adj = [set(s) for s in adj]
        best_edges = current_edges
    
    # Phase 2: Local search
    random.seed(42)
    for iteration in range(20000):
        if time.time() - start_time > time_limit:
            break
        if best_edges >= 64:
            break
        
        # Try: remove a random edge, then greedily add back
        existing = [(u, v) for u in range(n) for v in adj[u] if u < v]
        if not existing:
            break
        
        u_rem, v_rem = random.choice(existing)
        old_adj = [set(s) for s in adj]
        remove_edge(adj, u_rem, v_rem)
        
        # Greedily add all possible edges
        for u in range(n):
            for v in range(u + 1, n):
                if v not in adj[u]:
                    try_add_edge(adj, u, v)
        
        new_edges = edge_count(adj)
        if new_edges > best_edges and not is_3_colorable(adj):
            best_adj = [set(s) for s in adj]
            best_edges = new_edges
        else:
            adj = [set(s) for s in old_adj]
    
    adj = best_adj
    assert is_triangle_free(adj)
    assert not is_3_colorable(adj)
    
    return edges_from_adj(adj)

# EVOLVE-BLOCK-END
