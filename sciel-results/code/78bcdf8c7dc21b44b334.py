import random
import time

def construct():
    n = 20
    # adjacency list using sets
    adj = [set() for _ in range(n)]
    
    def can_add(u, v):
        if v in adj[u]:
            return False
            
        # Check pair (u, v)
        common_uv = adj[u] & adj[v]
        if len(common_uv) > 2:
            return False
            
        # Check pairs (u, w) for w in adj[v]
        # Adding (u, v) adds v to adj[u].
        # v is in adj[w] for all w in adj[v].
        # So v becomes a common neighbor of u and w.
        # Current common neighbors of u and w is adj[u] & adj[w].
        # New count is len(adj[u] & adj[w]) + 1 (for v)
        # We need new count <= 2, so old count <= 1.
        for w in adj[v]:
            if len(adj[u] & adj[w]) > 1:
                return False
                
        # Check pairs (v, w) for w in adj[u]
        # Adding (u, v) adds u to adj[v].
        # u is in adj[w] for all w in adj[u].
        # So u becomes a common neighbor of v and w.
        # Current common neighbors of v and w is adj[v] & adj[w].
        # New count is len(adj[v] & adj[w]) + 1 (for u)
        # We need new count <= 2, so old count <= 1.
        for w in adj[u]:
            if len(adj[v] & adj[w]) > 1:
                return False
                
        return True

    def add_edge(u, v):
        adj[u].add(v)
        adj[v].add(u)

    def remove_edge(u, v):
        adj[u].remove(v)
        adj[v].remove(u)

    def get_edges():
        edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.append([u, v])
        return edges

    def edge_count():
        return sum(len(s) for s in adj) // 2

    # Initialize with a random valid graph using greedy approach
    # Try to add all edges in random order
    edges_to_try = [(u, v) for u in range(n) for v in range(u+1, n)]
    random.shuffle(edges_to_try)
    
    for u, v in edges_to_try:
        if can_add(u, v):
            add_edge(u, v)
            
    best_edges = get_edges()
    best_count = edge_count()
    
    # Local search
    start_time = time.time()
    timeout = 110  # seconds, leave margin for overhead
    
    while time.time() - start_time < timeout:
        # Try to add any missing edge
        added = False
        missing_edges = [(u, v) for u in range(n) for v in range(u+1, n) if v not in adj[u]]
        random.shuffle(missing_edges)
        
        for u, v in missing_edges:
            if can_add(u, v):
                add_edge(u, v)
                added = True
                # Restart the missing edge scan? Or just continue?
                # Continuing is faster, but might miss better combinations.
                # Let's just continue for now.
        
        if edge_count() > best_count:
            best_count = edge_count()
            best_edges = get_edges()
            
        if not added:
            # Stuck. Remove a random edge and try again
            if edge_count() > 0:
                # Pick a random edge to remove
                all_edges = get_edges()
                random.shuffle(all_edges)
                u, v = all_edges[0]
                remove_edge(u, v)
            else:
                break

    return best_edges
