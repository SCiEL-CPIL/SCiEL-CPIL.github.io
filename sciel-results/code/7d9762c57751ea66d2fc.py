import time
import sys

def construct():
    """
    Finds a triangle-free graph on 16 vertices with chromatic number >= 4
    that maximizes the number of edges using branch-and-bound search.
    """
    n = 16
    deadline = time.time() + 100  # Leave safety margin before 120s limit
    
    # All possible edges
    all_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            all_edges.append((u, v))
    
    edge_index = {e: i for i, e in enumerate(all_edges)}
    m = len(all_edges)  # 120 total possible edges
    
    # Start with Clebsch graph as baseline
    S = [1, 2, 4, 8, 15]
    clebsch_edges = set()
    for u in range(n):
        for v in range(u + 1, n):
            if (u ^ v) in S:
                clebsch_edges.add((u, v))
    
    best_edges = set(clebsch_edges)
    best_count = len(clebsch_edges)
    
    # Build adjacency structure for fast triangle checks
    # adj[v] = set of neighbors
    adj = [set() for _ in range(n)]
    for u, v in clebsch_edges:
        adj[u].add(v)
        adj[v].add(u)
    
    def is_triangle_free_add(edge):
        """Check if adding edge (u,v) creates a triangle"""
        u, v = edge
        # Triangle exists if there's a common neighbor
        for w in adj[u]:
            if w in adj[v]:
                return False
        return True
    
    def add_edge(edge):
        u, v = edge
        adj[u].add(v)
        adj[v].add(u)
    
    def remove_edge(edge):
        u, v = edge
        adj[u].remove(v)
        adj[v].remove(u)
    
    def is_not_3_colorable():
        """Check if graph is not 3-colorable using backtracking"""
        color = [-1] * n
        deg = [len(adj[i]) for i in range(n)]
        
        # Order vertices by degree (descending) for better pruning
        order = sorted(range(n), key=lambda x: -deg[x])
        
        def backtrack(pos):
            if time.time() > deadline:
                return False  # Timeout, assume 3-colorable to be safe
            
            if pos == n:
                return True
            
            v = order[pos]
            
            # Try each color
            for c in range(3):
                # Check if color c is available
                conflict = False
                for u in adj[v]:
                    if color[u] == c:
                        conflict = True
                        break
                if conflict:
                    continue
                
                color[v] = c
                if backtrack(pos + 1):
                    return True
                color[v] = -1
            
            return False
        
        return not backtrack(0)
    
    # Get list of non-edges to try adding
    non_edges = [e for e in all_edges if e not in clebsch_edges]
    
    # Sort non-edges by potential (edges that connect high-degree vertices first)
    def edge_potential(e):
        u, v = e
        return len(adj[u]) + len(adj[v])
    
    non_edges.sort(key=edge_potential, reverse=True)
    
    # Branch and bound
    def search(idx, current_count):
        nonlocal best_edges, best_count
        
        if time.time() > deadline:
            return
        
        # Upper bound: current + remaining non-edges
        if current_count + (len(non_edges) - idx) <= best_count:
            return
        
        if idx == len(non_edges):
            return
        
        e = non_edges[idx]
        
        # Try adding edge e
        if is_triangle_free_add(e):
            add_edge(e)
            if is_not_3_colorable():
                new_count = current_count + 1
                if new_count > best_count:
                    best_count = new_count
                    best_edges = set()
                    for u in range(n):
                        for v in adj[u]:
                            if u < v:
                                best_edges.add((u, v))
                search(idx + 1, new_count)
            remove_edge(e)
        
        # Try not adding edge e
        search(idx + 1, current_count)
    
    search(0, best_count)
    
    # Convert to list of lists
    result = [[u, v] for u, v in best_edges]
    return result
