import random

def construct():
    n = 40
    m = 401
    
    def build_adj(edge_set):
        adj = [[False]*n for _ in range(n)]
        for u, v in edge_set:
            adj[u][v] = True
            adj[v][u] = True
        return adj
    
    def common_neighbors(adj, u, v):
        count = 0
        for w in range(n):
            if w != u and w != v and adj[u][w] and adj[v][w]:
                count += 1
        return count
    
    # Start with random graph
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(all_pairs)
    edge_set = set(all_pairs[:m])
    adj = build_adj(edge_set)
    
    # Compute book for each edge
    def compute_books(adj, edge_set):
        books = {}
        for u, v in edge_set:
            books[(u, v)] = common_neighbors(adj, u, v)
        return books
    
    books = compute_books(adj, edge_set)
    max_b = max(books.values())
    best_max = max_b
    best_edge_set = set(edge_set)
    
    # Local search
    for iteration in range(20000):
        if best_max <= 7:
            break
        
        # Find edge with max book
        max_edge = max(books, key=lambda e: books[e])
        max_b = books[max_edge]
        
        # Try to swap
        u, v = max_edge
        # Remove (u,v)
        adj[u][v] = False
        adj[v][u] = False
        edge_set.discard((min(u,v), max(u,v)))
        del books[max_edge]
        
        # Pick random non-edge
        non_edges = [e for e in all_pairs if e not in edge_set]
        if not non_edges:
            adj[u][v] = True
            adj[v][u] = True
            edge_set.add((min(u,v), max(u,v)))
            books[max_edge] = max_b
            break
        
        new_u, new_v = random.choice(non_edges)
        adj[new_u][new_v] = True
        adj[new_v][new_u] = True
        edge_set.add((min(new_u, new_v), max(new_u, new_v)))
        
        # Update books for affected edges
        # Edges affected: (new_u, new_v) and all edges (new_u, w) and (new_v, w) where w is neighbor
        # This is complex. Let me just recompute books for edges incident to new_u or new_v
        
        # Simpler: just compute new max by checking affected edges
        affected = set()
        affected.add((min(new_u, new_v), max(new_u, new_v)))
        for w in range(n):
            if w != new_u and w != new_v:
                if adj[new_u][w]:
                    affected.add((min(new_u, w), max(new_u, w)))
                if adj[new_v][w]:
                    affected.add((min(new_v, w), max(new_v, w)))
        
        new_books = {}
        for e in affected:
            a, b = e
            new_books[e] = common_neighbors(adj, a, b)
        
        # Update books dict
        for e in affected:
            if e in books:
                del books[e]
        books.update(new_books)
        
        new_max = max(books.values())
        
        if new_max <= max_b:
            best_max = min(best_max, new_max)
            if new_max < best_max:
                best_max = new_max
                best_edge_set = set(edge_set)
        else:
            # Revert
            adj[new_u][new_v] = False
            adj[new_v][new_u] = False
            edge_set.discard((min(new_u, new_v), max(new_u, new_v)))
            adj[u][v] = True
            adj[v][u] = True
            edge_set.add((min(u, v), max(u, v)))
            # Restore books
            for e in affected:
                if e in books:
                    del books[e]
            books[max_edge] = max_b
            # Recompute affected
            for e in affected:
                if e != max_edge and e in edge_set:
                    a, b = e
                    books[e] = common_neighbors(adj, a, b)
    
    return [[u, v] for u, v in best_edge_set]
