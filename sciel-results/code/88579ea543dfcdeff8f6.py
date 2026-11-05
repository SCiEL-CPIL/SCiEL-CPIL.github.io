"""Structural K20,20 with two K10 cliques per side + targeted local search."""
import time
import random

def construct():
    n = 40
    m = 401
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin

    # --- Structural initialization: K20,20 minus 9 edges + two K10s in A, two K10s in B ---
    # A = 0..19, B = 20..39
    # A1 = 0..9, A2 = 10..19, B1 = 20..29, B2 = 30..39
    # Remove 9 edges from bipartite, add internal edges in A1, A2, B1, B2
    
    edges = []
    edge_set = set()
    
    # Bipartite edges: all except 9 specific ones
    # Remove edges (0,20), (0,21), ..., (0,28) - 9 edges from vertex 0 to B1
    removed_bipartite = set()
    for i in range(9):
        removed_bipartite.add((0, 20 + i))
    
    for u in range(20):
        for v in range(20, 40):
            if (u, v) in removed_bipartite:
                continue
            edge_set.add((u, v))
            edges.append((u, v))
    
    # Add internal K10 in A1 (0..9)
    for i in range(10):
        for j in range(i+1, 10):
            edge_set.add((i, j))
            edges.append((i, j))
    
    # Add internal K10 in A2 (10..19)
    for i in range(10, 20):
        for j in range(i+1, 20):
            edge_set.add((i, j))
            edges.append((i, j))
    
    # Add internal K10 in B1 (20..29)
    for i in range(20, 30):
        for j in range(i+1, 30):
            edge_set.add((i, j))
            edges.append((i, j))
    
    # Add internal K10 in B2 (30..39)
    for i in range(30, 40):
        for j in range(i+1, 40):
            edge_set.add((i, j))
            edges.append((i, j))
    
    assert len(edge_set) == m, f"Expected {m} edges, got {len(edge_set)}"

    # --- Bitmask adjacency ---
    adj = [0] * n
    for u, v in edge_set:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def compute_max_book(adj, edge_set):
        mb = 0
        for u, v in edge_set:
            common = (adj[u] & adj[v]).bit_count()
            if common > mb:
                mb = common
        return mb

    def compute_all_books(adj, edge_set):
        books = {}
        for u, v in edge_set:
            books[(u, v)] = (adj[u] & adj[v]).bit_count()
        return books

    # Initial score
    best_max = compute_max_book(adj, edge_set)
    best_edge_set = set(edge_set)
    best_adj = list(adj)

    # --- Local search with targeted edge swaps ---
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    non_edges = [p for p in all_pairs if p not in edge_set]
    random.shuffle(non_edges)
    
    in_graph = [False] * (n * n)
    for u, v in edge_set:
        in_graph[u * n + v] = True
        in_graph[v * n + u] = True

    def add_edge(u, v):
        edge_set.add((u, v))
        in_graph[u * n + v] = True
        in_graph[v * n + u] = True
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(u, v):
        edge_set.discard((u, v))
        in_graph[u * n + v] = False
        in_graph[v * n + u] = False
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    # Local search: focus on reducing edges with max book
    improved = True
    no_improve_count = 0
    max_no_improve = 5
    
    while improved and time.time() < deadline:
        improved = False
        
        # Get current books and find max
        books = compute_all_books(adj, edge_set)
        current_max = max(books.values()) if books else 0
        
        # Find all edges with book >= current_max
        high_book_edges = [(u, v) for (u, v), bk in books.items() if bk >= current_max]
        random.shuffle(high_book_edges)
        
        # Try to improve each high-book edge
        for (u, v) in high_book_edges:
            if time.time() >= deadline:
                break
            
            bk = books[(u, v)]
            
            # Try swapping: remove (u,v), add a non-edge that has lower potential book
            # Focus on non-edges that don't create high books
            candidates = []
            for _ in range(200):
                if time.time() >= deadline:
                    break
                c, d = random.choice(non_edges)
                if (c, d) == (u, v) or (d, c) == (u, v):
                    continue
                
                # Estimate book size of (c,d) after removing (u,v)
                ac = adj[c]
                ad = adj[d]
                
                # Adjust for removal of (u,v)
                if c == u:
                    ac = adj[u] & ~(1 << v)
                elif c == v:
                    ac = adj[v] & ~(1 << u)
                
                if d == u:
                    ad = adj[u] & ~(1 << v)
                elif d == v:
                    ad = adj[v] & ~(1 << u)
                
                common_cd = (ac & ad).bit_count()
                
                # Only consider if this would reduce the book size
                if common_cd < bk:
                    candidates.append((c, d, common_cd))
            
            # Sort candidates by potential book size (lower is better)
            candidates.sort(key=lambda x: x[2])
            
            for c, d, common_cd in candidates[:10]:
                if time.time() >= deadline:
                    break
                
                # Perform the swap tentatively
                remove_edge(u, v)
                add_edge(c, d)
                
                # Recompute max book
                new_max = compute_max_book(adj, edge_set)
                
                if new_max < best_max:
                    best_max = new_max
                    best_edge_set = set(edge_set)
                    best_adj = list(adj)
                    improved = True
                    no_improve_count = 0
                    # Update non_edges list
                    try:
                        non_edges.remove((c, d))
                    except ValueError:
                        pass
                    non_edges.append((u, v))
                    random.shuffle(non_edges)
                    break  # Try next high-book edge
                else:
                    # Revert
                    remove_edge(c, d)
                    add_edge(u, v)
                    try:
                        non_edges.remove((u, v))
                    except ValueError:
                        pass
                    non_edges.append((c, d))
                    random.shuffle(non_edges)
            
            if time.time() >= deadline:
                break
        
        if not improved:
            no_improve_count += 1
            if no_improve_count >= max_no_improve:
                break

    # Return the best edge set found
    return [[u, v] for u, v in best_edge_set]
