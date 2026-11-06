"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a graph on 40 vertices with 401 edges minimizing the largest book.
    Strategy:
    1. Start with K20,20 + one extra edge (the seed).
    2. Use local search with edge swaps to reduce the maximum book size.
    3. Use a deadline guard to ensure we return within time limits.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s margin
    
    n = 40
    target_edges = 401
    
    # Initialize with K20,20 + one edge inside part A
    # Part A: 0-19, Part B: 20-39
    edges = set()
    for a in range(20):
        for b in range(20, 40):
            edges.add((a, b))
    edges.add((0, 1))
    
    # Convert to adjacency structure for fast lookups
    adj = [[False] * n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
    
    # Compute book sizes for all edges
    # book(u,v) = number of common neighbors of u and v
    def compute_books(adj, edges):
        books = {}
        for u, v in edges:
            count = 0
            for w in range(n):
                if adj[u][w] and adj[v][w]:
                    count += 1
            books[(u, v)] = count
        return books
    
    # Get current max book
    def get_max_book(books):
        return max(books.values()) if books else 0
    
    books = compute_books(adj, edges)
    best_max_book = get_max_book(books)
    best_edges = edges.copy()
    
    # Helper to compute book size of a specific edge
    def book_size(u, v, adj):
        count = 0
        for w in range(n):
            if adj[u][w] and adj[v][w]:
                count += 1
        return count
    
    # Local search: try to swap edges to reduce max book
    # Strategy: find an edge with high book, try to remove it and add a different edge
    # that doesn't increase the max book
    
    def try_improve(edges, adj, books, max_iter=100000):
        current_max = get_max_book(books)
        improved = False
        iter_count = 0
        
        # Sort edges by book size descending to prioritize high-book edges
        sorted_edges = sorted(edges, key=lambda e: books[e], reverse=True)
        
        for _ in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            iter_count += 1
            
            # Pick a random high-book edge to try to remove
            if not sorted_edges:
                break
            
            # Focus on edges with book size >= current_max or close to it
            high_book_edges = [e for e in sorted_edges if books[e] >= current_max - 1]
            if not high_book_edges:
                high_book_edges = sorted_edges[:10]
            
            u, v = random.choice(high_book_edges)
            u_book = books[(u, v)]
            
            if u_book <= 1:
                continue
            
            # Try to remove edge (u,v) and add a new edge (x,y)
            # The new edge should have low book size
            # Also, removing (u,v) should not increase other books too much
            
            # Find candidate new edges: non-edges with low book size
            # We need to find (x,y) not in edges, and (x,y) != (u,v)
            # such that book(x,y) is small
            
            # To be efficient, sample some candidate non-edges
            candidates = []
            for _ in range(20):  # Try 20 random non-edges
                x = random.randint(0, n - 1)
                y = random.randint(0, n - 1)
                if x == y or (x, y) in edges or (y, x) in edges:
                    continue
                if (x, y) == (u, v) or (x, y) == (v, u):
                    continue
                # Compute book size of (x,y)
                xy_book = 0
                for w in range(n):
                    if adj[x][w] and adj[y][w]:
                        xy_book += 1
                if xy_book < u_book:
                    candidates.append((x, y, xy_book))
            
            if not candidates:
                continue
            
            # Pick the candidate with smallest book size
            x, y, xy_book = min(candidates, key=lambda c: c[2])
            
            # Try the swap: remove (u,v), add (x,y)
            # Check if this improves or maintains the max book
            # We need to recompute books for affected edges
            
            # Edges affected by removing (u,v):
            # - All edges incident to u or v that share a common neighbor with u or v
            # This is complex, so let's just do a full recompute if the swap looks promising
            
            # Quick check: if xy_book < current_max and u_book == current_max,
            # and no other edge has book size == current_max, then we improve
            edges_with_max = [e for e in edges if books[e] == current_max]
            
            if len(edges_with_max) == 1 and xy_book < current_max:
                # This swap should improve the max book
                # Perform the swap
                edges.remove((u, v))
                edges.add((x, y))
                adj[u][v] = False
                adj[v][u] = False
                adj[x][y] = True
                adj[y][x] = True
                books = compute_books(adj, edges)
                new_max = get_max_book(books)
                if new_max < best_max_book:
                    best_max_book = new_max
                    best_edges = edges.copy()
                improved = True
                sorted_edges = sorted(edges, key=lambda e: books[e], reverse=True)
                continue
            
            # Even if not a clear improvement, try the swap and see
            # This is more expensive, so do it less frequently
            if random.random() < 0.1:
                # Perform the swap temporarily
                edges.remove((u, v))
                edges.add((x, y))
                adj[u][v] = False
                adj[v][u] = False
                adj[x][y] = True
                adj[y][x] = True
                books = compute_books(adj, edges)
                new_max = get_max_book(books)
                if new_max <= current_max:
                    if new_max < best_max_book:
                        best_max_book = new_max
                        best_edges = edges.copy()
                    improved = True
                    sorted_edges = sorted(edges, key=lambda e: books[e], reverse=True)
                else:
                    # Revert
                    edges.remove((x, y))
                    edges.add((u, v))
                    adj[u][v] = True
                    adj[v][u] = True
                    adj[x][y] = False
                    adj[y][x] = False
                    books = compute_books(adj, edges)
        
        return improved, best_edges
    
    # Run local search for a while
    while time.time() < deadline - 10.0:
        improved, best_edges = try_improve(edges, adj, books, max_iter=50000)
        if not improved:
            # Try a different random seed or restart
            # Let's try a few more iterations with different random choices
            for _ in range(3):
                if time.time() > deadline - 10.0:
                    break
                improved, best_edges = try_improve(edges, adj, books, max_iter=10000)
                if improved:
                    break
    
    # Return the best edges found
    result_edges = list(best_edges)
    
    # Ensure exactly 401 edges
    assert len(result_edges) == 401, f"Expected 401 edges, got {len(result_edges)}"
    
    # Convert to list of [u, v] format
    result = [[u, v] for u, v in result_edges]
    
    return result
# EVOLVE-BLOCK-END
