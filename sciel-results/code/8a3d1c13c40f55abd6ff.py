"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    """
    Construct a graph with 40 vertices and 401 edges minimizing the largest book.
    Strategy: Start with K20,20 + 1 edge, then use local search (edge swaps)
    to minimize the maximum book size.
    """
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s limit
    
    n = 40
    target_edges = 401
    
    # Initialize with K20,20 + one edge
    # Vertices 0-19 in part A, 20-39 in part B
    # All edges between A and B are present (400 edges)
    # Plus one edge within A: [0, 1]
    adj = np.zeros((n, n), dtype=np.int32)
    for a in range(20):
        for b in range(20, 40):
            adj[a, b] = 1
            adj[b, a] = 1
    adj[0, 1] = 1
    adj[1, 0] = 1
    
    def compute_books(adj):
        """Compute book size for each edge. Book size of edge (u,v) = number of common neighbors."""
        books = {}
        for u in range(n):
            for v in range(u+1, n):
                if adj[u, v]:
                    common = np.sum(adj[u, :] & adj[v, :])
                    books[(u, v)] = common
        return books
    
    def max_book(books):
        if not books:
            return 0
        return max(books.values())
    
    # Initial book computation
    books = compute_books(adj)
    current_max = max_book(books)
    best_max = current_max
    best_adj = adj.copy()
    
    # Local search: try to swap edges to reduce max book
    # An edge swap: remove edge (u,v) and add edge (x,y) where (x,y) is not currently an edge
    # This keeps edge count constant at 401
    
    # Precompute edge list for faster iteration
    def get_edges(adj):
        edges = set()
        for u in range(n):
            for v in range(u+1, n):
                if adj[u, v]:
                    edges.add((u, v))
        return edges
    
    edges = get_edges(adj)
    
    # For efficiency, we'll try targeted swaps
    # Focus on edges with high book sizes - try to remove them and replace with low-book edges
    
    random.seed(42)
    
    # Create a pool of potential non-edges
    non_edges = []
    for u in range(n):
        for v in range(u+1, n):
            if not adj[u, v]:
                non_edges.append((u, v))
    
    # Shuffle for deterministic but varied search
    random.shuffle(non_edges)
    
    # Track which non-edges we've tried recently
    tried_non_edges = set()
    
    # Main optimization loop
    while time.time() < deadline:
        # Find the edge with the maximum book
        max_edge = None
        max_val = 0
        for (u, v), val in books.items():
            if val > max_val:
                max_val = val
                max_edge = (u, v)
        
        if max_edge is None or max_val <= 4:
            # Already at theoretical lower bound or better
            break
        
        u, v = max_edge
        
        # Try to find a non-edge to add that will reduce the book size
        # When we remove (u,v), the book sizes of edges incident to u or v change
        # When we add (x,y), new triangles are formed
        
        # Strategy: try removing (u,v) and adding various non-edges
        # Pick non-edges that are "far" from u and v to minimize new triangle creation
        
        found_improvement = False
        attempts = 0
        max_attempts = 200  # limit per iteration
        
        while not found_improvement and attempts < max_attempts:
            # Pick a random non-edge
            idx = random.randint(0, len(non_edges) - 1)
            x, y = non_edges[idx]
            
            # Skip if we've tried this recently
            if (x, y) in tried_non_edges:
                attempts += 1
                continue
            
            # Check that adding (x,y) doesn't create too many new triangles
            # New triangles involving (x,y): common neighbors of x and y
            common_xy = np.sum(adj[x, :] & adj[y, :])
            
            # Also need to check: removing (u,v) and adding (x,y)
            # The book of (x,y) after addition = common_xy
            # But we also need to update books for other edges
            
            # Quick check: if common_xy >= max_val, this swap won't help
            if common_xy >= max_val:
                attempts += 1
                continue
            
            # Perform the swap temporarily
            # Remove (u,v)
            adj[u, v] = 0
            adj[v, u] = 0
            edges.discard((u, v))
            
            # Add (x,y)
            adj[x, y] = 1
            adj[y, x] = 1
            edges.add((x, y))
            
            # Recompute books (expensive but necessary for accuracy)
            new_books = compute_books(adj)
            new_max = max_book(new_books)
            
            if new_max < current_max:
                # Accept the swap
                current_max = new_max
                books = new_books
                tried_non_edges.add((x, y))
                found_improvement = True
                
                # Update best if improved
                if current_max < best_max:
                    best_max = current_max
                    best_adj = adj.copy()
            else:
                # Revert
                adj[u, v] = 1
                adj[v, u] = 1
                edges.add((u, v))
                
                adj[x, y] = 0
                adj[y, x] = 0
                edges.discard((x, y))
            
            attempts += 1
            
            if time.time() > deadline:
                break
        
        if time.time() > deadline:
            break
        
        # If no improvement found, try a different strategy
        if not found_improvement:
            # Try random restarts with small perturbations
            # Or try a different high-book edge
            # For now, just continue - the random selection should eventually find something
            
            # Occasionally, try a more aggressive approach:
            # Remove a high-book edge and add a random non-edge even if it doesn't immediately improve
            # This can help escape local minima
            if random.random() < 0.1:
                # Pick any high-book edge
                high_edges = [(e, b) for e, b in books.items() if b >= current_max - 1]
                if high_edges:
                    (u2, v2), _ = random.choice(high_edges)
                    idx = random.randint(0, len(non_edges) - 1)
                    x, y = non_edges[idx]
                    
                    # Remove (u2,v2)
                    adj[u2, v2] = 0
                    adj[v2, u2] = 0
                    edges.discard((u2, v2))
                    
                    # Add (x,y)
                    adj[x, y] = 1
                    adj[y, x] = 1
                    edges.add((x, y))
                    
                    new_books = compute_books(adj)
                    new_max = max_book(new_books)
                    
                    # Accept if not too bad (simulated annealing style)
                    if new_max <= current_max + 1:
                        current_max = new_max
                        books = new_books
                    else:
                        # Revert
                        adj[u2, v2] = 1
                        adj[v2, u2] = 1
                        edges.add((u2, v2))
                        
                        adj[x, y] = 0
                        adj[y, x] = 0
                        edges.discard((x, y))
    
    # Use the best adjacency matrix found
    final_adj = best_adj
    
    # Extract edges
    result = []
    for u in range(n):
        for v in range(u+1, n):
            if final_adj[u, v]:
                result.append([u, v])
    
    # Verify we have exactly 401 edges
    assert len(result) == 401, f"Expected 401 edges, got {len(result)}"
    
    return result
# EVOLVE-BLOCK-END
