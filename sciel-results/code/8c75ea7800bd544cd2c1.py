import random
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n = 40
    m = 401
    
    # Helper to check if edge exists
    def edge_exists(u, v):
        if u > v:
            u, v = v, u
        return (u, v) in edges_set

    # Initialize random graph
    edges = set()
    while len(edges) < m:
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u != v:
            if u > v:
                u, v = v, u
            edges.add((u, v))
    
    edges_list = list(edges)
    edges_set = set(edges)
    
    # Calculate book sizes
    def calc_books():
        books = {}
        # Adjacency list
        adj = [set() for _ in range(n)]
        for u, v in edges_set:
            adj[u].add(v)
            adj[v].add(u)
            
        for u, v in edges_set:
            common = adj[u].intersection(adj[v])
            books[(u, v)] = len(common)
        return books, adj

    books, adj = calc_books()
    max_book = max(books.values())
    best_edges = set(edges_set)
    best_max_book = max_book
    
    # Local search
    while time.time() < deadline:
        # Find edge with max book
        max_edge = None
        current_max = 0
        for e, b in books.items():
            if b > current_max:
                current_max = b
                max_edge = e
        
        if max_edge is None:
            break
            
        u, v = max_edge
        # Get common neighbors of u and v
        common_neighbors = list(adj[u].intersection(adj[v]))
        
        if not common_neighbors:
            # If max book is 0, we are done? But m=401 implies triangles exist.
            # If max book is 0, score is 20/0 which is infinity? No, score is seed/value.
            # Wait, the task says "score is the largest book".
            # "The score is the largest book... Minimize".
            # "search score is seed/value".
            # If value is 0, division by zero?
            # The theorem says score >= 40/6 ~ 6.67. So value should not be 0.
            # If we reach 0, it's a contradiction.
            pass

        improved = False
        # Try to remove an edge (u, w) where w is a common neighbor
        # This reduces the book size of (u, v)
        for w in common_neighbors:
            # Remove (u, w)
            edge_to_remove = (u, w) if u < w else (w, u)
            
            # We need to add a new edge
            # To minimize the increase in book sizes, pick a random non-edge
            # Or pick a non-edge with low potential book size
            # Let's just try a few random non-edges
            
            for _ in range(50): # Try 50 random additions
                x = random.randint(0, n - 1)
                y = random.randint(0, n - 1)
                if x == y:
                    continue
                if x > y:
                    x, y = y, x
                
                if (x, y) in edges_set:
                    continue
                
                # Simulate move
                edges_set.discard(edge_to_remove)
                edges_set.add((x, y))
                
                # Recalculate books
                new_books, new_adj = calc_books()
                new_max = max(new_books.values())
                
                if new_max < current_max:
                    # Accept move
                    max_book = new_max
                    books = new_books
                    adj = new_adj
                    if max_book < best_max_book:
                        best_max_book = max_book
                        best_edges = set(edges_set)
                    improved = True
                    break
                else:
                    # Reject move
                    edges_set.discard((x, y))
                    edges_set.add(edge_to_remove)
            
            if improved:
                break
        
        if not improved:
            # If we couldn't improve by removing from the max edge's neighbors,
            # maybe try swapping any edge?
            # Or just restart with a new random graph if stuck?
            # Given the small size, a random restart might be good if stuck for a while.
            # But let's try a few more random swaps first.
            
            # Pick a random edge to remove
            edge_to_remove = random.choice(list(edges_set))
            ru, rv = edge_to_remove
            
            # Remove it
            edges_set.discard(edge_to_remove)
            
            # Find a non-edge to add
            added = False
            for _ in range(50):
                x = random.randint(0, n - 1)
                y = random.randint(0, n - 1)
                if x == y:
                    continue
                if x > y:
                    x, y = y, x
                if (x, y) in edges_set:
                    continue
                
                edges_set.add((x, y))
                new_books, new_adj = calc_books()
                new_max = max(new_books.values())
                
                if new_max < max_book:
                    books = new_books
                    adj = new_adj
                    max_book = new_max
                    if max_book < best_max_book:
                        best_max_book = max_book
                        best_edges = set(edges_set)
                    added = True
                    break
                else:
                    edges_set.discard((x, y))
            
            if not added:
                # Revert removal
                edges_set.add(edge_to_remove)
                # If stuck, break or restart?
                # Let's break and rely on multiple runs if we had them, but we only have one.
                # Actually, we can just continue trying other edges.
                pass

    # Return the best result found
    # Convert set to list of lists
    return [list(e) for e in best_edges]
