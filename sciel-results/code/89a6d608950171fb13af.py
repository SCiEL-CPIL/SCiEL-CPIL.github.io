import random
import numpy as np

def construct():
    n = 40
    target_edges = 401
    
    # Deterministic initialization: Turan graph T_2(40) + 1 edge
    # Partition A: 0..19, Partition B: 20..39
    edges = set()
    for u in range(20):
        for v in range(20, 40):
            edges.add((u, v))
    
    # Add one edge inside A to reach 401 edges
    # This creates a "hot" edge
    edges.add((0, 1))
    
    def compute_books(edge_set):
        adj = np.zeros((n, n), dtype=int)
        for u, v in edge_set:
            adj[u, v] = 1
            adj[v, u] = 1
        common = adj @ adj
        max_book = 0
        edge_books = {}
        for u, v in edge_set:
            b = int(common[u, v])
            edge_books[(u, v)] = b
            if b > max_book:
                max_book = b
        return max_book, edge_books, adj, common

    max_book, edge_books, adj, common = compute_books(edges)
    
    # Local Search
    # We try to reduce the maximum book size.
    # Strategy: Find the edge with the max book. Try to break one of its triangles by swapping an edge.
    
    no_improve = 0
    max_iterations = 20000
    temperature = 1.0
    cooling = 0.999
    
    for it in range(max_iterations):
        if no_improve > 500:
            break
            
        # Find edge with max book
        current_max = max(edge_books.values())
        
        # If we reached the theoretical lower bound, stop
        if current_max <= 7:
            break
            
        # Get list of edges with max book
        max_edges = [e for e, b in edge_books.items() if b == current_max]
        
        if not max_edges:
            break
            
        # Pick a random max edge
        u, v = random.choice(max_edges)
        
        # Common neighbors of u and v
        common_neighbors = []
        for w in range(n):
            if w != u and w != v and adj[u, w] == 1 and adj[v, w] == 1:
                common_neighbors.append(w)
                
        if not common_neighbors:
            continue
            
        improved = False
        
        # Try removing an edge connected to a common neighbor
        # We shuffle common neighbors to try different ones
        random.shuffle(common_neighbors)
        
        for w in common_neighbors:
            # Try removing (u, w)
            # We need to add a new edge (x, y)
            # The new edge should be a non-edge
            # And it should not create a book larger than current_max
            
            # We try a few random non-edges
            found_swap = False
            for _ in range(10):
                # Pick random non-edge
                while True:
                    x, y = random.sample(range(n), 2)
                    if adj[x, y] == 0:
                        break
                    
                # Check if (x, y) is the edge we are removing (shouldn't be, but safe)
                if (x, y) == (u, w) or (y, x) == (u, w):
                    continue

                # Proposed change: remove (u, w), add (x, y)
                # Check validity:
                # 1. New book of (x, y) must be <= current_max
                # 2. Adding (x, y) increases book of (x, z) and (y, z) for common neighbors z
                #    These new books must be <= current_max
                
                # Calculate new book of (x, y)
                # common[x, y] in current graph
                new_book_xy = int(common[x, y])
                if new_book_xy > current_max:
                    continue
                
                # Check neighbors
                # Common neighbors of x and y in current graph
                # Note: removing (u, w) might change common neighbors if u or w are involved
                # But x, y are random, likely not u or w.
                # To be safe, we assume the change is local.
                # Actually, removing (u, w) decreases common neighbors of pairs involving u or w.
                # Adding (x, y) increases common neighbors of pairs involving x or y.
                
                # Let's just do a full recompute for the candidate if it passes the quick check.
                # n=40 is small.
                
                new_edges = set(edges)
                new_edges.remove((u, w))
                new_edges.add((x, y))
                
                new_max, _, _, _ = compute_books(new_edges)
                
                if new_max < current_max:
                    edges = new_edges
                    max_book = new_max
                    no_improve = 0
                    improved = True
                    found_swap = True
                    break
                elif new_max == current_max:
                    # Accept with probability based on temperature
                    if random.random() < temperature:
                        edges = new_edges
                        max_book = new_max
                        no_improve = 0
                        improved = True
                        found_swap = True
                        break
                # else reject
            
            if found_swap:
                break
                
        if not improved:
            no_improve += 1
            temperature *= cooling
        else:
            # Recompute books for next iteration
            max_book, edge_books, adj, common = compute_books(edges)

    return [list(e) for e in edges]
