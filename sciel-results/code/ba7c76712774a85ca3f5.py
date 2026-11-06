import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n = 40
    m = 401
    max_allowed_book = 7 # Target lower bound
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # We will try multiple random orderings of edges
    # and greedily add them if they don't violate the book constraint.
    # If we can't reach m edges, we might need to allow some violations
    # or use a repair step.
    
    best_edges = None
    best_max_book = 100
    
    rng = random.Random(42)
    
    for restart in range(50):
        if time.time() > deadline - 5:
            break
            
        # Random permutation of edges
        edges_to_try = all_edges[:]
        rng.shuffle(edges_to_try)
        
        current_edges = set()
        adj = [set() for _ in range(n)]
        book_sizes = {} # (u,v) -> count
        
        # To speed up, we track for each vertex, the neighbors.
        # When adding (u,v), we need to update books for (u,w) and (v,w).
        
        valid = True
        
        for u, v in edges_to_try:
            if len(current_edges) >= m:
                break
            
            # Check if adding (u,v) is valid
            # 1. Book of (u,v) itself: |N(u) & N(v)|
            common = len(adj[u] & adj[v])
            if common > max_allowed_book:
                continue
            
            # 2. Check if adding (u,v) pushes any existing book over the limit
            # Existing edges incident to u: (u, w) for w in N(u)
            # Existing edges incident to v: (v, w) for w in N(v)
            
            # The book of (u, w) increases by 1 if v is in N(w) (which it is, since (u,v) added? No.
            # The book of (u, w) is |N(u) & N(w)|.
            # Adding (u,v) adds v to N(u).
            # So N(u) becomes N(u) U {v}.
            # New book of (u, w) is | (N(u) U {v}) & N(w) | = |N(u) & N(w)| + (1 if v in N(w) else 0).
            # So we need to check if v is in N(w) for any w in N(u).
            # If v in N(w), then book(u,w) increases by 1.
            # We need to ensure book(u,w) + 1 <= max_allowed_book.
            
            can_add = True
            
            # Check neighbors of u
            for w in adj[u]:
                if w == v: continue
                # If v is in adj[w], then (u,w) and (v,w) form a triangle with (u,v)?
                # No, (u,v) is the new edge.
                # Triangle is (u,v,w).
                # This triangle contributes to book(u,v), book(u,w), book(v,w).
                # So if (u,w) and (v,w) exist, then adding (u,v) creates a triangle.
                # This increases book(u,w) by 1 and book(v,w) by 1.
                
                # So we need to check:
                # For all w in N(u): if v in N(w), then book(u,w) must be < max_allowed_book.
                # For all w in N(v): if u in N(w), then book(v,w) must be < max_allowed_book.
                
                if v in adj[w]:
                    if book_sizes.get((u, w) if u < w else (w, u), 0) >= max_allowed_book:
                        can_add = False
                        break
            
            if not can_add:
                continue
                
            for w in adj[v]:
                if w == u: continue
                if u in adj[w]:
                    if book_sizes.get((v, w) if v < w else (w, v), 0) >= max_allowed_book:
                        can_add = False
                        break
            
            if not can_add:
                continue
            
            # If we can add, update structures
            current_edges.add((u, v))
            adj[u].add(v)
            adj[v].add(u)
            
            # Update book sizes
            book_sizes[(u, v)] = common
            
            # Update books for (u, w) and (v, w)
            for w in adj[u]:
                if w == v: continue
                edge_key = (u, w) if u < w else (w, u)
                if v in adj[w]:
                    # Triangle (u,v,w) exists
                    book_sizes[edge_key] = book_sizes.get(edge_key, 0) + 1
            
            for w in adj[v]:
                if w == u: continue
                edge_key = (v, w) if v < w else (w, v)
                if u in adj[w]:
                    book_sizes[edge_key] = book_sizes.get(edge_key, 0) + 1
            
        if len(current_edges) == m:
            # We found a valid graph
            # Calculate actual max book to be sure
            actual_max = 0
            for u, v in current_edges:
                b = len(adj[u] & adj[v])
                if b > actual_max:
                    actual_max = b
            
            if actual_max < best_max_book:
                best_max_book = actual_max
                best_edges = list(current_edges)
                
            if best_max_book <= 7:
                break
        
    if best_edges is not None:
        return [[u, v] for u, v in best_edges]
    
    # Fallback: Return a random graph if greedy fails (should not happen often)
    edges = set(rng.sample(all_edges, m))
    return [[u, v] for u, v in edges]
