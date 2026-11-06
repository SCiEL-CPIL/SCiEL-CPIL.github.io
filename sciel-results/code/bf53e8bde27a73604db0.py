import numpy as np
import time
import random

def construct():
    n = 40
    m = 401
    deadline = time.time() + 115  # Safety margin
    
    # Initialize
    adj_masks = [0] * n
    edges = set()
    
    # Generate random graph
    while len(edges) < m:
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u != v and (u, v) not in edges:
            edges.add((u, v))
            edges.add((v, u))
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
            
    def calc_books():
        books = np.zeros((n, n), dtype=np.int32)
        for u, v in edges:
            if u < v:
                common = (adj_masks[u] & adj_masks[v]).bit_count()
                books[u, v] = common
                books[v, u] = common
        return books

    books = calc_books()
    current_max = int(np.max(books))
    
    best_adj_masks = adj_masks.copy()
    best_edges = edges.copy()
    best_max = current_max
    
    T = 10.0
    T_min = 0.1
    cooling = 0.999
    
    # Precompute list of all edges for random selection
    edge_list = list(edges)
    
    while time.time() < deadline:
        # Pick a random edge to remove
        u, v = random.choice(edge_list)
        
        # Try to find a non-edge to add
        # Sample 10 non-edges
        best_swap = None
        best_swap_max = current_max
        
        for _ in range(10):
            x = random.randint(0, n-1)
            y = random.randint(0, n-1)
            if x == y or (x, y) in edges or (y, x) in edges:
                continue
            
            # Simulate swap: remove (u,v), add (x,y)
            # Calculate new max book size
            
            # We need to update books for affected edges.
            # Affected edges are incident to u, v, x, y.
            
            # Let's calculate the new book size for all edges incident to u, v, x, y.
            # And check if the new max is lower.
            
            # To do this efficiently, we can:
            # 1. Identify all edges incident to u, v, x, y.
            # 2. For each such edge (a,b), calculate its new book size.
            # 3. The new global max is max(new_max_of_affected, old_max_if_old_max_edge_not_affected).
            
            # This is complex. Let's just do a full recalculation of the max for the candidate?
            # No, too slow.
            
            # Let's use the incremental update logic on a COPY of the books?
            # No, copying 40x40 array is fast (1600 ints).
            # 10 copies * 1000 steps = 10,000 copies. 10,000 * 1600 = 16M int copies.
            # This is fine.
            
            # Let's try:
            # 1. Copy books to temp_books.
            # 2. Apply remove (u,v) to temp_books.
            # 3. Apply add (x,y) to temp_books.
            # 4. Check max of temp_books.
            
            temp_books = books.copy()
            
            # Remove (u,v)
            # books[u,v] = 0
            temp_books[u, v] = 0
            temp_books[v, u] = 0
            
            # For w in common(u,v):
            #   temp_books[u,w] -= 1
            #   temp_books[v,w] -= 1
            
            common_uv = adj_masks[u] & adj_masks[v]
            for w in range(n):
                if (common_uv >> w) & 1:
                    temp_books[u, w] -= 1
                    temp_books[v, w] -= 1
            
            # Add (x,y)
            # temp_books[x,y] = common(x,y)
            # For w in common(x,y):
            #   temp_books[x,w] += 1
            #   temp_books[y,w] += 1
            
            # Note: If x or y is u or v, we must be careful.
            # The "common(x,y)" should be calculated in the NEW graph (after removing u,v).
            # And the updates to neighbors should also be in the NEW graph.
            
            # This incremental update on a copy is tricky if x,y overlap with u,v.
            
            # Given the complexity, let's just do a FULL recalculation of the max book size
            # for the candidate graph, but only for the edges that changed?
            # No, the max could be anywhere.
            
            # Let's go back to the bitmask approach for the candidate.
            # Create virtual masks for the candidate.
            # This is expensive to create.
            
            # Alternative:
            # Just do the swap on the real data, calculate max, and revert if bad.
            # Swap is O(n). Max calc is O(m*n) = 16k.
            # 10 candidates * 16k = 160k ops per step.
            # 115s / 160k ~ 700 steps.
            # This is acceptable.
            
            # Let's try this "try and revert" approach.
            
            # Save state
            # We need to save adj_masks and edges?
            # adj_masks is a list of ints. Copy is fast.
            # edges is a set. Copy is fast.
            # books is an array. Copy is fast.
            
            # But we don't need to save books if we recalculate it.
            # We only need to save adj_masks and edges.
            
            # Actually, we can just modify adj_masks and edges, calculate max, and revert.
            
            # Modify
            edges.remove((u, v))
            edges.remove((v, u))
            adj_masks[u] &= ~(1 << v)
            adj_masks[v] &= ~(1 << u)
            
            edges.add((x, y))
            edges.add((y, x))
            adj_masks[x] |= (1 << y)
            adj_masks[y] |= (1 << x)
            
            # Calculate max book
            new_max = 0
            for a, b in edges:
                if a < b:
                    common = (adj_masks[a] & adj_masks[b]).bit_count()
                    if common > new_max:
                        new_max = common
            
            if new_max < best_swap_max:
                best_swap_max = new_max
                best_swap = (x, y)
                
            # Revert
            edges.remove((x, y))
            edges.remove((y, x))
            adj_masks[x] &= ~(1 << y)
            adj_masks[y] &= ~(1 << x)
            
            edges.add((u, v))
            edges.add((v, u))
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
            
        # If we found a better swap
        if best_swap:
            x, y = best_swap
            # Apply swap
            edges.remove((u, v))
            edges.remove((v, u))
            adj_masks[u] &= ~(1 << v)
            adj_masks[v] &= ~(1 << u)
            
            edges.add((x, y))
            edges.add((y, x))
            adj_masks[x] |= (1 << y)
            adj_masks[y] |= (1 << x)
            
            # Update books
            books = calc_books()
            current_max = int(np.max(books))
            
            if current_max < best_max:
                best_max = current_max
                best_adj_masks = adj_masks.copy()
                best_edges = edges.copy()
                
            # Update edge_list
            edge_list = list(edges)
            
        T *= cooling
        if T < T_min:
            T = T_min
            
    # Return best
    res = []
    for u, v in best_edges:
        if u < v:
            res.append([u, v])
    return res
