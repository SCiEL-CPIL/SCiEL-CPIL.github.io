import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n = 40
    m = 401
    
    # Initialize with the structural construction from Inspiration 1
    # This is known to have max_book = 10 (based on score analysis)
    # We hope local search can reduce it to 7.
    
    edges = []
    edge_set = set()
    
    # Vertices 0..19 in A, 20..39 in B
    # Base: K20,20 (400 edges)
    # Remove 9 edges: (0, 20), (0, 21), ..., (0, 28)
    # Add 10 internal edges: (0,1), (0,2), ..., (0,10)
    
    for u in range(20):
        for v in range(20, 40):
            if u == 0 and 20 <= v <= 28:
                continue
            edges.append([u, v])
            edge_set.add((u, v))
    
    for i in range(1, 11):
        edges.append([0, i])
        edge_set.add((0, i))
        
    # Verify
    if len(edges) != m:
        # Fallback to random if construction failed (should not happen)
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_edges)
        edges = [list(e) for e in all_edges[:m]]
        edge_set = set(tuple(e) for e in edges)

    # Build adjacency masks
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        
    # Compute initial books
    books = {}
    for u, v in edges:
        common = bin(adj_mask[u] & adj_mask[v]).count('1')
        books[(u, v)] = common
        
    current_max_book = max(books.values())
    best_max_book = current_max_book
    best_edge_set = set(edge_set)
    best_adj_mask = list(adj_mask)
    
    if current_max_book <= 7:
        return [list(e) for e in best_edge_set]

    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    all_edges_set = set(all_edges)
    
    T = 2.0
    T_min = 0.01
    cooling = 0.999
    
    def get_affected_edges(e1, e2):
        """Get edges that might change book size when e1 is removed and e2 is added."""
        # Vertices involved
        u1, v1 = e1
        u2, v2 = e2
        verts = {u1, v1, u2, v2}
        
        affected = set()
        for x in verts:
            mask = adj_mask[x]
            while mask:
                y = (mask & -mask).bit_length() - 1
                mask &= mask - 1
                e = (min(x, y), max(x, y))
                affected.add(e)
        
        # Also include e2 itself
        affected.add(e2)
        return affected

    # Main SA Loop
    while time.time() < deadline and T > T_min:
        # Select edge to remove
        # Bias towards high book edges
        if random.random() < 0.7:
            # Find top 10% high book edges
            sorted_books = sorted(books.items(), key=lambda x: -x[1])
            top_k = max(1, len(sorted_books) // 10)
            e_remove = random.choice(sorted_books[:top_k])[0]
        else:
            e_remove = random.choice(list(edge_set))
            
        # Select edge to add
        # Bias towards edges with low common neighbors
        # We can't easily sort all non-edges, so pick random and check
        # Or maintain a list of non-edges? Too slow to update.
        # Just pick random non-edge.
        while True:
            e_add = random.choice(all_edges)
            if e_add not in edge_set:
                break
                
        if e_add == e_remove:
            T *= cooling
            continue
            
        # Calculate delta
        # Store old books for affected edges
        affected = get_affected_edges(e_remove, e_add)
        old_books = {e: books.get(e, 0) for e in affected}
        
        # Apply move temporarily
        u1, v1 = e_remove
        u2, v2 = e_add
        
        adj_mask[u1] &= ~(1 << v1)
        adj_mask[v1] &= ~(1 << u1)
        adj_mask[u2] |= (1 << v2)
        adj_mask[v2] |= (1 << u2)
        
        edge_set.remove(e_remove)
        edge_set.add(e_add)
        
        # Update books for affected edges
        new_max = 0
        for e in affected:
            if e == e_remove:
                continue # Removed
            x, y = e
            if e in edge_set:
                common = bin(adj_mask[x] & adj_mask[y]).count('1')
                books[e] = common
                if common > new_max:
                    new_max = common
            else:
                # Should not happen for affected edges that were previously in set
                # But e_add is new
                pass
        
        # We also need to check max of unaffected edges
        # This is expensive. 
        # Optimization: Keep a counter of how many edges have each book size?
        # Or just keep the global max and update it if the removed edge was the max.
        
        # Let's use a simpler approach: 
        # If the removed edge was the unique max, we must recalc global max.
        # Otherwise, new_max is max(new_max, old_global_max - 1 if old_global_max was removed?)
        
        # Actually, just recalc global max if the removed edge had the max book size.
        if old_books.get(e_remove, 0) == current_max_book:
             # Recalc global max
             new_global_max = max(books.values()) if books else 0
        else:
             # The max can only decrease if the removed edge was the max.
             # Since it wasn't, the max is at least current_max_book.
             # And at most max(current_max_book, new_max)
             new_global_max = max(current_max_book, new_max)
             
        delta = new_global_max - current_max_book
        
        # Accept/Reject
        if delta < 0 or random.random() < math.exp(-delta / T):
            # Accept
            current_max_book = new_global_max
            if current_max_book < best_max_book:
                best_max_book = current_max_book
                best_edge_set = set(edge_set)
                best_adj_mask = list(adj_mask)
                if best_max_book <= 7:
                    break
        else:
            # Revert
            adj_mask[u1] |= (1 << v1)
            adj_mask[v1] |= (1 << u1)
            adj_mask[u2] &= ~(1 << v2)
            adj_mask[v2] &= ~(1 << u2)
            
            edge_set.add(e_remove)
            edge_set.remove(e_add)
            
            # Revert books
            for e, b in old_books.items():
                if e == e_remove:
                    books[e] = b
                elif e in edge_set:
                    books[e] = b
                else:
                    # e_add was added then removed, so it shouldn't be in books
                    # But if it was in old_books, it means it was in the graph? 
                    # No, e_add was NOT in edge_set initially.
                    # So old_books[e_add] should be 0.
                    pass
        
        T *= cooling

    return [list(e) for e in best_edge_set]
