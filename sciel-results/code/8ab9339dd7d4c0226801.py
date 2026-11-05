import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n = 40
    m = 401
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def random_init():
        """Generate a random graph with m edges."""
        random.shuffle(all_edges)
        edges = all_edges[:m]
        edge_set = set(edges)
        adj_mask = [0] * n
        for u, v in edges:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
        return edge_set, adj_mask

    def compute_books(edge_set, adj_mask):
        """Compute book sizes for all edges."""
        books = {}
        for u, v in edge_set:
            books[(u, v)] = bin(adj_mask[u] & adj_mask[v]).count('1')
        return books

    def get_max_book(books):
        return max(books.values()) if books else 0

    # Main loop: Random restarts with Local Search
    best_book = float('inf')
    best_edge_set = None
    
    while time.time() < deadline:
        # 1. Initialize
        edge_set, adj_mask = random_init()
        books = compute_books(edge_set, adj_mask)
        current_max = get_max_book(books)
        
        if current_max < best_book:
            best_book = current_max
            best_edge_set = set(edge_set)
            
        # 2. Local Search
        stall_count = 0
        
        while time.time() < deadline - 1:
            # Find an edge to remove (one with high book size)
            max_b = current_max
            bad_edges = [e for e, b in books.items() if b == max_b]
            
            if not bad_edges:
                break
                
            # Pick a random bad edge
            e_remove = random.choice(bad_edges)
            u, v = e_remove
            
            # Find candidates to add
            # Sample non-edges
            found_improvement = False
            
            # Try a few random non-edges
            for _ in range(50): # Increased sample size
                e_add = random.choice(all_edges)
                if e_add in edge_set:
                    continue
                
                a, b = e_add
                
                # Check if this swap improves the max book
                # 1. Calculate book size of new edge (a,b)
                # Note: adj_mask still has (u,v) and not (a,b)
                # We need to simulate the swap
                
                # Save state
                mask_u = adj_mask[u]
                mask_v = adj_mask[v]
                mask_a = adj_mask[a]
                mask_b = adj_mask[b]
                
                # Apply swap to masks
                adj_mask[u] &= ~(1 << v)
                adj_mask[v] &= ~(1 << u)
                adj_mask[a] |= (1 << b)
                adj_mask[b] |= (1 << a)
                
                # Identify edges whose book size might change
                # These are edges incident to u, v, a, b
                # Plus the new edge (a,b)
                
                # We need to check if any of these edges now have book > max_b
                # Or if we reduced the count of edges with book == max_b
                
                new_max = 0
                # Check new edge
                b_new = bin(adj_mask[a] & adj_mask[b]).count('1')
                if b_new > new_max:
                    new_max = b_new
                
                # Check neighbors of u
                m = adj_mask[u]
                while m:
                    y = (m & -m).bit_length() - 1
                    m &= m - 1
                    e = (min(u, y), max(u, y))
                    if e in edge_set:
                        b_val = bin(adj_mask[u] & adj_mask[y]).count('1')
                        if b_val > new_max:
                            new_max = b_val
                            
                # Check neighbors of v
                m = adj_mask[v]
                while m:
                    y = (m & -m).bit_length() - 1
                    m &= m - 1
                    e = (min(v, y), max(v, y))
                    if e in edge_set:
                        b_val = bin(adj_mask[v] & adj_mask[y]).count('1')
                        if b_val > new_max:
                            new_max = b_val
                            
                # Check neighbors of a
                m = adj_mask[a]
                while m:
                    y = (m & -m).bit_length() - 1
                    m &= m - 1
                    e = (min(a, y), max(a, y))
                    if e in edge_set:
                        b_val = bin(adj_mask[a] & adj_mask[y]).count('1')
                        if b_val > new_max:
                            new_max = b_val
                            
                # Check neighbors of b
                m = adj_mask[b]
                while m:
                    y = (m & -m).bit_length() - 1
                    m &= m - 1
                    e = (min(b, y), max(b, y))
                    if e in edge_set:
                        b_val = bin(adj_mask[b] & adj_mask[y]).count('1')
                        if b_val > new_max:
                            new_max = b_val
                            
                # Revert masks
                adj_mask[u] = mask_u
                adj_mask[v] = mask_v
                adj_mask[a] = mask_a
                adj_mask[b] = mask_b
                
                # Decision
                if new_max < max_b:
                    # Improvement! Apply it
                    edge_set.remove(e_remove)
                    edge_set.add(e_add)
                    
                    # Apply to masks permanently
                    adj_mask[u] &= ~(1 << v)
                    adj_mask[v] &= ~(1 << u)
                    adj_mask[a] |= (1 << b)
                    adj_mask[b] |= (1 << a)
                    
                    # Update books
                    # Remove e_remove
                    del books[e_remove]
                    # Add e_add
                    books[e_add] = b_new
                    
                    # Update neighbors
                    # We need to update books for all edges incident to u,v,a,b
                    # that are in the graph.
                    
                    # Helper to update book for edge (x,y)
                    def update_book(x, y):
                        e = (min(x, y), max(x, y))
                        if e in edge_set:
                            books[e] = bin(adj_mask[x] & adj_mask[y]).count('1')
                        elif e in books:
                            del books[e]
                            
                    # Update u's neighbors
                    m = adj_mask[u]
                    while m:
                        y = (m & -m).bit_length() - 1
                        m &= m - 1
                        update_book(u, y)
                        
                    # Update v's neighbors
                    m = adj_mask[v]
                    while m:
                        y = (m & -m).bit_length() - 1
                        m &= m - 1
                        update_book(v, y)
                        
                    # Update a's neighbors
                    m = adj_mask[a]
                    while m:
                        y = (m & -m).bit_length() - 1
                        m &= m - 1
                        update_book(a, y)
                        
                    # Update b's neighbors
                    m = adj_mask[b]
                    while m:
                        y = (m & -m).bit_length() - 1
                        m &= m - 1
                        update_book(b, y)
                        
                    # Update the new edge itself (already done above, but just in case)
                    # books[e_add] = b_new
                    
                    # Recompute max
                    current_max = get_max_book(books)
                    
                    if current_max < best_book:
                        best_book = current_max
                        best_edge_set = set(edge_set)
                        
                    found_improvement = True
                    break
                elif new_max == max_b:
                    # Neutral move. We might accept it to escape local opt?
                    # For now, let's only accept improvements to be safe and fast.
                    # If we get stuck, we restart.
                    pass
            
            if found_improvement:
                stall_count = 0
            else:
                stall_count += 1
                if stall_count > 1000:
                    break # Restart
            
    return [list(e) for e in best_edge_set]
