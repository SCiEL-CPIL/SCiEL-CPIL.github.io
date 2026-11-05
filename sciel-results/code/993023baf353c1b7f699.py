import numpy as np
import time
import random

def construct():
    deadline = time.time() + 140.0
    n = 40
    target_edges = 401
    
    # Bitmask representation
    # adj_mask[u] is an integer where bit v is 1 if edge (u,v) exists
    adj_mask = [0] * n
    
    def add_edge(u, v):
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        
    def remove_edge(u, v):
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)
        
    def edge_count():
        return sum(m.bit_count() for m in adj_mask) // 2
        
    def get_edges():
        edges = []
        for u in range(n):
            m = adj_mask[u]
            while m:
                # Extract lowest set bit
                lsb = m & -m
                v = lsb.bit_length() - 1
                if u < v:
                    edges.append((u, v))
                m &= m - 1
        return edges

    def compute_books():
        """Compute book size for each edge."""
        books = {}
        for u in range(n):
            for v in range(u + 1, n):
                if (adj_mask[u] >> v) & 1:
                    common = (adj_mask[u] & adj_mask[v]).bit_count()
                    books[(u, v)] = common
        return books

    def max_book(books):
        if not books:
            return 0
        return max(books.values())

    # Initialization: Circulant graph
    random.seed(42)
    
    # Clear
    adj_mask = [0] * n
    for i in range(n):
        for d in range(1, 11):
            j = (i + d) % n
            if i < j:
                add_edge(i, j)
                
    # Adjust edge count to 401
    ec = edge_count()
    if ec < target_edges:
        # Add random edges
        while ec < target_edges:
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u != v and not ((adj_mask[u] >> v) & 1):
                add_edge(u, v)
                ec += 1
    elif ec > target_edges:
        edges = get_edges()
        random.shuffle(edges)
        for u, v in edges:
            if ec == target_edges:
                break
            remove_edge(u, v)
            ec -= 1

    # Initial books
    books = compute_books()
    best_books = {k: v for k, v in books.items()}
    best_max_book = max_book(best_books)
    best_adj_mask = adj_mask.copy()
    
    # Local Search with Simulated Annealing
    T = 10.0
    T_min = 0.01
    cooling = 0.999
    
    start_time = time.time()
    step = 0
    no_improve_count = 0
    max_no_improve = 5000
    
    while T > T_min and time.time() < deadline:
        step += 1
        
        # Get current edges for sampling
        # Optimization: maintain a list of edges?
        # For n=40, get_edges() is fast enough (401 edges)
        edges = get_edges()
        if len(edges) != target_edges:
            break
            
        u1, v1 = random.choice(edges)
        
        # Find a random non-edge to add
        # Strategy: pick random u2, v2
        found = False
        for _ in range(50):
            u2 = random.randint(0, n - 1)
            v2 = random.randint(0, n - 1)
            if u2 == v2:
                continue
            if u2 > v2:
                u2, v2 = v2, u2
            if (u2, v2) == (u1, v1):
                continue
            if not ((adj_mask[u2] >> v2) & 1):
                found = True
                break
                
        if not found:
            continue
            
        # Calculate impact
        # Remove (u1, v1), Add (u2, v2)
        
        # Affected vertices
        affected_verts = {u1, v1, u2, v2}
        
        # Edges that will change book size:
        # 1. Edges incident to u1 or v1 (because their neighbor sets change)
        # 2. Edges incident to u2 or v2 (because their neighbor sets change)
        # 3. The new edge (u2, v2) itself
        
        affected_edges = set()
        for x in affected_verts:
            m = adj_mask[x]
            while m:
                lsb = m & -m
                y = lsb.bit_length() - 1
                if x < y:
                    affected_edges.add((x, y))
                else:
                    affected_edges.add((y, x))
                m &= m - 1
        affected_edges.add((u2, v2))
        
        # Save old books
        old_books = {}
        for e in affected_edges:
            if e in books:
                old_books[e] = books[e]
                
        # Apply swap
        remove_edge(u1, v1)
        add_edge(u2, v2)
        
        # Update books for affected edges
        new_books = {}
        for e in affected_edges:
            a, b = e
            if (adj_mask[a] >> b) & 1:
                common = (adj_mask[a] & adj_mask[b]).bit_count()
                new_books[e] = common
            else:
                # If edge was removed, it's no longer in books
                # But we only care about edges that exist in the new state
                # The old edge (u1,v1) is removed, so it shouldn't be in new_books
                # Unless it was also in affected_edges and we are checking it
                # If it's not an edge, we don't store its book size
                pass
        
        # Update books dict
        for e in affected_edges:
            if e in books:
                del books[e]
        for e, val in new_books.items():
            books[e] = val
            
        # Check improvement
        current_max = max_book(books)
        
        if current_max < best_max_book:
            best_max_book = current_max
            best_books = {k: v for k, v in books.items()}
            best_adj_mask = adj_mask.copy()
            no_improve_count = 0
            # Lower temperature or keep?
            # T = max(T_min, T * 0.9)
        else:
            no_improve_count += 1
            # Accept if better or equal (for secondary metric? No, just max book)
            # Use SA: accept if current_max <= best_max_book?
            # If current_max == best_max_book, we might want to explore.
            # But if we are stuck at 8, we need to accept 8 to move around?
            # Standard SA: accept if delta < 0 or prob.
            # Here objective is max_book.
            # If current_max > best_max_book, delta > 0.
            # Accept with prob exp(-delta/T).
            
            if current_max > best_max_book:
                delta = current_max - best_max_book
                if random.random() < np.exp(-delta / T):
                    pass # Accept
                else:
                    # Reject
                    remove_edge(u2, v2)
                    add_edge(u1, v1)
                    # Restore books
                    for e in affected_edges:
                        if e in books:
                            del books[e]
                    for e, val in old_books.items():
                        books[e] = val
                    T *= cooling
                    continue
            else:
                # current_max <= best_max_book
                # Accept
                pass
                
        T *= cooling
        
        if no_improve_count > max_no_improve and step > 1000:
            # Restart with a small perturbation or break
            # Try to break out of local minimum
            # Randomly swap 5 edges
            for _ in range(5):
                edges_list = get_edges()
                if not edges_list: break
                ru1, rv1 = random.choice(edges_list)
                found_re = False
                for _ in range(50):
                    ru2 = random.randint(0, n - 1)
                    rv2 = random.randint(0, n - 1)
                    if ru2 == rv2: continue
                    if ru2 > rv2: ru2, rv2 = rv2, ru2
                    if (ru2, rv2) == (ru1, rv1): continue
                    if not ((adj_mask[ru2] >> rv2) & 1):
                        found_re = True
                        break
                if found_re:
                    remove_edge(ru1, rv1)
                    add_edge(ru2, rv2)
                    # Update books for these? 
                    # Simpler: recompute all books
                    books = compute_books()
            no_improve_count = 0
            T = 1.0 # Reheat
            
    # Restore best
    adj_mask = best_adj_mask
    
    # Extract edges
    result = []
    for i in range(n):
        m = adj_mask[i]
        while m:
            lsb = m & -m
            j = lsb.bit_length() - 1
            if i < j:
                result.append([i, j])
            m &= m - 1
            
    return result
