import random
import time
import math

def construct():
    n = 40
    m = 401
    deadline = time.time() + 110.0
    
    # Initialize
    edges = set()
    while len(edges) < m:
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u != v:
            if u > v: u, v = v, u
            edges.add((u, v))
            
    adj = [set() for _ in range(n)]
    adj_bits = [0] * n
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        adj_bits[u] |= (1 << v)
        adj_bits[v] |= (1 << u)
        
    books = {}
    book_counts = {}
    max_book = 0
    for u in range(n):
        for v in adj[u]:
            if u < v:
                common = adj_bits[u] & adj_bits[v]
                b = bin(common).count('1')
                books[(u, v)] = b
                book_counts[b] = book_counts.get(b, 0) + 1
                if b > max_book:
                    max_book = b
                    
    best_edges = set(edges)
    best_max_book = max_book
    
    T = 5.0
    decay = 0.995
    
    while time.time() < deadline:
        # Find max book edge
        # We track max_book
        # Find an edge with book size == max_book
        # We can iterate books
        target_edge = None
        for e, b in books.items():
            if b == max_book:
                target_edge = e
                break
                
        if target_edge is None:
            break
            
        u, v = target_edge
        
        common_neighbors = [x for x in adj[u] if x in adj[v]]
        if not common_neighbors:
            break
            
        w = random.choice(common_neighbors)
        
        non_neighbors = [x for x in range(n) if x not in adj[u] and x != v]
        if not non_neighbors:
            continue
            
        good_z = [z for z in non_neighbors if z not in adj[v]]
        if good_z:
            z = random.choice(good_z)
        else:
            z = random.choice(non_neighbors)
            
        # Save state
        adj_copy = [s.copy() for s in adj]
        adj_bits_copy = adj_bits[:]
        books_copy = books.copy()
        book_counts_copy = book_counts.copy()
        max_book_copy = max_book
        
        # Perform swap
        # Remove (u, w)
        adj[u].remove(w)
        adj[w].remove(u)
        adj_bits[u] &= ~(1 << w)
        adj_bits[w] &= ~(1 << u)
        
        # Add (u, z)
        adj[u].add(z)
        adj[z].add(u)
        adj_bits[u] |= (1 << z)
        adj_bits[z] |= (1 << u)
        
        # Update books
        # Affected vertices: u, w, z
        # Edges incident to these vertices
        affected = set()
        for vert in [u, w, z]:
            for nei in adj[vert]:
                e = (min(vert, nei), max(vert, nei))
                affected.add(e)
                
        for e in affected:
            if e in books:
                old_b = books[e]
                book_counts[old_b] -= 1
                if book_counts[old_b] == 0:
                    del book_counts[old_b]
                del books[e]
                
            a, b = e
            if a in adj[b]:
                new_b = bin(adj_bits[a] & adj_bits[b]).count('1')
                books[e] = new_b
                book_counts[new_b] = book_counts.get(new_b, 0) + 1
                if new_b > max_book:
                    max_book = new_b
            else:
                # Should not happen if e was in affected and we only added/removed valid edges
                # But if e was (u,w), it is removed.
                # If e was (u,z), it is added.
                # If e was (w,z), it might be added/removed? No, (w,z) status unchanged.
                # Wait, if (w,z) was not an edge, and we didn't touch it, it's not in books.
                # If it was an edge, it's in books.
                # My affected set includes all edges incident to u, w, z.
                # So if (w,z) is an edge, it is in affected.
                # Its book size might change if N(w) or N(z) changed.
                # N(w) lost u. N(z) gained u.
                # So b(w,z) = |N(w) \cap N(z)|.
                # u is in N(z) now. Is u in N(w)? No, we removed (u,w).
                # So u is not in intersection.
                # Did u used to be in intersection?
                # Before: u in N(w). u in N(z)? No, (u,z) was not edge.
                # So u was not in intersection before.
                # So b(w,z) unchanged?
                # Let's check.
                # Old N(w) had u. Old N(z) did not have u.
                # New N(w) no u. New N(z) has u.
                # Intersection change:
                # u was in N(w) but not N(z) -> not in intersection.
                # u is not in N(w) but in N(z) -> not in intersection.
                # So u doesn't affect intersection.
                # What about other vertices?
                # N(w) changed by -u. N(z) changed by +u.
                # For any x != u:
                # x in N(w) old? x in N(w) new? Same.
                # x in N(z) old? x in N(z) new? Same.
                # So intersection for x != u is same.
                # So b(w,z) is unchanged.
                # But my code recalculates it. That's fine.
                pass
        
        # Update max_book if it decreased
        while max_book > 0 and book_counts.get(max_book, 0) == 0:
            max_book -= 1
            
        # Acceptance
        if max_book <= max_book_copy:
            if max_book < best_max_book:
                best_max_book = max_book
                best_edges = set()
                for u2 in range(n):
                    for v2 in adj[u2]:
                        if u2 < v2:
                            best_edges.add((u2, v2))
        else:
            delta = max_book - max_book_copy
            if random.random() < math.exp(-delta / T):
                if max_book < best_max_book:
                    best_max_book = max_book
                    best_edges = set()
                    for u2 in range(n):
                        for v2 in adj[u2]:
                            if u2 < v2:
                                best_edges.add((u2, v2))
            else:
                # Revert
                adj = adj_copy
                adj_bits = adj_bits_copy
                books = books_copy
                book_counts = book_counts_copy
                max_book = max_book_copy
        
        T *= decay
        if T < 0.1:
            break
            
    return [list(e) for e in best_edges]
