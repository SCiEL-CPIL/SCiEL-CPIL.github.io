import time
import random
import numpy as np

def construct():
    start = time.time()
    deadline = 135.0
    
    n = 40
    target = 401
    
    # Use numpy adjacency matrix for fast operations
    adj = np.zeros((n, n), dtype=np.uint8)
    edges = set()
    
    def add(u, v):
        if u == v: return False
        if u > v: u, v = v, u
        if (u, v) in edges: return False
        edges.add((u, v))
        adj[u, v] = 1
        adj[v, u] = 1
        return True
    
    def remove(u, v):
        if u == v: return False
        if u > v: u, v = v, u
        if (u, v) not in edges: return False
        edges.remove((u, v))
        adj[u, v] = 0
        adj[v, u] = 0
        return True
    
    def compute_books():
        """Return dict: edge -> book size"""
        books = {}
        edge_list = list(edges)
        for u, v in edge_list:
            cnt = int(np.sum(adj[u] & adj[v]))
            books[(u, v)] = cnt
        return books
    
    def max_book(books):
        if not books:
            return 0
        return max(books.values())
    
    # Initialize: K20,20 + 1 intra edge
    for a in range(20):
        for b in range(20, 40):
            add(a, b)
    add(0, 1)
    
    books = compute_books()
    current_max = max_book(books)
    best_max = current_max
    best_edges = set(edges)
    best_books = dict(books)
    
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Count how many edges have each book size
    from collections import Counter
    book_counts = Counter(books.values())
    
    def update_book_counts(books, old_books):
        """Update book_counts when books change"""
        # Remove old
        for e, cnt in old_books.items():
            if e in books:
                # Check if changed
                pass
        # Simpler: just recompute
        book_counts.clear()
        for cnt in books.values():
            book_counts[cnt] += 1
    
    def get_edges_with_max(books, max_val):
        """Get list of edges that have book size == max_val"""
        return [e for e, c in books.items() if c == max_val]
    
    while time.time() < deadline:
        # Focus on edges that have the maximum book size
        max_edges = get_edges_with_max(books, current_max)
        
        if not max_edges:
            break
        
        # Pick a random edge with max book size to remove
        u, v = random.choice(max_edges)
        remove(u, v)
        old_books = dict(books)
        del books[(u, v)]
        
        found_improvement = False
        
        # Try to add edges that don't increase the max book
        # Prefer edges that would have low book size
        attempts = 0
        max_attempts = 500
        while attempts < max_attempts and time.time() < deadline:
            a, b = random.choice(all_pairs)
            if (a, b) in edges:
                attempts += 1
                continue
            
            # Quick check: what would the book size be if we added this edge?
            # Book size of (a,b) = number of common neighbors of a and b
            potential_book = int(np.sum(adj[a] & adj[b]))
            
            # If potential_book < current_max, adding it won't increase max
            # (but might decrease if it was the max... no, it's a new edge)
            # Actually, adding an edge can only increase book sizes of existing edges
            # that share a vertex with it, or create new triangles.
            # This is complex. Let's just try it.
            
            add(a, b)
            
            # Recompute books for affected edges only
            # An edge (x,y) is affected if it shares a vertex with (a,b)
            # i.e., x==a or x==b or y==a or y==b
            affected = set()
            for x, y in edges:
                if x == a or x == b or y == a or y == b:
                    affected.add((x, y))
            affected.add((a, b))
            
            new_books = dict(books)
            for e in affected:
                x, y = e
                if x > y: x, y = y, x
                cnt = int(np.sum(adj[x] & adj[y]))
                new_books[(x, y)] = cnt
            
            new_max = max(new_books.values()) if new_books else 0
            
            if new_max < current_max:
                current_max = new_max
                books = new_books
                found_improvement = True
                break
            elif new_max == current_max and random.random() < 0.2:
                # Accept with some probability for exploration
                books = new_books
                break
            else:
                # Revert
                remove(a, b)
                books = dict(old_books)
                # Restore the removed edge
                add(u, v)
                # But we already removed (u,v), so we need to handle this carefully
                # Let's just break and retry
                break
            
            attempts += 1
        
        if not found_improvement:
            # Restore the removed edge if it was removed
            if (u, v) not in edges:
                add(u, v)
                books = dict(old_books)
        
        if current_max < best_max:
            best_max = current_max
            best_edges = set(edges)
            best_books = dict(books)
            if best_max <= 7:
                break
    
    edges = best_edges
    return [[u, v] for u, v in edges]
