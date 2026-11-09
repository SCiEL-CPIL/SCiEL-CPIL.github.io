import time
import random
import numpy as np
from collections import Counter

def construct():
    start = time.time()
    deadline = 145.0
    
    n = 40
    target = 401
    
    # Use bitmasks for fast common-neighbor counting
    adj = [0] * n
    edges = set()
    
    def add(u, v):
        if u == v:
            return False
        if u > v:
            u, v = v, u
        if (u, v) in edges:
            return False
        edges.add((u, v))
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        return True
    
    def remove(u, v):
        if u == v:
            return False
        if u > v:
            u, v = v, u
        if (u, v) not in edges:
            return False
        edges.remove((u, v))
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        return True
    
    def book_size(u, v):
        """Number of common neighbors of u and v = triangles containing edge (u,v)"""
        return bin(adj[u] & adj[v]).count('1')
    
    def all_books():
        """Return dict of edge -> book size for all edges"""
        books = {}
        for u, v in edges:
            books[(u, v)] = book_size(u, v)
        return books
    
    def max_book_from_books(books):
        if not books:
            return 0
        return max(books.values())
    
    def get_max_edges(books, max_val):
        return [e for e, c in books.items() if c == max_val]
    
    # Strategy: Use a random 401-edge graph with controlled structure
    # Start with a bipartite-like base and add edges strategically
    
    # Initialize with K20,20 + 1 intra edge (known good start)
    for a in range(20):
        for b in range(20, 40):
            add(a, b)
    add(0, 1)
    
    books = all_books()
    current_max = max_book_from_books(books)
    best_max = current_max
    best_edges = set(edges)
    best_books = dict(books)
    
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Track book size distribution
    book_dist = Counter(books.values())
    
    # Local search: try to reduce max book by removing max-book edges
    # and replacing with edges that have low book potential
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        
        # Get edges with maximum book size
        max_edges = get_max_edges(books, current_max)
        if not max_edges:
            break
        
        # Try each max-book edge for removal (or random subset for speed)
        if len(max_edges) > 10:
            candidates = random.sample(max_edges, 10)
        else:
            candidates = max_edges
        
        found_improvement = False
        for u, v in candidates:
            if time.time() > deadline - 5:
                break
            
            # Remove the max-book edge
            remove(u, v)
            old_books = dict(books)
            del books[(u, v)]
            
            # Try to add a replacement edge
            # Prefer edges between vertices that currently have low common-neighbor counts
            # Sample multiple candidates
            best_new_max = current_max
            best_add_pair = None
            best_new_books = None
            
            # Try several random additions
            for _ in range(100):
                if time.time() > deadline - 5:
                    break
                a, b = random.choice(all_pairs)
                if (a, b) in edges:
                    continue
                
                # Quick estimate: book size if we add (a,b)
                # = number of common neighbors of a and b in current graph
                pot_book = bin(adj[a] & adj[b]).count('1')
                
                # Skip if this would create a book larger than current_max
                if pot_book >= current_max:
                    continue
                
                add(a, b)
                
                # Recompute books for affected edges only
                # Affected: edges sharing a vertex with (a,b)
                affected = set()
                for x, y in edges:
                    if x == a or x == b or y == a or y == b:
                        affected.add((x, y))
                affected.add((a, b))
                
                new_books = dict(books)
                for e in affected:
                    x, y = e
                    new_books[e] = book_size(x, y)
                
                new_max = max_book_from_books(new_books)
                
                if new_max < best_new_max:
                    best_new_max = new_max
                    best_add_pair = (a, b)
                    best_new_books = new_books
                    if new_max < current_max:
                        break
                else:
                    remove(a, b)
            
            # Apply best improvement
            if best_new_books is not None and best_new_max <= current_max:
                if best_add_pair is not None:
                    # Ensure the edge is added
                    if best_add_pair not in edges:
                        add(*best_add_pair)
                books = best_new_books
                current_max = best_new_max
                found_improvement = True
                if current_max < best_max:
                    best_max = current_max
                    best_edges = set(edges)
                    best_books = dict(books)
                    if best_max <= 7:
                        return [[u, v] for u, v in best_edges]
                break
            else:
                # Restore removed edge
                if (u, v) not in edges:
                    add(u, v)
                books = old_books
        
        if not found_improvement:
            # Try a different strategy: random restart with perturbation
            if random.random() < 0.1 and time.time() < deadline - 20:
                # Perturb: remove a random edge, add a random edge
                if edges:
                    ru, rv = random.choice(list(edges))
                    remove(ru, rv)
                    for _ in range(50):
                        a, b = random.choice(all_pairs)
                        if (a, b) not in edges:
                            add(a, b)
                            break
                    books = all_books()
                    current_max = max_book_from_books(books)
    
    edges = best_edges
    return [[u, v] for u, v in edges]
