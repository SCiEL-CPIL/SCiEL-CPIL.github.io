import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    n = 40
    target_edges = 401

    # Start with a Turán graph T_2(40) = K_{20,20} (400 edges) + 1 intra edge
    # This gives a book size of at most 20 (for the intra edge) and 0 for cross edges.
    # We want to minimize the max book.
    
    # Better strategy: Start from a random graph with 401 edges and use local search.
    # Or start from Turan + 1 and try to reduce the max book.
    
    # The max book for T_2(40) + 1 intra edge (say edge (0,1) within part 0):
    # For edge (0,1): common neighbors are vertices in part 1 that connect to both 0 and 1.
    # In K_{20,20}, every vertex in part 0 connects to every vertex in part 1.
    # So common neighbors of (0,1) = all 20 vertices in part 1. Book size = 20.
    # For cross edges (i in part0, j in part1): common neighbors = vertices in part0 that connect to j AND vertices in part1 that connect to i.
    # j connects to all 20 in part0, i connects to all 20 in part1.
    # Common neighbors of (i,j) = (part0 \ {i}) ∩ N(j) ∪ (part1 \ {j}) ∩ N(i)
    # = 19 + 19 = 38? No wait.
    # N(i) = all 20 vertices in part1. N(j) = all 20 vertices in part0.
    # N(i) ∩ N(j) = (part1) ∩ (part0) = empty set. So book size = 0 for cross edges.
    # Only the intra edge has book size 20.
    
    # So initial max book = 20. We want to reduce this.
    
    # Strategy: Use simulated annealing / local search to minimize max book.
    # Start from T_2(40) + 1, then try edge swaps to reduce max book.
    
    adj = np.zeros((n, n), dtype=np.uint8)
    edges = set()
    
    def add_edge(u, v):
        if u == v:
            return False
        if u > v:
            u, v = v, u
        if (u, v) in edges:
            return False
        edges.add((u, v))
        adj[u, v] = 1
        adj[v, u] = 1
        return True
    
    def remove_edge(u, v):
        if u == v:
            return False
        if u > v:
            u, v = v, u
        if (u, v) not in edges:
            return False
        edges.remove((u, v))
        adj[u, v] = 0
        adj[v, u] = 0
        return True
    
    def compute_max_book():
        """Compute the maximum book size over all edges."""
        max_book = 0
        for u, v in edges:
            cnt = int(np.sum(adj[u] & adj[v]))
            if cnt > max_book:
                max_book = cnt
        return max_book
    
    def compute_all_books():
        """Compute book sizes for all edges."""
        books = {}
        for u, v in edges:
            cnt = int(np.sum(adj[u] & adj[v]))
            books[(u, v)] = cnt
        return books
    
    def max_book_of_books(books):
        if not books:
            return 0
        return max(books.values())
    
    # Initialize: K_{20,20} + 1 intra edge
    part_a = list(range(20))
    part_b = list(range(20, 40))
    
    for a in part_a:
        for b in part_b:
            add_edge(a, b)
    # Add one intra edge in part_a
    intra_edge = (0, 1)
    add_edge(*intra_edge)
    
    current_books = compute_all_books()
    current_max = max_book_of_books(current_books)
    best_max = current_max
    best_edges = set(edges)
    
    all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    rng = random.Random(42)
    
    while time.time() < deadline:
        # Try to find an edge swap that reduces the max book
        # Pick a random edge from the current graph
        if not edges:
            break
        
        u1, v1 = rng.choice(list(edges))
        
        # Remove it
        remove_edge(u1, v1)
        
        # Try to add a random non-edge
        u2, v2 = rng.choice(all_possible)
        if (u2, v2) in edges or (u2, v2) == (u1, v1):
            # Not a valid swap, re-add and continue
            add_edge(u1, v1)
            continue
        
        add_edge(u2, v2)
        
        # Recompute books
        new_books = compute_all_books()
        new_max = max_book_of_books(new_books)
        
        # Accept if better, or with probability if worse (simulated annealing)
        if new_max < current_max:
            current_max = new_max
            current_books = new_books
            if new_max < best_max:
                best_max = new_max
                best_edges = set(edges)
                if best_max <= 7:
                    break
        elif new_max == current_max and rng.random() < 0.3:
            # Accept equal with some probability for exploration
            current_books = new_books
        else:
            # Revert
            remove_edge(u2, v2)
            add_edge(u1, v1)
            current_books = compute_all_books()
            current_max = max_book_of_books(current_books)
    
    edges = best_edges
    return [[u, v] for u, v in edges]
