"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n = 40
    target_edges = 401
    total_possible_edges = n * (n - 1) // 2

    # 1. Construct a 20-regular graph using cyclic shifts (blow-up of C5)
    # This is a quasi-random structure with good spectral properties.
    A = np.zeros((n, n), dtype=bool)
    edges = []
    non_edges = []
    
    # 8 vertices per part, 5 parts
    part_size = 8
    # Connect part i to part (i+1)%5 and (i-1)%5 with a perfect matching
    # We use a cyclic shift for the matching to ensure regularity and spread
    for i in range(5):
        start_i = i * part_size
        # Next part
        start_next = ((i + 1) % 5) * part_size
        # Prev part
        start_prev = ((i - 1) % 5) * part_size
        
        for k in range(part_size):
            u = start_i + k
            v_next = start_next + ((k + 1) % part_size)
            v_prev = start_prev + ((k - 1) % part_size)
            
            A[u, v_next] = True
            A[v_next, u] = True
            edges.append((u, v_next))
            
            A[u, v_prev] = True
            A[v_prev, u] = True
            edges.append((u, v_prev))
            
            # Also need to add the other direction for the list to be complete?
            # No, we just need unique undirected edges.
            # The loop covers all edges from part i to i+1 and i-1.
            # Since i goes 0..4, edge (0,1) is added when i=0 (0->1) and i=1 (1->0? No, 1->0 is i=1, prev=0).
            # Wait, if I add (u, v_next) for all i, I cover all edges between i and i+1.
            # Do I add duplicates?
            # i=0: 0->1. i=1: 1->2. ... i=4: 4->0.
            # These are all distinct pairs of parts.
            # Within the pair (part 0, part 1), I add 8 edges.
            # Are there any other edges between part 0 and part 1? No.
            # So this constructs exactly 5 * 8 * 2 = 80 edges?
            # Wait.
            # For each i, I add 8 edges to next and 8 edges to prev.
            # Total 16 edges per part.
            # 5 parts * 16 = 80 edges?
            # No, 5 parts * 8 vertices * 2 connections = 80 edges total?
            # Let's check degrees.
            # Each vertex connects to 1 in next, 1 in prev. Degree 2.
            # Total edges = 40 * 2 / 2 = 40.
            # I need 401 edges.
            # So this is just a skeleton. I need to add 361 more edges.
            
    # The above construction gives 40 edges.
    # Let's restart the initialization logic to be cleaner.
    
    A = np.zeros((n, n), dtype=bool)
    edges = []
    non_edges = []
    
    # Build the 2-regular base (C5 blow-up)
    for i in range(5):
        start_i = i * part_size
        start_next = ((i + 1) % 5) * part_size
        start_prev = ((i - 1) % 5) * part_size
        
        for k in range(part_size):
            u = start_i + k
            v_next = start_next + ((k + 1) % part_size)
            v_prev = start_prev + ((k - 1) % part_size)
            
            A[u, v_next] = True
            A[v_next, u] = True
            edges.append((u, v_next))
            
            A[u, v_prev] = True
            A[v_prev, u] = True
            edges.append((u, v_prev))
            
    # Now we have 40 edges. We need 401.
    # Let's fill the rest randomly but ensuring we don't duplicate.
    # Actually, let's just initialize with a random graph of 401 edges but seeded with this structure?
    # Or better: Start with the 2-regular graph, then add random edges until we reach 401.
    
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(all_pairs)
    
    # Remove the existing edges from the candidate list
    existing_set = set(edges)
    candidates = [p for p in all_pairs if p not in existing_set]
    
    # Add random edges until we hit target
    needed = target_edges - len(edges)
    for p in candidates[:needed]:
        u, v = p
        A[u, v] = True
        A[v, u] = True
        edges.append(p)
        
    # Build non_edges list
    edge_set = set(edges)
    non_edges = [p for p in all_pairs if p not in edge_set]

    def compute_all_books(A):
        """Compute book size for every possible edge (including non-edges)."""
        books = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(i+1, n):
                books[i, j] = np.sum(A[i] & A[j])
                books[j, i] = books[i, j]
        return books

    def get_max_book(books, edge_set):
        """Get max book size among edges in edge_set."""
        max_b = 0
        for e in edge_set:
            u, v = e
            if books[u, v] > max_b:
                max_b = books[u, v]
        return max_b

    def update_books_after_swap(books, A, removed_edge, added_edge):
        """
        Incrementally update book sizes.
        """
        u, v = removed_edge
        a, b = added_edge
        
        affected_nodes = [u, v, a, b]
        
        # Recompute book sizes for all edges incident to these nodes
        for node in affected_nodes:
            for other in range(n):
                if other != node:
                    # We update both (node, other) and (other, node)
                    # But books is symmetric.
                    # We only need to update if (node, other) is an edge or non-edge?
                    # The task asks for max book over EDGES.
                    # But books for non-edges are useful for heuristic selection.
                    # So we update all pairs involving affected nodes.
                    books[node, other] = np.sum(A[node] & A[other])
                    books[other, node] = books[node, other]
        
        return books

    # Initial books
    books = compute_all_books(A)
    current_max = get_max_book(books, set(edges))
    best_A = A.copy()
    best_max = current_max
    best_books = books.copy()
    
    # Local Search with Simulated Annealing flavor
    # We want to minimize max book.
    # Move: Swap an edge (u,v) with a non-edge (a,b).
    # Constraint: We must maintain exactly 401 edges.
    
    T = 1.0
    cooling_rate = 0.9995
    min_T = 1e-4
    steps = 0
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() < deadline:
        steps += 1
        
        # Pick a random edge to remove
        u, v = random.choice(edges)
        
        # Pick a random non-edge to add
        # We bias towards non-edges with low book size
        # But random is fine for SA.
        a, b = random.choice(non_edges)
        
        # Perform swap
        A[u, v] = False
        A[v, u] = False
        A[a, b] = True
        A[b, a] = True
        
        # Update books
        books = update_books_after_swap(books, A, (u, v), (a, b))
        
        # Compute new max
        new_max = get_max_book(books, set(edges))
        
        delta = new_max - current_max
        
        if delta < 0 or (T > min_T and random.random() < math.exp(-delta / T)):
            # Accept
            current_max = new_max
            
            # Update lists
            # Replace (u,v) in edges with (a,b)
            idx = edges.index((u, v))
            edges[idx] = (a, b)
            
            # Replace (a,b) in non_edges with (u,v)
            idx = non_edges.index((a, b))
            non_edges[idx] = (u, v)
            
            if current_max < best_max:
                best_max = current_max
                best_A = A.copy()
                best_books = books.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
                
            T *= cooling_rate
        else:
            # Reject
            A[u, v] = True
            A[v, u] = True
            A[a, b] = False
            A[b, a] = False
            
            books = update_books_after_swap(books, A, (a, b), (u, v))
            
            no_improve_count += 1
            T *= cooling_rate
        
        # If stuck for too long, restart or perturb
        if no_improve_count > max_no_improve:
            # Perturb: do 5 random swaps
            for _ in range(5):
                u, v = random.choice(edges)
                a, b = random.choice(non_edges)
                
                A[u, v] = False
                A[v, u] = False
                A[a, b] = True
                A[b, a] = True
                
                idx = edges.index((u, v))
                edges[idx] = (a, b)
                
                idx = non_edges.index((a, b))
                non_edges[idx] = (u, v)
            
            books = compute_all_books(A)
            current_max = get_max_book(books, set(edges))
            
            if current_max < best_max:
                best_max = current_max
                best_A = A.copy()
                best_books = books.copy()
            
            no_improve_count = 0
            T = max(T, 0.5) # Reheat slightly

    # Convert best_A to list of edges
    result = []
    for i in range(n):
        for j in range(i+1, n):
            if best_A[i, j]:
                result.append([i, j])
    
    # Verify count
    if len(result) != 401:
        # Fallback: build a valid graph
        result = []
        all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_pairs)
        result = [list(e) for e in all_pairs[:target_edges]]
        
    return result
# EVOLVE-BLOCK-END
