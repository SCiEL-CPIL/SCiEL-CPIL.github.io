"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n = 40
    target_edges = 401
    total_possible_edges = n * (n - 1) // 2
    
    # 1. Initialize with a nearly regular graph
    # Degrees: 38 vertices of degree 20, 2 vertices of degree 21
    # Sum of degrees = 38*20 + 2*21 = 760 + 42 = 802. Edges = 401.
    
    # Generate a random 20-regular graph on 40 vertices
    # We can use a simple heuristic: random matching of stubs
    # Or just start with a random graph and regularize it?
    # Simpler: Start with a random graph, then perform swaps to equalize degrees.
    
    A = np.zeros((n, n), dtype=bool)
    edges = []
    non_edges = []
    
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(all_pairs)
    
    for u, v in all_pairs[:target_edges]:
        A[u, v] = True
        A[v, u] = True
        edges.append((u, v))
    
    for u, v in all_pairs[target_edges:]:
        non_edges.append((u, v))
        
    # Regularization Phase: Try to make degrees as equal as possible
    # Target degrees: 20 or 21
    def compute_degrees(A):
        return np.sum(A, axis=1)
        
    def get_max_book_from_books(books, edge_set):
        max_b = 0
        for e in edge_set:
            if e in books:
                b = books[e]
                if b > max_b:
                    max_b = b
        return max_b
        
    def compute_all_books(A):
        books = {}
        for i in range(n):
            for j in range(i+1, n):
                books[(i, j)] = np.sum(A[i] & A[j])
        return books
        
    # Perform some swaps to regularize
    for _ in range(1000):
        if time.time() > deadline: break
        degs = compute_degrees(A)
        # Find vertex with max degree and min degree
        u = np.argmax(degs)
        v = np.argmin(degs)
        if degs[u] - degs[v] <= 1:
            continue
            
        # Find a neighbor of u that is not a neighbor of v
        # And a non-neighbor of u that is a neighbor of v
        # Swap (u, w1) and (v, w2) -> (u, w2) and (v, w1)
        # This keeps degrees same? No.
        # deg(u) decreases by 1, deg(v) increases by 1?
        # Remove (u, w1), add (v, w1). deg(u)-1, deg(v)+1.
        # Remove (v, w2), add (u, w2). deg(v)-1, deg(u)+1.
        # Net change: degs unchanged.
        # We want to reduce max deg and increase min deg.
        # So we need a move that transfers degree from u to v.
        # Remove (u, w1) where w1 is neighbor of u, not v.
        # Add (v, w1).
        # This reduces deg(u) by 1, increases deg(v) by 1.
        # But we must keep edge count constant. So we must add an edge elsewhere?
        # No, we are just swapping edges in the set.
        # If we remove (u, w1) and add (v, w1), we change the graph.
        # But we need to maintain 401 edges.
        # So we must also remove an edge incident to v and add one incident to u?
        # This is a 2-swap.
        
        # Find w1 in N(u) \ N(v)
        # Find w2 in N(v) \ N(u)
        # If we swap (u, w1) with (v, w2) -> remove both, add (u, w2) and (v, w1)?
        # No, that's not a valid swap if (u, w2) or (v, w1) exist.
        
        # Simpler: Just use the random initialization and rely on SA to improve.
        # The regularization is nice but might be complex to implement correctly with edge count constraint.
        pass

    books = compute_all_books(A)
    current_max = get_max_book_from_books(books, set(edges))
    best_A = A.copy()
    best_max = current_max
    
    # Simulated Annealing
    T = 1.0
    cooling_rate = 0.995
    
    max_iterations = 50000
    iterations = 0
    
    def update_books_after_swap(books, A, removed_edge, added_edge):
        u, v = removed_edge
        a, b = added_edge
        affected_nodes = set([u, v, a, b])
        for node in affected_nodes:
            for other in range(n):
                if other != node:
                    edge = (min(node, other), max(node, other))
                    books[edge] = np.sum(A[node] & A[other])
        return books

    while time.time() < deadline and iterations < max_iterations:
        iterations += 1
        
        # Pick a random edge to remove
        u, v = random.choice(edges)
        
        # Pick a random non-edge to add
        a, b = random.choice(non_edges)
        
        # Perform swap
        A[u, v] = False
        A[v, u] = False
        A[a, b] = True
        A[b, a] = True
        
        edges.remove((u, v))
        edges.append((a, b))
        non_edges.remove((a, b))
        non_edges.append((u, v))
        
        books = update_books_after_swap(books, A, (u, v), (a, b))
        new_max = get_max_book_from_books(books, set(edges))
        
        delta = new_max - current_max
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_max = new_max
            if current_max < best_max:
                best_max = current_max
                best_A = A.copy()
        else:
            # Revert
            A[u, v] = True
            A[v, u] = True
            A[a, b] = False
            A[b, a] = False
            
            edges.remove((a, b))
            edges.append((u, v))
            non_edges.remove((u, v))
            non_edges.append((a, b))
            
            books = update_books_after_swap(books, A, (a, b), (u, v))
            
        T *= cooling_rate
        if T < 0.01:
            T = 0.01

    # Convert best_A to list of edges
    result = []
    for i in range(n):
        for j in range(i+1, n):
            if best_A[i, j]:
                result.append([i, j])
                
    if len(result) != 401:
        result = []
        all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_pairs)
        result = [list(e) for e in all_pairs[:target_edges]]
        
    return result
# EVOLVE-BLOCK-END
