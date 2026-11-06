"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

# EVOLVE-BLOCK-START
def construct():
    n = 40
    m = 401
    deadline = time.time() + 145.0  # safety margin before 150s kill
    
    # Start with K20,20 + one edge
    edges = [[a, b] for a in range(20) for b in range(20, 40)]
    edges.append([0, 1])
    
    # Build adjacency matrix
    A = np.zeros((n, n), dtype=np.int32)
    for u, v in edges:
        A[u, v] = 1
        A[v, u] = 1
    
    # Compute books: for each edge, count common neighbors
    # book[e] = number of triangles containing edge e
    # For edge (u,v), book = (A[u] & A[v]).sum()
    
    def compute_max_book(A):
        max_book = 0
        for i in range(n):
            for j in range(i+1, n):
                if A[i, j]:
                    b = np.sum(A[i] & A[j])
                    if b > max_book:
                        max_book = b
        return max_book
    
    current_max = compute_max_book(A)
    best_max = current_max
    best_A = A.copy()
    
    # Simulated annealing with edge swaps
    # Move: remove edge (u1,v1), add edge (u2,v2) where (u2,v2) not in graph
    # Keep edge count constant at 401
    
    def get_edges_from_A(A):
        es = []
        for i in range(n):
            for j in range(i+1, n):
                if A[i, j]:
                    es.append((i, j))
        return es
    
    def remove_edge(A, u, v):
        A[u, v] = 0
        A[v, u] = 0
    
    def add_edge(A, u, v):
        A[u, v] = 1
        A[v, u] = 1
    
    # Precompute for faster book delta computation
    # When we remove (u1,v1) and add (u2,v2):
    # Books change for edges that share a vertex with the removed/added edges
    
    def book_of_edge(A, u, v):
        return int(np.sum(A[u] & A[v]))
    
    # For efficiency, track books for all edges
    # books[(i,j)] for i<j where A[i,j]=1
    
    def all_books(A):
        bk = {}
        for i in range(n):
            for j in range(i+1, n):
                if A[i, j]:
                    bk[(i,j)] = int(np.sum(A[i] & A[j]))
        return bk
    
    current_books = all_books(A)
    current_max = max(current_books.values()) if current_books else 0
    best_max = current_max
    best_A = A.copy()
    
    T = 5.0
    T_min = 0.01
    cooling = 0.9999
    
    random.seed(42)
    
    iteration = 0
    while time.time() < deadline and T > T_min:
        iteration += 1
        
        # Pick a random edge to remove
        edge_list = []
        for (i,j), b in current_books.items():
            edge_list.append((i,j))
        
        if not edge_list:
            break
        
        u1, v1 = random.choice(edge_list)
        
        # Pick a random non-edge to add
        non_edges = []
        for i in range(n):
            for j in range(i+1, n):
                if not A[i, j] and (i,j) != (u1,v1) and (j,i) != (u1,v1):
                    non_edges.append((i,j))
        
        if not non_edges:
            break
        
        u2, v2 = random.choice(non_edges)
        if u2 > v2:
            u2, v2 = v2, u2
        
        # Compute delta in max book
        # Edges affected: those sharing a vertex with (u1,v1) or (u2,v2)
        
        # First, tentatively apply the move
        old_books = {}
        # Save books for edges that will change
        affected = set()
        for i in range(n):
            if A[i, u1] or A[i, v1] or A[i, u2] or A[i, v2]:
                for j in range(i+1, n):
                    if A[i, j]:
                        affected.add((i,j))
        # Also the new edge
        affected.add((u2, v2))
        
        # Compute old max
        old_max = current_max
        
        # Apply move
        remove_edge(A, u1, v1)
        add_edge(A, u2, v2)
        
        # Recompute books for affected edges
        new_books = {}
        for (i, j) in affected:
            if A[i, j]:
                new_books[(i,j)] = int(np.sum(A[i] & A[j]))
        
        # Update current_books
        for (i, j) in affected:
            if (i, j) in current_books:
                if not A[i, j]:
                    del current_books[(i,j)]
                else:
                    current_books[(i,j)] = new_books.get((i,j), 0)
        
        if A[u2, v2]:
            current_books[(u2, v2)] = int(np.sum(A[u2] & A[v2]))
        
        # Remove (u1,v1) from books if it was there
        if (u1, v1) in current_books:
            del current_books[(u1, v1)]
        
        new_max = max(current_books.values()) if current_books else 0
        
        # Accept or reject
        delta = new_max - old_max
        if delta <= 0 or random.random() < np.exp(-delta / T):
            # Accept
            if new_max < best_max:
                best_max = new_max
                best_A = A.copy()
        else:
            # Reject: undo
            remove_edge(A, u2, v2)
            add_edge(A, u1, v1)
            # Restore books
            for (i, j) in affected:
                if A[i, j]:
                    current_books[(i,j)] = int(np.sum(A[i] & A[j]))
                elif (i, j) in current_books:
                    del current_books[(i,j)]
            # Ensure (u1,v1) is in books
            current_books[(u1, v1)] = int(np.sum(A[u1] & A[v1]))
            # Ensure (u2,v2) is NOT in books
            if (u2, v2) in current_books:
                del current_books[(u2, v2)]
            current_max = old_max
        
        T *= cooling
        
        # Every 500 iterations, do a full recompute to avoid drift
        if iteration % 500 == 0:
            current_books = all_books(A)
            current_max = max(current_books.values()) if current_books else 0
    
    # Final: use best_A
    final_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_A[i, j]:
                final_edges.append([i, j])
    
    return final_edges
# EVOLVE-BLOCK-END
