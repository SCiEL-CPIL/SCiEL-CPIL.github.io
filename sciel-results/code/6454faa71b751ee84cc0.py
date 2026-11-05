import numpy as np
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    n = 40
    target_edges = 401
    
    # Start with a random regular-ish graph
    # Average degree = 2*401/40 = 20.05
    # Let's start with a 20-regular graph (400 edges) and add 1
    
    # Generate a random 20-regular graph
    # We can use a simple method: random permutation based
    # Or just random edges until we have 400, then check degrees and fix
    
    # Actually, let's start with a circulant graph for symmetry
    # Degree 20 circulant on 40 vertices
    # Differences: pick 10 differences from 1..19 such that sum of counts is 20
    # For even n, if we include n/2, it contributes 1 to degree, others contribute 2
    # Let's pick differences: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    # Each contributes 2 edges (i to i+d and i to i-d), except if d=20 (not here)
    # So degree = 2 * 10 = 20.
    # Edges: (i, (i+d)%40) for d in 1..10, i in 0..39
    # But (i, i+d) is same as (i+d, i), so we need to be careful not to double count
    # Standard circulant: for each d, edges are (i, (i+d)%n) for i in 0..n-1
    # If d != n/2, this creates n edges, but each undirected edge is counted twice?
    # No, (i, i+d) and (i+d, i) are different pairs in the list if we iterate all i.
    # Wait, edge (u,v) is distinct from (v,u).
    # If we list all (i, (i+d)%n), we get n edges.
    # For d=1: (0,1), (1,2), ..., (39,0). These are 40 distinct undirected edges.
    # So 10 differences * 40 edges = 400 edges.
    # Degree of each vertex: connected to +/-1, +/-2, ..., +/-10. Total 20 neighbors.
    # This is a 20-regular graph with 400 edges.
    
    diffs = list(range(1, 11))
    edges = set()
    for d in diffs:
        for i in range(n):
            u, v = i, (i + d) % n
            if u > v:
                u, v = v, u
            edges.add((u, v))
    
    # Check count
    assert len(edges) == 400, f"Expected 400, got {len(edges)}"
    
    # Add 1 random edge
    all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
    missing = [e for e in all_possible if e not in edges]
    if missing:
        e_add = random.choice(missing)
        edges.add(e_add)
    
    # Now we have 401 edges.
    # Local Search: Simulated Annealing or Hill Climbing
    # Objective: Minimize max book size.
    # Book size of edge (u,v) = number of common neighbors of u and v.
    
    # Precompute adjacency matrix
    adj = np.zeros((n, n), dtype=np.int32)
    edge_list = list(edges)
    for u, v in edge_list:
        adj[u, v] = 1
        adj[v, u] = 1
    
    # Calculate book sizes
    # Book(u,v) = sum(w) adj[u,w]*adj[v,w]
    # We can compute this for all edges.
    
    def get_books(adj):
        books = {}
        for u, v in edge_list:
            # Common neighbors
            # (adj[u] & adj[v]).sum()
            b = np.dot(adj[u], adj[v])
            books[(u, v)] = b
        return books
    
    books = get_books(adj)
    max_book = max(books.values())
    
    # Optimization loop
    best_max_book = max_book
    best_edges = set(edge_list)
    best_adj = adj.copy()
    best_books = books.copy()
    
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    # Precompute potential moves?
    # A move is: remove edge (u,v), add edge (x,y) where (x,y) not in E.
    # This changes degrees and books.
    # To be efficient, we should pick a high-book edge to remove and a low-book potential edge to add?
    # Or just random swap.
    
    # Let's try:
    # 1. Pick an edge (u,v) from the current graph. Weight by book size?
    # 2. Pick a non-edge (x,y).
    # 3. Swap.
    # 4. Evaluate delta in max book.
    
    # To make it faster, let's maintain the book sizes incrementally?
    # That's complex. For 40 vertices, 401 edges, full recalculation is:
    # 401 edges * (40*40 dot product) ~ 401 * 1600 ops = 641k ops.
    # In Python/NumPy, this might be ~0.1ms - 1ms per evaluation.
    # 100k evaluations in 140s -> 1.4ms per eval. Feasible.
    
    # Let's optimize the book calculation using matrix multiplication?
    # Books for all pairs (u,v) is (A @ A.T) - A (since diagonal is degree, and we want off-diagonal common neighbors)
    # Actually, (A @ A.T)[u,v] is the number of common neighbors of u and v.
    # This is exactly the book size for edge (u,v) if (u,v) is an edge.
    # So we can compute B = A @ A.T.
    # Then for each edge (u,v), book size is B[u,v].
    # This is one matrix multiplication of 40x40 matrices. Very fast.
    
    def calc_max_book(adj):
        B = adj @ adj.T
        max_b = 0
        for u, v in edge_list:
            if B[u, v] > max_b:
                max_b = B[u, v]
        return max_b, B
    
    current_max_book, current_B = calc_max_book(adj)
    
    # List of non-edges for random selection
    non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if (i, j) not in edges]
    
    # Random swaps
    for step in range(200000):
        if time.time() > deadline:
            break
        
        # Pick random edge to remove
        u, v = random.choice(edge_list)
        
        # Pick random non-edge to add
        # We need to ensure (x,y) is not (u,v) and not in edges
        # Since we pick from non_edges, it's not in edges.
        # But we must ensure we don't add (u,v) itself? No, (u,v) is in edges, so it's not in non_edges.
        # However, the set of non_edges changes.
        # For efficiency, let's just pick random x, y and check.
        x, y = random.sample(range(n), 2)
        if x > y:
            x, y = y, x
        
        if (x, y) == (u, v):
            continue
        if (x, y) in edges:
            continue
            
        # Perform swap
        edges.remove((u, v))
        edges.add((x, y))
        
        adj[u, v] = 0
        adj[v, u] = 0
        adj[x, y] = 1
        adj[y, x] = 1
        
        new_max_book, new_B = calc_max_book(adj)
        
        # Accept if better or with probability exp(-delta/T)
        delta = new_max_book - current_max_book
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_max_book = new_max_book
            current_B = new_B
            if current_max_book < best_max_book:
                best_max_book = current_max_book
                best_edges = set(edges)
                best_adj = adj.copy()
        else:
            # Revert
            edges.remove((x, y))
            edges.add((u, v))
            adj[u, v] = 1
            adj[v, u] = 1
            adj[x, y] = 0
            adj[y, x] = 0
        
        T *= cooling
        if T < T_min:
            T = T_min
            
    # Return best found
    # Convert set of tuples to list of lists
    return [list(e) for e in best_edges]
