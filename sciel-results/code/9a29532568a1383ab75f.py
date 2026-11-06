import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    n = 40
    target_edges = 401
    
    def make_initial_graph():
        # Start with K_20,20
        A = np.zeros((n, n), dtype=np.int8)
        for i in range(20):
            for j in range(20, 40):
                A[i, j] = 1
                A[j, i] = 1
        # Add one edge within partition 0
        A[0, 1] = 1
        A[1, 0] = 1
        return A

    def compute_books(A):
        # books[u, v] = number of triangles containing (u, v)
        # This is the number of common neighbors of u and v
        # A @ A gives counts of walks of length 2.
        # (A @ A)[u, v] = sum_k A[u, k] A[k, v]
        # Since graph is undirected, this is common neighbors.
        # Note: For u != v, (A @ A)[u, v] is exactly the number of common neighbors.
        # The book size is this value for edges (u, v).
        A2 = A @ A
        # We only care about edges where A[u, v] == 1
        books = A2 * A
        return books

    def get_edges(A):
        return [(i, j) for i in range(n) for j in range(i+1, n) if A[i, j] == 1]

    def get_nonedges(A):
        return [(i, j) for i in range(n) for j in range(i+1, n) if A[i, j] == 0]

    def local_search(A, max_iters=20000):
        A = A.copy()
        books = compute_books(A)
        current_score = int(np.max(books))
        best_score = current_score
        best_A = A.copy()
        
        T = 1.0
        T_min = 0.01
        cool = 0.9995
        
        edges = get_edges(A)
        nonedges = get_nonedges(A)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            if not edges or not nonedges:
                break
            
            # Random swap
            u1, v1 = random.choice(edges)
            u2, v2 = random.choice(nonedges)
            
            # Calculate new book sizes for affected edges
            # Affected edges are those incident to u1, v1, u2, or v2
            # And the new edge (u2, v2) itself
            # And the removed edge (u1, v1) is no longer an edge
            
            # We need to update books for:
            # 1. Edges (x, u1) for x in N(u1) \ {v1}
            # 2. Edges (x, v1) for x in N(v1) \ {u1}
            # 3. Edges (x, u2) for x in N(u2)
            # 4. Edges (x, v2) for x in N(v2)
            # 5. The new edge (u2, v2)
            
            # To do this efficiently, we can just recompute the common neighbors
            # for these specific pairs in the NEW graph.
            
            # Let's simulate the swap on a temporary view or just calculate deltas.
            # Removing (u1, v1):
            #   N(u1) loses v1. N(v1) loses u1.
            # Adding (u2, v2):
            #   N(u2) gains v2. N(v2) gains u2.
            
            # For any edge (a, b), new_book(a, b) = |N_new(a) \cap N_new(b)|
            
            # Let's collect the set of edges to recompute
            recompute_edges = set()
            
            # Edges incident to u1
            for x in range(n):
                if A[u1, x] == 1 and x != v1:
                    recompute_edges.add((min(u1, x), max(u1, x)))
            # Edges incident to v1
            for x in range(n):
                if A[v1, x] == 1 and x != u1:
                    recompute_edges.add((min(v1, x), max(v1, x)))
            # Edges incident to u2
            for x in range(n):
                if A[u2, x] == 1:
                    recompute_edges.add((min(u2, x), max(u2, x)))
            # Edges incident to v2
            for x in range(n):
                if A[v2, x] == 1:
                    recompute_edges.add((min(v2, x), max(v2, x)))
            # The new edge itself
            recompute_edges.add((min(u2, v2), max(u2, v2)))
            
            # Now, simulate the swap on a temporary adjacency matrix?
            # No, we can just adjust the neighbor sets logically.
            # But it's easier to just apply the swap to A, recompute books for these edges,
            # check the max, and if rejected, revert.
            
            # Apply swap
            A[u1, v1] = 0
            A[v1, u1] = 0
            A[u2, v2] = 1
            A[v2, u2] = 1
            
            # Recompute books for recompute_edges
            new_books_vals = {}
            max_new_book = 0
            # We need to know the max book over ALL edges.
            # Most edges are unaffected. Their book size is the same as in `books`.
            # So, new_max = max( max(books for unaffected edges), max(new_books_vals) )
            
            # To get max of unaffected edges, we can just take the old max,
            # but if the old max edge was affected, we need to be careful.
            # Actually, if the old max edge is in recompute_edges, its value changes.
            # If it's not, its value stays the same.
            
            # Let's just compute the max over the recompute_edges,
            # and compare with the old max.
            # But the old max might have been on an affected edge that decreased.
            # So we need the second highest? No.
            
            # Safe way:
            # 1. Store old books for recompute_edges.
            # 2. Compute new books for recompute_edges.
            # 3. The new global max is max( max(new_books for recompute_edges), max(books for non-recompute edges) )
            
            # We don't have a list of non-recompute edges handy without iterating.
            # However, n=40, so total edges ~ 780.
            # Iterating over all edges to find max is O(E) = 780 ops.
            # Doing this for 20000 iterations is 15.6 million ops. Very fast.
            
            # So, let's just recompute the max book size from scratch for the whole graph?
            # books = A @ A * A. np.max(books).
            # Matrix multiplication 40x40 is very fast.
            # 20000 * (40^3) = 128 million ops. Might be too slow in Python loop?
            # NumPy is C-optimized. 40^3 is small.
            # Let's try the "recompute all books" approach for simplicity and robustness.
            
            new_books = compute_books(A)
            new_score = int(np.max(new_books))
            
            delta = new_score - current_score
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                # Accept
                current_score = new_score
                books = new_books
                if current_score < best_score:
                    best_score = current_score
                    best_A = A.copy()
                
                # Update edge lists
                edges = get_edges(A)
                nonedges = get_nonedges(A)
            else:
                # Reject
                A[u1, v1] = 1
                A[v1, u1] = 1
                A[u2, v2] = 0
                A[v2, u2] = 0
            
            T = max(T * cool, T_min)
        
        return best_A, best_score

    best_A = None
    best_score = float('inf')
    
    # Multiple restarts
    num_restarts = 10
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        A = make_initial_graph()
        
        # Randomize slightly to break symmetry
        # Perform some random swaps
        for _ in range(100):
            edges = get_edges(A)
            nonedges = get_nonedges(A)
            if not edges or not nonedges: break
            u1, v1 = random.choice(edges)
            u2, v2 = random.choice(nonedges)
            A[u1, v1] = 0
            A[v1, u1] = 0
            A[u2, v2] = 1
            A[v2, u2] = 1
            
        A, score = local_search(A, max_iters=5000)
        
        if score < best_score:
            best_score = score
            best_A = A.copy()

    if best_A is None:
        # Fallback
        A = make_initial_graph()
        best_A = A

    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_A[i, j] == 1:
                edges.append([i, j])
    
    return edges
