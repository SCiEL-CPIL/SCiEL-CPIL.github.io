import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n = 40
    target_edges = 401
    
    # Helper to compute book sizes for all edges
    def compute_books(A):
        # A is n x n adjacency matrix
        # Book size for edge (i, j) is sum_k A[i,k] * A[j,k]
        # This is (A @ A)[i, j]
        A2 = A @ A
        # We only care about edges where A[i,j] == 1
        # Create a matrix of book sizes, 0 for non-edges
        books = A2 * A
        # Diagonal is 0 in books because A[i,i]=0
        return books

    # Helper to get max book size and the edge with it
    def get_max_book(books):
        max_val = np.max(books)
        # Find one edge with this max value
        # We don't strictly need the edge, just the value
        return int(max_val)

    # Generate a random 20-regular graph
    def generate_random_regular_graph(n, d):
        # Use K_20,20 as start
        A = np.zeros((n, n), dtype=int)
        for i in range(20):
            for j in range(20, 40):
                A[i, j] = 1
                A[j, i] = 1
        
        # Randomize with swaps
        # Pre-compute list of edges for faster sampling? 
        # Or just use random vertices and check.
        # Since n is small, checking is fast.
        
        for _ in range(2000):
            # Pick 4 distinct vertices
            # We need to swap (a,b) and (c,d) to (a,c) and (b,d)
            # Conditions: A[a,b]=1, A[c,d]=1, A[a,c]=0, A[b,d]=0
            # And a,b,c,d distinct.
            
            # Try random pairs
            for _ in range(10):
                a, b = np.random.choice(n, 2, replace=False)
                c, d = np.random.choice(n, 2, replace=False)
                
                # Ensure distinct
                if len(set([a, b, c, d])) < 4:
                    continue
                    
                if A[a, b] == 1 and A[c, d] == 1 and A[a, c] == 0 and A[b, d] == 0:
                    A[a, b] = 0; A[b, a] = 0
                    A[c, d] = 0; A[d, c] = 0
                    A[a, c] = 1; A[c, a] = 1
                    A[b, d] = 1; A[d, b] = 1
                    break
        return A

    def add_best_edge(A):
        # Find non-edge (u,v) with min common neighbors
        A2 = A @ A
        # Mask non-edges and diagonal
        mask = (A == 0) & (np.eye(n, dtype=int) == 0)
        A2_masked = np.where(mask, A2, np.inf)
        min_val = np.min(A2_masked)
        u, v = np.unravel_index(np.argmin(A2_masked), A2_masked.shape)
        A[u, v] = 1
        A[v, u] = 1
        return A, int(min_val)

    best_A = None
    best_score = float('inf')
    
    # Try a few random starts
    for i in range(10):
        if time.time() > deadline:
            break
            
        A = generate_random_regular_graph(n, 20)
        A, _ = add_best_edge(A)
        
        # Initial book computation
        books = compute_books(A)
        current_score = get_max_book(books)
        
        if current_score < best_score:
            best_score = current_score
            best_A = A.copy()
            best_books = books.copy()
        else:
            # Keep the current one for local search if it's close?
            # Just use the new one for local search
            pass
            
        current_A = A.copy()
        current_books = books.copy()
        current_score = get_max_book(current_books)
        
        # Local Search
        # We will perform edge swaps.
        # To update books efficiently:
        # When swapping (u1, v1) out and (u2, v2) in:
        # 1. Remove (u1, v1).
        #    This affects book sizes of edges incident to u1 or v1.
        #    Specifically, for any edge (x, y) incident to u1 (i.e., y=u1), 
        #    book(x, u1) changes because u1's neighbor set changed (lost v1).
        #    Wait, book(x, y) = common(x, y).
        #    If we remove edge (u1, v1):
        #    - v1 is no longer a neighbor of u1.
        #    - u1 is no longer a neighbor of v1.
        #    - For any edge (x, u1), common(x, u1) decreases by 1 if x was a neighbor of v1?
        #      No. common(x, u1) = sum_k A[x,k] A[u1,k].
        #      The term for k=v1 was A[x,v1] * A[u1,v1]. Since A[u1,v1] was 1, it was A[x,v1].
        #      Now A[u1,v1] is 0. So common(x, u1) decreases by A[x,v1].
        #      So for all x such that A[x,u1]==1, we need to update book(x, u1).
        #      The change is -A[x,v1].
        #    - Similarly for edges incident to v1.
        #    - Also, the edge (u1, v1) itself is removed, so its book size is no longer relevant.
        #      But we are tracking max book size of *existing* edges.
        #      So we just update the books of remaining edges.
        
        # 2. Add (u2, v2).
        #    - u2 is now a neighbor of v2.
        #    - v2 is now a neighbor of u2.
        #    - For any edge (x, u2), common(x, u2) increases by A[x,v2].
        #    - For any edge (x, v2), common(x, v2) increases by A[x,u2].
        #    - The new edge (u2, v2) has book size common(u2, v2).
        
        # Let's implement a function to update books given a swap.
        def update_books_after_swap(A, books, u1, v1, u2, v2):
            # Remove (u1, v1)
            # For all neighbors x of u1 (x != v1):
            #   books[x, u1] -= A[x, v1]
            # For all neighbors x of v1 (x != u1):
            #   books[x, v1] -= A[x, u1]
            # Note: books is symmetric.
            
            # Get neighbors of u1
            neigh_u1 = np.where(A[u1, :] == 1)[0]
            for x in neigh_u1:
                if x == v1: continue
                # x is neighbor of u1. Edge (x, u1) exists.
                # common(x, u1) changes by -A[x, v1]
                books[x, u1] -= A[x, v1]
                books[u1, x] -= A[x, v1]
                
            # Get neighbors of v1
            neigh_v1 = np.where(A[v1, :] == 1)[0]
            for x in neigh_v1:
                if x == u1: continue
                # x is neighbor of v1. Edge (x, v1) exists.
                # common(x, v1) changes by -A[x, u1]
                books[x, v1] -= A[x, u1]
                books[v1, x] -= A[x, u1]
                
            # The edge (u1, v1) is removed. 
            # We don't care about its book size anymore, but we should zero it out 
            # or leave it? 
            # get_max_book uses A to mask. 
            # If we don't update A yet, we can't use A to mask correctly in intermediate steps?
            # We should update A first? 
            # No, we need old A to compute changes? 
            # The changes depend on the state BEFORE the swap for the "remove" part?
            # Actually, the change in common(x, u1) when removing (u1, v1) 
            # is -A[x, v1] (where A is the state BEFORE removal).
            # So we use the current A (which still has the edge) to calculate the delta.
            # Then we update A.
            
            A[u1, v1] = 0
            A[v1, u1] = 0
            
            # Add (u2, v2)
            # For all neighbors x of u2 (x != v2):
            #   books[x, u2] += A[x, v2] (A is state AFTER removal, BEFORE addition? 
            #   Wait. When adding (u2, v2), the new commonality is with the current neighbors.
            #   The term added to common(x, u2) is A[x, v2] * A[u2, v2] (which becomes 1).
            #   So we add A[x, v2] from the state BEFORE addition.
            
            neigh_u2 = np.where(A[u2, :] == 1)[0]
            for x in neigh_u2:
                if x == v2: continue
                books[x, u2] += A[x, v2]
                books[u2, x] += A[x, v2]
                
            neigh_v2 = np.where(A[v2, :] == 1)[0]
            for x in neigh_v2:
                if x == u2: continue
                books[x, v2] += A[x, u2]
                books[v2, x] += A[x, u2]
                
            # New edge (u2, v2)
            # common(u2, v2) = sum_k A[u2,k] A[v2,k]
            # We can compute this directly
            new_book = np.sum(A[u2, :] * A[v2, :])
            books[u2, v2] = new_book
            books[v2, u2] = new_book
            
            A[u2, v2] = 1
            A[v2, u2] = 1
            
            return A, books

        T = 1.0
        for step in range(2000):
            if time.time() > deadline:
                break
                
            # Pick random edge to remove
            # Get list of edges
            edges = np.argwhere(current_A == 1)
            edges = edges[edges[:, 0] < edges[:, 1]]
            if len(edges) == 0: break
            e_rem = edges[np.random.randint(len(edges))]
            u1, v1 = e_rem
            
            # Pick random non-edge to add
            # To pick a non-edge, we can pick random vertices and check
            # Or use a list of non-edges. 
            # Since graph is dense (401/780), non-edges are ~379.
            # Picking random u, v and checking is fast.
            found = False
            for _ in range(10):
                u2, v2 = np.random.choice(n, 2, replace=False)
                if current_A[u2, v2] == 0:
                    found = True
                    break
            
            if not found:
                continue
                
            # Apply move
            new_A, new_books = update_books_after_swap(current_A.copy(), current_books.copy(), u1, v1, u2, v2)
            
            new_score = get_max_book(new_books)
            
            delta = new_score - current_score
            if delta < 0 or np.random.random() < np.exp(-delta / T):
                current_A = new_A
                current_books = new_books
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_A = current_A.copy()
            # else: revert (do nothing, as we used copies)
            
            T *= 0.995

    # Convert best_A to list of edges
    if best_A is None:
        # Fallback
        return [[a, b] for a in range(20) for b in range(20, 40)] + [[0, 1]]
        
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_A[i, j] == 1:
                edges.append([i, j])
                
    return edges
