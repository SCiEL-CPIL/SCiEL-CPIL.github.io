import time
import random
import numpy as np

def construct():
    N = 40
    M = 401
    deadline = time.time() + 110
    
    # Initialize a near-regular graph
    # Target degree ~ 10
    def init_regular():
        A = np.zeros((N, N), dtype=int)
        degrees = np.zeros(N, dtype=int)
        all_edges = [(i, j) for i in range(N) for j in range(i+1, N)]
        random.shuffle(all_edges)
        
        # Greedy: pick edges that minimize the max degree difference
        # We want to fill up to M edges.
        # Simple heuristic: sort by current degrees sum, pick lowest.
        # But re-sorting is expensive. 
        # Alternative: just pick random, but bias towards low degree vertices.
        
        # Let's try a specific construction:
        # 1. Create a 10-regular graph if possible (N=40, 10-regular is possible since 40*10 is even).
        # 2. Add 1 extra edge.
        # 10-regular graph:
        # Connect i to (i+1)%40, (i+2)%40, ..., (i+5)%40? No, that's not regular in a simple way.
        # Standard 10-regular: 
        # For each i, connect to i+1, i+2, i+3, i+4, i+5, i-1, i-2, i-3, i-4, i-5 mod 40?
        # This creates a circulant graph.
        # Edges: (i, (i+k)%N) for k in 1..5.
        # Total edges: 40 * 5 = 200. This is 10-regular.
        # We need 401 edges.
        # So start with 200 edges (10-regular).
        # Add 201 more edges.
        # To keep it near-regular, we should add edges between vertices with lowest degrees.
        
        A = np.zeros((N, N), dtype=int)
        for i in range(N):
            for k in range(1, 6):
                j = (i + k) % N
                if i != j:
                    A[i, j] = 1
                    A[j, i] = 1
        
        # Current edges: 200. Degrees: 10 each.
        # Need 201 more edges.
        degrees = A.sum(axis=1)
        
        # Generate all non-edges
        non_edges = [(i, j) for i in range(N) for j in range(i+1, N) if A[i, j] == 0]
        random.shuffle(non_edges)
        
        # Add edges greedily to balance degrees
        for u, v in non_edges:
            if M - np.count_nonzero(A) == 0:
                break
            # Check if adding this edge keeps degrees balanced
            # We just add it if it doesn't exceed some limit, or just add random ones?
            # Since we start at 10, and need to add ~5 per vertex, 
            # random selection from non-edges will naturally balance if we avoid high-degree vertices.
            # Let's pick the non-edge with minimum degrees[u] + degrees[v]
            # This is O(E_non * N) if we do it naively, but we can just sort once?
            # No, degrees change.
            
            # Optimization: Just pick random non-edges, but if a vertex has degree > 12, skip?
            # Or simpler: just pick the best available.
            # Since 201 edges is small, we can afford a loop.
            
            # Find best u,v
            best_u, best_v = -1, -1
            best_score = float('inf')
            # Sample a subset to speed up? No, 201 * 780 is small.
            # But we already shuffled non_edges.
            # Let's just iterate through the shuffled list and pick the first one that satisfies a condition?
            # No, let's just pick the global min sum for each of the 201 additions.
            
            # To avoid O(201 * 780), we can just add random ones, but this might create high degrees.
            # Let's try adding random ones first, then polish.
            pass 

        # Let's stick to the random shuffle of non-edges and add them.
        # The initial 10-regular structure is good.
        # Adding 201 random edges will result in degrees ~ 10 + 201/20 = 15? No, 201/40 = 5.
        # So degrees will be around 15.
        # This is fine.
        
        # Actually, let's just use the random graph initialization but with a bias?
        # The current program uses random graph.
        # The inspiration says "sequential degree-constrained".
        
        # Let's implement the "pick min degree sum" properly for the 201 additions.
        edges_added = 0
        while edges_added < 201:
            # Find min degree sum pair among non-edges
            min_sum = float('inf')
            best_pair = None
            # Iterate all non-edges
            for u, v in non_edges:
                if A[u, v] == 1: continue # might have been added
                s = degrees[u] + degrees[v]
                if s < min_sum:
                    min_sum = s
                    best_pair = (u, v)
            if best_pair is None:
                break
            u, v = best_pair
            A[u, v] = 1
            A[v, u] = 1
            degrees[u] += 1
            degrees[v] += 1
            edges_added += 1
            
        return A

    A = init_regular()
    
    def compute_max_book(A):
        # A is NxN int matrix
        common = A @ A.T
        # Mask out non-edges
        # books[i,j] = common[i,j] if A[i,j]==1 else 0
        # We want max of books[i,j]
        # Note: diagonal is degree, we should ignore it or mask it.
        # A[i,i] is 0, so common[i,i] is degree. 
        # But we only care about edges i!=j.
        # Since A[i,i]=0, the product A@A.T diagonal is degree.
        # We can just set diagonal to 0 before max?
        # Or use a mask.
        mask = A.copy()
        np.fill_diagonal(mask, 0)
        books = common * mask
        return int(books.max())
    
    best_A = A.copy()
    best_max_book = compute_max_book(best_A)
    
    # Local Search
    # Use vectorized evaluation of swaps?
    # For each step:
    # 1. Find edge (u,v) with max book.
    # 2. Generate K random non-edges (x,y).
    # 3. For each (x,y), create A_swap, compute max book.
    # 4. If any improvement, accept the best one.
    
    edges_list = [(i, j) for i in range(N) for j in range(i+1, N) if A[i, j] == 1]
    non_edges_list = [(i, j) for i in range(N) for j in range(i+1, N) if A[i, j] == 0]
    
    # We need to update edges_list and non_edges_list as we swap.
    # This is O(N^2) to rebuild.
    # Given N=40, rebuilding is fast.
    
    while time.time() < deadline:
        # Find max edge
        common = A @ A.T
        mask = A.copy()
        np.fill_diagonal(mask, 0)
        books = common * mask
        max_book = int(books.max())
        
        if max_book < best_max_book:
            best_max_book = max_book
            best_A = A.copy()
            
        if max_book <= 7:
            break
            
        # Find the edge with max book
        # There might be multiple. Pick one.
        max_edge = None
        max_val = -1
        for i in range(N):
            for j in range(i+1, N):
                if A[i, j] == 1:
                    if books[i, j] > max_val:
                        max_val = books[i, j]
                        max_edge = (i, j)
        
        u, v = max_edge
        improved = False
        
        # Try 100 random swaps
        # We can batch these?
        # Creating 100 copies of A and computing A@A.T for each is 100 * 40^3 ~ 640k ops.
        # This is very fast in NumPy.
        
        candidates = []
        # Sample non-edges
        # To avoid picking (u,v) itself (which is an edge now, so not in non_edges_list if we update it)
        # We should update non_edges_list periodically or just check A[x,y]==0
        
        for _ in range(100):
            x = random.randint(0, N-1)
            y = random.randint(0, N-1)
            if x == y: continue
            if x > y: x, y = y, x
            if x == u and y == v: continue
            if A[x, y] == 1: continue
            
            # Simulate swap
            A[x, y] = 1
            A[y, x] = 1
            A[u, v] = 0
            A[v, u] = 0
            
            new_common = A @ A.T
            new_mask = A.copy()
            np.fill_diagonal(new_mask, 0)
            new_books = new_common * new_mask
            new_max = int(new_books.max())
            
            if new_max < max_val:
                candidates.append((new_max, x, y))
                # Revert
                A[x, y] = 0
                A[y, x] = 0
                A[u, v] = 1
                A[v, u] = 1
                # Don't break immediately, might find even better?
                # But if we find one < max_val, it's an improvement.
                # We can break to save time, or keep looking for the best improvement.
                # Let's keep looking for the best among the 100.
            else:
                # Revert
                A[x, y] = 0
                A[y, x] = 0
                A[u, v] = 1
                A[v, u] = 1
        
        if candidates:
            # Pick the best candidate
            best_candidate = min(candidates, key=lambda t: t[0])
            new_max, x, y = best_candidate
            A[x, y] = 1
            A[y, x] = 1
            A[u, v] = 0
            A[v, u] = 0
            improved = True
        else:
            # No improvement found in 100 tries
            # Try a random restart? Or just continue?
            # If we are stuck, maybe randomize a bit?
            # Or just break and restart?
            # The current program breaks after 20 stuck steps.
            # Let's count stuck steps.
            pass
            
    # Convert best_A to edge list
    edges = []
    for i in range(N):
        for j in range(i+1, N):
            if best_A[i, j] == 1:
                edges.append([i, j])
    return edges
