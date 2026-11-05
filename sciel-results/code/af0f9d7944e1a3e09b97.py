import time
import random
import numpy as np

def construct():
    n = 40
    deadline = time.time() + 110.0  # Safety margin
    
    # --- C6 Detection using Bitmask BFS ---
    # We represent the graph as a list of bitmasks for neighbors.
    # To check if adding (u, v) creates a C6, we need to check if there is a path of length 5 between u and v.
    # Path length 5: u - x - y - z - w - v
    # This is equivalent to: v is in the set of vertices reachable from u in exactly 5 steps?
    # Actually, a C6 is a cycle. Adding (u,v) creates a C6 if there is a path of length 5 between u and v in the graph WITHOUT (u,v).
    # Let's precompute reachability sets for BFS.
    
    # For speed, we maintain adjacency as a list of integers (bitmasks).
    # adj[i] is a bitmask of neighbors of i.
    
    def is_c6_free(adj, u, v):
        """
        Check if adding edge (u, v) creates a C6.
        This is true if there is a path of length 5 from u to v in the graph G without edge (u,v).
        Since we are checking before adding, (u,v) is not in adj.
        
        Path: u -> x -> y -> z -> w -> v
        We can perform a BFS from u up to depth 4, and check if any node at depth 4 is adjacent to v.
        Or BFS from both ends?
        Depth 5 is small. 
        Let's do a BFS from u to depth 4.
        If any node at distance 4 from u is a neighbor of v, and the path is simple, we have a C6.
        Note: Simple path requirement is crucial. 
        However, for small n=40, we can just check (A^5)[u,v] > 0 using matrix multiplication?
        Matrix multiplication is O(n^3) ~ 64k ops. Very fast.
        But we need to update A^5? 
        Actually, let's stick to the matrix power method but optimize the search.
        The previous program used np.linalg.matrix_power(A, 5).
        Let's use a faster C6 check:
        A C6 exists between u and v if there is a path of length 5.
        We can compute the set of vertices reachable from u in exactly k steps.
        Let S_0 = {u}
        S_{k+1} = (Union of neighbors of S_k) \ S_{<k}  (for simple paths)
        If v is in S_5, then a C6 is formed.
        
        Let's implement this BFS check. It is O(n * E) worst case per check? No, BFS is O(V+E).
        V=40, E~100-200. Very fast.
        """
        # Bitmask BFS
        # visited[i] = set of vertices visited at depth i?
        # We just need to know if we can reach v in exactly 5 steps with a simple path.
        
        # Since n is small, we can just use standard BFS with path tracking or just check distances?
        # No, distance 5 is not enough, we need a specific path of length 5.
        # If dist(u,v) < 5, adding edge might create a shorter cycle, but we only care about C6.
        # If dist(u,v) == 5, adding edge creates a C6.
        # If dist(u,v) > 5, no C6.
        # If dist(u,v) < 5, adding edge creates a cycle of length dist+1. 
        # We need to avoid C6. So we must avoid dist(u,v) == 5.
        # What if there is a path of length 5 but the shortest path is 3? 
        # Then adding (u,v) creates a C4. The problem forbids C6. Does it forbid C4?
        # "containing no six-cycle". It does NOT forbid C4.
        # So, adding (u,v) creates a C6 IF AND ONLY IF there is a simple path of length 5 between u and v.
        
        # So we need to check existence of a simple path of length 5.
        # This is harder than just checking shortest path.
        # However, if we maintain the graph as C6-free, we know no C6 exists.
        # When we add (u,v), we create a C6 if there is a path of length 5.
        
        # Let's use the matrix power check but optimized.
        # A^5[u,v] counts walks of length 5.
        # If A^5[u,v] > 0, there is a walk. Does it imply a simple path?
        # Not necessarily. But for sparse graphs, it's a good proxy.
        # Actually, the seed program used this and got 1.0 (105 edges).
        # The issue with the seed was the search strategy (greedy + random swap).
        
        # Let's use a more robust check:
        # Check if v is reachable from u in exactly 5 steps via a simple path.
        # We can do a DFS with pruning? Or BFS with state (node, depth, visited_mask)?
        # State space: 40 * 5 * 2^40 is too big.
        
        # Alternative: 
        # We know the graph is C6-free.
        # If we add (u,v), we create a C6 iff there is a path of length 5.
        # Let's compute the number of paths of length 5 using matrix multiplication on the fly?
        # For n=40, A^5 is fast.
        
        # Let's try a different approach for the check:
        # Precompute A2, A3? No, A changes.
        
        # Let's stick to the matrix power check but make the SEARCH better.
        # The seed's search was weak.
        
    # --- Strategy: Simulated Annealing with Matrix Power Check ---
    
    # 1. Initialize with a good structure.
    # Polarity graph of projective plane of order 4 has 21 vertices, 84 edges.
    # Blow up each vertex to 2 vertices? 
    # If we replace each vertex v with 2 vertices v0, v1, and connect v0-v1, and connect v0, v1 to all neighbors of v...
    # This creates K2s.
    # Let's try a random regular graph or a known extremal graph.
    # Best known ex(40, C6) is 105.
    # Let's start with a random graph with p=0.25 (approx 100 edges) and repair?
    # Or start with a grid-like structure?
    
    # Let's use a "Randomized Greedy" initialization with a specific order, then Simulated Annealing.
    
    # To make the C6 check faster and more accurate for simple paths, 
    # we can use the following fact:
    # In a C6-free graph, the number of paths of length 5 is 0.
    # If we add (u,v), we create a C6 if the number of paths of length 5 between u and v is > 0.
    # We can compute this by:
    # count = 0
    # for x in neighbors(u):
    #   for y in neighbors(x):
    #     if y == u: continue
    #     for z in neighbors(y):
    #       if z == x or z == u: continue
    #       for w in neighbors(z):
    #         if w == y or w == x or w == u: continue
    #         if w == v: count += 1
    # This is O(deg(u)*deg(x)*deg(y)*deg(z)) which is too slow.
    
    # Let's use the matrix power check: 
    # P = A^5
    # If P[u,v] > 0, reject.
    # This allows walks that might not be simple paths.
    # However, if there is a walk of length 5, does it imply a C6?
    # A walk of length 5 u-x-y-z-w-v.
    # If it's not simple, it contains a cycle of length < 6.
    # If the graph is already C6-free, any new C6 must be formed by the new edge + a path of length 5.
    # If the path of length 5 is not simple, it contains a smaller cycle.
    # But we are allowed to have smaller cycles!
    # So, if there is a non-simple walk of length 5, it doesn't necessarily mean there is a simple path of length 5.
    # Example: u-x-u-x-y-v is a walk of length 5? No, length 5 has 5 edges.
    # u-x-u-x-y is 4 edges.
    # u-x-u-x-y-z is 5 edges. Path ends at z. If z=v, then u-x-u-x-y-v.
    # This walk uses edge (u,x) twice.
    # Does this create a C6? No. It creates a C4 (u-x-u-x is not a simple cycle).
    # So the matrix power check is a SUFFICIENT condition (if >0, likely C6, but might be false positive if only non-simple walks exist).
    # It is NOT a necessary condition? No, if there is a simple path, there is a walk. So if simple path exists, P[u,v] > 0.
    # So P[u,v] == 0 implies NO simple path of length 5. Safe to add.
    # P[u,v] > 0 implies MIGHT be a simple path. We should verify?
    # For speed, we can assume P[u,v] > 0 means "reject". This might miss some valid additions (false negatives for safety), leading to a suboptimal graph, but it's safe (no C6s).
    # To maximize edges, we want to minimize false rejections.
    # Let's use the matrix power check.
    
    def compute_A5(mat):
        # mat is numpy array
        A2 = mat @ mat
        A3 = A2 @ mat
        A4 = A3 @ mat
        A5 = A4 @ mat
        return A5

    # Initialize
    A = np.zeros((n, n), dtype=np.int64)
    
    # Start with a random graph with edge prob ~ 0.25
    random.seed(42)
    for u in range(n):
        for v in range(u + 1, n):
            if random.random() < 0.25:
                A[u, v] = 1
                A[v, u] = 1
                
    # Repair: Remove edges that create C6?
    # Actually, let's start with empty and add greedily?
    # No, let's start with a known good structure if possible.
    # Let's try a random regular graph of degree 5 (100 edges).
    # Generate a random 5-regular graph.
    
    def generate_random_regular_graph(n, k, seed=1):
        random.seed(seed)
        adj = np.zeros((n, n), dtype=np.int64)
        # Configuration model
        half_edges = []
        for i in range(n):
            for _ in range(k):
                half_edges.append(i)
        random.shuffle(half_edges)
        
        # Pair them up
        for i in range(0, len(half_edges), 2):
            u = half_edges[i]
            v = half_edges[i+1]
            if u != v and adj[u, v] == 0:
                adj[u, v] = 1
                adj[v, u] = 1
            else:
                # Self loop or multi-edge, retry?
                # For simplicity, just accept and fix later or restart
                pass
                
        # Check degrees
        degs = np.sum(adj, axis=1)
        if not np.all(degs == k):
            # Fallback to random
            for u in range(n):
                for v in range(u+1, n):
                    if random.random() < k/n:
                        adj[u, v] = 1
                        adj[v, u] = 1
        return adj

    A = generate_random_regular_graph(n, 5)
    A5 = compute_A5(A)
    
    current_edges = np.sum(A) // 2
    best_edges = current_edges
    best_A = A.copy()
    
    # Simulated Annealing
    temp = 100.0
    cooling = 0.999
    
    start_time = time.time()
    
    while time.time() < deadline:
        # Pick a random move
        # 1. Add an edge
        # 2. Remove an edge
        # 3. Swap an edge (remove one, add another)
        
        move_type = random.random()
        
        if move_type < 0.3:
            # Add edge
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v or A[u, v] == 1:
                continue
            
            if A5[u, v] == 0:
                # Safe to add
                A[u, v] = 1
                A[v, u] = 1
                A5 = compute_A5(A)
                current_edges += 1
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_A = A.copy()
            else:
                # Not safe. Accept with probability exp(-1/temp) to escape?
                # No, adding an unsafe edge creates a C6, which is INVALID.
                # We cannot add unsafe edges.
                # So we must find a different way to escape.
                # Maybe remove an edge first?
                pass
                
        elif move_type < 0.6:
            # Remove edge
            if current_edges == 0: continue
            # Pick random edge
            edges_list = np.argwhere(A > 0)
            if len(edges_list) == 0: continue
            idx = random.randint(0, len(edges_list) - 1)
            u, v = edges_list[idx]
            if u > v: u, v = v, u
            
            A[u, v] = 0
            A[v, u] = 0
            A5 = compute_A5(A)
            current_edges -= 1
            
        else:
            # Swap: Remove (u,v), Add (x,y)
            if current_edges == 0: continue
            edges_list = np.argwhere(A > 0)
            if len(edges_list) == 0: continue
            idx = random.randint(0, len(edges_list) - 1)
            u, v = edges_list[idx]
            if u > v: u, v = v, u
            
            # Save state
            A_old = A.copy()
            A5_old = A5.copy()
            
            # Remove
            A[u, v] = 0
            A[v, u] = 0
            A5 = compute_A5(A)
            current_edges -= 1
            
            # Try to add random edge
            improved = False
            for _ in range(10):
                x = random.randint(0, n - 1)
                y = random.randint(0, n - 1)
                if x == y or A[x, y] == 1:
                    continue
                if A5[x, y] == 0:
                    A[x, y] = 1
                    A[y, x] = 1
                    A5 = compute_A5(A)
                    current_edges += 1
                    improved = True
                    break
            
            if not improved:
                # Revert
                A = A_old
                A5 = A5_old
                current_edges += 1
            else:
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_A = A.copy()
        
        # Cooling
        temp *= cooling
        if temp < 0.1:
            temp = 100.0 # Restart temperature
            
    # Final greedy pass
    # Try to add any remaining safe edges
    changed = True
    while changed:
        changed = False
        for u in range(n):
            for v in range(u + 1, n):
                if A[u, v] == 0:
                    if A5[u, v] == 0:
                        A[u, v] = 1
                        A[v, u] = 1
                        A5 = compute_A5(A)
                        current_edges += 1
                        changed = True
                        if current_edges > best_edges:
                            best_edges = current_edges
                            best_A = A.copy()
                        
    # Return best
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_A[u, v] == 1:
                edges.append([u, v])
    return edges
