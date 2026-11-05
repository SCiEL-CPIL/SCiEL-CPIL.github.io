import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n = 40
    m = 401
    
    # 1. Generate a highly symmetric base graph using a circulant construction.
    # We want a graph with 400 edges (20-regular) plus 1 extra edge.
    # A circulant graph C(n, S) where S is a set of offsets.
    # For n=40, if we pick offsets {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, 
    # each vertex i is connected to i±1 ... i±10.
    # This creates a 20-regular graph with 40*20/2 = 400 edges.
    # This graph is very regular and symmetric, likely having uniform book sizes.
    
    base_offsets = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    base_edges = set()
    for i in range(n):
        for d in base_offsets:
            j = (i + d) % n
            if i < j:
                base_edges.add((i, j))
            elif i > j:
                base_edges.add((j, i))
    
    # We have 400 edges. We need 1 more.
    # We will pick the "best" extra edge to add, or just add one and let local search fix it.
    # Let's pick a random non-edge to start, or the one that minimizes immediate book increase.
    # Actually, let's just pick one and let the local search handle the optimization.
    # To be safe, let's try a few candidates for the 401st edge.
    
    all_possible_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    existing = base_edges
    non_edges = [e for e in all_possible_edges if e not in existing]
    
    best_edges = None
    best_score = float('inf')
    
    # Try adding a few different non-edges as the 401st edge
    random.seed(42)
    candidate_extra_edges = random.sample(non_edges, min(20, len(non_edges)))
    
    for extra_e in candidate_extra_edges:
        if time.time() > deadline - 20: # Leave time for local search
            break
            
        current_edges = list(base_edges)
        current_edges.append(extra_e)
        
        # Build adjacency and book sizes
        adj = [set() for _ in range(n)]
        for u, v in current_edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Calculate book sizes
        book_sizes = {}
        max_book = 0
        for u, v in current_edges:
            common = len(adj[u] & adj[v])
            book_sizes[(u, v)] = common
            if common > max_book:
                max_book = common
        
        # Local Search: 1-flip moves
        # Replace edge (u, v) with (u, w) or (v, w) where w is not currently connected.
        # This changes the degree of v (decreases) and w (increases).
        # To keep the graph near-regular, we might prefer moves that preserve degrees?
        # But 1-flip is simpler. Let's try 1-flip first.
        # If 1-flip is too disruptive, we can try 2-flips (swap endpoints of two edges).
        
        # For speed, we will do a limited number of local search steps.
        improved = True
        max_iterations = 5000 # Cap iterations per restart to avoid getting stuck
        iter_count = 0
        
        while improved and iter_count < max_iterations:
            if time.time() > deadline - 5:
                break
                
            improved = False
            iter_count += 1
            
            # Find the edge with the maximum book size
            worst_edge = None
            worst_book = -1
            for e, b in book_sizes.items():
                if b > worst_book:
                    worst_book = b
                    worst_edge = e
            
            if worst_book == 0:
                break
                
            u, v = worst_edge
            
            # Try to flip one endpoint
            # Candidate new neighbors for u: vertices not in adj[u]
            # Candidate new neighbors for v: vertices not in adj[v]
            
            # To speed up, we can just try a few random candidates or scan all.
            # Scanning all 40 vertices is fast.
            
            best_delta = 0
            best_move = None
            
            # Try moving v to w
            for w in range(n):
                if w == u: continue
                if w in adj[v]: continue # Already connected
                
                # Proposed move: remove (u, v), add (u, w)
                # Check if (u, w) is already an edge? No, w not in adj[u] is not checked yet.
                if w in adj[u]: continue # (u, w) would be a duplicate or self loop? No, w!=u. If w in adj[u], (u,w) exists.
                
                # Calculate change in book sizes
                # Removing (u, v):
                # - book_sizes[(u, v)] becomes 0 (edge removed)
                # - For each x in adj[u] \ {v}: if x in adj[v], book_sizes[(u, x)] -= 1, book_sizes[(v, x)] -= 1
                # - For each x in adj[v] \ {u}: if x in adj[u], book_sizes[(u, x)] -= 1, book_sizes[(v, x)] -= 1
                #   (Note: x in adj[u] & adj[v] are the common neighbors. The triangle (u,v,x) is broken.
                #    Edges (u,x) and (v,x) lose 1 triangle.)
                
                # Adding (u, w):
                # - New edge (u, w). Its book size = |adj[u] ∩ adj[w]| (after removal of v from adj[u] and adding w? No, adj sets are updated).
                #   Let's compute new common neighbors: (adj[u] \ {v}) ∩ adj[w]
                # - For each x in (adj[u] \ {v}) ∩ adj[w]:
                #   - book_sizes[(u, x)] += 1
                #   - book_sizes[(w, x)] += 1 (if (w,x) is an edge. Since x in adj[w] and x in adj[u], (w,x) exists if x!=u. x!=u because u not in adj[u].)
                
                # This calculation is O(degree^2) which is ~400. 40 candidates * 400 = 16000 ops per step.
                # 5000 steps * 16000 = 80M ops. Might be too slow in Python.
                
                # Optimization: 
                # We only care if the max book size decreases.
                # Current max is worst_book.
                # If we remove (u,v) which has book size worst_book, the max might drop if no other edge has that size.
                # But adding (u,w) might create a new edge with high book size.
                
                # Let's estimate the new book size of the new edge (u, w).
                # common_uw = len((adj[u] - {v}) & adj[w])
                # If common_uw >= worst_book, this move is likely bad (unless it reduces other high books significantly).
                # Let's just compute the exact delta for the few most promising w?
                # Or just do a simple heuristic: try to reduce the degree of high-book edges?
                
                # Given the complexity, let's try a simpler local search:
                # Swap endpoints of two edges (2-flip).
                # Pick another edge (x, y).
                # Try to replace (u, v) and (x, y) with (u, x) and (v, y) or (u, y) and (v, x).
                # This preserves degrees.
                
                pass # Skipping complex 1-flip calculation for now to ensure speed.
            
            # Let's try a different, faster local search:
            # Identify edges with book size > target (e.g., current max - 1).
            # Try to break triangles by removing an edge from a high-book edge and adding a random non-edge.
            # This is basically what we were doing, but let's make it faster by only evaluating a subset of moves.
            
            # Actually, the initial circulant graph is likely very good.
            # Let's just evaluate the initial circulant + random edge and see the score.
            # If it's already <= 8, we might not need heavy local search.
            
            break # Break out of local search loop for now to prioritize speed and get a baseline.
        
        if max_book < best_score:
            best_score = max_book
            best_edges = current_edges
            
            # If we found a very good score, we can stop early?
            # The theoretical bound is ~6.67. So 7 is the target.
            if best_score <= 7:
                # Try to verify if we can get lower? Unlikely.
                # Let's do a quick check if 7 is achievable.
                pass

    # If local search was skipped, we might have a suboptimal result if the random extra edge was bad.
    # Let's do a quick pass: try all non-edges for the 401st slot and pick the best one.
    # There are 780 - 400 = 380 non-edges.
    # Calculating book size for each is O(m * n) ~ 400 * 40 = 16000.
    # 380 * 16000 = 6M ops. Very fast.
    
    final_best_edges = None
    final_best_score = float('inf')
    
    for extra_e in non_edges:
        if time.time() > deadline - 10:
            break
            
        current_edges = list(base_edges)
        current_edges.append(extra_e)
        
        adj = [set() for _ in range(n)]
        for u, v in current_edges:
            adj[u].add(v)
            adj[v].add(u)
        
        max_book = 0
        for u, v in current_edges:
            common = len(adj[u] & adj[v])
            if common > max_book:
                max_book = common
        
        if max_book < final_best_score:
            final_best_score = max_book
            final_best_edges = current_edges
            
            # If we hit the theoretical lower bound, we can stop?
            # Bound is 40/6 = 6.66, so 7 is the integer minimum.
            if final_best_score == 7:
                break

    if final_best_edges:
        return [[u, v] for u, v in final_best_edges]
    elif best_edges:
        return [[u, v] for u, v in best_edges]
    else:
        # Fallback
        return [[0, 1]] + [[i, j] for i in range(n) for j in range(i+1, n) if i < 20 and j >= 20]
