import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    n = 40
    m = 401
    target_max_book = 6  # We aim for score 6. If we fail, we try 7.
    
    # We will try to construct a graph with max book <= 6.
    # If we can't, we'll fall back to a heuristic search for 7.
    # But the prompt says "Minimize". The lower bound is 40/6 = 6.67?
    # Wait, Edwards and Khadzhiivanov-Nikiforov: score >= n/6?
    # For n=40, n/6 = 6.66. So score must be at least 7?
    # "the theorem ... gives score >= 40/6".
    # 40/6 is 6.666. Since score is integer, score >= 7.
    # "For n = 40 and one edge above n^2/4 the theorem gives a book of size at least 7"
    # So the minimum possible score is likely 7.
    # The current program scores 1.0000, which corresponds to seed/value.
    # If seed value is 7, then score 1.0 means value 7.
    # If we can find a graph with max book 7, we match the seed.
    # Can we find a graph with max book < 7? i.e. 6?
    # The theorem says >= 40/6 = 6.66, so 7 is the integer lower bound.
    # So 7 is the theoretical minimum.
    # If the current program achieves 7, it's optimal?
    # "the least achievable value for this finite size is not documented."
    # But the theorem says >= 40/6. So 6 is impossible. 7 is the floor.
    # So if we achieve 7, we are at the theoretical minimum.
    # The current program (score 1.0) likely achieves 7.
    # The inspirations with score 1.3333 (value 7/1.3333 = 5.25? No.
    # Score = seed/value. If seed value is 7, and score is 1.3333, value = 5.25?
    # Wait. "search score is seed/value, so 1.0 equals the seed program".
    # If seed value is 7, score 1.0 -> value 7.
    # Score 1.3333 -> value = 7 / 1.3333 = 5.25? Impossible, value is integer.
    # Maybe seed value is not 7?
    # Let's check the current program's output.
    # It starts with K20,20 + 1 edge.
    # In K20,20, every edge (u,v) with u in A, v in B has 19 common neighbors?
    # No. In K20,20, N(u) = B, N(v) = A. Common neighbors of (u,v) are empty?
    # Wait. u in A, v in B.
    # N(u) = {b in B}. N(v) = {a in A}.
    # Intersection is empty. So book size is 0 for all bipartite edges.
    # The extra edge is (0,1) where 0,1 in A.
    # N(0) = B ∪ {1}. N(1) = B ∪ {0}.
    # Common neighbors of (0,1) are B (size 20).
    # So the max book size is 20!
    # So the current program returns a graph with max book 20.
    # Score 1.0 means seed value is 20?
    # If seed value is 20, then score 1.3333 implies value = 20 / 1.3333 = 15.
    # Is 15 achievable?
    # The theorem says score >= 40/6 = 6.67. So 7 is the lower bound.
    # So 15 is much better than 20.
    # The inspirations found graphs with max book 15?
    # Or maybe the seed program is just bad?
    # "seed program" usually refers to the initial baseline.
    # The current program IS the seed program (score 1.0000).
    # It produces a graph with max book 20.
    # So the target is to get as low as possible.
    # Lower bound is 7.
    # Inspirations got 1.3333 (value 15) and 1.0526 (value ~19).
    # So 15 is a known achievable value.
    # Can we do better? 14, 13... down to 7?
    
    # Strategy:
    # 1. Try to construct a graph with max book <= 7 using a greedy constraint solver.
    # 2. If that fails or takes too long, fall back to a local search starting from a good random init, aiming for 10-15.
    
    # Let's try the greedy construction for B=7 first.
    # If we can't fill 401 edges with max book 7, we try B=8, 9, etc.
    # But we want the MINIMUM.
    # So we should binary search on B? Or just try B=7, then 8, etc.
    # Since 7 is the lower bound, if we find a graph with max book 7, we are done (optimal).
    # If not, try 8, etc.
    
    # However, constructing such a graph is hard.
    # A simpler approach: Randomized Local Search starting from a random graph, minimizing max book.
    # The inspirations used SA or Hill Climbing.
    # Let's implement a fast Hill Climber / Simulated Annealing that specifically targets the max book.
    
    # To be safe and fast, we will:
    # 1. Initialize with a random graph of 401 edges.
    # 2. Use a local search to minimize the maximum book size.
    # 3. We will use a "steepest descent" on the max book, with random restarts if stuck.
    
    # Better: Use the "bipartite double cover" idea from the directive?
    # "fix a partition ... K20,20 + 1 intra ... enumerate ... to minimize induced book"
    # K20,20 has max book 0 for cross edges, but intra edges have huge books.
    # If we add 1 intra edge, max book is 20.
    # If we remove some cross edges to allow more intra edges?
    # No, we must have 401 edges.
    # K20,20 has 400 edges. We need 1 more.
    # If we add 1 intra edge, we have 401 edges. Max book is 20 (for that intra edge).
    # Can we do better?
    # What if we don't use K20,20?
    # What if we use a random graph G(40, 401)?
    # Expected degree is 401*2/40 = 20.05.
    # Expected triangles per edge?
    # Number of triangles in G(n, p) is approx n^3 p^3 / 6.
    # p = 401/780 = 0.514.
    # Triangles ~ 40^3 * 0.514^3 / 6 = 64000 * 0.136 / 6 ~ 1450 triangles.
    # Total edges 401.
    # Average triangles per edge = 1450 * 3 / 401 ~ 10.8.
    # So a random graph likely has max book around 15-20.
    # This matches the seed (20) and inspirations (15).
    
    # To get lower, we need to distribute triangles more evenly or reduce total triangles.
    # Reducing total triangles while keeping 401 edges is hard (Motzkin-Straus?).
    # But we can try to minimize the MAX by balancing.
    
    # Let's implement a robust local search:
    # 1. Start with random graph.
    # 2. Calculate book sizes.
    # 3. While time < deadline:
    #    - Find the edge with max book.
    #    - Try to swap it with a random non-edge to reduce the max book.
    #    - Accept if max book decreases.
    #    - If max book stays same, accept if sum of squares decreases (smoothing).
    #    - If stuck, do a random perturbation (restart or random walk).
    
    # This is similar to the current program but potentially more effective if tuned.
    # The current program breaks if no improvement is found in a batch of 500 swaps.
    # This is too conservative. We should allow neutral moves or worsening moves (SA).
    
    # Let's implement Simulated Annealing.
    
    def random_graph(n, m):
        edges = set()
        while len(edges) < m:
            u, v = random.sample(range(n), 2)
            if u > v: u, v = v, u
            edges.add((u, v))
        return edges
    
    def calc_books(edges, n):
        adj = np.zeros((n, n), dtype=bool)
        for u, v in edges:
            adj[u, v] = True
            adj[v, u] = True
        books = {}
        for u, v in edges:
            books[(u, v)] = np.sum(adj[u] & adj[v])
        return books, adj
    
    def get_metrics(books):
        if not books: return 0, 0
        max_b = max(books.values())
        sum_sq = sum(v*v for v in books.values())
        return max_b, sum_sq
    
    best_edges = None
    best_max_book = float('inf')
    best_sum_sq = float('inf')
    
    # We will run multiple restarts.
    # Each restart runs SA for a fixed time or steps.
    
    num_restarts = 10
    time_per_restart = (deadline - time.time()) / num_restarts
    
    for r in range(num_restarts):
        if time.time() > deadline - 2: break
        
        # Initialize
        edges = random_graph(n, m)
        books, adj = calc_books(edges, n)
        curr_max, curr_sq = get_metrics(books)
        
        # SA Parameters
        T = 10.0
        cooling = 0.95
        min_T = 0.1
        steps = 2000 # Fixed steps per restart to control time
        
        for step in range(steps):
            if time.time() > deadline - 2: break
            
            # Pick random edge to remove
            u, v = random.choice(list(edges))
            
            # Pick random non-edge to add
            found = False
            for _ in range(10):
                a, b = random.sample(range(n), 2)
                if not adj[a, b]:
                    found = True
                    break
            if not found: continue
            
            # Evaluate swap
            # Remove (u,v), Add (a,b)
            # Affected edges: incident to u, v, a, b
            
            # We will just recompute books for affected edges and update global metrics.
            # To be fast, we'll use the adj matrix.
            
            # Save old state of adj for affected?
            # Just modify adj, compute, revert.
            
            adj[u, v] = False
            adj[v, u] = False
            adj[a, b] = True
            adj[b, a] = True
            
            # Identify affected edges
            affected = set()
            for z in range(n):
                if adj[u, z]: affected.add((min(u,z), max(u,z)))
                if adj[v, z]: affected.add((min(v,z), max(v,z)))
                if adj[a, z]: affected.add((min(a,z), max(a,z)))
                if adj[b, z]: affected.add((min(b,z), max(b,z)))
            
            # Update books
            # We need to be careful: (u,v) is removed, (a,b) is added.
            # Other affected edges are just recalculated.
            
            new_books = dict(books)
            
            # Remove old (u,v)
            if (u, v) in new_books: del new_books[(u, v)]
            elif (v, u) in new_books: del new_books[(v, u)]
            
            # Add/Update (a,b)
            key_ab = (min(a,b), max(a,b))
            new_books[key_ab] = np.sum(adj[a] & adj[b])
            
            # Update other affected
            for e in affected:
                if e == key_ab: continue
                if e in edges or (e[0] == a and e[1] == b) or (e[0] == b and e[1] == a):
                    # If it's the new edge, already handled.
                    # If it's an existing edge, recalculate.
                    # Note: `edges` is the OLD set.
                    # The new set is edges - {(u,v)} + {(a,b)}.
                    # So if e is in old edges and e != (u,v), it's in new edges.
                    # If e is (a,b), it's in new edges.
                    # If e is not in old edges and e != (a,b), it's NOT in new edges.
                    # But `affected` contains edges incident to u,v,a,b in the NEW adj.
                    # So `affected` might contain edges that are NOT in the new edge set?
                    # No, `affected` is derived from `adj` which is the NEW adj.
                    # So `affected` contains exactly the edges incident to u,v,a,b in the NEW graph.
                    # Are all these edges in the NEW edge set?
                    # Yes, by definition of adjacency.
                    # So we just recalculate book size for all e in affected.
                    # Wait, if e was not in old edges and e != (a,b), can it be in affected?
                    # Only if it became an edge? No, we only added (a,b).
                    # So if e is in affected and e != (a,b), it must have been in old edges.
                    # So we can just recalculate.
                    
                    x, y = e
                    new_books[e] = np.sum(adj[x] & adj[y])
            
            # Revert adj
            adj[u, v] = True
            adj[v, u] = True
            adj[a, b] = False
            adj[b, a] = False
            
            new_max, new_sq = get_metrics(new_books)
            
            # Calculate delta
            delta_max = new_max - curr_max
            delta_sq = new_sq - curr_sq
            
            # Acceptance criteria
            accept = False
            if delta_max < 0:
                accept = True
            elif delta_max == 0 and delta_sq < 0:
                accept = True
            elif T > min_T:
                # SA acceptance
                # Use combined cost? Or just max book?
                # If we only use max book, we might get stuck at high max but low sq.
                # Let's use a composite score: max_book * 100 + sum_sq / 1000?
                # Or just accept if max_book <= curr_max and sq improves?
                # To escape local minima of max_book, we must allow worsening max_book.
                # Let's use cost = max_book * 1000 + sum_sq
                old_cost = curr_max * 1000 + curr_sq
                new_cost = new_max * 1000 + new_sq
                delta_cost = new_cost - old_cost
                if delta_cost < 0:
                    accept = True
                else:
                    prob = np.exp(-delta_cost / T)
                    if random.random() < prob:
                        accept = True
            
            if accept:
                # Apply swap
                edges.discard((u, v))
                edges.add(key_ab)
                books = new_books
                curr_max, curr_sq = new_max, new_sq
                
                # Update best
                if curr_max < best_max_book or (curr_max == best_max_book and curr_sq < best_sum_sq):
                    best_max_book = curr_max
                    best_sum_sq = curr_sq
                    best_edges = set(edges)
            
            T *= cooling
        
    # If we found a good graph, return it.
    # If not, return the random one? Or the seed?
    # We should return the best found.
    
    if best_edges is None:
        # Fallback to seed-like graph
        edges = set()
        for a in range(20):
            for b in range(20, 40):
                edges.add((a, b))
        edges.add((0, 1))
        best_edges = edges
    
    result = [[u, v] for u, v in best_edges]
    
    # Sanity check
    if len(result) != 401:
        return [[a, b] for a in range(20) for b in range(20, 40)] + [[0, 1]]
        
    return result
