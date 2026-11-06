import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Initial construction: K_{6,10} + matching of size 5 in the part of size 10
    n = 16
    # Part A: 0..5 (size 6)
    # Part B: 6..15 (size 10)
    # Edges: All (A, B) pairs
    edges = set()
    for u in range(6):
        for v in range(6, 16):
            edges.add((u, v))
    
    # Add matching in B: (6,7), (8,9), (10,11), (12,13), (14,15)
    # Wait, this creates 5 edges. Total 60 + 5 = 65.
    match_edges = [(6,7), (8,9), (10,11), (12,13), (14,15)]
    for u, v in match_edges:
        edges.add((u, v))
    
    # Convert to list for return later, but use set for search
    # We need to maintain exactly 65 edges.
    
    # Helper to compute score
    def compute_score(edge_set):
        # Build adjacency bitmasks
        adj = [0] * n
        deg = [0] * n
        for u, v in edge_set:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            deg[u] += 1
            deg[v] += 1
        
        max_sum = 0
        # Find all triangles
        # Iterate u < v < w
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    # Check common neighbors w > v
                    common = adj[u] & adj[v]
                    # Only consider w > v to avoid duplicates
                    common &= ~((1 << (v + 1)) - 1)
                    while common:
                        lsb = common & -common
                        w = lsb.bit_length() - 1
                        s = deg[u] + deg[v] + deg[w]
                        if s > max_sum:
                            max_sum = s
                        common ^= lsb
        return max_sum

    current_edges = set(edges)
    current_score = compute_score(current_edges)
    
    # Local Search
    # We will try to swap edges.
    # Move: remove (u,v), add (x,y).
    # Constraints: (x,y) not in edges, (u,v) in edges.
    # Also, simple graph.
    
    # To speed up, we can focus on triangles with max score.
    # But recomputing score from scratch is O(E * N) or similar.
    # With N=16, it's fast.
    
    best_score = current_score
    best_edges = set(current_edges)
    
    # List of all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Precompute edge existence for faster lookup? Set is fine.
    
    # Search loop
    # We try random swaps.
    # If we find a better score, we accept and restart? Or continue?
    # Since the landscape might be rugged, random restarts with local improvement might be good.
    # But we have a time limit.
    
    # Let's do a simple hill climbing with random restarts if stuck.
    # Actually, just iterate through random moves until deadline.
    
    # To make it smarter, we can prioritize moves that affect high-degree vertices in critical triangles.
    # But for now, random swaps are a good baseline.
    
    # Optimization: 
    # Instead of random swaps, we can try to "fix" the worst triangles.
    # If a triangle (u,v,w) has sum S, we want to reduce deg(u), deg(v) or deg(w).
    # Removing an edge incident to u, v, or w reduces their degree by 1.
    # Adding an edge elsewhere keeps total edges constant.
    
    # Let's implement a move generator that:
    # 1. Picks a random edge (u,v) to remove.
    # 2. Picks a random non-edge (x,y) to add.
    # 3. Checks if (x,y) is valid (not already present).
    # 4. Computes new score.
    # 5. Accepts if new_score < current_score.
    #    If new_score == current_score, accept with some probability (simulated annealing style) or just accept to explore.
    
    # Given the small N, we can be more aggressive.
    
    # Let's track the best found.
    
    # Initial score is 24.
    # Can we get 23?
    # 23 is odd. Sum of 3 integers.
    # If we can balance degrees better.
    
    # Let's run the search.
    
    # To speed up, we can cache some triangle info?
    # No, N=16 is tiny.
    
    # One improvement: 
    # The current graph has triangles formed by (a, b1, b2) where (b1,b2) is a matching edge.
    # deg(a)=10, deg(b1)=7, deg(b2)=7. Sum=24.
    # If we remove an edge from the matching, say (6,7), and add an edge (0,1) in A.
    # New degrees:
    # deg(6)=6, deg(7)=6.
    # deg(0)=11, deg(1)=11.
    # Other A vertices: 10.
    # Other B vertices: 7.
    # Triangles:
    # Old triangles involving (6,7) are gone.
    # New triangles involving (0,1): (0,1,b) for all b in B.
    # Sum: deg(0)+deg(1)+deg(b) = 11+11+7 = 29. Much worse.
    
    # What if we add an edge (6,8) in B instead of (0,1)?
    # Remove (6,7). Add (6,8).
    # deg(6)=7 (was 7, lost 7, gained 8). Wait.
    # Initial deg(6)=7 (5 from K? No, 6 from K? A has 6 vertices, so deg_B = 6. Plus 1 matching = 7).
    # Remove (6,7): deg(6)=6, deg(7)=6.
    # Add (6,8): deg(6)=7, deg(8)=8.
    # deg(7)=6.
    # deg(8)=8.
    # Other B: 7.
    # A: 10.
    # Triangles:
    # (6,8) is an edge. Triangles (a, 6, 8) for a in A.
    # Sum: 10 + 7 + 8 = 25. Worse than 24.
    
    # It seems 24 is a local minimum for this family.
    # Maybe a different starting graph?
    # K_{5,11} + 10 edges?
    # K_{5,11} has 55 edges. Need 10 more.
    # If we add 10 edges in B (size 11).
    # Max degree in B from internal edges: 10 (if star).
    # deg_A = 11.
    # deg_B = 5 + internal_deg.
    # Triangle sum: 11 + (5+d1) + (5+d2) = 21 + d1 + d2.
    # If we add a matching of size 5 in B (10 vertices used, 1 unused).
    # d1=1, d2=1. Sum = 23.
    # Let's check this construction.
    
    # Construction 2:
    # A: 0..4 (size 5)
    # B: 5..15 (size 11)
    # Edges: All (A, B) -> 5*11 = 55.
    # Add matching in B: (5,6), (7,8), (9,10), (11,12), (13,14). (15 is unmatched in B-internal).
    # Total edges: 55 + 5 = 60. Wait. 55+5=60. We need 65.
    # We need 5 MORE edges.
    # Add 5 more edges in B.
    # If we add them as a matching, we need 5 disjoint edges.
    # We have 11 vertices in B. We used 10 in the first matching.
    # We can only add 1 more disjoint edge if we use the 11th vertex and one of the others? No, disjoint means new vertices.
    # We have 1 vertex left (15). We can't make a disjoint edge.
    # So we must add edges that share vertices or are in A.
    
    # If we add edges in A:
    # A has 5 vertices. Max edges in A is 10.
    # If we add 5 edges in A (e.g. a cycle or star).
    # deg_A increases.
    # Let's try adding 5 edges in B that form a star or something.
    # If we add edges incident to 15?
    # Connect 15 to 5,6,7,8,9?
    # deg(15) = 5 (from K) + 5 = 10.
    # deg(5) = 5 (from K) + 1 (match 5-6) + 1 (15-5) = 7.
    # Triangle (a, 5, 15): sum = deg(a) + deg(5) + deg(15) = 11 + 7 + 10 = 28. Worse.
    
    # What if we start with K_{5,11} and add 10 edges in B such that max internal degree is small?
    # We need to add 10 edges to B (size 11).
    # Average internal degree = 20/11 = 1.8.
    # If we use a matching of size 5 (10 edges? No, matching of size 5 has 5 edges).
    # We need 10 edges.
    # A 2-regular graph on 11 vertices? Cycle C11?
    # Internal degree 2 for all.
    # deg_B = 5 + 2 = 7.
    # deg_A = 11.
    # Triangle sum: 11 + 7 + 7 = 25. Worse than 24.
    
    # So K_{6,10} + matching (score 24) seems better than K_{5,11} + C11 (score 25).
    
    # What about K_{7,9}?
    # 63 edges. Need 2 more.
    # We analyzed this: score 25 or 27.
    
    # So 24 is the best known simple construction.
    # Can we tweak K_{6,10} + matching to get 23?
    # Score 24 comes from 10+7+7.
    # To get 23, we need sums like 10+7+6, 9+8+6, etc.
    # This requires reducing some degrees.
    # If we remove an edge from A-B, we lose a bipartite connection.
    # But we must add an edge elsewhere.
    
    # Let's let the local search find it if it exists.
    # The search space is small enough that a good local search might find it.
    
    # Implementation details for speed:
    # Use bitmasks.
    # Precompute list of edges as tuples.
    
    # We will use a random restart local search.
    # Start with the 24-score graph.
    # Try random edge swaps.
    # If we find 23, we return it.
    # If we don't find anything better, we return the best found (likely 24).
    
    # One more idea: 
    # Maybe a non-bipartite starting point?
    # But K_{m,n} is a very strong base.
    
    # Let's code the search.
    
    def get_edges_list(edge_set):
        return list(edge_set)
        
    # We need to be careful with the deadline.
    
    # To improve search efficiency, we can use the fact that only triangles involving changed vertices need re-evaluation?
    # No, degrees change globally if we move an edge.
    # But N=16, so full recompute is ~16^3/6 * 16 ops? No, triangle count is small.
    # Number of triangles in K_{6,10}+matching:
    # 5 matching edges * 6 A vertices = 30 triangles.
    # Very fast to recompute.
    
    # So full recompute is fine.
    
    # Let's run the search.
    
    # To avoid getting stuck in local minima, we can add some "noise" or random restarts.
    # But if 23 doesn't exist, we waste time.
    # We should check if 23 is theoretically possible.
    # Sum of degrees = 2*65 = 130.
    # Average degree = 130/16 = 8.125.
    # If we have a triangle with sum 23, e.g. 8+8+7.
    # This is close to average.
    # It is plausible.
    
    # Let's write the code.
    
    # Random seed for reproducibility in debug, but evaluator might not care.
    # We should use time-based seed or fixed?
    # Fixed is safer for deterministic behavior if we want to ensure we find the best known.
    # But random search is better for finding new ones.
    # We'll use a fixed seed for the initial phase, then random.
    
    # Actually, let's just use random.
    
    # One specific move: 
    # Swap an edge from the matching in B with an edge in A.
    # We tried this and it was worse.
    
    # What if we swap an edge from the matching in B with an edge between A and B?
    # Remove (6,7). Add (0, 15)?
    # (0,15) is already in K_{6,10}. So can't add.
    # Add (0, 6)? Already in K.
    # We can only add edges NOT in the graph.
    # Non-edges are:
    # - Edges within A (15 possible)
    # - Edges within B not in matching (45 - 5 = 40 possible)
    # - Edges between A and B? No, K_{6,10} is complete. So no non-edges between A and B.
    
    # So we can only swap edges within B or within A.
    # Since A has no edges, we can only remove edges from B (matching) and add edges in B (non-matching) or in A.
    
    # If we remove (6,7) and add (6,8):
    # We analyzed this: score 25.
    
    # If we remove (6,7) and add (0,1) in A:
    # We analyzed this: score 29.
    
    # So simple swaps from 24 lead to worse scores.
    # This suggests 24 is a local minimum.
    
    # To escape, we might need a 2-move sequence.
    # Remove (6,7), add (6,8) -> score 25.
    # Then from 25, can we go to 23?
    # Probably not directly.
    
    # Given the time constraint and the difficulty of finding 23, I will implement the search but expect to return 24.
    # However, the prompt says "Scores below 24 would beat the published construction".
    # And "Minimize".
    # If I return 24, the score is 26/24 = 1.083.
    # If I return 26, score is 1.0.
    # So returning 24 is an improvement.
    
    # I will ensure the code reliably returns the 24-score graph.
    # And I will add a limited local search to see if it finds 23.
    
    # Code structure:
    # 1. Build K_{6,10} + matching.
    # 2. Compute score (should be 24).
    # 3. Run local search for a few seconds.
    # 4. Return best found.
    
    # To make the local search effective, we can try all single-edge swaps from the current state?
    # Number of edges = 65.
    # Number of non-edges = 120 - 65 = 55.
    # Total swaps = 65 * 55 = 3575.
    # Each swap score computation is fast.
    # 3575 * (time to compute score).
    # Time to compute score: ~30 triangles * 3 deg lookups. Very fast.
    # So we can try ALL single-edge swaps in one iteration.
    # This is a complete local search.
    
    # If no single-edge swap improves the score, we are at a local minimum.
    # We can then try 2-edge swaps?
    # 65*55 * 65*55 is too big.
    # But we can try 2-edge swaps involving the edges that were "almost" good?
    # Or just restart with a different random graph?
    
    # Let's do complete single-edge local search first.
    
    # If stuck at 24, we return 24.
    
    # Let's verify the score of K_{6,10}+matching again.
    # A: 0-5. B: 6-15.
    # Edges: (a,b) for all a,b.
    # Matching: (6,7), (8,9), (10,11), (12,13), (14,15).
    # deg(a) = 10.
    # deg(b_matched) = 6 (from A) + 1 (match) = 7.
    # Triangle (a, b1, b2) where (b1,b2) is matching.
    # Sum = 10 + 7 + 7 = 24.
    # Are there other triangles?
    # No edges in A.
    # No other edges in B.
    # So only these 30 triangles.
    # Max sum is 24.
    
    # So the initial score is exactly 24.
    
    # I will implement the complete local search.
    
    # If the search finds 23, great.
    # If not, it returns 24.
    
    # One detail: The prompt says "Scores below 24 would beat the published construction".
    # This implies 24 is the published construction's score.
    # And the seed is 26.
    # So 24 is an improvement.
    
    # I will write the code now.

    # To ensure we don't time out, I will limit the number of local search iterations.
    # One full pass of single-edge swaps is ~3500 evaluations.
    # Each eval is < 1ms. So one pass is < 4s.
    # We can do several passes.
    
    # I'll add a loop that runs until deadline.

    # Let's refine the score computation to be as fast as possible.
    
    def fast_score(edge_set):
        adj = [0] * n
        deg = [0] * n
        for u, v in edge_set:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            deg[u] += 1
            deg[v] += 1
        
        max_sum = 0
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    common = adj[u] & adj[v]
                    common &= ~((1 << (v + 1)) - 1)
                    while common:
                        lsb = common & -common
                        w = lsb.bit_length() - 1
                        s = deg[u] + deg[v] + deg[w]
                        if s > max_sum:
                            max_sum = s
                        common ^= lsb
        return max_sum

    # We can precompute the list of all possible edges and non-edges.
    
    # Start search.
    
    # Note: The problem asks for a single function construct().
    
    # I will put everything inside construct().
    
    # One potential issue: random seed.
    # I will use a fixed seed for the first few restarts to ensure we explore the neighborhood of the known good solution.
    
    # Let's go.

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n = 16
    
    # Initial construction: K_{6,10} + matching of size 5 in B
    edges = set()
    for u in range(6):
        for v in range(6, 16):
            edges.add((u, v))
    
    match_edges = [(6,7), (8,9), (10,11), (12,13), (14,15)]
    for u, v in match_edges:
        edges.add((u, v))
    
    # Score function
    def compute_score(edge_set):
        adj = [0] * n
        deg = [0] * n
        for u, v in edge_set:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            deg[u] += 1
            deg[v] += 1
        
        max_sum = 0
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    common = adj[u] & adj[v]
                    common &= ~((1 << (v + 1)) - 1)
                    while common:
                        lsb = common & -common
                        w = lsb.bit_length() - 1
                        s = deg[u] + deg[v] + deg[w]
                        if s > max_sum:
                            max_sum = s
                        common ^= lsb
        return max_sum

    current_edges = set(edges)
    current_score = compute_score(current_edges)
    
    best_score = current_score
    best_edges = set(current_edges)
    
    # List of all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Local Search
    # We try to improve by swapping edges.
    # To avoid getting stuck, we use a simple hill climbing with random restarts if stuck.
    # But first, let's do a complete local search on single-edge swaps.
    
    # We will iterate through all possible swaps (remove e1, add e2).
    # If score improves, we accept and restart the scan.
    # If we complete a scan without improvement, we are at a local minimum.
    # Then we try a random 2-edge swap or restart.
    
    # Given time constraints, we will do several passes of single-edge swaps.
    
    # Precompute edge existence for faster check?
    # Set lookup is O(1).
    
    # To speed up, we can store edges in a list and use a set for lookup.
    
    # Let's run the search loop.
    
    while time.time() < deadline:
        improved = False
        # Shuffle the order of edges to avoid bias
        edge_list = list(current_edges)
        random.shuffle(edge_list)
        
        for e1 in edge_list:
            if time.time() > deadline:
                break
            # Try to remove e1
            e1_set = {e1[0], e1[1]}
            # Try to add e2
            # We need to pick a non-edge e2
            # To avoid iterating all 55 non-edges for each e1 (which is 65*55=3575),
            # we can just iterate all possible e2 and check if it's in current_edges.
            # 120 * 65 = 7800 checks. Still fast.
            
            for e2 in all_edges:
                if e2 in current_edges:
                    continue
                if e2 == e1:
                    continue
                
                # Perform swap
                new_edges = set(current_edges)
                new_edges.remove(e1)
                new_edges.add(e2)
                
                new_score = compute_score(new_edges)
                
                if new_score < current_score:
                    current_edges = new_edges
                    current_score = new_score
                    improved = True
                    if current_score < best_score:
                        best_score = current_score
                        best_edges = set(current_edges)
                    break # Restart the loop with the new state
                elif new_score == current_score:
                    # Optional: accept equal score moves to explore
                    # But this might lead to wandering.
                    # We'll stick to strict improvement for now to ensure progress.
                    pass
        
        if not improved:
            # Local minimum reached.
            # Try a random restart?
            # Or try a 2-edge swap?
            # Given the time, let's just break and return the best found.
            # Or try a few random restarts from the initial state?
            # No, the initial state is the best known.
            # If we can't improve from it, we return it.
            
            # Let's try a few random 2-edge swaps to see if we can escape.
            # Pick 2 random edges to remove, 2 random non-edges to add.
            # This is expensive.
            # Let's just break.
            break
    
    # Return the best edges found
    # Convert to list of lists
    return {'n': 16, 'edges': [[u, v] for u, v in best_edges]}
# EVOLVE-BLOCK-END
