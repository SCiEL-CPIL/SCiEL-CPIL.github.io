import time
import random
import sys
from itertools import combinations

def construct():
    deadline = time.time() + 285.0  # Safety margin before 300s limit

    # --- Paley Graph Fallback (37 vertices) ---
    def get_paley_edges(q):
        residues = {(x * x) % q for x in range(1, q)}
        edges = []
        for i in range(q):
            for j in range(i + 1, q):
                if (j - i) % q in residues:
                    edges.append((i, j))
        return q, edges

    # --- Verification ---
    def count_violations_fast(adj, n):
        # Returns (k5_count, alpha5_count)
        # Optimized for speed using bitmasks where possible, but explicit loops for clarity in 5-sets
        # For n=42, C(42,5) ~ 850k. 10 checks each = 8.5M ops. ~1-2s in Python.
        k5 = 0
        alpha5 = 0
        # Pre-compute list of 5-combinations for this n to avoid overhead in loop?
        # For a single check, generating on the fly is fine, but if we check many times, pre-compute.
        # Since we are doing SA, we check incrementally or occasionally.
        # This function is for final verification or initial state.
        
        # A faster check: 
        # For each 5-set, count edges.
        # If 10 -> K5
        # If 0 -> Alpha5
        
        # We can speed this up by pre-computing the 5-combinations once per n.
        # But n varies. Let's just use itertools.combinations.
        
        for c in combinations(range(n), 5):
            v0, v1, v2, v3, v4 = c
            cnt = 0
            if adj[v0] & (1 << v1): cnt += 1
            if adj[v0] & (1 << v2): cnt += 1
            if adj[v0] & (1 << v3): cnt += 1
            if adj[v0] & (1 << v4): cnt += 1
            if adj[v1] & (1 << v2): cnt += 1
            if adj[v1] & (1 << v3): cnt += 1
            if adj[v1] & (1 << v4): cnt += 1
            if adj[v2] & (1 << v3): cnt += 1
            if adj[v2] & (1 << v4): cnt += 1
            if adj[v3] & (1 << v4): cnt += 1
            
            if cnt == 10:
                k5 += 1
            elif cnt == 0:
                alpha5 += 1
                
        return k5 + alpha5

    def is_valid(adj, n):
        return count_violations_fast(adj, n) == 0

    def get_delta(adj, n, u, v):
        # Calculate change in violation count if edge (u,v) is flipped.
        # This is O(n^3) which is ~42^3 = 74k ops. Very fast.
        # We need to check all triples (a,b,c) from V \ {u,v}
        # A 5-set {u,v,a,b,c} is affected only if the other 3 vertices (a,b,c) form a triangle (for K5) or independent triangle (for Alpha5) with respect to u and v.
        
        edge_exists = (adj[u] >> v) & 1
        delta = 0
        
        # Iterate over all triples a < b < c from vertices excluding u and v
        # To make it faster, we can precompute lists of vertices not u,v?
        # Or just iterate and skip.
        
        # Optimization: 
        # If we add edge (u,v):
        #   We create a K5 if {u,v,a,b,c} becomes a K5. This requires a,b,c to be a triangle AND all connected to u and v.
        #   We destroy an Alpha5 if {u,v,a,b,c} was an Alpha5. This requires a,b,c to be independent AND all NOT connected to u and v.
        
        # If we remove edge (u,v):
        #   We destroy a K5 if {u,v,a,b,c} was a K5.
        #   We create an Alpha5 if {u,v,a,b,c} becomes an Alpha5.
        
        # Let's just check the 9 edges for each triple.
        # 9 edge checks * 74k triples = 666k ops. ~0.1s per step? No, 666k ops is very fast in Python, maybe 0.05s.
        # SA might do 1000-5000 steps in 30s. 5000 * 0.05 = 250s. Too slow.
        
        # We need a faster delta calculation or a different search strategy.
        
        # Alternative: Local Search with "Repair" strategy.
        # 1. Generate random graph or extend Paley.
        # 2. Find a violation (K5 or Alpha5).
        # 3. Flip a random edge within that violation.
        # 4. Repeat.
        # This avoids calculating delta for random edges. It only calculates delta (or just flips) for edges in known violations.
        # But we need to find violations efficiently.
        
        # Let's stick to the "Find violation and flip" strategy (Inspiration 1) but optimize the violation finder.
        # Finding a violation is O(n^5) if done naively, but we can stop at the first one.
        # In a dense graph with few violations, finding one might be fast.
        # If the graph is far from valid, there are many violations, so finding one is fast.
        # If it's close, there are few, so it's slower.
        
        # Let's implement a hybrid:
        # 1. Try to extend Paley 37 to 42/43.
        # 2. Use Local Search: 
        #    - Find a violation.
        #    - If none, return valid.
        #    - If found, pick a random edge in the violation to flip.
        #    - To avoid cycling, keep a small memory or just rely on randomness.
        
        return 0 # Not used in the new strategy, placeholder

    def find_violation(adj, n):
        # Returns a 5-tuple of vertices forming a K5 or Alpha5, or None if valid.
        # Iterates through combinations.
        # To speed up, we can shuffle the order of combinations?
        # Or just iterate.
        
        # Optimization: Pre-compute combinations for n?
        # For n=42, list of 850k tuples. Memory ~ 850k * 5 * 8 bytes = 34MB. Acceptable.
        # But generating them takes time.
        # Let's generate on the fly but use a pre-computed list if n is fixed in the loop.
        
        # Since we will call this many times, let's assume n is fixed for the search phase.
        # We will pass the pre-computed combinations.
        
        # Actually, let's just use itertools.combinations. The overhead is small compared to the 10 edge checks.
        
        for c in combinations(range(n), 5):
            v0, v1, v2, v3, v4 = c
            cnt = 0
            if adj[v0] & (1 << v1): cnt += 1
            if adj[v0] & (1 << v2): cnt += 1
            if adj[v0] & (1 << v3): cnt += 1
            if adj[v0] & (1 << v4): cnt += 1
            if adj[v1] & (1 << v2): cnt += 1
            if adj[v1] & (1 << v3): cnt += 1
            if adj[v1] & (1 << v4): cnt += 1
            if adj[v2] & (1 << v3): cnt += 1
            if adj[v2] & (1 << v4): cnt += 1
            if adj[v3] & (1 << v4): cnt += 1
            
            if cnt == 10:
                return c
            if cnt == 0:
                return c
        return None

    def local_search(n, time_limit):
        start_time = time.time()
        
        # Initialization: 
        # Strategy A: Random graph
        # Strategy B: Paley 37 + random 5 vertices (for n=42)
        
        adj = [0] * n
        
        # If n == 42, use Paley 37 base
        if n == 42:
            # Initialize 0-36 with Paley
            residues = {(x * x) % 37 for x in range(1, 37)}
            for i in range(37):
                for j in range(i + 1, 37):
                    if (j - i) % 37 in residues:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            # Initialize 37-41 randomly
            for i in range(37, n):
                for j in range(n):
                    if i == j: continue
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
        else:
            # Random initialization for other n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)

        # Main Loop
        while time.time() - start_time < time_limit:
            violation = find_violation(adj, n)
            if violation is None:
                return adj
            
            # Pick a random edge from the violation to flip
            v0, v1, v2, v3, v4 = violation
            pairs = [(v0, v1), (v0, v2), (v0, v3), (v0, v4),
                     (v1, v2), (v1, v3), (v1, v4),
                     (v2, v3), (v2, v4),
                     (v3, v4)]
            
            u, v = random.choice(pairs)
            
            # Flip edge
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
            
        return None # Failed to find valid graph

    # --- Main Search ---
    
    # We prioritize n=42 (Exoo's bound) and n=43 (target for improvement).
    # We will try n=43 first with a time budget, then n=42, then fallback to 37.
    
    # Time allocation:
    # 300s total.
    # Try n=43 for 100s.
    # Try n=42 for 100s.
    # Fallback 37 (instant).
    
    # However, finding a valid 43-vertex graph is extremely hard (R(5,5) <= 46, so 43 is possible but unknown if it exists? 
    # Wait, R(5,5) > 42 is known. R(5,5) <= 46. 
    # So a 42-vertex graph exists. A 43-vertex graph might exist.
    # The seed is 37. Exoo found 42.
    # If we find 42, score is 42/37 = 1.13.
    # If we find 43, score is 43/37 = 1.16.
    
    # Let's try 42 first as it is known to exist, so higher probability of success.
    # Then try 43 with remaining time.
    
    # Strategy:
    # 1. Attempt to find a 42-vertex graph. (Budget: 120s)
    # 2. If found, try to extend to 43 or find a different 43-vertex graph. (Budget: 150s)
    # 3. If 43 found, return it.
    # 4. If 42 found but not 43, return 42.
    # 5. If neither, return 37.
    
    best_n = 37
    best_edges = get_paley_edges(37)[1]
    best_adj = None
    
    # Reconstruct adj from edges for 37
    adj_37 = [0] * 37
    for u, v in best_edges:
        adj_37[u] |= (1 << v)
        adj_37[v] |= (1 << u)
    best_adj = adj_37

    # Try 42
    if time.time() < deadline - 150:
        budget_42 = min(120.0, deadline - time.time() - 150)
        if budget_42 > 10:
            adj_42 = local_search(42, budget_42)
            if adj_42 is not None:
                if is_valid(adj_42, 42):
                    edges_42 = []
                    for i in range(42):
                        for j in range(i + 1, 42):
                            if adj_42[i] & (1 << j):
                                edges_42.append((i, j))
                    best_n = 42
                    best_edges = edges_42
                    best_adj = adj_42

    # Try 43
    if time.time() < deadline - 10:
        budget_43 = deadline - time.time() - 5
        if budget_43 > 10:
            # We can start from the 42 graph if we found one, by adding a random 43rd vertex.
            # Or just run local_search(43, budget_43) which does its own init.
            # Let's modify local_search to allow a seed?
            # For now, let's just run a fresh search for 43.
            # To improve odds, we can bias the initialization of 43 to be close to a known 42 graph if we have one.
            
            adj_43 = None
            
            # If we have a 42 graph, extend it
            if best_n == 42:
                # Create a 43 vertex graph by adding vertex 42
                adj_43 = [0] * 43
                # Copy 42 graph
                for i in range(42):
                    adj_43[i] = best_adj[i]
                
                # Connect vertex 42 to 0-41 randomly
                for i in range(42):
                    if random.random() < 0.5:
                        adj_43[42] |= (1 << i)
                        adj_43[i] |= (1 << 42)
                
                # Now run local search on this specific adj
                # We need a version of local search that takes an initial adj
                
                def local_search_from_seed(adj_init, n, time_limit):
                    start_time = time.time()
                    adj = adj_init[:]
                    
                    while time.time() - start_time < time_limit:
                        violation = find_violation(adj, n)
                        if violation is None:
                            return adj
                        
                        v0, v1, v2, v3, v4 = violation
                        pairs = [(v0, v1), (v0, v2), (v0, v3), (v0, v4),
                                 (v1, v2), (v1, v3), (v1, v4),
                                 (v2, v3), (v2, v4),
                                 (v3, v4)]
                        
                        u, v = random.choice(pairs)
                        
                        adj[u] ^= (1 << v)
                        adj[v] ^= (1 << u)
                    
                    return None

                adj_43 = local_search_from_seed(adj_43, 43, budget_43)
            
            if adj_43 is None or not is_valid(adj_43, 43):
                # Try fresh random search for 43 if time permits
                if time.time() < deadline - 10:
                    adj_43 = local_search(43, deadline - time.time() - 5)
            
            if adj_43 is not None and is_valid(adj_43, 43):
                edges_43 = []
                for i in range(43):
                    for j in range(i + 1, 43):
                        if adj_43[i] & (1 << j):
                            edges_43.append((i, j))
                best_n = 43
                best_edges = edges_43

    return best_n, best_edges
