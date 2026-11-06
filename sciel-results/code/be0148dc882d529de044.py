import random
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 115.0  # 115s limit, 5s margin
    
    n = 18
    # Initialize with K_{6,6,6}
    # Parts: 0-5, 6-11, 12-17
    adj = [0] * n
    for i in range(n):
        for j in range(i+1, n):
            part_i = i // 6
            part_j = j // 6
            if part_i != part_j:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    best_adj = adj[:]
    best_score = 12  # Known score for K_{6,6,6}
    
    # Precompute vertex lists for iteration
    vertices = list(range(n))
    
    def is_k4_free(adj):
        # Check all 4-subsets
        for a in range(n):
            for b in range(a+1, n):
                if not (adj[a] & (1 << b)): continue
                for c in range(b+1, n):
                    if not (adj[a] & (1 << c)) or not (adj[b] & (1 << c)): continue
                    for d in range(c+1, n):
                        if (adj[a] & (1 << d)) and (adj[b] & (1 << d)) and (adj[c] & (1 << d)):
                            return False
        return True

    def is_triangle_free_subset(adj, mask):
        # Check if subgraph induced by mask is triangle-free
        # Iterate vertices in mask
        u = 0
        while mask:
            if mask & 1:
                # u is in mask
                # Get neighbors of u in mask
                neighbors = adj[u] & mask
                # Check if neighbors contain an edge
                v = 0
                while neighbors:
                    if neighbors & 1:
                        # v is in neighbors
                        # Check if v has any neighbor in neighbors
                        # adj[v] & neighbors will include u? No, u is not in neighbors (simple graph)
                        # But adj[v] & neighbors might include other vertices
                        if adj[v] & neighbors:
                            return False
                    v += 1
                    neighbors >>= 1
            u += 1
            mask >>= 1
        return True

    def calc_alpha2(adj):
        # Find max size of triangle-free induced subgraph
        # We know max possible is 18, min is 1
        # Check sizes from 11 down to 1
        # If we find a triangle-free set of size k, return k
        # We want to minimize this, so we find the max k that exists.
        # If no set of size 11 is triangle-free, check 10, etc.
        # Optimization: Randomly sample subsets of size 11? 
        # No, exact is better for n=18.
        # C(18, 11) = 31824.
        # For each, is_triangle_free_subset is fast.
        
        # Check size 11
        # Generate all 11-subsets
        # Using combinations is slow in Python for 30k calls?
        # Bitmask iteration is faster.
        
        # Iterate all masks with popcount 11
        # There are 31824 such masks.
        # We can generate them recursively or use a precomputed list?
        # Precomputing 31824 masks is cheap.
        
        # Actually, we can just check if ANY 11-subset is triangle-free.
        # If yes, alpha2 >= 11.
        # We want the MAX size.
        # So we check 11, 10, ...
        
        # To speed up, we can stop early if we find one.
        
        # Helper to iterate k-subsets
        def iter_k_subsets(n, k):
            # Yield bitmasks of k-subsets
            # Simple recursive generator
            if k == 0:
                yield 0
                return
            for i in range(n, k-1, -1):
                for sub in iter_k_subsets(i-1, k-1):
                    yield sub | (1 << (i-1))
            
        # This generator might be slow.
        # Let's use a simpler approach: 
        # Since we are doing local search, we can just check the current graph's alpha2.
        # And maybe sample some subsets if time is tight.
        # But exact is preferred.
        
        # Let's optimize the check.
        # If we find a triangle-free set of size 11, alpha2 is at least 11.
        # We want to know the EXACT max.
        
        # Check 11
        # If none, check 10.
        
        # Precomputed list of 11-subsets?
        # 31824 integers. Memory is fine.
        # But generating them takes time.
        
        # Alternative:
        # Randomly check 1000 subsets of size 11.
        # If any is triangle-free, return 11 (or higher if we check 12, but we know 12 is max for seed).
        # If none found in 1000, assume < 11? Risky.
        
        # Let's do exact for 11.
        # Use itertools.combinations?
        # 31824 combinations. For each, check triangle-free.
        # Check is ~18^2 operations. 31824 * 324 ~ 10M ops.
        # In Python, 10M ops takes ~1s.
        # So 1s per evaluation.
        # We can do ~100 evaluations in 100s.
        # This is feasible.
        
        import itertools
        # Check size 11
        for combo in itertools.combinations(range(n), 11):
            mask = 0
            for v in combo:
                mask |= (1 << v)
            if is_triangle_free_subset(adj, mask):
                return 11 # Or higher? We know max is 12 for seed.
                         # If we improve, it might be 10.
                         # So if 11 is not found, check 10.
        
        # Check size 10
        for combo in itertools.combinations(range(n), 10):
            mask = 0
            for v in combo:
                mask |= (1 << v)
            if is_triangle_free_subset(adj, mask):
                return 10
        
        # Check size 9
        for combo in itertools.combinations(range(n), 9):
            mask = 0
            for v in combo:
                mask |= (1 << v)
            if is_triangle_free_subset(adj, mask):
                return 9
        
        # ... down to 1
        return 1

    # Local Search
    current_adj = best_adj[:]
    current_score = best_score
    
    # Precompute edge list for random selection
    def get_edges(adj):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i] & (1 << j):
                    edges.append((i, j))
        return edges

    def get_non_edges(adj):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if not (adj[i] & (1 << j)):
                    edges.append((i, j))
        return edges

    while time.time() < deadline:
        # Random move
        if random.random() < 0.5:
            # Add edge
            non_edges = get_non_edges(current_adj)
            if not non_edges: continue
            u, v = random.choice(non_edges)
            # Try adding
            new_adj = current_adj[:]
            new_adj[u] |= (1 << v)
            new_adj[v] |= (1 << u)
            if is_k4_free(new_adj):
                current_adj = new_adj
                score = calc_alpha2(current_adj)
                if score < current_score:
                    current_score = score
                    best_adj = current_adj[:]
                    best_score = score
        else:
            # Remove edge
            edges = get_edges(current_adj)
            if not edges: continue
            u, v = random.choice(edges)
            new_adj = current_adj[:]
            new_adj[u] &= ~(1 << v)
            new_adj[v] &= ~(1 << u)
            current_adj = new_adj
            score = calc_alpha2(current_adj)
            if score < current_score:
                current_score = score
                best_adj = current_adj[:]
                best_score = score
            # Simulated annealing acceptance?
            # If score is worse, accept with small prob?
            # For now, greedy.

    # Convert best_adj to edge list
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i] & (1 << j):
                edges.append([i, j])
                
    return edges
