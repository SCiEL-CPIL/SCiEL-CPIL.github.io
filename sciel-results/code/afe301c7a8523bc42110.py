import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    def is_triangle_free(n, edges):
        # Efficient check using adjacency matrix or set
        # For n <= 41, O(n^3) is fine if optimized, or O(n^2) with bitsets
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        for u in range(n):
            # If any two neighbors of u are connected, it's a triangle
            # Neighbors of u:
            # Check all pairs of neighbors?
            # Or: for each v in neighbors(u), if adj[u] & adj[v] != 0, then triangle.
            # But we must exclude u itself from adj[v] check?
            # adj[v] has u set. So adj[u] & adj[v] will have u set? No, adj[u] doesn't have u.
            # So if there is any common neighbor w != u, then adj[u] & adj[v] will have w.
            # Since adj[u] has v, and adj[v] has u.
            # Let's just check: for each v in range(n):
            #   if adj[u] & (1 << v):
            #      common = adj[u] & adj[v]
            #      if common & ~(1 << u) & ~(1 << v): # if there is a third vertex
            #         return False
            # This is O(n^2) per u, so O(n^3) total. 41^3 ~ 68k ops. Very fast.
            for v in range(n):
                if adj[u] & (1 << v):
                    # Check if u and v share a common neighbor w
                    common = adj[u] & adj[v]
                    # Remove u and v from common
                    common &= ~(1 << u)
                    common &= ~(1 << v)
                    if common:
                        return False
        return True

    def has_independent_set_10(n, edges):
        # Branch and bound for independent set of size 10
        # Use bitsets for speed
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        # Max independent set size < 10
        # We want to find if there is an independent set of size 10.
        # If we find one, return True.
        
        # Greedy upper bound:
        # We can use a simple recursive search with pruning.
        # Order vertices by degree?
        
        def backtrack(candidates, size):
            if size == 10:
                return True
            # Pruning: if size + popcount(candidates) < 10, return False
            if size + bin(candidates).count('1') < 10:
                return False
            
            # Pick a vertex
            # Heuristic: pick vertex with lowest degree in candidate subgraph?
            # Or just first one.
            # Let's pick the first available vertex
            # Find lowest set bit
            if candidates == 0:
                return False
            v = (candidates & -candidates).bit_length() - 1
            candidates &= ~(1 << v)
            
            # Option 1: Include v
            # New candidates: neighbors of v removed
            new_candidates = candidates & ~adj[v]
            if backtrack(new_candidates, size + 1):
                return True
            
            # Option 2: Exclude v
            if backtrack(candidates, size):
                return True
                
            return False
        
        return backtrack((1 << n) - 1, 0)

    def validate(n, edges):
        if not is_triangle_free(n, edges):
            return False
        if has_independent_set_10(n, edges):
            return False
        return True

    # Known baseline
    best_n = 26
    best_edges = [[i, j] for i in range(26) for j in range(i + 1, 26) if (j - i) % 26 % 3 == 1]
    
    # Try to find larger n
    # Start from 40 down to 27
    for n in range(40, 26, -1):
        if time.time() > deadline - 5:
            break
            
        # Try several random restarts
        found = False
        for attempt in range(1000): # Limit attempts to prevent infinite loop if validation is slow
            if time.time() > deadline - 2:
                break
                
            # Generate random graph
            # To make it triangle-free, we can start with a random graph and remove triangles?
            # Or start with a random set of edges and check?
            # Better: Start with a random graph, then remove edges to ensure triangle-free.
            # This might result in a very sparse graph, which has large independent sets.
            # We want a dense triangle-free graph to minimize independent set size.
            
            # Strategy: Randomly add edges, but ensure no triangle is formed.
            edges = []
            adj = [0] * n
            # Shuffle all possible edges
            all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
            random.shuffle(all_edges)
            
            for u, v in all_edges:
                # Check if adding (u, v) creates a triangle
                # Triangle exists if there is w such that (u,w) and (v,w) are in edges
                # common = adj[u] & adj[v]
                # If common != 0, then adding (u,v) creates a triangle.
                if (adj[u] & adj[v]) == 0:
                    edges.append((u, v))
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            
            # Now we have a maximal triangle-free graph (random greedy)
            # Check if it has independent set of size 10
            # Convert edges list to list of lists for validate
            edge_list = [[u, v] for u, v in edges]
            
            # Quick check: if number of edges is too low, alpha is likely high
            # Turan's theorem bound?
            # Just run the checker
            if validate(n, edge_list):
                best_n = n
                best_edges = edge_list
                found = True
                break # Found a valid graph for this n, move to next n? 
                # Actually, if we find one for n, we can stop searching for this n and try n+1?
                # No, we are iterating n from 40 down. If we find one for 40, we can return immediately.
                # But we might find one for 39 first.
                # Since we iterate 40, 39, 38..., if we find one for 40, we return.
                # If we don't find one for 40, we try 39.
                # So if found=True, we can break the inner loop and return?
                # Yes, because we want to MAXIMIZE n. If we found a valid graph for n, and we are iterating downwards, 
                # the first valid n we find is the maximum?
                # Not necessarily. We might fail to find a valid graph for 40 due to bad luck in random search, 
                # but one might exist. However, if we find one for 39, it's better than 26.
                # We should try to find the largest possible.
                # So if we find one for 40, return.
                # If not, try 39.
                # If we find one for 39, should we keep looking for 40?
                # We already tried 40. If we didn't find one in 1000 attempts, it's unlikely we will in the remaining time.
                # So we can return the best found so far.
                
                # Let's return immediately if we find a valid graph for the current n, 
                # because we are going from high to low.
                return {'n': n, 'edges': best_edges}
        
        # If we didn't find one for this n, continue to n-1

    return {'n': best_n, 'edges': best_edges}
