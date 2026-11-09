import time
import random
import itertools

def _has_independent_set_10(adj_mask, n):
    """
    Checks if there is an independent set of size 10.
    adj_mask[i] is a bitmask of neighbors of i.
    Uses branch and bound.
    """
    # Heuristic: If max degree is low, or graph is sparse, alpha is large.
    # We want to prove NO independent set of size 10.
    # So we search for one. If found, return True.
    
    def dfs(available_mask, size):
        if size == 10:
            return True
        if available_mask == 0:
            return False
        
        # Pick a vertex with minimum degree in the induced subgraph?
        # Or just pick the first one.
        # Better: Pick vertex v.
        # If we include v, we must exclude its neighbors.
        # If we exclude v, we continue with available_mask without v.
        
        # Optimization: If popcount(available_mask) < 10 - size, return False.
        # We need 10 - size more vertices.
        if bin(available_mask).count('1') < 10 - size:
            return False
            
        # Pick a vertex v from available_mask
        # Choose v with max degree in available_mask to prune?
        # Or min degree?
        # Let's pick the lowest set bit for simplicity, or random.
        # To be fast, pick v such that it removes many neighbors.
        
        # Let's iterate through candidates
        # We can pick one vertex v.
        # Branch 1: Include v. New available = available_mask & ~adj_mask[v] & ~(1<<v)
        # Branch 2: Exclude v. New available = available_mask & ~(1<<v)
        
        # To speed up, we can order vertices.
        # Let's just pick the first available vertex.
        v_bit = available_mask & -available_mask
        v = v_bit.bit_length() - 1
        
        # Include v
        new_avail = available_mask & ~adj_mask[v] & ~v_bit
        if dfs(new_avail, size + 1):
            return True
            
        # Exclude v
        new_avail = available_mask & ~v_bit
        if dfs(new_avail, size):
            return True
            
        return False

    return dfs((1 << n) - 1, 0)

def construct():
    start_time = time.time()
    deadline = start_time + 110 # 10s margin
    
    # Try to find a graph with n > 26
    # We know n=39 is possible.
    # We'll try to construct one using random restarts + local search.
    
    best_n = 26
    best_edges = [[i, j] for i in range(26) for j in range(i + 1, 26) if (j - i) % 26 % 3 == 1]
    
    for target_n in [39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27]:
        if time.time() > deadline:
            break
            
        # Try to find a valid graph on target_n vertices
        found = False
        attempts = 0
        max_attempts = 20 # Limit attempts per n to avoid timeout
        
        while attempts < max_attempts and time.time() < deadline and not found:
            attempts += 1
            # Generate a random maximal triangle-free graph
            adj = [0] * target_n
            # Randomly add edges
            # To ensure maximal, we can just add all non-conflicting edges in random order?
            # Or start empty and add random edges until no more can be added.
            
            # A better way: Start with a random bipartite-like structure?
            # No, just random.
            edges_to_add = []
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    edges_to_add.append((i, j))
            random.shuffle(edges_to_add)
            
            for u, v in edges_to_add:
                # Check if adding (u,v) creates a triangle
                # Common neighbors of u and v must be empty
                common = adj[u] & adj[v]
                if common == 0:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            
            # Now we have a maximal triangle-free graph.
            # Check alpha
            if not _has_independent_set_10(adj, target_n):
                # Found a valid graph
                best_n = target_n
                best_edges = []
                for i in range(target_n):
                    for j in range(i + 1, target_n):
                        if adj[i] & (1 << j):
                            best_edges.append([i, j])
                found = True
                break
                
            # If not found, we could try local search to reduce alpha
            # But for now, just restart.
            
        if found:
            break
            
    return {'n': best_n, 'edges': best_edges}
