import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Seed graph: Andrasfai graph And(9)
    n_seed = 26
    edges_seed = [[i, j] for i in range(n_seed) for j in range(i + 1, n_seed) if (j - i) % n_seed % 3 == 1]
    best_result = {'n': n_seed, 'edges': edges_seed}
    
    # Helper: Check if graph has independent set of size k
    # adj: list of bitmasks
    # k: target size
    def has_independent_set(adj, k):
        n = len(adj)
        # Recursive search with memoization is hard due to state size, use simple backtracking
        # Order vertices by degree (ascending) to prune? 
        # For now, simple bitset backtracking
        
        def backtrack(candidates, current_size):
            if current_size == k:
                return True
            if candidates == 0:
                return False
            # Pruning: if max possible size < k, return False
            # popcount is fast in Python 3.8+ via int.bit_count()
            if current_size + candidates.bit_count() < k:
                return False
            
            # Pick a vertex
            # Heuristic: pick vertex with lowest degree in the induced subgraph of candidates?
            # For simplicity, pick lowest index
            v = (candidates & -candidates).bit_length() - 1
            
            # Branch 1: Include v
            # New candidates: remove v and its neighbors
            new_candidates_include = candidates & ~(1 << v) & ~adj[v]
            if backtrack(new_candidates_include, current_size + 1):
                return True
            
            # Branch 2: Exclude v
            new_candidates_exclude = candidates & ~(1 << v)
            if backtrack(new_candidates_exclude, current_size):
                return True
            
            return False
        
        initial_candidates = (1 << n) - 1
        return backtrack(initial_candidates, 0)

    # Helper: Make graph triangle-free
    # adj: list of bitmasks
    def remove_triangles(adj):
        n = len(adj)
        changed = True
        while changed:
            changed = False
            for u in range(n):
                for v in range(u + 1, n):
                    if (adj[u] >> v) & 1:
                        # Check if edge (u,v) is in a triangle
                        common = adj[u] & adj[v]
                        if common:
                            # Remove edge (u,v)
                            adj[u] &= ~(1 << v)
                            adj[v] &= ~(1 << u)
                            changed = True
        return adj

    # Helper: Local search to minimize alpha
    def local_search(adj, time_limit):
        n = len(adj)
        start = time.time()
        
        # Initial triangle-free
        adj = remove_triangles(adj)
        
        while time.time() - start < time_limit:
            # Check if alpha < 10
            if not has_independent_set(adj, 10):
                return adj
            
            # Try to add an edge to reduce alpha
            # Find an independent set S of size >= 10
            # We don't have the set, just the boolean.
            # Heuristic: Randomly pick two non-adjacent vertices u, v.
            # If adding (u,v) doesn't create triangle, add it.
            # To be smarter, pick u, v from a large independent set.
            # Since we don't have the set, we just try random pairs.
            
            # Optimization: Try to add edges that are "safe"
            # A safe edge is one where adj[u] & adj[v] == 0
            # We want to add edges that connect vertices that are far apart in the independence number sense.
            
            # Simple random edge addition
            added = False
            for _ in range(100): # Limit attempts per loop
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v:
                    continue
                if (adj[u] >> v) & 1:
                    continue
                # Check if adding (u,v) creates triangle
                if adj[u] & adj[v]:
                    continue
                
                # Safe to add
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                added = True
                break
            
            if not added:
                # Stuck. No safe edge to add.
                # Must remove an edge to unlock.
                # Remove a random edge.
                # Find any edge
                edge_found = False
                for u in range(n):
                    if adj[u]:
                        v = (adj[u] & -adj[u]).bit_length() - 1
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                        edge_found = True
                        break
                if not edge_found:
                    break # Empty graph, alpha=n, won't improve
            
            # Re-check triangles (removal might not create, but addition did? No, we checked.
            # But removal might make other edges unsafe? No, removal only increases safety.)
            # However, if we removed an edge, we might have created a "gap" that allows adding other edges.
            # The loop will try to add again.

        return adj

    # Try to find a graph for n = 39 down to 27
    for n in range(39, 26, -1):
        if time.time() > deadline - 5:
            break
            
        # Time limit for this n
        time_limit = 10.0
        found = False
        
        # Random restarts
        max_restarts = 50
        restart_time = time_limit / max_restarts
        
        for r in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            # Random graph
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            # Local search
            adj = local_search(adj, restart_time)
            
            # Final check
            if not has_independent_set(adj, 10):
                # Verify triangle-free (should be)
                is_tf = True
                for i in range(n):
                    for j in range(i + 1, n):
                        if (adj[i] >> j) & 1:
                            if adj[i] & adj[j]:
                                is_tf = False
                                break
                    if not is_tf:
                        break
                
                if is_tf:
                    edges = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if (adj[i] >> j) & 1:
                                edges.append([i, j])
                    best_result = {'n': n, 'edges': edges}
                    found = True
                    break
        
        if found:
            break # Found best possible n for this run, return it
            
    return best_result

# EVOLVE-BLOCK-END
