import time
import random
import sys

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    def is_c5_free(adj, n):
        # Check if graph contains a 5-cycle
        # Simple DFS/BFS from each node to find cycles of length 5?
        # Faster: For each edge (u,v), check if there is a path of length 3 from u to v.
        # If yes, u-v + path = 5-cycle.
        # Path of length 3: u-x-y-v. x != y, x != v, y != u.
        # This is O(n^2) per edge if done naively, but we can precompute 2-hop neighbors.
        # Let's use a simpler check: enumerate all 5-tuples? No, too slow.
        
        # Efficient check:
        # For each u, for each v in adj[u]:
        #   For each x in adj[v]:
        #     For each y in adj[x]:
        #       If u in adj[y] and y != u and y != v and x != u:
        #         Found C5.
        # This is O(sum deg(u)*deg(v)*deg(x)) which is roughly O(n * d^3).
        # For n=35, d~10, this is 35 * 1000 = 35000 ops. Very fast.
        
        for u in range(n):
            for v in adj[u]:
                for x in adj[v]:
                    if x == u: continue
                    for y in adj[x]:
                        if y == u or y == v: continue
                        if u in adj[y]:
                            return False
        return True

    def find_c5_edge(adj, n):
        # Returns an edge (u,v) that is part of a C5, or None
        for u in range(n):
            for v in adj[u]:
                if u > v: continue # Check each edge once
                # Check if removing (u,v) breaks all C5s involving it? 
                # Actually, just check if (u,v) is in a C5.
                # Path u-x-y-v length 3.
                for x in adj[u]:
                    if x == v: continue
                    for y in adj[x]:
                        if y == u or y == v: continue
                        if v in adj[y]:
                            return (u, v)
        return None

    def has_independent_set_size_k(adj, n, k=9):
        # Backtracking to find independent set of size k
        # Order vertices by degree (heuristic)
        # Simple recursive search with pruning
        
        # We want to know if there EXISTS an independent set of size k.
        # If we find one, return True.
        # Pruning: If remaining vertices < k - current_size, return False.
        
        # Use a set for current independent set?
        # Or just list of vertices in set.
        # To speed up, sort vertices by degree ascending? No, high degree vertices are harder to include.
        # Standard heuristic: try vertices with low degree first? Or high?
        # For finding a large independent set, picking low degree vertices is often good.
        
        # Let's use a simple recursive approach.
        # State: current_set, available_vertices
        # To avoid global state complexity, we pass lists.
        
        # Optimization: Precompute neighbors as sets for fast intersection?
        # adj is list of lists. Let's make a set version for the check function.
        adj_sets = [set(adj[i]) for i in range(n)]
        
        def backtrack(current_set, available):
            if len(current_set) == k:
                return True
            if not available:
                return False
            if len(available) < k - len(current_set):
                return False
            
            # Pick a vertex from available
            # Heuristic: Pick vertex with min degree in induced subgraph?
            # For simplicity, pick first one, or random?
            # Let's pick the vertex with the smallest degree in the remaining available set.
            # This tends to leave more options for others.
            
            # Calculate degrees in available subgraph
            # This is O(|available|^2) which might be slow if called many times.
            # Given n=35, it's okay.
            
            min_deg = float('inf')
            min_v = -1
            for v in available:
                deg = 0
                for u in available:
                    if u != v and u in adj_sets[v]:
                        deg += 1
                if deg < min_deg:
                    min_deg = deg
                    min_v = v
            
            v = min_v
            
            # Branch 1: Include v
            new_set = current_set + [v]
            # New available: vertices in available that are not neighbors of v and not v
            new_available = [u for u in available if u != v and u not in adj_sets[v]]
            if backtrack(new_set, new_available):
                return True
            
            # Branch 2: Exclude v
            # We must exclude v.
            new_available2 = [u for u in available if u != v]
            if backtrack(current_set, new_available2):
                return True
            
            return False
        
        initial_available = list(range(n))
        return backtrack([], initial_available)

    def try_construct(n, max_restarts=100):
        for r in range(max_restarts):
            if time.time() > deadline:
                return None
            
            # Initialize random graph
            # Use a moderate probability to start dense enough for low alpha
            p = 0.5 
            adj = [[] for _ in range(n)]
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < p:
                        adj[i].append(j)
                        adj[j].append(i)
            
            # Phase 1: Remove edges to make C5-free
            # We repeatedly find a C5 and remove a random edge from it
            # To be efficient, we can just check and remove.
            # Since n is small, we can just loop.
            for _ in range(100): # Max iterations to remove edges
                c5_edge = find_c5_edge(adj, n)
                if c5_edge is None:
                    break
                u, v = c5_edge
                # Remove edge (u,v)
                adj[u].remove(v)
                adj[v].remove(u)
            
            if find_c5_edge(adj, n) is not None:
                continue # Failed to remove all C5s in 100 steps? Unlikely.

            # Phase 2: Check Alpha
            if not has_independent_set_size_k(adj, n, 9):
                # Valid graph!
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append([i, j])
                return {'n': n, 'edges': edges}
            
            # Phase 3: Try to add edges to reduce Alpha
            # If alpha >= 9, try to add edges that don't create C5.
            # This might reduce alpha.
            changed = True
            while changed:
                changed = False
                # Try to add a random non-edge
                # Iterate through all non-edges? Or pick random?
                # Let's pick random non-edges to add
                for i in range(n):
                    for j in range(i+1, n):
                        if j not in adj[i]:
                            # Check if adding (i,j) creates C5
                            # Temporarily add
                            adj[i].append(j)
                            adj[j].append(i)
                            if find_c5_edge(adj, n) is None:
                                changed = True
                                # Keep it
                            else:
                                # Revert
                                adj[i].remove(j)
                                adj[j].remove(i)
                            # If we found one good edge, we could break and restart loop?
                            # Or just continue adding as many as possible.
                            # Let's break to re-evaluate alpha? No, adding edges only helps alpha.
                            # But we need to check if we broke C5.
                            # The check above ensures we only add if no C5 is created.
                            # So we can just add all possible edges that don't create C5.
                            # But doing this for all pairs is O(n^2) checks.
                            # Let's just add one random valid edge per iteration to keep it simple.
                            if changed:
                                break
                    if changed:
                        break
            
            if not changed:
                continue # Stuck, restart
                
            # Re-check alpha after adding edges
            if not has_independent_set_size_k(adj, n, 9):
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append([i, j])
                return {'n': n, 'edges': edges}
            
            # If still invalid, restart

    # Try to maximize n
    for n in range(35, 27, -1):
        res = try_construct(n)
        if res:
            return res
            
    # Fallback to baseline
    return {'n': 28, 'edges': [[4 * b + i, 4 * b + j] for b in range(7) for i in range(4) for j in range(i + 1, 4)]}

# EVOLVE-BLOCK-END
