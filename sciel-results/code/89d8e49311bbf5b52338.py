import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second margin
    
    def is_c4_free(n, edges):
        # Build adjacency list
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Check for C4: no two vertices have more than 1 common neighbor
        for u in range(n):
            for v in range(u + 1, n):
                common = adj[u].intersection(adj[v])
                if len(common) > 1:
                    return False
        return True

    def check_alpha_less_than(n, edges, k=11):
        # Check if there is an independent set of size k
        # Use a simple branch and bound
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Sort vertices by degree
        deg = [len(adj[i]) for i in range(n)]
        order = sorted(range(n), key=lambda x: deg[x], reverse=True)
        
        # Backtracking search for independent set of size k
        def backtrack(start_idx, current_set, candidates):
            if len(current_set) >= k:
                return True
            if start_idx >= len(candidates):
                return False
            
            # Pruning: if remaining candidates + current size < k, fail
            if len(current_set) + (len(candidates) - start_idx) < k:
                return False
            
            for i in range(start_idx, len(candidates)):
                v = candidates[i]
                # If v can be added
                if all(v not in adj[u] for u in current_set):
                    if backtrack(i + 1, current_set + [v], candidates):
                        return True
            return False
        
        # Initial candidates are all vertices
        # We can optimize by only considering vertices with low degree first?
        # Actually, for finding a large independent set, high degree vertices are less likely to be in it?
        # No, we want to find IF it exists.
        # Standard Max Independent Set heuristic.
        
        # For n=43, k=11, this might be slow if alpha is close to 11.
        # But we expect alpha to be small.
        
        # A faster check: Randomized greedy to find large independent set.
        # If we can't find one of size 11 in many tries, assume it's not there.
        
        best_alpha = 0
        for _ in range(50):
            remaining = list(range(n))
            current = []
            while remaining:
                # Pick a random vertex from remaining
                idx = random.randint(0, len(remaining) - 1)
                v = remaining[idx]
                current.append(v)
                # Remove v and its neighbors from remaining
                neighbors = adj[v]
                remaining = [u for u in remaining if u != v and u not in neighbors]
            best_alpha = max(best_alpha, len(current))
            
        if best_alpha >= 11:
            return False
        return True

    def greedy_construct(n_target):
        edges = []
        adj = [set() for _ in range(n_target)]
        
        for i in range(n_target):
            if time.time() > deadline:
                break
            
            # Build conflict graph H on vertices 0..i-1
            # H has edge (u, v) if |N(u) intersect N(v)| > 0
            # We need to find an independent set S in H
            # S will be the neighbors of i
            
            if i == 0:
                continue
            
            # Compute H
            H_adj = [set() for _ in range(i)]
            for u in range(i):
                for v in range(u + 1, i):
                    if adj[u].intersection(adj[v]):
                        H_adj[u].add(v)
                        H_adj[v].add(u)
            
            # Find a large independent set in H
            # Greedy: pick vertex with min degree in H, add to S, remove from H
            remaining_H = list(range(i))
            S = []
            
            # Randomize order to get different structures
            random.shuffle(remaining_H)
            
            while remaining_H:
                # Find vertex with min degree in remaining_H
                min_deg = float('inf')
                min_idx = -1
                for u in remaining_H:
                    deg_u = len([v for v in remaining_H if v in H_adj[u]])
                    if deg_u < min_deg:
                        min_deg = deg_u
                        min_idx = u
                
                if min_idx == -1:
                    break
                    
                v = remaining_H[min_idx]
                S.append(v)
                
                # Remove v and its neighbors from remaining_H
                neighbors_in_H = H_adj[v]
                remaining_H = [u for u in remaining_H if u != v and u not in neighbors_in_H]
            
            # Connect i to all in S
            for v in S:
                edges.append([i, v])
                adj[i].add(v)
                adj[v].add(i)
                
        return n_target, edges

    # Try to construct a graph with n=43
    # We might need to restart if alpha is too large
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        n, edges = greedy_construct(43)
        
        # Verify C4-free
        if not is_c4_free(n, edges):
            continue
            
        # Check alpha
        if check_alpha_less_than(n, edges, 11):
            return {'n': n, 'edges': edges}
            
    # If we failed to get a good graph with 43, try smaller n
    for n_target in range(42, 29, -1):
        if time.time() > deadline:
            break
        n, edges = greedy_construct(n_target)
        if is_c4_free(n, edges) and check_alpha_less_than(n, edges, 11):
            return {'n': n, 'edges': edges}
            
    # Fallback to seed
    edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        edges += [[a, b], [a, c], [b, c]]
    return {'n': 30, 'edges': edges}

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
