import time
import random
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def solve_graph(n, edges, time_limit):
        if not edges:
            return []
            
        # Build adjacency and degree
        adj = defaultdict(set)
        degree = defaultdict(int)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            degree[u] += 1
            degree[v] += 1
            
        # Initial paths: each edge is a path
        # Represent path as tuple of vertices
        initial_paths = [tuple(sorted((u, v))) for u, v in edges]
        
        best_paths = initial_paths[:]
        best_len = len(best_paths)
        
        # Heuristic: Greedy merge with randomization (Simulated Annealing-like)
        # We try to merge two paths P1 and P2 if they are edge-disjoint (always true)
        # and vertex-disjoint except for one shared endpoint.
        # If P1 ends at x and P2 starts at x, we can merge them into P1 + P2[1:].
        
        def can_merge(p1, p2):
            # p1 and p2 are tuples
            # Check if they share exactly one vertex, and that vertex is an endpoint of both.
            # Since they are edge-disjoint paths in the same graph, they can share vertices.
            # But for a simple path cover, we want to merge them into a single simple path.
            # This requires that the internal vertices of the new path are distinct.
            
            # Endpoints of p1: p1[0], p1[-1]
            # Endpoints of p2: p2[0], p2[-1]
            
            # Possible merges:
            # 1. p1[-1] == p2[0] -> p1 + p2[1:]
            # 2. p1[-1] == p2[-1] -> p1 + p2[-2::-1] (reverse p2)
            # 3. p1[0] == p2[0] -> p1[1:] + p2
            # 4. p1[0] == p2[-1] -> p1[1::-1] + p2 (reverse p1)
            
            # We need to ensure the resulting path is simple (no repeated vertices).
            
            def is_simple(path):
                return len(set(path)) == len(path)
                
            candidates = []
            
            if p1[-1] == p2[0]:
                merged = p1 + p2[1:]
                if is_simple(merged):
                    candidates.append(merged)
            if p1[-1] == p2[-1]:
                merged = p1 + p2[-2::-1]
                if is_simple(merged):
                    candidates.append(merged)
            if p1[0] == p2[0]:
                merged = p1[1:] + p2
                if is_simple(merged):
                    candidates.append(merged)
            if p1[0] == p2[-1]:
                merged = p1[1::-1] + p2
                if is_simple(merged):
                    candidates.append(merged)
                
            return candidates

        # Local Search: Randomized greedy
        # We maintain a list of paths.
        # We repeatedly pick two random paths and try to merge them.
        # If successful, we replace them with the merged path.
        # We use a temperature parameter to accept worse solutions? No, merging always reduces count.
        # The challenge is getting stuck in local minima where no merge is possible, but a sequence of swaps/merges would allow more merges.
        # Since we only merge, we never increase the count. This is a "hill climbing" on the number of paths (minimizing).
        # To escape local minima, we can use "restarts" with different random orderings or perturbations.
        
        # A more powerful approach: 
        # 1. Greedily merge as much as possible.
        # 2. If stuck, try "unmerging" a path and re-merging differently? 
        #    Or simply restart with a different random seed/order.
        
        # Given the small size (N=20, 25, 30), we can afford multiple restarts.
        
        def greedy_merge(paths, rng):
            # Sort paths by length descending? Or random?
            # Let's try random order to break symmetries.
            paths = paths[:]
            # Shuffle
            rng.shuffle(paths)
            
            merged = True
            while merged:
                merged = False
                # Try all pairs? O(N^2 * L). N is number of paths, L is path length.
                # N starts at ~M (edges). M is small (< 100).
                # We can do this efficiently.
                
                # We need to be careful: if we merge p1 and p2, we remove them and add p12.
                # We should try to merge as much as possible.
                
                # To avoid O(N^2) per step being too slow if N is large, 
                # but N is at most 100, so N^2 is 10,000. 
                # Each merge check is O(L). L is at most N.
                # So O(N^3) roughly. 100^3 = 1,000,000 operations. Very fast.
                
                i = 0
                while i < len(paths):
                    j = i + 1
                    while j < len(paths):
                        cands = can_merge(paths[i], paths[j])
                        if cands:
                            # Pick a candidate (usually only one, or symmetric)
                            new_path = cands[0]
                            # Remove i and j, insert new_path
                            # Note: removing from list while iterating is tricky.
                            # Let's collect new list.
                            new_paths = []
                            for k in range(len(paths)):
                                if k == i or k == j:
                                    continue
                                new_paths.append(paths[k])
                            new_paths.append(new_path)
                            paths = new_paths
                            merged = True
                            # Restart the outer loop because indices changed
                            i = 0
                            break
                        j += 1
                    if merged:
                        continue
                    i += 1
            return paths

        # Strategy:
        # 1. Run greedy_merge with several different random seeds/orderings.
        # 2. Keep the best result.
        
        best_result = initial_paths[:]
        
        # Time budget per restart
        # Total time limit for this graph: time_limit
        # Let's allow 20 restarts if time permits.
        
        num_restarts = 50
        restart_time_budget = time_limit / num_restarts
        
        for r in range(num_restarts):
            if time.time() > start_time + time_limit:
                break
                
            rng = random.Random(r)
            # Initial paths are fixed (edges). 
            # The randomness comes from the order in which we attempt merges.
            # My greedy_merge shuffles the list.
            
            res = greedy_merge(initial_paths, rng)
            if len(res) < len(best_result):
                best_result = res
                # If we hit the theoretical lower bound, stop?
                # Lower bound: max(1, (sum(deg-1) for deg>1) / 2 + isolated edges?)
                # Actually, Gallai's bound is floor((n-1)/2) or similar?
                # For now, just keep improving.
                
            # Early stop if we can't do better than 1? (Hamiltonian path)
            if len(best_result) == 1:
                break

        return best_result

    results = []
    specs = [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]
    
    # Allocate time
    # Graph 1: 20 nodes, ~20-30 edges. Fast.
    # Graph 2: 25 nodes, ~30-40 edges. Fast.
    # Graph 3: 30 nodes, ~40-50 edges. Fast.
    # Total 120s. We have plenty of time for 50 restarts on each.
    
    for spec in specs:
        if time.time() > deadline:
            n, p, seed = spec
            edges = instance(n, p, seed)
            results.append([list(e) for e in edges])
            continue
            
        n, p, seed = spec
        edges = instance(n, p, seed)
        
        # Allocate remaining time / remaining graphs
        remaining_time = deadline - time.time()
        graphs_left = specs.index(spec) - (-1) # 0, 1, 2
        # Actually, let's just give each graph 1/3 of the remaining time? 
        # Or fixed time per graph?
        # 3 graphs. 110s total. ~36s each.
        time_for_graph = min(35.0, remaining_time / (specs.index(spec) + 1))
        
        paths = solve_graph(n, edges, time_for_graph)
        # Convert tuples to lists as required by "vertex lists"
        results.append([list(p) for p in paths])

    return results
