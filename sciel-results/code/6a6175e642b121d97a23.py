import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = set()
        # Path edges
        for i in range(n - 1):
            u, v = order[i], order[i + 1]
            edges.add((min(u, v), max(u, v)))
        # Random edges
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return list(edges)

    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    all_results = []

    def try_merge(paths):
        changed = True
        while changed:
            changed = False
            for i in range(len(paths)):
                if not paths[i]:
                    continue
                for j in range(i + 1, len(paths)):
                    if not paths[j]:
                        continue
                    p1, p2 = paths[i], paths[j]
                    set1 = set(p1)
                    set2 = set(p2)
                    common = set1.intersection(set2)
                    if len(common) != 1:
                        continue
                    cv = list(common)[0]
                    if cv in (p1[0], p1[-1]) and cv in (p2[0], p2[-1]):
                        # Try to merge
                        new_path = None
                        if p1[-1] == p2[0]:
                            new_path = p1 + p2[1:]
                        elif p1[0] == p2[-1]:
                            new_path = p2 + p1[1:]
                        elif p1[-1] == p2[-1]:
                            new_path = p1 + p2[-2::-1]
                        elif p1[0] == p2[0]:
                            new_path = p2[::-1] + p1[1:]
                        
                        if new_path and len(new_path) == len(set(new_path)):
                            paths[i] = new_path
                            paths[j] = []
                            changed = True
                            break
                if changed:
                    break
            if changed:
                paths = [p for p in paths if p]
        return paths

    for spec in specs:
        n, p, seed = spec
        edges = instance(n, p, seed)
        
        # Build adjacency
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Calculate lower bound
        deg = {i: 0 for i in range(n)}
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        odd_count = sum(1 for d in deg.values() if d % 2 != 0)
        lower_bound = max(1, odd_count // 2)
        
        best_paths = None
        best_count = float('inf')
        
        # Heuristic: Greedy Longest Path with Randomized Restart
        max_restarts = 500
        # If graph is small, allow more restarts
        if n <= 20:
            max_restarts = 1000
        elif n <= 25:
            max_restarts = 800
        else:
            max_restarts = 600

        # Shuffle start order for diversity in each restart?
        # The greedy path finder will try all starts, so start order doesn't matter for the *search*.
        # However, the *result* of the greedy path might depend on tie-breaking.
        # We can introduce randomness in tie-breaking.

        for r in range(max_restarts):
            if time.time() > deadline - 5.0:
                break
            if best_count == lower_bound:
                break
                
            curr_edges = set(edges)
            curr_paths = []
            
            # Randomize seed for tie-breaking in this restart
            rng_restart = random.Random(r + seed)
            
            while curr_edges:
                if time.time() > deadline - 2.0:
                    curr_paths = None
                    break
                
                # Find the longest simple path in curr_edges
                # We try starting from every vertex with degree > 0
                active_vertices = [v for v in range(n) if adj[v] and any((min(v, nb), max(v, nb)) in curr_edges for nb in adj[v])]
                
                if not active_vertices:
                    break
                
                best_path_in_iter = []
                
                # To save time, we can skip vertices that are likely not to yield long paths?
                # But N is small. Let's try all.
                for s in active_vertices:
                    # Greedy Warnsdorff from s
                    path = [s]
                    visited = {s}
                    curr_node = s
                    
                    while True:
                        candidates = []
                        for nb in adj[curr_node]:
                            if nb in visited:
                                continue
                            e = (min(curr_node, nb), max(curr_node, nb))
                            if e in curr_edges:
                                # Calculate unvisited degree of nb in curr_edges
                                unvis_deg = 0
                                for m in adj[nb]:
                                    if m in visited:
                                        continue
                                    e2 = (min(nb, m), max(nb, m))
                                    if e2 in curr_edges:
                                        unvis_deg += 1
                                candidates.append((unvis_deg, nb))
                        
                        if not candidates:
                            break
                        
                        # Sort by unvisited degree ascending (Warnsdorff)
                        candidates.sort(key=lambda x: x[0])
                        
                        # Randomize tie-breaking among those with minimum degree
                        min_deg = candidates[0][0]
                        min_candidates = [c for c in candidates if c[0] == min_deg]
                        next_node = rng_restart.choice(min_candidates)[1]
                        
                        path.append(next_node)
                        visited.add(next_node)
                        curr_node = next_node
                    
                    if len(path) > len(best_path_in_iter):
                        best_path_in_iter = path[:]

                if best_path_in_iter and len(best_path_in_iter) > 1:
                    curr_paths.append(best_path_in_iter)
                    # Remove edges from curr_edges
                    for i in range(len(best_path_in_iter) - 1):
                        u, v = best_path_in_iter[i], best_path_in_iter[i + 1]
                        e = (min(u, v), max(u, v))
                        curr_edges.discard(e)
                else:
                    # Should not happen if curr_edges is not empty, but safe fallback
                    # This implies no path of length > 1 could be found, which is impossible if there's an edge.
                    # Unless the graph is disconnected? But we start from active vertices.
                    # Wait, if active_vertices is not empty, we find at least one edge.
                    # The loop `while True` in greedy always takes at least one step if candidates exist.
                    # So best_path_in_iter should always have length >= 2.
                    if curr_edges:
                        e = next(iter(curr_edges))
                        curr_paths.append(list(e))
                        curr_edges.discard(e)
            
            if curr_paths and len(curr_paths) < best_count:
                # Apply merging heuristic
                merged_paths = try_merge(curr_paths)
                if len(merged_paths) < best_count:
                    best_count = len(merged_paths)
                    best_paths = merged_paths

        # If no best_paths found due to timeout, fallback to edge list
        if best_paths is None:
            best_paths = [list(e) for e in edges]
            # Try to merge even the fallback
            best_paths = try_merge(best_paths)
            
        all_results.append(best_paths)

    return all_results
