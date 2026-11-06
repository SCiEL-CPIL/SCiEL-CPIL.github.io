import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin for 120s limit

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
        
        # Restart counts
        if n <= 20:
            max_restarts = 200
        elif n <= 25:
            max_restarts = 100
        else:
            max_restarts = 50

        def merge_paths(paths):
            """Merge paths that share an endpoint."""
            changed = True
            while changed:
                changed = False
                new_paths = []
                used = [False] * len(paths)
                
                for i in range(len(paths)):
                    if used[i]:
                        continue
                    p_i = paths[i]
                    if len(p_i) < 2:
                        new_paths.append(p_i)
                        continue
                        
                    for j in range(i + 1, len(paths)):
                        if used[j]:
                            continue
                        p_j = paths[j]
                        
                        # Check if they can be merged
                        merged = None
                        # Case 1: p_i end == p_j start
                        if p_i[-1] == p_j[0]:
                            candidate = p_i + p_j[1:]
                            if len(candidate) == len(set(candidate)):
                                merged = candidate
                        # Case 2: p_i end == p_j end
                        if merged is None and p_i[-1] == p_j[-1]:
                            candidate = p_i + list(reversed(p_j[1:]))
                            if len(candidate) == len(set(candidate)):
                                merged = candidate
                        # Case 3: p_i start == p_j start
                        if merged is None and p_i[0] == p_j[0]:
                            candidate = list(reversed(p_i[1:])) + p_j
                            if len(candidate) == len(set(candidate)):
                                merged = candidate
                        # Case 4: p_i start == p_j end
                        if merged is None and p_i[0] == p_j[-1]:
                            candidate = p_i[1:] + list(reversed(p_j))
                            if len(candidate) == len(set(candidate)):
                                merged = candidate

                        if merged:
                            new_paths.append(merged)
                            used[i] = True
                            used[j] = True
                            changed = True
                            break # Restart loop for this i
                    
                    if not used[i]:
                        new_paths.append(p_i)
                
                paths = new_paths
                
            return paths

        for r in range(max_restarts):
            if time.time() > deadline - 5.0:
                break
            if best_count == lower_bound:
                break
                
            curr_edges = set(edges)
            curr_paths = []
            
            # Randomize tie-breaking
            rng_local = random.Random(r)
            
            while curr_edges:
                if time.time() > deadline - 2.0:
                    break
                
                best_path = None
                best_len = 0
                
                # Calculate degrees
                deg_curr = defaultdict(int)
                for u, v in curr_edges:
                    deg_curr[u] += 1
                    deg_curr[v] += 1
                
                active_nodes = [u for u in deg_curr if deg_curr[u] > 0]
                if not active_nodes:
                    break
                
                # Sort by degree
                active_nodes.sort(key=lambda x: deg_curr[x])
                
                # Try top k starts
                k_starts = 10
                for s in active_nodes[:k_starts]:
                    # Beam Search
                    beam = [([s], {s})]
                    beam_width = 20
                    
                    for d in range(n):
                        new_beam = []
                        for path, visited in beam:
                            u = path[-1]
                            candidates = []
                            for nb in adj[u]:
                                if nb in visited:
                                    continue
                                e = (min(u, nb), max(u, nb))
                                if e in curr_edges:
                                    # Score: prefer neighbors with fewer remaining options (Warnsdorff-like)
                                    score = 0
                                    for m in adj[nb]:
                                        if m in visited:
                                            continue
                                        e2 = (min(nb, m), max(nb, m))
                                        if e2 in curr_edges:
                                            score += 1
                                    # Add noise
                                    score += rng_local.random() * 0.1
                                    candidates.append((score, nb))
                            
                            if not candidates:
                                if len(path) > best_len:
                                    best_len = len(path)
                                    best_path = path[:]
                                continue
                            
                            candidates.sort()
                            for _, nb in candidates[:beam_width]:
                                new_beam.append((path + [nb], visited | {nb}))
                        
                        if not new_beam:
                            break
                            
                        new_beam.sort(key=lambda x: -len(x[0]))
                        beam = new_beam[:beam_width]
                        
                        if beam and len(beam[0][0]) > best_len:
                            best_len = len(beam[0][0])
                            best_path = beam[0][0][:]
                    
                if best_path and len(best_path) > 1:
                    # Verify
                    valid = True
                    for i in range(len(best_path) - 1):
                        u, v = best_path[i], best_path[i+1]
                        e = (min(u,v), max(u,v))
                        if e not in curr_edges:
                            valid = False
                            break
                    if valid:
                        curr_paths.append(best_path)
                        for i in range(len(best_path) - 1):
                            u, v = best_path[i], best_path[i+1]
                            e = (min(u,v), max(u,v))
                            curr_edges.discard(e)
                    else:
                        e = next(iter(curr_edges))
                        curr_paths.append(list(e))
                        curr_edges.discard(e)
                else:
                    if curr_edges:
                        e = next(iter(curr_edges))
                        curr_paths.append(list(e))
                        curr_edges.discard(e)
            
            if curr_paths:
                # Merge
                merged_paths = merge_paths(curr_paths)
                if len(merged_paths) < best_count:
                    best_count = len(merged_paths)
                    best_paths = merged_paths
                    
        if best_paths is None:
            best_paths = [list(e) for e in edges]
            
        all_results.append(best_paths)

    return all_results
