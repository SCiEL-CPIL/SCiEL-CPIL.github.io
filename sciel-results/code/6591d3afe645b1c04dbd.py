import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin for 150s limit

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
        
        remaining_edges = set(edges)
        best_paths = None
        best_count = float('inf')
        
        # Heuristic: Greedy Longest Path with Restart
        # We try multiple random restarts to find a minimal path cover
        
        max_restarts = 50
        # If graph is small, allow more restarts
        if n <= 20:
            max_restarts = 100
        elif n <= 25:
            max_restarts = 80
        else:
            max_restarts = 60

        def extend_path_bidirectional(start_node, curr_edges_set, adj_dict, max_depth=30, beam_width=8):
            """Extend a path bidirectionally from start_node using beam search."""
            # Forward beam
            fwd_beam = [([start_node], {start_node})]
            # Backward beam (empty initially, will be populated)
            
            best_path = [start_node]
            
            # First, do a forward beam search
            for d in range(max_depth):
                new_fwd = []
                for path, visited in fwd_beam:
                    node = path[-1]
                    valid_neighbors = []
                    for nb in adj_dict[node]:
                        if nb in visited:
                            continue
                        e = (min(node, nb), max(node, nb))
                        if e in curr_edges_set:
                            valid_neighbors.append(nb)
                    
                    if not valid_neighbors:
                        if len(path) > len(best_path):
                            best_path = path[:]
                        continue
                    
                    # Score by Warnsdorff's rule
                    scored = []
                    for nb in valid_neighbors:
                        count = 0
                        for m in adj_dict[nb]:
                            if m in visited:
                                continue
                            e2 = (min(nb, m), max(nb, m))
                            if e2 in curr_edges_set:
                                count += 1
                        scored.append((count, nb))
                    scored.sort()
                    
                    for _, nb in scored[:beam_width]:
                        new_path = path + [nb]
                        new_visited = visited | {nb}
                        new_fwd.append((new_path, new_visited))
                
                if not new_fwd:
                    break
                
                new_fwd.sort(key=lambda x: -len(x[0]))
                fwd_beam = new_fwd[:beam_width]
                
                longest = max(len(p) for p, v in fwd_beam)
                if longest > len(best_path):
                    best_path = fwd_beam[0][0][:]
            
            # Now try to extend from the other end of best_path
            if len(best_path) > 1:
                end_node = best_path[0]
                visited_set = set(best_path)
                # Try extending backwards from the first node
                back_beam = [([best_path[0]], visited_set)]
                for d in range(max_depth // 2):
                    new_back = []
                    for path, visited in back_beam:
                        node = path[-1]
                        valid_neighbors = []
                        for nb in adj_dict[node]:
                            if nb in visited:
                                continue
                            e = (min(node, nb), max(node, nb))
                            if e in curr_edges_set:
                                valid_neighbors.append(nb)
                        
                        if not valid_neighbors:
                            continue
                        
                        scored = []
                        for nb in valid_neighbors:
                            count = 0
                            for m in adj_dict[nb]:
                                if m in visited:
                                    continue
                                e2 = (min(nb, m), max(nb, m))
                                if e2 in curr_edges_set:
                                    count += 1
                            scored.append((count, nb))
                        scored.sort()
                        
                        for _, nb in scored[:beam_width]:
                            new_path = path + [nb]
                            new_visited = visited | {nb}
                            new_back.append((new_path, new_visited))
                    
                    if not new_back:
                        break
                    
                    new_back.sort(key=lambda x: -len(x[0]))
                    back_beam = new_back[:beam_width]
                    
                    if back_beam:
                        # Combine: reverse the back extension and prepend to best_path
                        extended = list(reversed(back_beam[0][0])) + best_path[1:]
                        # Check if this is a valid simple path
                        if len(extended) == len(set(extended)) and len(extended) > len(best_path):
                            best_path = extended
            
            return best_path

        for r in range(max_restarts):
            if time.time() > deadline - 5.0:
                break
            if best_count == lower_bound:
                break
                
            curr_edges = set(remaining_edges)
            curr_paths = []
            
            # Randomize start selection order for diversity
            starts = list(range(n))
            random.shuffle(starts)
            
            # Greedy extraction
            while curr_edges:
                if time.time() > deadline - 2.0:
                    curr_paths = None
                    break
                
                # Find longest simple path in curr_edges
                # Use beam search
                best_path = None
                beam_width = 10
                max_depth = n
                
                # Try to start from high degree vertices first?
                # Or just iterate through all possible starts?
                # For speed, we can limit starts to those with remaining degree > 0
                active_starts = [s for s in starts if adj[s] and any((min(s, nb), max(s, nb)) in curr_edges for nb in adj[s])]
                
                if not active_starts:
                    break
                    
                # Pick a few random starts from active_starts to save time?
                # Or just try the first few?
                # Let's try the first 10 active starts
                
                for s in active_starts[:10]:
                    # Use bidirectional extension
                    local_best = extend_path_bidirectional(s, curr_edges, adj, max_depth=n, beam_width=10)
                    if local_best:
                        if best_path is None or len(local_best) > len(best_path):
                            best_path = local_best

                if best_path and len(best_path) > 1:
                    # Verify the path is valid (all edges exist and are in curr_edges)
                    valid = True
                    for i in range(len(best_path) - 1):
                        u, v = best_path[i], best_path[i + 1]
                        e = (min(u, v), max(u, v))
                        if e not in curr_edges:
                            valid = False
                            break
                    if valid:
                        curr_paths.append(best_path)
                        for i in range(len(best_path) - 1):
                            u, v = best_path[i], best_path[i + 1]
                            e = (min(u, v), max(u, v))
                            curr_edges.discard(e)
                    else:
                        # Fallback: find a valid path
                        if curr_edges:
                            e = next(iter(curr_edges))
                            curr_paths.append(list(e))
                            curr_edges.discard(e)
                else:
                    # Single edge fallback
                    if curr_edges:
                        e = next(iter(curr_edges))
                        curr_paths.append(list(e))
                        curr_edges.discard(e)
            
            if curr_paths and len(curr_paths) < best_count:
                best_count = len(curr_paths)
                best_paths = curr_paths
                
        # If no best_paths found due to timeout, fallback to edge list
        if best_paths is None:
            best_paths = [list(e) for e in edges]
            
        all_results.append(best_paths)

    return all_results
