import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    DEADLINE = 145.0

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))

    def decompose_graph(n, edges, time_limit):
        if not edges:
            return []

        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        edge_set = set(edges)
        
        # Calculate number of odd degree vertices
        degrees = defaultdict(int)
        for u, v in edges:
            degrees[u] += 1
            degrees[v] += 1
        odd_count = sum(1 for d in degrees.values() if d % 2 == 1)
        
        # Lower bound for trails is odd_count // 2. 
        # For simple paths, it's often higher. 
        # We start from max(1, odd_count // 2)
        
        start_k = max(1, odd_count // 2)
        
        # We will try to find a solution with K paths.
        # We use a recursive backtracking search.
        
        def find_paths(remaining_edges, k, current_paths, timeout):
            """
            Recursively find k simple paths covering remaining_edges.
            Returns a list of paths if successful, None if not.
            """
            if time.time() > timeout:
                return None
            
            if k == 0:
                if not remaining_edges:
                    return current_paths
                return None
            
            if not remaining_edges:
                # We have covered all edges with fewer than k paths.
                # This is a valid solution (we can just return what we have).
                # But we want to minimize, so if we found a solution with < k, 
                # it's better. However, we are iterating k upwards, so the first 
                # success is the minimum.
                # Wait, if we reach here, it means we used < k paths.
                # This is fine.
                return current_paths

            # Heuristic: Pick an edge to be part of the next path.
            # To reduce branching, we pick an edge and try to extend it into a path.
            
            # Choose a "pivot" edge. 
            # Strategy: Pick an edge incident to a vertex with high degree in remaining graph?
            # Or just pick a random edge to encourage diversity in restarts?
            # For deterministic backtracking, we should be systematic.
            
            # Let's pick the first edge in the set (sorted for determinism)
            # But we need to generate candidate paths containing this edge.
            
            # A better heuristic: 
            # 1. Find all maximal simple paths in the remaining graph? Too expensive.
            # 2. Generate a set of candidate paths using greedy heuristics.
            
            candidates = []
            
            # Generate candidate paths
            # We try to find long paths.
            # We pick a starting node.
            
            # To ensure we don't miss solutions, we need to be careful.
            # However, with small graphs, a greedy set of candidates might work.
            
            # Let's try this: 
            # Pick a node with the highest degree in the remaining graph.
            # Try to find long paths starting from it.
            
            rem_adj = defaultdict(set)
            for u, v in remaining_edges:
                rem_adj[u].add(v)
                rem_adj[v].add(u)
                
            if not rem_adj:
                return current_paths

            # Find node with max degree
            max_deg_node = max(rem_adj.keys(), key=lambda x: len(rem_adj[x]))
            
            # Generate paths starting from max_deg_node
            # We use a DFS to find simple paths up to a certain length or until stuck.
            
            def dfs_path(start, remaining, max_depth=10):
                """Find simple paths starting from start."""
                paths = []
                def _dfs(cur, path, visited, used_edges):
                    if time.time() > timeout:
                        return
                    # Record path if length > 1
                    if len(path) > 1:
                        paths.append(list(path))
                    
                    if len(path) >= max_depth:
                        return
                        
                    for nb in rem_adj[cur]:
                        e = (min(cur, nb), max(cur, nb))
                        if e in remaining and e not in used_edges and nb not in visited:
                            _dfs(nb, path + [nb], visited | {nb}, used_edges | {e})
                
                _dfs(max_deg_node, [max_deg_node], {max_deg_node}, set())
                return paths

            # This might generate too many paths.
            # Let's limit the number of candidates.
            
            # Alternative: Just pick ONE good path, remove it, and recurse.
            # If it fails, backtrack.
            # To backtrack, we need to try other paths.
            
            # Let's generate a small set of diverse paths.
            # 1. Greedy Longest
            # 2. Greedy Shortest (to cover hard edges?)
            
            # For now, let's just try a few random walks/paths.
            
            candidate_paths = []
            
            # Try to find a path using DFS from max_deg_node
            # We'll collect a few distinct ones.
            
            def get_candidate_paths():
                cands = []
                # Try a few random starts if max_deg is not unique or to add diversity
                starts = list(rem_adj.keys())
                random.shuffle(starts)
                
                for s in starts[:5]: # Try 5 starts
                    # DFS for one long path
                    stack = [(s, [s], {s}, set())]
                    best = None
                    while stack:
                        cur, path, vis, used = stack.pop()
                        if len(path) > (len(best) if best else 0):
                            best = list(path)
                        
                        if len(path) > 8: # Limit depth for speed
                            continue
                            
                        for nb in rem_adj[cur]:
                            e = (min(cur, nb), max(cur, nb))
                            if e in remaining and e not in used and nb not in vis:
                                stack.append((nb, path + [nb], vis | {nb}, used | {e}))
                    if best and len(best) > 1:
                        cands.append(best)
                return cands

            cands = get_candidate_paths()
            
            # Also try the specific edge (max_deg_node, neighbor)
            if not cands:
                # Fallback: single edge
                for nb in rem_adj[max_deg_node]:
                    e = (min(max_deg_node, nb), max(max_deg_node, nb))
                    if e in remaining_edges:
                        cands.append([max_deg_node, nb])
                        break
            
            # Sort candidates by length descending
            cands.sort(key=len, reverse=True)
            
            # Limit candidates to avoid explosion
            cands = cands[:10]
            
            for path in cands:
                # Verify path is valid in remaining_edges
                valid = True
                path_edges = set()
                for i in range(len(path) - 1):
                    e = (min(path[i], path[i+1]), max(path[i], path[i+1]))
                    if e not in remaining_edges:
                        valid = False
                        break
                    path_edges.add(e)
                
                if not valid:
                    continue
                
                new_remaining = remaining_edges - path_edges
                result = find_paths(new_remaining, k - 1, current_paths + [path], timeout)
                if result is not None:
                    return result
            
            return None

        # Try increasing K
        best_result = None
        for k in range(start_k, n // 2 + 1):
            timeout = min(time_limit, time.time() + 10) # 10s per K attempt
            result = find_paths(edge_set, k, [], timeout)
            if result is not None:
                best_result = result
                break
        
        if best_result is None:
            # Fallback: single edges
            best_result = [[u, v] for u, v in edges]
            
        return best_result

    results = []
    for idx, spec in enumerate(specs):
        if time.time() > DEADLINE - 10:
            n, p, seed = spec
            edges = instance(n, p, seed)
            results.append([[u, v] for u, v in edges])
            continue

        n, p, seed = spec
        edges = instance(n, p, seed)

        remaining_time = DEADLINE - time.time() - 5
        time_for_this = remaining_time / (len(specs) - idx)

        paths = decompose_graph(n, edges, time_limit=time_for_this)
        results.append(paths)

    return results
