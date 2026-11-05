import time
import random
from collections import defaultdict
from itertools import combinations

def construct():
    """
    Decomposes the edges of three fixed instance graphs into the minimum number of
    edge-disjoint simple paths.
    
    Returns:
        A list of three lists of paths. Each path is a list of vertices.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def validate_paths(paths, edges):
        """Validate that paths are simple, edge-disjoint, and cover all edges."""
        if not paths:
            return len(edges) == 0
        
        edge_set = set(edges)
        used_edges = set()
        
        for path in paths:
            if len(path) < 2:
                continue
            # Check simple path
            if len(set(path)) != len(path):
                return False
            
            # Check edges exist and are not reused
            for j in range(len(path) - 1):
                u, v = path[j], path[j+1]
                e = tuple(sorted((u, v)))
                if e not in edge_set:
                    return False
                if e in used_edges:
                    return False
                used_edges.add(e)
        
        # Check all edges are covered
        if used_edges != edge_set:
            return False
        
        return True

    def max_matching_bipartite(left_nodes, right_nodes, adjacency):
        """
        Find maximum bipartite matching using Hopcroft-Karp algorithm.
        left_nodes, right_nodes: lists of node IDs
        adjacency: dict mapping left node -> set of right nodes
        Returns: list of (left, right) pairs in the matching
        """
        if not left_nodes or not right_nodes:
            return []
        
        # Build node index maps
        left_index = {node: i for i, node in enumerate(left_nodes)}
        right_index = {node: i for i, node in enumerate(right_nodes)}
        
        n_left = len(left_nodes)
        n_right = len(right_nodes)
        
        # Build adjacency list in indexed form
        adj = [[] for _ in range(n_left)]
        for u in left_nodes:
            ui = left_index[u]
            for v in adjacency.get(u, set()):
                if v in right_index:
                    adj[ui].append(right_index[v])
        
        # Hopcroft-Karp
        match_left = [-1] * n_left
        match_right = [-1] * n_right
        dist = [0] * n_left
        
        def bfs():
            queue = []
            for u in range(n_left):
                if match_left[u] == -1:
                    dist[u] = 0
                    queue.append(u)
                else:
                    dist[u] = float('inf')
            
            found = False
            while queue:
                u = queue.pop(0)
                for v in adj[u]:
                    if match_right[v] == -1:
                        found = True
                    elif dist[match_right[v]] == float('inf'):
                        dist[match_right[v]] = dist[u] + 1
                        queue.append(match_right[v])
            return found
        
        def dfs(u):
            for v in adj[u]:
                if match_right[v] == -1 or (dist[match_right[v]] == dist[u] + 1 and dfs(match_right[v])):
                    match_left[u] = v
                    match_right[v] = u
                    return True
            dist[u] = float('inf')
            return False
        
        matching = []
        while bfs():
            for u in range(n_left):
                if match_left[u] == -1:
                    if dfs(u):
                        pass
        
        for u in range(n_left):
            if match_left[u] != -1:
                matching.append((left_nodes[u], right_nodes[match_left[u]]))
        
        return matching

    def greedy_initial_decomposition(edges, seed):
        """
        Create an initial path decomposition using greedy approach.
        Returns list of paths.
        """
        rng = random.Random(seed)
        remaining = set(edges)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        paths = []
        
        while remaining:
            # Pick a random edge to start
            edge = rng.choice(list(remaining))
            remaining.remove(edge)
            u, v = edge
            path = [u, v]
            
            # Extend the path greedily
            changed = True
            while changed:
                changed = False
                # Try extending from both ends
                for end in [0, 1]:  # 0 = front, 1 = back
                    if end == 0:
                        current = path[0]
                        candidates = [nb for nb in adj[current] 
                                      if nb not in path and ((current, nb) in remaining or (nb, current) in remaining)]
                        if candidates:
                            nb = rng.choice(candidates)
                            if (current, nb) in remaining:
                                remaining.remove((current, nb))
                            else:
                                remaining.remove((nb, current))
                            path.insert(0, nb)
                            changed = True
                    else:
                        current = path[-1]
                        candidates = [nb for nb in adj[current] 
                                      if nb not in path and ((current, nb) in remaining or (nb, current) in remaining)]
                        if candidates:
                            nb = rng.choice(candidates)
                            if (current, nb) in remaining:
                                remaining.remove((current, nb))
                            else:
                                remaining.remove((nb, current))
                            path.append(nb)
                            changed = True
                
                if not changed:
                    break
            
            paths.append(path)
        
        return paths

    def merge_paths_with_matching(paths, edges, adj, used_edges, max_iterations=200):
        """
        Iteratively merge path endpoints using maximum matching.
        paths: list of paths (each is a list of vertices)
        edges: set of all edges
        adj: adjacency dict
        used_edges: set of edges already used in paths
        Returns: merged list of paths
        """
        if not paths:
            return []
        
        current_paths = [list(p) for p in paths]
        current_used = set(used_edges)
        
        for iteration in range(max_iterations):
            if time.time() > deadline - 5:
                break
            
            path_count = len(current_paths)
            if path_count <= 1:
                break
            
            # Build interaction graph
            interact = defaultdict(set)
            merge_info = {} # (i, j) -> list of valid (pos_i, pos_j, edge) tuples
            
            for i in range(path_count):
                for j in range(i + 1, path_count):
                    p_i = current_paths[i]
                    p_j = current_paths[j]
                    
                    set_i = set(p_i)
                    set_j = set(p_j)
                    if set_i & set_j:
                        continue
                    
                    valid_merges = []
                    
                    # Try i_end - j_start
                    e = tuple(sorted((p_i[-1], p_j[0])))
                    if e in edges and e not in current_used:
                        valid_merges.append((1, 0, e))
                    
                    # Try i_end - j_end
                    e = tuple(sorted((p_i[-1], p_j[-1])))
                    if e in edges and e not in current_used:
                        valid_merges.append((1, 1, e))
                        
                    # Try i_start - j_start
                    e = tuple(sorted((p_i[0], p_j[0])))
                    if e in edges and e not in current_used:
                        valid_merges.append((0, 0, e))
                            
                    # Try i_start - j_end
                    e = tuple(sorted((p_i[0], p_j[-1])))
                    if e in edges and e not in current_used:
                        valid_merges.append((0, 1, e))
                    
                    if valid_merges:
                        interact[i].add(j)
                        interact[j].add(i)
                        merge_info[(i, j)] = valid_merges

            if not interact:
                break
                
            # Find Maximum Matching in the interaction graph
            match = {} # path_idx -> matched_path_idx
            
            def dfs_augment(u, visited):
                for v in interact[u]:
                    if v in visited:
                        continue
                    visited.add(v)
                    if v not in match or dfs_augment(match[v], visited):
                        match[u] = v
                        match[v] = u
                        return True
                return False
            
            # Randomize the order of DFS attempts to explore different matchings
            nodes_to_visit = list(range(path_count))
            random.Random(iteration).shuffle(nodes_to_visit)
            
            for u in nodes_to_visit:
                if u not in match:
                    dfs_augment(u, set())
            
            if not match:
                break
            
            # Apply matches
            new_paths = {}
            for i in range(path_count):
                new_paths[i] = list(current_paths[i])
            
            removed = set()
            applied_merges = 0
            
            for i, j in list(match.items()):
                if i in removed or j in removed:
                    continue
                if i > j: # Ensure we process each pair once
                    continue
                
                valid_merges = merge_info.get((i, j), [])
                if not valid_merges:
                    continue
                    
                # Try all valid merges for this pair, pick the first one that works
                merged_success = False
                for pos_i, pos_j, edge in valid_merges:
                    p_i = new_paths[i]
                    p_j = new_paths[j]
                    
                    # Orient paths
                    if pos_i == 1:
                        p_i = list(reversed(p_i))
                    if pos_j == 1:
                        p_j = list(reversed(p_j))
                    
                    # Now we connect p_i[-1] to p_j[0]
                    if tuple(sorted((p_i[-1], p_j[0]))) != edge:
                        continue
                 
                    merged = p_i + p_j[1:]
                    
                    if len(set(merged)) != len(merged):
                        continue
                    
                    # Check validity of merged edges
                    valid = True
                    for k in range(len(merged) - 1):
                        e = tuple(sorted((merged[k], merged[k+1])))
                        if e not in edges:
                            valid = False
                            break
                    
                    if not valid:
                        continue
                    
                    new_paths[i] = merged
                    del new_paths[j]
                    removed.add(j)
                    current_used.add(edge)
                    applied_merges += 1
                    merged_success = True
                    break
                
                if not merged_success:
                    # If no merge worked for this pair, it's a problem, but we just skip it
                    pass
            
            current_paths = [new_paths[i] for i in sorted(new_paths.keys())]
            
            if applied_merges == 0:
                break
        
        return current_paths

    def local_search_improve(paths, edges, adj, used_edges, max_iterations=50):
        """
        Try to reduce path count by splitting and recombining paths.
        This is a placeholder for now, as the matching step is the primary driver.
        """
        return paths

    def solve_instance(n, p, seed):
        edges = instance(n, p, seed)
        if not edges:
            return []
        
        # Build adjacency and degrees
        adj = defaultdict(set)
        degrees = defaultdict(int)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            degrees[u] += 1
            degrees[v] += 1
            
        odd_vertices = [v for v in degrees if degrees[v] % 2 != 0]
        num_odd = len(odd_vertices)
        
        # Theoretical Lower Bound
        if num_odd == 0:
            lb = 1 if len(edges) == 1 else 2
        else:
            lb = num_odd // 2
        
        best_paths = None
        best_count = float('inf')
        
        # Strategy: Try multiple random initial decompositions, then merge with matching
        # Increase attempts as matching is now more efficient
        max_attempts = 500
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            seed_val = seed * 10000 + attempt
            initial_paths = greedy_initial_decomposition(edges, seed_val)
            
            if not initial_paths:
                continue
            
            # Track used edges
            used_edges = set()
            for path in initial_paths:
                for j in range(len(path) - 1):
                    u, v = path[j], path[j+1]
                    e = tuple(sorted((u, v)))
                    used_edges.add(e)
            
            # Merge with matching
            merged_paths = merge_paths_with_matching(initial_paths, edges, adj, used_edges)
            
            # Validate
            if validate_paths(merged_paths, edges):
                if len(merged_paths) < best_count:
                    best_count = len(merged_paths)
                    best_paths = merged_paths
                    
                    # If we reached the lower bound, we can stop
                    if len(merged_paths) <= lb:
                        break
        
        # Fallback to simple greedy if matching approach didn't work well
        if best_paths is None:
            for i in range(100):
                if time.time() > deadline - 5:
                    break
                paths = greedy_initial_decomposition(edges, seed * 10000 + 10000 + i)
                if validate_paths(paths, edges):
                    if len(paths) < best_count:
                        best_count = len(paths)
                        best_paths = paths
            
            if best_paths is None:
                best_paths = [[u, v] for u, v in edges]
                best_count = len(best_paths)

        return best_paths

    specs = [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]
    results = []
    
    for spec in specs:
        n, p, seed = spec
        # Check time before starting instance
        if time.time() > deadline - 10:
            # Not enough time, use trivial decomposition for remaining
            edges = instance(n, p, seed)
            results.append([[u, v] for u, v in edges])
            continue
            
        paths = solve_instance(n, p, seed)
        results.append(paths)
        
    # Fallback
    while len(results) < 3:
        n, p, seed = specs[len(results)]
        edges = instance(n, p, seed)
        results.append([[u, v] for u, v in edges])

    return results

if __name__ == "__main__":
    res = construct()
    total = sum(len(r) for r in res)
    print(f"Total paths: {total}")
    for i, r in enumerate(res):
        print(f"Instance {i}: {len(r)} paths")
