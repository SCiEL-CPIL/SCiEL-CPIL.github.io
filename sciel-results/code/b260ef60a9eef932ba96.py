"""Path decompositions with iterative matching-based path merging."""
import time
from collections import defaultdict

def construct():
    _t0 = time.time()
    _DEADLINE = 145.0  # safety margin before 150s kill

    def instance(n, p, seed):
        from random import Random
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def eulerian_decompose(edges):
        """Initial decomposition via Eulerian trail on doubled-edge multigraph."""
        if not edges:
            return []
        n = 0
        for (u, v) in edges:
            if u > n: n = u
            if v > n: n = v
        n += 1
        adj = [[] for _ in range(n)]
        for i, (u, v) in enumerate(edges):
            adj[u].append((v, 2 * i))
            adj[v].append((u, 2 * i))
            adj[u].append((v, 2 * i + 1))
            adj[v].append((u, 2 * i + 1))
        used = [False] * (2 * len(edges))
        start = 0
        for x in range(n):
            if adj[x]:
                start = x
                break
        stack = [(start, None)]
        circuit = []
        while stack:
            v, eid = stack[-1]
            found = -1
            for k in range(len(adj[v])):
                if not used[adj[v][k][1]]:
                    found = k
                    break
            if found == -1:
                circuit.append(v)
                stack.pop()
            else:
                u, eid2 = adj[v][found]
                used[eid2] = True
                adj[v].pop(found)
                stack.append((u, eid2))
        circuit.reverse()
        # Group into paths by cutting at repeated vertices
        paths = []
        cur = [circuit[0]]
        for i in range(1, len(circuit)):
            if circuit[i] == cur[-1]:
                if len(cur) > 1:
                    paths.append(cur)
                cur = [circuit[i]]
            else:
                cur.append(circuit[i])
        if len(cur) > 1:
            paths.append(cur)
        return paths

    def max_matching_bipartite(left_nodes, right_nodes, adj_map):
        """
        Find maximum cardinality matching in bipartite graph.
        left_nodes: list of left vertices
        right_nodes: list of right vertices  
        adj_map: dict mapping left vertex -> list of right vertices
        Returns list of (left, right) pairs.
        Uses Hopcroft-Karp style augmenting paths.
        """
        if not left_nodes or not right_nodes:
            return []
        
        # Build adjacency
        from_left = {u: [] for u in left_nodes}
        for u in left_nodes:
            for v in adj_map.get(u, []):
                if v in set(right_nodes):
                    from_left[u].append(v)
        
        match_l = {}  # left -> right
        match_r = {}  # right -> left
        
        def dfs(u, visited):
            for v in from_left.get(u, []):
                if v in visited:
                    continue
                visited.add(v)
                if v not in match_r or dfs(match_r[v], visited):
                    match_l[u] = v
                    match_r[v] = u
                    return True
            return False
        
        result = []
        for u in left_nodes:
            visited = set()
            if dfs(u, visited):
                result.append((u, match_l[u]))
        
        return result

    def merge_paths_with_matching(paths, edges_set, time_limit=None):
        """
        Iteratively merge paths using maximum matching on endpoint compatibility graph.
        Two path endpoints can be merged if there's an edge between them and the merge
        doesn't create a cycle (i.e., the other endpoint of the second path isn't already
        in the first path).
        """
        if time_limit is None:
            time_limit = time.time() + 5.0
        
        paths = [list(p) for p in paths]
        
        # Precompute edge set for O(1) lookup
        edge_set = set()
        for e in edges_set:
            edge_set.add(tuple(sorted(e)))
        
        def can_merge(path1, end1_idx, path2, end2_idx):
            """
            Check if we can merge path1 at end1_idx with path2 at end2_idx.
            end1_idx: 0 for front, 1 for back
            end2_idx: 0 for front, 1 for back
            """
            if path1 is path2:
                return False
            
            # Get the endpoints to connect
            if end1_idx == 0:
                v1 = path1[0]
            else:
                v1 = path1[-1]
            
            if end2_idx == 0:
                v2 = path2[0]
            else:
                v2 = path2[-1]
            
            # Check if edge exists
            e = tuple(sorted((v1, v2)))
            if e not in edge_set:
                return False
            
            # Check if merge creates a cycle: the other endpoint of path2 
            # must not be in path1
            if end2_idx == 0:
                other_end2 = path2[-1]
            else:
                other_end2 = path2[0]
            
            if other_end2 in path1:
                return False
            
            # Also check the other direction for symmetry
            if end1_idx == 0:
                other_end1 = path1[-1]
            else:
                other_end1 = path1[0]
            
            # Actually, we need to check that connecting v1-v2 doesn't create a cycle
            # The merged path would go from other_end1 to other_end2
            # This is a cycle if other_end1 == other_end2, which can't happen since
            # they're in different paths. But we need to ensure no vertex appears twice.
            # Since paths are simple and disjoint in vertices (they cover disjoint edge sets),
            # the only issue is if the connection creates a cycle within the merged structure.
            # Actually, two paths sharing no vertices can always be merged at an edge
            # without creating a cycle, as long as the edge exists.
            
            return True

        # Iteratively find and apply maximum matchings
        changed = True
        while changed and time.time() < time_limit:
            changed = False
            
            # Build compatibility graph between path endpoints
            # Left side: (path_idx, end_idx) for all path endpoints
            # Right side: (path_idx, end_idx) for all path endpoints
            # But we need to avoid matching a path to itself
            
            # For simplicity, let's just try all pairs and find a maximal set of non-conflicting merges
            # A merge conflicts if it uses the same path endpoint twice
            
            # Collect all possible merges
            possible_merges = []
            n_paths = len(paths)
            for i in range(n_paths):
                for j in range(i + 1, n_paths):
                    # Try merging back of i with front of j
                    if can_merge(paths[i], 1, paths[j], 0):
                        possible_merges.append((i, 1, j, 0))
                    # Try merging back of i with back of j
                    if can_merge(paths[i], 1, paths[j], 1):
                        possible_merges.append((i, 1, j, 1))
                    # Try merging front of i with front of j
                    if can_merge(paths[i], 0, paths[j], 0):
                        possible_merges.append((i, 0, j, 0))
                    # Try merging front of i with back of j
                    if can_merge(paths[i], 0, paths[j], 1):
                        possible_merges.append((i, 0, j, 1))
            
            if not possible_merges:
                break
            
            # Find a maximal set of non-conflicting merges
            # A merge conflicts if it uses the same (path, end) pair
            used_endpoints = set()
            selected_merges = []
            
            # Sort by some heuristic - prefer merges that connect high-degree endpoints
            # For now, just greedily select
            for merge in possible_merges:
                i, end_i, j, end_j = merge
                key_i = (i, end_i)
                key_j = (j, end_j)
                if key_i not in used_endpoints and key_j not in used_endpoints:
                    used_endpoints.add(key_i)
                    used_endpoints.add(key_j)
                    selected_merges.append(merge)
            
            if not selected_merges:
                break
            
            # Apply the merges
            new_paths = []
            path_map = {}  # old index -> new index or None if merged
            for idx in range(n_paths):
                path_map[idx] = idx
            
            # Create new paths list, marking which ones are merged
            merged_into = {}  # (i, end_i) -> (j, end_j) meaning i's end_i connects to j's end_j
            
            for merge in selected_merges:
                i, end_i, j, end_j = merge
                merged_into[(i, end_i)] = (j, end_j)
                merged_into[(j, end_j)] = (i, end_i)
            
            # Build new paths
            # For each path, check if its endpoints are merged
            # If a path has both endpoints merged, it's fully absorbed
            # If one endpoint is merged, it extends another path
            
            # Simpler approach: process merges one by one
            active_paths = [list(p) for p in paths]
            active_set = set(range(n_paths))
            
            for merge in selected_merges:
                i, end_i, j, end_j = merge
                if i not in active_set or j not in active_set:
                    continue
                
                p_i = active_paths[i]
                p_j = active_paths[j]
                
                # Determine how to concatenate
                if end_i == 1 and end_j == 0:
                    # p_i back connects to p_j front
                    new_path = p_i + p_j
                elif end_i == 1 and end_j == 1:
                    # p_i back connects to p_j back
                    new_path = p_i + list(reversed(p_j))
                elif end_i == 0 and end_j == 0:
                    # p_i front connects to p_j front
                    new_path = list(reversed(p_i)) + p_j
                else:  # end_i == 0 and end_j == 1
                    # p_i front connects to p_j back
                    new_path = list(reversed(p_i)) + list(reversed(p_j))
                
                # Verify simplicity
                if len(set(new_path)) != len(new_path):
                    continue  # Skip if creates cycle
                
                active_paths[i] = new_path
                active_set.discard(j)
                changed = True
            
            # Collect active paths
            paths = [active_paths[i] for i in range(n_paths) if i in active_set]
        
        return paths

    def greedy_extend(paths, edges_set, time_limit=None):
        """Try to extend paths to absorb more edges."""
        if time_limit is None:
            time_limit = time.time() + 5.0
        
        remaining_edges = set()
        for path in paths:
            for k in range(len(path) - 1):
                e = tuple(sorted((path[k], path[k + 1])))
                remaining_edges.add(e)
        
        unused_edges = edges_set - remaining_edges
        if not unused_edges:
            return paths
        
        # Build adjacency for unused edges
        adj = defaultdict(list)
        for u, v in unused_edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Try to extend each path
        changed = True
        while changed and time.time() < time_limit:
            changed = False
            for i, path in enumerate(paths):
                # Try extending from back
                if path:
                    back = path[-1]
                    for neighbor in adj.get(back, []):
                        if neighbor not in path:
                            path.append(neighbor)
                            e = tuple(sorted((back, neighbor)))
                            if e in unused_edges:
                                unused_edges.discard(e)
                            changed = True
                            break
                # Try extending from front
                if path:
                    front = path[0]
                    for neighbor in adj.get(front, []):
                        if neighbor not in path:
                            path.insert(0, neighbor)
                            e = tuple(sorted((front, neighbor)))
                            if e in unused_edges:
                                unused_edges.discard(e)
                            changed = True
                            break
        
        return paths

    def validate_and_fix(paths, edges_set):
        """Validate paths and fix any issues."""
        seen_edges = set()
        valid_paths = []
        for path in paths:
            if len(path) < 2:
                continue
            path_ok = True
            for j in range(len(path) - 1):
                e = tuple(sorted((path[j], path[j + 1])))
                if e in seen_edges or e not in edges_set:
                    path_ok = False
                    break
                seen_edges.add(e)
            if path_ok:
                valid_paths.append(path)
        
        # Add any missing edges as single-edge paths
        for e in edges_set:
            if e not in seen_edges:
                valid_paths.append(list(e))
                seen_edges.add(e)
        
        return valid_paths

    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    results = []
    
    for spec in specs:
        if time.time() > _DEADLINE:
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
        
        edges = instance(*spec)
        edges_set = set(edges)
        
        try:
            # Step 1: Initial Eulerian decomposition
            paths = eulerian_decompose(edges)
            
            # Step 2: Greedy extension to absorb any missed edges
            paths = greedy_extend(paths, edges_set, time.time() + 10.0)
            
            # Step 3: Merge paths using matching
            paths = merge_paths_with_matching(paths, edges_set, time.time() + 20.0)
            
            # Step 4: Final validation and fix
            paths = validate_and_fix(paths, edges_set)
            
            results.append(paths)
        except Exception:
            results.append([list(e) for e in edges])
    
    return results
