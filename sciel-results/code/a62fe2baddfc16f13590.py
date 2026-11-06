"""Rewritten program: Matching-based path decomposition with 2-opt and extension for longest path."""
import time
import sys
import random

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    start_time = time.time()
    deadline = start_time + 80.0
    
    # Find largest connected component
    visited_comp = [False] * n
    comp_sizes = []
    for i in range(n):
        if not visited_comp[i]:
            stack = [i]
            visited_comp[i] = True
            size = 0
            while stack:
                node = stack.pop()
                size += 1
                for nb in adj[node]:
                    if not visited_comp[nb]:
                        visited_comp[nb] = True
                        stack.append(nb)
            comp_sizes.append((size, i))
    
    if not comp_sizes:
        return [0]
        
    comp_sizes.sort(reverse=True)
    largest_comp_size, start_node = comp_sizes[0]
    
    # Identify nodes in largest component
    visited_comp = [False] * n
    largest_comp_nodes = set()
    stack = [start_node]
    visited_comp[start_node] = True
    while stack:
        node = stack.pop()
        largest_comp_nodes.add(node)
        for nb in adj[node]:
            if not visited_comp[nb]:
                visited_comp[nb] = True
                stack.append(nb)
    
    comp_nodes = list(largest_comp_nodes)
    node_to_idx = {node: i for i, node in enumerate(comp_nodes)}
    m = len(comp_nodes)
    
    # Build adjacency within component
    comp_adj = [set() for _ in range(m)]
    for i, u in enumerate(comp_nodes):
        for v in adj[u]:
            if v in node_to_idx:
                comp_adj[i].add(node_to_idx[v])
    
    def find_longest_path():
        """Find a long path using matching-based decomposition + local search."""
        
        def max_matching_bipartite():
            """Find maximum matching in bipartite graph (left=nodes, right=nodes, edge if adjacent)."""
            # Use Hopcroft-Karp or simple DFS-based matching
            match_left = [-1] * m
            match_right = [-1] * m
            
            def dfs_augment(u, visited):
                for v in comp_adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        if match_right[v] == -1 or dfs_augment(match_right[v], visited):
                            match_left[u] = v
                            match_right[v] = u
                            return True
                return False
            
            for u in range(m):
                visited = [False] * m
                dfs_augment(u, visited)
            
            # Extract matched edges
            matched_edges = set()
            for u in range(m):
                if match_left[u] != -1:
                    matched_edges.add((u, match_left[u]))
                    matched_edges.add((match_left[u], u))
            return matched_edges
        
        def build_paths_from_matching(matched_edges):
            """Build vertex-disjoint paths from matching edges."""
            in_path = [False] * m
            paths = []
            
            for i in range(m):
                if in_path[i]:
                    continue
                
                # Check if i is a leaf (degree 0 or 1 in matching)
                deg = sum(1 for (a, b) in matched_edges if a == i or b == i)
                if deg == 0:
                    paths.append([i])
                    in_path[i] = True
                elif deg == 1:
                    # Start from leaf i
                    path = [i]
                    in_path[i] = True
                    current = i
                    while True:
                        # Find next node
                        next_node = None
                        for (a, b) in matched_edges:
                            if a == current and not in_path[b]:
                                next_node = b
                                break
                            elif b == current and not in_path[a]:
                                next_node = a
                                break
                        if next_node is None:
                            break
                        path.append(next_node)
                        in_path[next_node] = True
                        current = next_node
                    paths.append(path)
            
            return paths
        
        def concatenate_paths(paths):
            """Concatenate paths into a single path, trying to merge where possible."""
            if not paths:
                return []
            if len(paths) == 1:
                return paths[0]
            
            # Sort paths by length descending
            paths.sort(key=len, reverse=True)
            
            # Start with longest path
            result = list(paths[0])
            used_paths = [True] + [False] * (len(paths) - 1)
            
            # Try to extend or merge other paths
            for i in range(1, len(paths)):
                if used_paths[i]:
                    continue
                p = paths[i]
                
                # Try to merge p into result
                merged = False
                
                # Try appending p to end of result
                if result and p and result[-1] in comp_adj[p[0]]:
                    result.extend(p)
                    used_paths[i] = True
                    merged = True
                elif result and p and result[-1] in comp_adj[p[-1]]:
                    result.extend(p[::-1])
                    used_paths[i] = True
                    merged = True
                # Try prepending p to start of result
                elif result and p and result[0] in comp_adj[p[-1]]:
                    result = p + result
                    used_paths[i] = True
                    merged = True
                elif result and p and result[0] in comp_adj[p[0]]:
                    result = p[::-1] + result
                    used_paths[i] = True
                    merged = True
                # Try inserting p between two consecutive nodes in result
                else:
                    for j in range(len(result) - 1):
                        if result[j] in comp_adj[p[0]] and result[j+1] in comp_adj[p[-1]]:
                            result = result[:j+1] + p + result[j+1:]
                            used_paths[i] = True
                            merged = True
                            break
                        elif result[j] in comp_adj[p[-1]] and result[j+1] in comp_adj[p[0]]:
                            result = result[:j+1] + p[::-1] + result[j+1:]
                            used_paths[i] = True
                            merged = True
                            break
                
                if not merged:
                    # Just append to end (may create discontinuity, but we'll fix later)
                    result.extend(p)
                    used_paths[i] = True
            
            return result
        
        def two_opt(path):
            """Apply 2-opt to improve path."""
            if len(path) < 4:
                return path
            
            improved = True
            while improved:
                improved = False
                for i in range(len(path) - 1):
                    if time.time() > deadline:
                        return path
                    for j in range(i + 2, len(path)):
                        if time.time() > deadline:
                            return path
                        # Check if reversing path[i+1:j+1] improves
                        if path[i] in comp_adj[path[j+1]] if j+1 < len(path) else False:
                            continue
                        # Try reversing segment
                        new_path = path[:i+1] + path[i+1:j+1][::-1] + path[j+1:]
                        # Verify all consecutive edges exist
                        valid = True
                        for k in range(len(new_path) - 1):
                            if new_path[k] not in comp_adj[new_path[k+1]]:
                                valid = False
                                break
                        if valid and len(new_path) > len(path):
                            path = new_path
                            improved = True
                            break
                    if improved:
                        break
            return path
        
        def extend_both_ends(path):
            """Try to extend path at both ends."""
            changed = True
            while changed:
                changed = False
                # Extend at start
                if path:
                    for nb in comp_adj[path[0]]:
                        if nb not in set(path):
                            path = [nb] + path
                            changed = True
                            break
                # Extend at end
                if path:
                    for nb in comp_adj[path[-1]]:
                        if nb not in set(path):
                            path = path + [nb]
                            changed = True
                            break
            return path
        
        # Step 1: Find maximum matching
        matched_edges = max_matching_bipartite()
        
        # Step 2: Build paths from matching
        paths = build_paths_from_matching(matched_edges)
        
        # Step 3: Concatenate paths
        path = concatenate_paths(paths)
        
        # Step 4: Apply local search improvements
        path = extend_both_ends(path)
        path = two_opt(path)
        path = extend_both_ends(path)
        
        return path
    
    # Try multiple random perturbations of the matching to find better paths
    best_path = []
    rng = random.Random(748)
    
    for attempt in range(20):
        if time.time() > deadline:
            break
        
        # Perturb matching: randomly remove some edges
        matched_edges = max_matching_bipartite()
        
        # Randomly remove some matched edges to create different path structures
        if attempt > 0:
            edges_list = list(matched_edges)
            remove_count = min(len(edges_list) // 4, 10)
            for _ in range(remove_count):
                if edges_list:
                    idx = rng.randrange(len(edges_list))
                    edges_list.pop(idx)
            matched_edges = set(edges_list)
        
        paths = build_paths_from_matching(matched_edges)
        path = concatenate_paths(paths)
        path = extend_both_ends(path)
        path = two_opt(path)
        path = extend_both_ends(path)
        
        if len(path) > len(best_path):
            best_path = path
        
        # Early termination if we found a Hamiltonian path
        if len(best_path) == m:
            break
    
    # Convert back to original node labels
    result = [comp_nodes[i] for i in best_path]
    
    # Verify path validity
    for i in range(len(result) - 1):
        if result[i+1] not in adj[result[i]]:
            # Path is invalid, fall back to a simple valid path
            result = [comp_nodes[0]]
            break
    
    if not result:
        result = [comp_nodes[0]]
    
    return result
