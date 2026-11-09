<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque
from random import Random
import heapq

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s limit
    
    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def bfs_shortest_paths(adj, n):
        """Compute shortest path distances from all vertices to all others using BFS."""
        dist = [[None] * n for _ in range(n)]
        for src in range(n):
            if src not in adj or not adj[src]:
                continue
            dist[src][src] = 0
            queue = deque([src])
            while queue:
                u = queue.popleft()
                for v in adj.get(u, []):
                    if dist[src][v] is None:
                        dist[src][v] = dist[src][u] + 1
                        queue.append(v)
        return dist
    
    def find_eulerian_circuit(adj, n):
        """Find an Eulerian circuit using Hierholzer's algorithm."""
        # Work with a copy of adjacency
        adj_copy = {v: list(neighbors) for v, neighbors in adj.items()}
        
        # Find a vertex with non-empty adjacency
        start = None
        for v in adj_copy:
            if adj_copy[v]:
                start = v
                break
        if start is None:
            return []
        
        # Hierholzer's algorithm
        stack = [start]
        circuit = []
        
        while stack:
            v = stack[-1]
            if adj_copy[v]:
                next_v = adj_copy[v].pop()
                stack.append(next_v)
            else:
                circuit.append(stack.pop())
        
        circuit.reverse()
        return circuit
    
    def solve_with_pairing(edges, virtual_pairs, n):
        """Given edges and virtual pairs, find path decomposition."""
        adj = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Add virtual edges
        virtual_set = set()
        for u, v in virtual_pairs:
            adj[u].append(v)
            adj[v].append(u)
            virtual_set.add(tuple(sorted((u, v))))
        
        # Find Eulerian circuit on augmented graph
        circuit = find_eulerian_circuit(adj, n)
        
        if not circuit:
            return [[u, v] for u, v in edges]
        
        # Split circuit at virtual edges to get paths
        paths = []
        current_path = [circuit[0]]
        for i in range(1, len(circuit)):
            u = circuit[i-1]
            v = circuit[i]
            edge = tuple(sorted((u, v)))
            current_path.append(v)
            if edge in virtual_set:
                if len(current_path) >= 2:
                    paths.append(current_path[:])
                current_path = [circuit[i]]
        
        # Handle wrap-around
        if len(circuit) > 1:
            u = circuit[-1]
            v = circuit[0]
            edge = tuple(sorted((u, v)))
            if edge in virtual_set:
                if len(current_path) >= 2:
                    paths.append(current_path[:])
                current_path = []
            else:
                current_path.append(circuit[0])
                if len(current_path) >= 2:
                    paths.append(current_path[:])
        
        return paths
    
    def try_improve_paths(paths, edges, n, edge_set):
        """Try to merge paths to reduce count."""
        if len(paths) <= 1:
            return paths
        
        # Build edge set for quick lookup
        used_edges = set()
        for path in paths:
            for i in range(len(path)-1):
                u, v = path[i], path[i+1]
                used_edges.add(tuple(sorted((u, v))))
        
        # Try to merge paths: if path1 ends at vertex that path2 starts at, 
        # and the resulting path is simple, merge them
        improved = True
        while improved:
            improved = False
            new_paths = []
            merged_indices = set()
            
            for i in range(len(paths)):
                if i in merged_indices:
                    continue
                path_i = paths[i]
                merged = False
                
                for j in range(i+1, len(paths)):
                    if j in merged_indices:
                        continue
                    path_j = paths[j]
                    
                    # Try merging: path_i + path_j (if end of i == start of j)
                    if path_i[-1] == path_j[0]:
                        candidate = path_i + path_j[1:]
                        # Check simplicity
                        if len(set(candidate)) == len(candidate):
                            # Check edge validity
                            valid = True
                            for k in range(len(candidate)-1):
                                u, v = candidate[k], candidate[k+1]
                                if tuple(sorted((u, v))) not in edge_set:
                                    valid = False
                                    break
                            if valid:
                                new_paths.append(candidate)
                                merged_indices.add(j)
                                improved = True
                                merged = True
                                break
                    
                    # Try merging: path_j + path_i (if end of j == start of i)
                    elif path_j[-1] == path_i[0]:
                        candidate = path_j + path_i[1:]
                        if len(set(candidate)) == len(candidate):
                            valid = True
                            for k in range(len(candidate)-1):
                                u, v = candidate[k], candidate[k+1]
                                if tuple(sorted((u, v))) not in edge_set:
                                    valid = False
                                    break
                            if valid:
                                new_paths.append(candidate)
                                merged_indices.add(i)
                                improved = True
                                merged = True
                                break
                    
                    # Try reversing path_j and merging
                    path_j_rev = path_j[::-1]
                    if path_i[-1] == path_j_rev[0]:
                        candidate = path_i + path_j_rev[1:]
                        if len(set(candidate)) == len(candidate):
                            valid = True
                            for k in range(len(candidate)-1):
                                u, v = candidate[k], candidate[k+1]
                                if tuple(sorted((u, v))) not in edge_set:
                                    valid = False
                                    break
                            if valid:
                                new_paths.append(candidate)
                                merged_indices.add(j)
                                improved = True
                                merged = True
                                break
                    
                    if path_j_rev[-1] == path_i[0]:
                        candidate = path_j_rev + path_i[1:]
                        if len(set(candidate)) == len(candidate):
                            valid = True
                            for k in range(len(candidate)-1):
                                u, v = candidate[k], candidate[k+1]
                                if tuple(sorted((u, v))) not in edge_set:
                                    valid = False
                                    break
                            if valid:
                                new_paths.append(candidate)
                                merged_indices.add(i)
                                improved = True
                                merged = True
                                break
                
                if not merged:
                    new_paths.append(path_i)
            
            paths = new_paths
        
        return paths
    
    def find_eulerian_trails_improved(edges, n, time_limit=30.0):
        """Find minimum number of edge-disjoint paths covering all edges using improved pairing."""
        if not edges:
            return []
        
        edge_set = set(tuple(sorted(e)) for e in edges)
        
        # Build adjacency list
        adj = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Find odd-degree vertices
        odd_vertices = [v for v in adj if len(adj[v]) % 2 == 1]
        
        # If no odd vertices, we can find one Eulerian circuit
        if not odd_vertices:
            circuit = find_eulerian_circuit(adj, n)
            return [circuit] if circuit else [[u, v] for u, v in edges]
        
        # Compute shortest path distances
        dist = bfs_shortest_paths(adj, n)
        
        # Minimum weight perfect matching on odd vertices
        # Use greedy approach: repeatedly pick the closest pair
        remaining = list(odd_vertices)
        virtual_pairs = []
        
        while len(remaining) >= 2:
            best_pair = None
            best_dist = float('inf')
            for i in range(len(remaining)):
                for j in range(i+1, len(remaining)):
                    u, v = remaining[i], remaining[j]
                    d = dist[u][v] if dist[u][v] is not None else float('inf')
                    if d < best_dist:
                        best_dist = d
                        best_pair = (u, v)
            
            if best_pair:
                u, v = best_pair
                virtual_pairs.append((u, v))
                remaining.remove(u)
                remaining.remove(v)
        
        # Solve with this pairing
        paths = solve_with_pairing(edges, virtual_pairs, n)
        
        # Try to improve by merging
        if time.time() < start_time + time_limit:
            paths = try_improve_paths(paths, edges, n, edge_set)
        
        return paths
    
    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > deadline:
            # Fallback to trivial solution if out of time
            edges = instance(*spec)
            results.append([[list(e)] for e in edges])
            continue
        
        edges = instance(*spec)
        n = max(max(e) for e in edges) + 1 if edges else 0
        edge_set = set(tuple(sorted(e)) for e in edges)
        
        # Try improved algorithm
        paths = find_eulerian_trails_improved(edges, n, time_limit=35.0)
        
        # Verify and clean up paths
        valid_paths = []
        used_edges = set()
        
        for path in paths:
            if len(path) < 2:
                continue
            # Check if path uses only valid edges and is simple
            path_edges = []
            valid = True
            seen_in_path = set()
            for i in range(len(path)-1):
                u, v = path[i], path[i+1]
                edge = tuple(sorted((u, v)))
                if edge not in edge_set:
                    valid = False
                    break
                if u in seen_in_path or v in seen_in_path:
                    valid = False
                    break
                seen_in_path.add(u)
                seen_in_path.add(v)
                path_edges.append(edge)
            
            if not valid:
                continue
            
            # Check edge-disjointness
            if any(e in used_edges for e in path_edges):
                continue
            
            used_edges.update(path_edges)
            valid_paths.append(path)
        
        # If we didn't cover all edges, add remaining edges as single-edge paths
        remaining_edges = edge_set - used_edges
        for u, v in remaining_edges:
            valid_paths.append([u, v])
        
        results.append(valid_paths)
    
    return results
# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque
from random import Random
import heapq

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s limit
    
    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def greedy_longest_path(adj, start, used_edges):
        """Find the longest simple path starting from `start` using DFS with backtracking."""
        best_path = [start]
        best_len = 0
        
        def dfs(path, visited):
            nonlocal best_path, best_len
            if len(path) > best_len:
                best_path = path[:]
                best_len = len(path)
            
            current = path[-1]
            for neighbor in adj.get(current, []):
                if neighbor not in visited:
                    edge = tuple(sorted((current, neighbor)))
                    if edge not in used_edges:
                        visited.add(neighbor)
                        path.append(neighbor)
                        dfs(path, visited)
                        path.pop()
                        visited.discard(neighbor)
        
        dfs([start], {start})
        return best_path
    
    def find_longest_path_from(start, adj, used_edges, n):
        """Find longest simple path starting from start using iterative DFS."""
        best_path = [start]
        best_len = 0
        
        # Use iterative DFS to avoid recursion depth issues
        stack = [(start, {start}, [start])]
        
        while stack:
            current, visited, path = stack.pop()
            
            if len(path) > best_len:
                best_path = path[:]
                best_len = len(path)
            
            for neighbor in adj.get(current, []):
                if neighbor not in visited:
                    edge = tuple(sorted((current, neighbor)))
                    if edge not in used_edges:
                        new_visited = visited | {neighbor}
                        new_path = path + [neighbor]
                        stack.append((neighbor, new_visited, new_path))
        
        return best_path
    
    def greedy_path_cover(edges, n, time_limit=30.0):
        """Greedy approach: repeatedly extract longest simple path from remaining edges."""
        if not edges:
            return []
        
        edge_set = set(tuple(sorted(e)) for e in edges)
        
        # Build adjacency list
        adj = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        remaining_edges = set(edge_set)
        paths = []
        
        while remaining_edges:
            if time.time() > start_time + time_limit:
                break
            
            # Find a good starting vertex: prefer vertices with highest degree in remaining graph
            # Try multiple starting points and pick the best path
            candidates = []
            for v in range(n):
                if adj[v]:
                    # Count available edges from this vertex
                    available = 0
                    for u in adj[v]:
                        if tuple(sorted((v, u))) in remaining_edges:
                            available += 1
                    if available > 0:
                        candidates.append((available, v))
            
            if not candidates:
                break
            
            # Sort by degree descending, try top few
            candidates.sort(reverse=True)
            
            best_path = None
            best_len = 0
            
            # Try top 5 starting vertices
            for _, start_v in candidates[:5]:
                # Find longest path from this start
                path = find_longest_path_from(start_v, adj, remaining_edges, n)
                if len(path) > best_len:
                    best_len = len(path)
                    best_path = path
            
            if not best_path or len(best_path) < 2:
                # Fallback: just take any remaining edge
                edge = next(iter(remaining_edges))
                best_path = [edge[0], edge[1]]
            
            # Extract this path's edges from remaining
            for i in range(len(best_path) - 1):
                u, v = best_path[i], best_path[i+1]
                remaining_edges.discard(tuple(sorted((u, v))))
            
            paths.append(best_path)
        
        # Add any remaining edges as single-edge paths
        for u, v in remaining_edges:
            paths.append([u, v])
        
        return paths
    
    def try_merge_paths(paths, edge_set):
        """Try to merge paths that share endpoints."""
        if len(paths) <= 1:
            return paths
        
        improved = True
        while improved:
            improved = False
            new_paths = []
            merged = set()
            
            for i in range(len(paths)):
                if i in merged:
                    continue
                path_i = paths[i]
                merged_flag = False
                
                for j in range(i + 1, len(paths)):
                    if j in merged:
                        continue
                    path_j = paths[j]
                    
                    # Try all 4 combinations of orientations
                    for p1 in [path_i, path_i[::-1]]:
                        for p2 in [path_j, path_j[::-1]]:
                            # Check if we can concatenate
                            if p1[-1] == p2[0]:
                                candidate = p1 + p2[1:]
                                if len(set(candidate)) == len(candidate):
                                    # Verify all edges exist
                                    valid = True
                                    for k in range(len(candidate) - 1):
                                        u, v = candidate[k], candidate[k+1]
                                        if tuple(sorted((u, v))) not in edge_set:
                                            valid = False
                                            break
                                    if valid:
                                        new_paths.append(candidate)
                                        merged.add(j)
                                        improved = True
                                        merged_flag = True
                                        break
                        if merged_flag:
                            break
                    if merged_flag:
                        break
                
                if not merged_flag:
                    new_paths.append(path_i)
            
            paths = new_paths
        
        return paths
    
    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > deadline:
            # Fallback to trivial solution if out of time
            edges = instance(*spec)
            results.append([[u, v] for u, v in edges])
            continue
        
        edges = instance(*spec)
        n = max(max(e) for e in edges) + 1 if edges else 0
        edge_set = set(tuple(sorted(e)) for e in edges)
        
        # Use greedy longest path extraction
        paths = greedy_path_cover(edges, n, time_limit=35.0)
        
        # Try to merge paths
        if time.time() < deadline - 5:
            paths = try_merge_paths(paths, edge_set)
        
        # Verify and clean up
        valid_paths = []
        used_edges = set()
        
        for path in paths:
            if len(path) < 2:
                continue
            path_edges = []
            valid = True
            seen = set()
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                edge = tuple(sorted((u, v)))
                if edge not in edge_set:
                    valid = False
                    break
                if u in seen or v in seen:
                    valid = False
                    break
                seen.add(u)
                seen.add(v)
                path_edges.append(edge)
            
            if not valid:
                continue
            if any(e in used_edges for e in path_edges):
                continue
            
            used_edges.update(path_edges)
            valid_paths.append(path)
        
        # Add remaining edges
        for u, v in (edge_set - used_edges):
            valid_paths.append([u, v])
        
        results.append(valid_paths)
    
    return results
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
