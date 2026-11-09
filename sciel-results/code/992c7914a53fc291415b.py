import random
import time
from collections import defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def bfs_shortest_path(start, end, adj):
        """Find shortest path from start to end in unweighted graph."""
        if start == end:
            return [start]
        from collections import deque
        visited = {start}
        parent = {start: None}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            if node == end:
                # Reconstruct path
                path = []
                curr = end
                while curr is not None:
                    path.append(curr)
                    curr = parent[curr]
                return path[::-1]
            for nb in adj[node]:
                if nb not in visited:
                    visited.add(nb)
                    parent[nb] = node
                    queue.append(nb)
        return None

    def find_path_with_edges(start, end, remaining_edges, adj, max_depth=60):
        """Find a simple path from start to end using only edges in remaining_edges."""
        if start == end:
            return [start]
        
        visited = {start}
        path = [start]
        
        def dfs(node, path, visited):
            if node == end:
                return list(path)
            if len(path) > max_depth:
                return None
            
            neighbors = list(adj[node])
            # Sort neighbors: prefer those with fewer remaining options
            def neighbor_score(nb):
                count = 0
                for nb2 in adj[nb]:
                    e = tuple(sorted((nb, nb2)))
                    if e in remaining_edges and nb2 not in visited:
                        count += 1
                return count
            
            neighbors.sort(key=neighbor_score)
            
            for nb in neighbors:
                if nb in visited:
                    continue
                e = tuple(sorted((node, nb)))
                if e not in remaining_edges:
                    continue
                
                # Try this edge
                remaining_edges.discard(e)
                visited.add(nb)
                path.append(nb)
                
                result = dfs(nb, path, visited)
                if result is not None:
                    return result
                
                # Backtrack
                path.pop()
                visited.discard(nb)
                remaining_edges.add(e)
            
            return None
        
        return dfs(start, path, visited)

    def decompose(edges):
        if not edges:
            return []
        
        # Build adjacency
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        remaining_edges = set(edges)
        paths = []
        
        # Compute degrees
        degree = defaultdict(int)
        for u, v in remaining_edges:
            degree[u] += 1
            degree[v] += 1
        
        # Find odd-degree vertices
        odd_vertices = sorted([v for v in degree if degree[v] % 2 == 1])
        
        # Use minimum-weight matching to pair odd vertices optimally
        # The weight is the shortest path distance between them
        if len(odd_vertices) >= 2:
            # Build distance matrix
            m = len(odd_vertices)
            # Map vertex to index
            vertex_to_idx = {v: i for i, v in enumerate(odd_vertices)}
            
            # Compute pairwise shortest path distances
            dist_matrix = np.zeros((m, m))
            for i in range(m):
                for j in range(i+1, m):
                    path = bfs_shortest_path(odd_vertices[i], odd_vertices[j], adj)
                    if path is not None:
                        d = len(path) - 1
                        dist_matrix[i, j] = d
                        dist_matrix[j, i] = d
                    else:
                        dist_matrix[i, j] = 1000  # Large penalty
                        dist_matrix[j, i] = 1000
            
            # Find minimum weight perfect matching
            row_ind, col_ind = linear_sum_assignment(dist_matrix)
            
            # Extract pairs
            pairs = []
            for r, c in zip(row_ind, col_ind):
                if r != c:
                    pairs.append((odd_vertices[r], odd_vertices[c]))
            
            # Remove duplicates (matching is symmetric)
            seen = set()
            unique_pairs = []
            for a, b in pairs:
                key = (min(a, b), max(a, b))
                if key not in seen:
                    seen.add(key)
                    unique_pairs.append((a, b))
            
            # For each pair, try to find a path
            for (a, b) in unique_pairs:
                if time.time() - start_time > time_limit - 5:
                    break
                path = find_path_with_edges(a, b, remaining_edges, adj, max_depth=50)
                if path and len(path) >= 2:
                    paths.append(path)
                    # Update degrees
                    for i in range(len(path) - 1):
                        degree[path[i]] -= 1
                        degree[path[i + 1]] -= 1
                else:
                    # Can't find path between a and b; will handle with greedy later
                    pass
        
        # Greedy: cover remaining edges
        while remaining_edges:
            if time.time() - start_time > time_limit - 2:
                # Emergency: add remaining edges as individual paths
                for e in remaining_edges:
                    paths.append(list(e))
                break
            
            # Recompute degrees for remaining edges
            rem_degree = defaultdict(int)
            for u, v in remaining_edges:
                rem_degree[u] += 1
                rem_degree[v] += 1
            
            # Pick an edge to start a path
            # Choose edge where sum of endpoint degrees is highest
            best_edge = None
            best_key = -1
            for e in remaining_edges:
                u, v = e
                k = rem_degree[u] + rem_degree[v]
                if k > best_key:
                    best_key = k
                    best_edge = e
            
            if best_edge is None:
                break
            
            u, v = best_edge
            # Extend path from both ends
            path = [u, v]
            remaining_edges.discard(best_edge)
            path_set = set(path)
            
            # Extend from u end
            while True:
                current = path[0]
                candidates = [nb for nb in adj[current] 
                             if tuple(sorted((current, nb))) in remaining_edges 
                             and nb not in path_set]
                if not candidates:
                    break
                # Pick neighbor with highest remaining degree
                candidates.sort(key=lambda nb: -rem_degree[nb])
                next_node = candidates[0]
                path.insert(0, next_node)
                path_set.add(next_node)
                remaining_edges.discard(tuple(sorted((current, next_node))))
                rem_degree[current] -= 1
                rem_degree[next_node] -= 1
            
            # Extend from v end
            while True:
                current = path[-1]
                candidates = [nb for nb in adj[current] 
                             if tuple(sorted((current, nb))) in remaining_edges 
                             and nb not in path_set]
                if not candidates:
                    break
                candidates.sort(key=lambda nb: -rem_degree[nb])
                next_node = candidates[0]
                path.append(next_node)
                path_set.add(next_node)
                remaining_edges.discard(tuple(sorted((current, next_node))))
                rem_degree[current] -= 1
                rem_degree[next_node] -= 1
            
            paths.append(path)
        
        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = instance(*spec)
        paths = decompose(edges)
        results.append(paths)
        
    return results
