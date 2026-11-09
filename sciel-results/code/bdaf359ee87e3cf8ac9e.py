import random
import time
from collections import defaultdict, deque
import numpy as np
from scipy.optimize import linear_sum_assignment

def construct():
    start_time = time.time()
    time_limit = 145.0

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def bfs_shortest_path(start, end, adj):
        if start == end:
            return [start]
        visited = {start}
        parent = {start: None}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            if node == end:
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

    def find_matching_pairs(odd_vertices, adj):
        if not odd_vertices:
            return []
        m = len(odd_vertices)
        if m == 0:
            return []
        
        # Compute distance matrix
        dist_matrix = np.zeros((m, m))
        for i in range(m):
            for j in range(i + 1, m):
                path = bfs_shortest_path(odd_vertices[i], odd_vertices[j], adj)
                if path is not None:
                    d = len(path) - 1
                    dist_matrix[i, j] = d
                    dist_matrix[j, i] = d
                else:
                    dist_matrix[i, j] = 1000
                    dist_matrix[j, i] = 1000
        
        row_ind, col_ind = linear_sum_assignment(dist_matrix)
        
        pairs = []
        seen = set()
        for r, c in zip(row_ind, col_ind):
            if r != c:
                key = (min(r, c), max(r, c))
                if key not in seen:
                    seen.add(key)
                    pairs.append((odd_vertices[r], odd_vertices[c]))
        return pairs

    def extract_cycles(adj, remaining_edges):
        """
        Decompose the Eulerian subgraph (defined by remaining_edges) into edge-disjoint cycles.
        Returns a list of cycles, where each cycle is a list of vertices.
        """
        # Build local adjacency for remaining edges
        local_adj = defaultdict(set)
        for u, v in remaining_edges:
            local_adj[u].add(v)
            local_adj[v].add(u)
        
        cycles = []
        edges_to_visit = set(remaining_edges)
        
        # We need to find all edge-disjoint cycles.
        # We can iterate through edges, and for each unvisited edge, find a cycle containing it.
        # To do this efficiently, we can use Hierholzer's on the subgraph formed by unvisited edges.
        
        # Since the graph is Eulerian, we can just run Hierholzer's until no edges are left.
        # But Hierholzer's finds ONE circuit. We need to decompose into cycles.
        # Standard approach:
        # 1. Pick an edge.
        # 2. Follow it to find a cycle.
        # 3. Remove cycle edges.
        # 4. Repeat.
        
        # To find a cycle containing a specific edge (u,v):
        # We can use DFS/BFS to find a path from v back to u in the remaining graph.
        
        while edges_to_visit:
            # Pick an arbitrary edge to start
            u, v = next(iter(edges_to_visit))
            
            # Find a path from v to u in the remaining graph
            # This path + the edge (u,v) forms a cycle.
            # To ensure the cycle is simple, we need a simple path from v to u.
            # Since the remaining graph is Eulerian (all even degrees), such a path always exists?
            # Yes, because u and v are connected in the Eulerian graph, and we can find a simple path.
            
            # Use BFS to find shortest path from v to u
            path = bfs_shortest_path(v, u, local_adj)
            if path is None:
                # Should not happen in connected Eulerian component, but handle just in case
                # If not connected, the graph is not Eulerian?
                # If the graph is disconnected, we handle components separately.
                # But instance_graph is connected. Removing edges might disconnect it?
                # Yes. But each component is still Eulerian.
                # So if BFS fails, it means u and v are in different components?
                # But they share an edge, so they are in the same component.
                # So BFS should succeed.
                break
            
            # Cycle is u -> path[0] ... path[-1] -> u
            # path is v ... u
            cycle_vertices = [u] + path
            # The edges in this cycle are:
            # (u, v) and edges in path.
            
            # Remove these edges from edges_to_visit and local_adj
            # Edge (u, v)
            edges_to_visit.discard((u, v))
            edges_to_visit.discard((v, u))
            
            # Edges in path
            for i in range(len(path) - 1):
                a, b = path[i], path[i + 1]
                edges_to_visit.discard((a, b))
                edges_to_visit.discard((b, a))
                
            # Update local_adj?
            # We can just rebuild local_adj from edges_to_visit for the next iteration?
            # That is O(E^2). E is small (~100). So it's fine.
            
            cycles.append(cycle_vertices)
            
            # Rebuild local_adj for next iteration
            local_adj = defaultdict(set)
            for a, b in edges_to_visit:
                local_adj[a].add(b)
                local_adj[b].add(a)
                
        return cycles

    def cycle_to_paths(cycle):
        """
        Convert a simple cycle (list of vertices) into a list of simple paths covering all its edges.
        Returns 2 paths: 
        1. v0 -> v1 -> ... -> vk-1
        2. vk-1 -> v0 (single edge)
        """
        if len(cycle) <= 1:
            return []
        
        # cycle is [v0, v1, ..., vk-1, v0]
        # We want to cover edges (v0,v1), ..., (vk-1,v0)
        
        # Path 1: v0 to vk-1
        path1 = cycle[:-1]
        
        # Path 2: vk-1 to v0
        path2 = [cycle[-1], cycle[0]]
        
        return [path1, path2]

    def decompose(edges):
        if not edges:
            return []
        
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        remaining_edges = set(edges)
        paths = []
        
        # Phase 1: Handle Odd Vertices
        while True:
            if time.time() - start_time > time_limit - 5:
                break
            
            # Compute degrees in remaining_edges
            degree = defaultdict(int)
            for u, v in remaining_edges:
                degree[u] += 1
                degree[v] += 1
            
            odd_vertices = [v for v in degree if degree[v] % 2 == 1]
            
            if not odd_vertices:
                break
            
            # Find matching pairs
            pairs = find_matching_pairs(odd_vertices, adj)
            
            if not pairs:
                # Should not happen if graph is connected and has odd vertices
                break
            
            # For each pair, find shortest path and add to result
            for a, b in pairs:
                path = bfs_shortest_path(a, b, adj)
                if path:
                    paths.append(path)
                    # Remove edges from remaining_edges
                    for i in range(len(path) - 1):
                        u, v = path[i], path[i + 1]
                        remaining_edges.discard(tuple(sorted((u, v))))
                    # Update adj? 
                    # We can just use remaining_edges to check availability in BFS?
                    # bfs_shortest_path uses adj. We need to update adj or pass remaining_edges.
                    # Let's update adj to reflect remaining edges for accurate BFS in next iterations.
                    # Actually, we should rebuild adj or filter neighbors in BFS.
                    # Rebuilding adj is safer.
            
            # Rebuild adj from remaining_edges
            adj = defaultdict(set)
            for u, v in remaining_edges:
                adj[u].add(v)
                adj[v].add(u)
                
            # Check if we made progress
            # If not, break to avoid infinite loop
            # (This is handled by the while condition on odd_vertices)

        # Phase 2: Handle Eulerian Part
        if remaining_edges:
            cycles = extract_cycles(adj, remaining_edges)
            for cycle in cycles:
                if time.time() - start_time > time_limit - 2:
                    break
                paths_in_cycle = cycle_to_paths(cycle)
                paths.extend(paths_in_cycle)
        
        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = instance(*spec)
        paths = decompose(edges)
        results.append(paths)
        
    return results
