import time
import sys
from collections import defaultdict, deque
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    from random import Random

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def bfs_shortest_path(adj, src, dst):
        """BFS to find shortest path from src to dst in unweighted graph."""
        if src == dst:
            return [src]
        parent = {src: None}
        queue = deque([src])
        while queue:
            u = queue.popleft()
            if u == dst:
                # Reconstruct path
                path = []
                cur = dst
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                path.reverse()
                return path
            for v in adj.get(u, []):
                if v not in parent:
                    parent[v] = u
                    queue.append(v)
        return None  # Should not happen if graph is connected

    def find_eulerian_trails(adj, edge_set):
        """
        Find Eulerian trails in a multigraph.
        adj: dict of vertex -> list of neighbors (with duplicates for multi-edges)
        edge_set: set of (u,v) tuples for original edges (for tracking)
        Returns list of trails (vertex sequences).
        """
        # Build adjacency with edge counts
        # We need to handle multi-edges properly
        # Let's use a different approach: represent edges as (u, v, id)
        # But since we're working with a multigraph after augmentation,
        # let's just use adjacency lists with counts
        
        # Actually, let's use a simpler approach for Eulerian trails:
        # 1. Build adjacency list
        # 2. Find vertices with odd degree
        # 3. Start from odd-degree vertices (or any vertex if all even)
        # 4. Use Hierholzer's algorithm
        
        # Build adjacency with edge tracking
        # For simplicity, let's use a list of (neighbor, edge_id) pairs
        # But we need to be careful with multi-edges
        
        # Let's use a different strategy:
        # Since the graph is small, let's just use a greedy approach
        # to find trails that cover all edges
        
        # Actually, let me implement proper Eulerian trail finding
        # using Hierholzer's algorithm with proper edge tracking
        
        # Build adjacency as list of (neighbor, edge_index)
        edges_list = list(edge_set)
        edge_index = {e: i for i, e in enumerate(edges_list)}
        
        # Build adjacency with edge indices
        adj_with_edges = defaultdict(list)
        for i, (u, v) in enumerate(edges_list):
            adj_with_edges[u].append((v, i))
            adj_with_edges[v].append((u, i))
        
        # Find vertices with odd degree
        odd_vertices = [v for v in adj_with_edges if len(adj_with_edges[v]) % 2 == 1]
        
        # For Eulerian trail, we start from an odd vertex if exists, else any vertex
        # For Eulerian circuit, all vertices have even degree
        # We may have multiple connected components, but our graph is connected
        
        # Hierholzer's algorithm
        # We need to modify adjacency to remove edges as we use them
        # Let's use a copy of the adjacency with mutable lists
        
        adj_copy = {v: list(neighbors) for v, neighbors in adj_with_edges.items()}
        
        def hierholzer(start):
            """Find Eulerian trail starting from 'start'."""
            path = []
            stack = [start]
            while stack:
                v = stack[-1]
                if adj_copy[v]:
                    # Take the next available edge
                    next_v, edge_i = adj_copy[v].pop()
                    # Remove the reverse edge
                    # Find and remove the edge from next_v's list
                    for i, (nv, ei) in enumerate(adj_copy[next_v]):
                        if ei == edge_i:
                            adj_copy[next_v].pop(i)
                            break
                    stack.append(next_v)
                else:
                    # Backtrack
                    path.append(stack.pop())
            path.reverse()
            return path
        
        # Find all trails
        trails = []
        # Start from odd vertices first, then even vertices with remaining edges
        start_vertices = odd_vertices[:]
        # Add any vertices with remaining edges that aren't already in start_vertices
        for v in adj_copy:
            if adj_copy[v] and v not in start_vertices:
                start_vertices.append(v)
        
        for start in start_vertices:
            if adj_copy[start]:
                trail = hierholzer(start)
                if len(trail) > 1:
                    trails.append(trail)
        
        return trails

    def repair_trail_to_simple_path(trail, adj):
        """
        Convert a trail (which may repeat vertices) into a simple path
        by shortcutting repeated vertices using existing edges.
        """
        if len(trail) <= 2:
            return trail
        
        # Try to make it simple by removing cycles
        # If a vertex appears multiple times, we can shortcut
        seen = {}
        result = []
        for i, v in enumerate(trail):
            if v in seen:
                # We've seen this vertex before
                # We can potentially shortcut from the previous occurrence
                # to here, but we need to make sure the shortcut is valid
                # For now, let's just keep the first occurrence and skip
                # Actually, we need to be more careful
                pass
            else:
                seen[v] = i
                result.append(v)
        
        # The above is too simplistic. Let's use a different approach:
        # Greedily build a simple path by extending from the start
        if not trail:
            return trail
        
        path = [trail[0]]
        used_vertices = {trail[0]}
        current = trail[0]
        
        # Try to follow the trail, but if we hit a used vertex, shortcut
        for i in range(1, len(trail)):
            next_v = trail[i]
            if next_v in used_vertices:
                # We need to find an alternative
                # Try to find a neighbor of current that is not used
                # and is connected to next_v
                # For simplicity, let's just stop here if we can't extend
                # Actually, let's try to find a shortcut
                # Find a vertex in path that is adjacent to next_v
                shortcut_found = False
                for j in range(len(path) - 1, -1, -1):
                    if (path[j], next_v) in adj or (next_v, path[j]) in adj:
                        # We can shortcut to path[j]
                        # But we need to make sure the edge exists
                        # Check if edge (path[j], next_v) exists in original graph
                        edge = tuple(sorted((path[j], next_v)))
                        # We need to check if this edge is available
                        # For now, let's just accept it
                        path = path[:j+1]
                        path.append(next_v)
                        used_vertices = set(path)
                        current = next_v
                        shortcut_found = True
                        break
                if not shortcut_found:
                    # Can't extend, stop
                    break
            else:
                # Check if edge (current, next_v) exists
                edge = tuple(sorted((current, next_v)))
                # We assume the trail uses valid edges, so this should be fine
                path.append(next_v)
                used_vertices.add(next_v)
                current = next_v
        
        return path

    def decompose_graph(edges):
        """
        Decompose graph into edge-disjoint simple paths covering all edges.
        Strategy: 
        1. Build the graph
        2. Find odd-degree vertices
        3. Pair them up with shortest paths (augment the graph)
        4. Find Eulerian trails in the augmented graph
        5. Repair each trail to a simple path
        """
        if not edges:
            return []
        
        edge_set = set(tuple(sorted(e)) for e in edges)
        n = max(max(e) for e in edges) + 1
        
        # Build adjacency
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Find odd-degree vertices
        odd_vertices = [v for v in range(n) if len(adj[v]) % 2 == 1]
        
        # Pair up odd vertices with shortest paths
        # For simplicity, pair them in order (this is not optimal but works)
        # A better approach: use minimum weight matching, but for small graphs,
        # greedy pairing is sufficient
        
        # Build a list of edges to add
        augmentation_edges = []
        if len(odd_vertices) > 0:
            # Pair up odd vertices
            # Use a simple greedy: pair consecutive odd vertices
            # This is not optimal but should work for small graphs
            for i in range(0, len(odd_vertices) - 1, 2):
                u, v = odd_vertices[i], odd_vertices[i + 1]
                # Find shortest path between u and v
                path = bfs_shortest_path(adj, u, v)
                if path:
                    for j in range(len(path) - 1):
                        augmentation_edges.append(tuple(sorted((path[j], path[j + 1]))))
        
        # Build augmented graph
        augmented_edges = set(edge_set)
        for e in augmentation_edges:
            augmented_edges.add(e)
        
        # Find Eulerian trails in the augmented graph
        trails = find_eulerian_trails(adj, augmented_edges)
        
        # Repair each trail to a simple path
        paths = []
        for trail in trails:
            simple_path = repair_trail_to_simple_path(trail, adj)
            if len(simple_path) > 1:
                paths.append(simple_path)
        
        # Verify that we cover all original edges
        # If not, add missing edges as single-edge paths
        covered_edges = set()
        for path in paths:
            for i in range(len(path) - 1):
                covered_edges.add(tuple(sorted((path[i], path[i + 1]))))
        
        for e in edge_set:
            if e not in covered_edges:
                paths.append([e[0], e[1]])
        
        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > deadline:
            break
        edges = instance(*spec)
        paths = decompose_graph(edges)
        results.append(paths)
    
    # If we ran out of time, fill remaining with trivial edge paths
    while len(results) < 3:
        spec = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))[len(results)]
        edges = instance(*spec)
        results.append([[u, v] for u, v in edges])
    
    return results
