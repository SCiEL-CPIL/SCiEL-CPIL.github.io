"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
from collections import deque

def construct():
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_connected_component(start, unused_edges, adj):
        """Find all edges in the connected component containing start."""
        comp_edges = set()
        visited = set()
        stack = [start]
        visited.add(start)
        
        while stack:
            node = stack.pop()
            for neighbor in adj.get(node, []):
                edge = (node, neighbor) if node < neighbor else (neighbor, node)
                if edge in unused_edges and edge not in comp_edges:
                    comp_edges.add(edge)
                    if neighbor not in visited:
                        visited.add(neighbor)
                        stack.append(neighbor)
        
        # Get all vertices in this component
        comp_vertices = set()
        for u, v in comp_edges:
            comp_vertices.add(u)
            comp_vertices.add(v)
            
        return comp_edges, comp_vertices

    def get_odd_vertices(comp_edges):
        """Get vertices with odd degree in the subgraph defined by comp_edges."""
        deg = {}
        for u, v in comp_edges:
            deg[u] = deg.get(u, 0) + 1
            deg[v] = deg.get(v, 0) + 1
        return {v for v, d in deg.items() if d % 2 != 0}

    def find_eulerian_trail(comp_edges, adj):
        """Find an Eulerian trail in the subgraph defined by comp_edges.
        Assumes the component is connected and has 0 or 2 odd-degree vertices."""
        if not comp_edges:
            return []
            
        # Find start vertex: an odd-degree vertex if exists, else any vertex
        odd_verts = get_odd_vertices(comp_edges)
        if odd_verts:
            start = next(iter(odd_verts))
        else:
            start = next(iter(next(iter(comp_edges))))
            
        # Hierholzer's algorithm
        # We need to work with a local adjacency list for the component
        local_adj = {}
        for u, v in comp_edges:
            if u not in local_adj: local_adj[u] = []
            if v not in local_adj: local_adj[v] = []
            local_adj[u].append(v)
            local_adj[v].append(u)
            
        # We need to track which edges are used. Since it's Eulerian, we use each exactly once.
        # We can just pop from the lists.
        # But we need to ensure we don't use edges outside comp_edges.
        # The local_adj only contains edges in comp_edges.
        
        stack = [start]
        trail = []
        used_edges = set() # To be safe, though popping should handle it if we are careful
        
        # Standard Hierholzer with explicit edge tracking to avoid issues with multi-edges (though here simple graph)
        # Actually, for simple graphs, we can just use the adjacency lists and remove used edges.
        
        # Let's use a modified Hierholzer that tracks remaining edges in the component
        remaining = set(comp_edges)
        
        # Re-build adjacency for remaining edges
        rem_adj = {v: [] for v in local_adj}
        for u, v in remaining:
            rem_adj[u].append(v)
            rem_adj[v].append(u)
            
        stack = [start]
        path = []
        
        while stack:
            v = stack[-1]
            if rem_adj[v]:
                w = rem_adj[v].pop()
                # Remove edge (v, w) from remaining
                edge = (v, w) if v < w else (w, v)
                remaining.discard(edge)
                rem_adj[w].remove(v)
                stack.append(w)
            else:
                path.append(stack.pop())
                
        path.reverse()
        return path

    def extend_path(start, unused_edges, adj):
        """Greedy extension of a path starting from 'start'.
        Returns the path and the set of edges used."""
        path = [start]
        used_edges = set()
        current = start
        
        while True:
            extended = False
            for neighbor in adj.get(current, []):
                edge = (current, neighbor) if current < neighbor else (neighbor, current)
                if edge in unused_edges:
                    # Check if neighbor is already in path (simple path constraint)
                    if neighbor not in path:
                        unused_edges.remove(edge)
                        used_edges.add(edge)
                        path.append(neighbor)
                        current = neighbor
                        extended = True
                        break
            if not extended:
                break
                
        return path, used_edges

    def decompose_paths(edges):
        if not edges:
            return []
        
        # Build global adjacency list
        adj = {}
        for u, v in edges:
            if u not in adj:
                adj[u] = []
            if v not in adj:
                adj[v] = []
            adj[u].append(v)
            adj[v].append(u)
        
        unused_edges = set((u, v) if u < v else (v, u) for u, v in edges)
        paths = []
        
        while unused_edges:
            # Find a connected component
            # Pick an arbitrary edge to start
            start_edge = next(iter(unused_edges))
            start_node = start_edge[0]
            
            comp_edges, comp_vertices = find_connected_component(start_node, unused_edges, adj)
            
            odd_verts = get_odd_vertices(comp_edges)
            
            if len(odd_verts) <= 2:
                # Can form a single Eulerian trail
                trail = find_eulerian_trail(comp_edges, adj)
                if trail:
                    paths.append(trail)
                    # Remove all edges in this component from unused
                    for e in comp_edges:
                        unused_edges.discard(e)
            else:
                # Cannot form a single Eulerian trail.
                # Strategy: Find a leaf in the component if possible.
                leaf = None
                for v in comp_vertices:
                    # Check degree in comp_edges
                    deg = sum(1 for u, w in comp_edges if u == v or w == v)
                    if deg == 1:
                        leaf = v
                        break
                        
                if leaf:
                    # Start path from leaf
                    path, used_edges = extend_path(leaf, unused_edges, adj)
                    paths.append(path)
                    # Remove used edges from unused
                    for e in used_edges:
                        unused_edges.discard(e)
                else:
                    # No leaves. Pick an edge incident to an odd vertex.
                    # Or just pick any edge from the component.
                    # Let's pick an edge incident to an odd vertex to "balance" it somewhat?
                    # Or just pick the first edge in comp_edges.
                    edge = next(iter(comp_edges))
                    u, v = edge
                    # Prefer starting from an odd vertex if one is u or v
                    if u in odd_verts:
                        start_node = u
                    elif v in odd_verts:
                        start_node = v
                    else:
                        start_node = u
                        
                    # Remove the starting edge? No, extend_path will pick it.
                    # But extend_path starts from start_node and picks neighbors.
                    # We need to ensure it picks the edge we want if possible, or just let it pick.
                    path, used_edges = extend_path(start_node, unused_edges, adj)
                    paths.append(path)
                    for e in used_edges:
                        unused_edges.discard(e)
                        
        return paths

    result = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = instance(*spec)
        paths = decompose_paths(edges)
        result.append(paths)
        
    return result
# EVOLVE-BLOCK-END
