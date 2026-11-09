"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque
import random

def construct():
    start_time = time.time()
    deadline = 135.0  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def build_adjacency(edges):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def bfs_shortest_path(adj, start, end):
        """BFS to find shortest path from start to end, return path or None."""
        if start == end:
            return [start]
        visited = {start}
        parent = {start: None}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor in adj[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent[neighbor] = node
                    if neighbor == end:
                        # Reconstruct path
                        path = []
                        curr = end
                        while curr is not None:
                            path.append(curr)
                            curr = parent[curr]
                        path.reverse()
                        return path
                    queue.append(neighbor)
        return None

    def compute_all_pairs_shortest_paths(adj, vertices):
        """Compute shortest path distances and parents for all pairs."""
        dist = {}
        parent = {}
        for src in vertices:
            dist[src] = {}
            parent[src] = {}
            # BFS from src
            visited = {src}
            parent[src][src] = None
            dist[src][src] = 0
            queue = deque([(src, 0)])
            while queue:
                node, d = queue.popleft()
                for neighbor in adj[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        dist[src][neighbor] = d + 1
                        parent[src][neighbor] = node
                        queue.append((neighbor, d + 1))
        return dist, parent

    def get_path_from_parent(parent, src, dst):
        """Reconstruct path from src to dst using parent pointers."""
        if dst not in parent[src]:
            return None
        path = []
        curr = dst
        while curr is not None:
            path.append(curr)
            curr = parent[src][curr]
        path.reverse()
        return path

    def greedy_path_cover(edges, adj, dist, parent, vertices, max_paths_hint=None):
        """
        Greedy path cover: repeatedly merge endpoints of paths using shortest paths.
        """
        if not edges:
            return []

        # Start with each edge as its own path
        paths = []
        for u, v in edges:
            if u < v:
                paths.append([u, v])
            else:
                paths.append([v, u])

        # Track edges used in paths
        used_edges = set()
        for path in paths:
            for i in range(len(path) - 1):
                e = tuple(sorted((path[i], path[i+1])))
                used_edges.add(e)

        # For each path, track its endpoints
        # path_endpoints[i] = (start, end) of path i
        # We want to merge paths by connecting end of one to start of another
        # using a shortest path that doesn't use already-used edges

        # Greedy: repeatedly find the best merge
        # A merge connects path A's end to path B's start via a shortest path
        # The new path is A + shortest_path + B (reversed if needed)

        # Actually, let's think about this differently.
        # We have a set of paths. We want to merge them.
        # Merging path P1 (end at v1) and path P2 (start at v2) via shortest path v1->v2
        # creates a new path. The cost is the length of the shortest path.
        # But we need the shortest path to use only unused edges.

        # This is complex. Let's use a simpler heuristic:
        # 1. Start with each edge as a path.
        # 2. Repeatedly find two paths that can be merged with the shortest connecting path.
        # 3. Merge them.

        # For efficiency, let's use a priority queue of potential merges.

        import heapq

        # State: list of paths, each path is a list of vertices
        # We'll track which edges are used
        # For merging, we need to find shortest path between endpoint of one path
        # and endpoint of another, using only unused edges.

        # This is expensive. Let's use a different approach:
        # Use the fact that we have all-pairs shortest paths (ignoring edge usage).
        # This gives a lower bound. For actual merging, we need to check if the
        # shortest path uses only unused edges.

        # Alternative: use a flow-based or matching-based approach.
        # But for now, let's do a simple greedy with local search.

        # Simple greedy:
        # 1. Compute degree of each vertex in the graph.
        # 2. Vertices with odd degree must be endpoints of paths.
        # 3. Pair up odd-degree vertices using shortest paths.
        # 4. The paths formed by these pairings plus the original edges form the cover.

        # Actually, the standard approach for minimum path cover in an undirected graph:
        # The minimum number of edge-disjoint paths that cover all edges is at least
        # max(1, ceil(odd_degree_vertices / 2)) where odd_degree_vertices is the number
        # of vertices with odd degree.
        # This is because each path has exactly two endpoints, and each endpoint must
        # be a vertex of odd degree (since internal vertices have even degree in the
        # subgraph formed by the path).

        # Wait, that's not quite right. Let me think again.
        # In a path, the two endpoints have degree 1 in the path, and internal vertices
        # have degree 2. So in the union of all paths, a vertex's degree in the original
        # graph equals the sum of its degrees in each path. If a vertex is an endpoint
        # of k paths and an internal vertex of l paths, its degree is k + 2l.
        # So the number of endpoints of a vertex (counting multiplicity) has the same
        # parity as the vertex's degree in the original graph.

        # The minimum number of paths is at least ceil(odd_count / 2) where odd_count
        # is the number of vertices with odd degree.

        # To achieve this bound, we need to pair up odd-degree vertices and find
        # edge-disjoint paths between each pair. This is equivalent to finding a
        # minimum-weight perfect matching on the odd-degree vertices, where the weight
        # of an edge (u, v) is the shortest path distance between u and v.

        # But the paths must be edge-disjoint, which makes this more complex.

        # For a practical approach, let's use the following heuristic:
        # 1. Find all odd-degree vertices.
        # 2. Pair them up greedily using shortest paths.
        # 3. For each pair, find a shortest path using unused edges.
        # 4. The union of all these paths covers all edges.

        # Let's implement this.

        # Compute degrees
        degree = defaultdict(int)
        for u, v in edges:
            degree[u] += 1
            degree[v] += 1

        odd_vertices = [v for v in vertices if degree[v] % 2 == 1]

        if not odd_vertices:
            # All vertices have even degree, so we can cover with 1 path (Eulerian)
            # But we need to find an Eulerian path/circuit.
            # For now, return a single path that traverses all edges.
            # This requires finding an Eulerian trail.
            return _find_eulerian_trail(edges, adj)

        # Pair up odd vertices using greedy matching with shortest paths
        # Sort odd vertices by degree (highest first) to prioritize
        odd_vertices_sorted = sorted(odd_vertices, key=lambda v: degree[v], reverse=True)

        # Greedy pairing
        used_in_matching = set()
        pairs = []
        for v in odd_vertices_sorted:
            if v in used_in_matching:
                continue
            # Find the closest unused odd vertex
            best_w = None
            best_dist = float('inf')
            for w in odd_vertices_sorted:
                if w == v or w in used_in_matching:
                    continue
                if v in dist and w in dist[v]:
                    d = dist[v][w]
                    if d < best_dist:
                        best_dist = d
                        best_w = w
            if best_w is not None:
                pairs.append((v, best_w))
                used_in_matching.add(v)
                used_in_matching.add(best_w)

        # For each pair, find a shortest path using unused edges
        used_edges = set()
        result_paths = []
        for u, v in pairs:
            # Find shortest path from u to v using only unused edges
            path = _find_shortest_path_unused_edges(adj, u, v, used_edges)
            if path is not None:
                # Mark edges as used
                for i in range(len(path) - 1):
                    e = tuple(sorted((path[i], path[i+1])))
                    used_edges.add(e)
                result_paths.append(path)
            else:
                # Fallback: use direct path if possible
                # This shouldn't happen for connected graphs
                pass

        # Now, the remaining edges (not used in the above paths) need to be covered.
        # They form subgraphs where all vertices have even degree (since we removed
        # paths between odd-degree vertices). So each connected component is Eulerian.
        # We can find Eulerian circuits for each component.

        remaining_edges = set(tuple(sorted(e)) for e in edges) - used_edges
        remaining_edges = set(remaining_edges)

        # Find connected components of remaining edges
        remaining_adj = defaultdict(set)
        for u, v in remaining_edges:
            remaining_adj[u].add(v)
            remaining_adj[v].add(u)

        visited = set()
        for v in remaining_adj:
            if v not in visited:
                # Find Eulerian circuit in this component
                circuit = _find_eulerian_circuit(remaining_adj, v)
                if circuit:
                    result_paths.append(circuit)
                for node in remaining_adj:
                    if node in remaining_adj and node not in visited:
                        # BFS to mark visited
                        pass

        # Mark all remaining edges as visited
        for u, v in remaining_edges:
            visited.add(u)
            visited.add(v)

        return result_paths

    def _find_eulerian_trail(edges, adj):
        """Find an Eulerian trail in a connected graph."""
        if not edges:
            return []

        # Find start vertex (odd degree or any vertex if all even)
        degree = defaultdict(int)
        for u, v in edges:
            degree[u] += 1
            degree[v] += 1

        start = None
        for v in degree:
            if degree[v] % 2 == 1:
                start = v
                break
        if start is None:
            start = list(degree.keys())[0] if degree else None

        if start is None:
            return []

        # Hierholzer's algorithm
        edge_set = set(tuple(sorted(e)) for e in edges)
        adj_dict = defaultdict(list)
        for u, v in edge_set:
            adj_dict[u].append(v)
            adj_dict[v].append(u)

        stack = [start]
        trail = []
        while stack:
            v = stack[-1]
            if adj_dict[v]:
                w = adj_dict[v].pop()
                # Remove edge (v, w)
                e = tuple(sorted((v, w)))
                if e in edge_set:
                    edge_set.remove(e)
                stack.append(w)
            else:
                trail.append(stack.pop())

        trail.reverse()
        return trail

    def _find_eulerian_circuit(adj, start):
        """Find an Eulerian circuit starting from start in a connected component."""
        if not adj[start]:
            return [start]

        # Hierholzer's algorithm
        # We need to work with a mutable adjacency structure
        adj_copy = defaultdict(list)
        for u in adj:
            adj_copy[u] = list(adj[u])

        stack = [start]
        circuit = []
        while stack:
            v = stack[-1]
            if adj_copy[v]:
                w = adj_copy[v].pop()
                stack.append(w)
            else:
                circuit.append(stack.pop())

        circuit.reverse()
        return circuit

    def _find_shortest_path_unused_edges(adj, start, end, used_edges):
        """Find shortest path from start to end using only unused edges."""
        if start == end:
            return [start]
        visited = {start}
        parent = {start: None}
        queue = deque([(start, None)])
        while queue:
            node, prev_edge = queue.popleft()
            for neighbor in adj[node]:
                e = tuple(sorted((node, neighbor)))
                if e in used_edges:
                    continue
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent[neighbor] = node
                    if neighbor == end:
                        path = []
                        curr = end
                        while curr is not None:
                            path.append(curr)
                            curr = parent[curr]
                        path.reverse()
                        return path
                    queue.append((neighbor, e))
        return None

    # Main loop
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    results = []

    for n, p, seed in specs:
        if time.time() > deadline - 10:
            # Not enough time, return what we have
            break

        edges = instance(n, p, seed)
        adj = build_adjacency(edges)
        vertices = list(range(n))

        # Compute all-pairs shortest paths
        dist, parent = compute_all_pairs_shortest_paths(adj, vertices)

        # Try greedy path cover
        paths = greedy_path_cover(edges, adj, dist, parent, vertices)

        results.append(paths)

    return results

# EVOLVE-BLOCK-END
