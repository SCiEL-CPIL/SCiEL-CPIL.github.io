import time
import random
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def max_flow_unit(n, adj, source, sink):
        """
        Edmonds-Karp max flow for unit capacities.
        adj: dict of dict {u: {v: cap}}
        """
        # Copy adj to residual graph
        residual = defaultdict(lambda: defaultdict(int))
        for u in adj:
            for v in adj[u]:
                residual[u][v] += adj[u][v]
                residual[v][u] += 0 # Ensure reverse exists
        
        flow = 0
        while True:
            # BFS to find augmenting path
            parent = {source: None}
            queue = deque([source])
            found = False
            while queue and not found:
                u = queue.popleft()
                for v in residual[u]:
                    if residual[u][v] > 0 and v not in parent:
                        parent[v] = u
                        if v == sink:
                            found = True
                            break
                        queue.append(v)
            
            if not found:
                break
            
            # Find min capacity on path (always 1 for unit capacities, but let's be safe)
            v = sink
            min_cap = float('inf')
            while v != source:
                u = parent[v]
                min_cap = min(min_cap, residual[u][v])
                v = u
            
            # Update residual graph
            v = sink
            while v != source:
                u = parent[v]
                residual[u][v] -= min_cap
                residual[v][u] += min_cap
                v = u
            
            flow += min_cap
            
        return flow, residual

    def find_eulerian_circuit(adj, start):
        """
        Find an Eulerian circuit starting at 'start' in a graph where all degrees are even.
        Returns a list of vertices.
        """
        if not adj[start]:
            return [start]
        
        circuit = []
        stack = [start]
        visited_edges = set()
        
        while stack:
            u = stack[-1]
            # Find an unvisited edge from u
            next_v = None
            for v in adj[u]:
                e = tuple(sorted((u, v)))
                if e not in visited_edges:
                    next_v = v
                    break
            
            if next_v is None:
                circuit.append(stack.pop())
            else:
                e = tuple(sorted((u, next_v)))
                visited_edges.add(e)
                # We need to remove the edge from both directions?
                # Since we use visited_edges set, we just mark it.
                # But we need to be careful not to pick it again.
                stack.append(next_v)
        
        return circuit[::-1]

    def solve_graph(edges, seed):
        if not edges:
            return []
        
        edge_set = set(tuple(sorted(e)) for e in edges)
        adj = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Calculate degrees
        degrees = {v: len(adj[v]) for v in adj}
        odd_vertices = [v for v in degrees if degrees[v] % 2 != 0]
        
        if not odd_vertices:
            # Eulerian
            # Find all components
            visited = set()
            paths = []
            for v in adj:
                if v not in visited:
                    # BFS to find component
                    comp = set()
                    queue = deque([v])
                    visited.add(v)
                    while queue:
                        u = queue.popleft()
                        comp.add(u)
                        for w in adj[u]:
                            if w not in visited:
                                visited.add(w)
                                queue.append(w)
                    
                    # Check if component has edges
                    has_edges = False
                    for u in comp:
                        if adj[u]:
                            has_edges = True
                            break
                    if has_edges:
                        # Find Eulerian circuit
                        # Start with any vertex in comp
                        start = list(comp)[0]
                        circuit = find_eulerian_circuit(adj, start)
                        # Convert circuit to path
                        # Circuit is a cycle. Remove last vertex (which is same as first) to make it a path?
                        # No, a circuit v0, v1, ..., vk, v0.
                        # Path can be v0, v1, ..., vk.
                        # This covers all edges.
                        paths.append(circuit[:-1])
            
            return paths
        
        # Odd vertices exist
        # Try Max Flow approach
        n = max(adj.keys()) + 1
        source = n
        sink = n + 1
        
        flow_adj = defaultdict(lambda: defaultdict(int))
        for u in adj:
            for v in adj[u]:
                if u < v: # Avoid double counting
                    flow_adj[u][v] = 1
                    flow_adj[v][u] = 1
        
        for v in odd_vertices:
            flow_adj[source][v] = 1
            flow_adj[v][sink] = 1
        
        max_f, residual = max_flow_unit(n+2, flow_adj, source, sink)
        
        if max_f == len(odd_vertices) // 2:
            # Extract paths
            # The flow is from source to sink.
            # Trace paths from source to sink in the residual graph?
            # No, the flow is in the original graph.
            # We can find the paths by looking at the edges that have flow.
            # Since it's unit capacity, an edge has flow 1 if it was used.
            # We can reconstruct the paths by following the flow.
            
            # Build a list of edges used in the flow
            flow_edges = []
            for u in flow_adj:
                if u == source or u == sink:
                    continue
                for v in flow_adj[u]:
                    if v == source or v == sink:
                        continue
                    # Check if flow was sent u->v
                    # In residual graph, if residual[u][v] < flow_adj[u][v], then flow was sent.
                    # Actually, easier: just check if the edge is part of the flow.
                    # We can trace from source.
                    pass
            
            # Simpler: Just find paths in the graph induced by the flow edges.
            # The flow edges form a collection of paths connecting odd vertices.
            # We can find them by DFS.
            
            # Reconstruct flow edges
            used_edges = set()
            for u in flow_adj:
                if u == source or u == sink:
                    continue
                for v in flow_adj[u]:
                    if v == source or v == sink:
                        continue
                    if u < v:
                        # Check if this edge has flow
                        # If residual[u][v] is 0 and original cap was 1, it might be used.
                        # But residual is modified.
                        # Let's just use the fact that we have a max flow.
                        # We can identify the paths by looking at the degrees in the flow subgraph.
                        # Vertices with flow-degree 1 are endpoints (odd vertices).
                        # Vertices with flow-degree 2 are internal.
                        
                        # Let's just collect all edges that have flow.
                        # We can check if the edge is in the residual graph with 0 capacity?
                        # No, if cap was 1 and flow is 1, residual is 0.
                        # If cap was 1 and flow is 0, residual is 1.
                        # So if residual[u][v] == 0, it was used?
                        # Wait, residual[v][u] would be 1.
                        # So if residual[u][v] == 0 and residual[v][u] == 1, flow was u->v.
                        # Or if residual[u][v] == 1 and residual[v][u] == 0, flow was v->u.
                        
                        if residual[u][v] == 0 and residual[v][u] == 1:
                            used_edges.add(tuple(sorted((u, v))))
                        elif residual[u][v] == 1 and residual[v][u] == 0:
                            used_edges.add(tuple(sorted((u, v))))
            
            # Now we have a set of edges `used_edges` which form disjoint paths.
            # Find these paths.
            flow_paths = []
            remaining_flow_edges = set(used_edges)
            
            # Find start vertices (odd vertices in the flow subgraph)
            flow_adj_list = defaultdict(list)
            for u, v in used_edges:
                flow_adj_list[u].append(v)
                flow_adj_list[v].append(u)
            
            for v in odd_vertices:
                if flow_adj_list[v]:
                    # Start a path
                    path = [v]
                    visited = {v}
                    current = v
                    while True:
                        next_v = None
                        for w in flow_adj_list[current]:
                            if w not in visited:
                                next_v = w
                                break
                        if next_v is None:
                            break
                        visited.add(next_v)
                        path.append(next_v)
                        current = next_v
                    flow_paths.append(path)
            
            # Remove these edges from the main graph
            remaining_edges = edge_set - used_edges
            
            # Decompose remaining graph
            rem_adj = defaultdict(list)
            for u, v in remaining_edges:
                rem_adj[u].append(v)
                rem_adj[v].append(u)
            
            # Find Eulerian circuits in remaining graph
            visited_rem = set()
            rem_paths = []
            for v in rem_adj:
                if v not in visited_rem:
                    comp = set()
                    queue = deque([v])
                    visited_rem.add(v)
                    while queue:
                        u = queue.popleft()
                        comp.add(u)
                        for w in rem_adj[u]:
                            if w not in visited_rem:
                                visited_rem.add(w)
                                queue.append(w)
                    
                    has_edges = False
                    for u in comp:
                        if rem_adj[u]:
                            has_edges = True
                            break
                    if has_edges:
                        start = list(comp)[0]
                        circuit = find_eulerian_circuit(rem_adj, start)
                        rem_paths.append(circuit[:-1])
            
            return flow_paths + rem_paths
        
        # Fallback if max flow failed or not full
        # Use the original greedy method
        rng = random.Random(seed + 100)
        paths = []
        available = set(edge_set)
        
        def find_greedy_path(adj, available_edges, rng):
            if not available_edges:
                return []
            degrees = defaultdict(int)
            for u, v in available_edges:
                degrees[u] += 1
                degrees[v] += 1
            if not degrees:
                return []
            start_candidates = list(degrees.keys())
            rng.shuffle(start_candidates)
            best_path = []
            for start in start_candidates[:2]:
                path = [start]
                visited = {start}
                current = start
                while True:
                    neighbors = []
                    for v in adj[current]:
                        e = tuple(sorted((current, v)))
                        if e in available_edges and v not in visited:
                            neighbors.append(v)
                    if not neighbors:
                        break
                    next_node = rng.choice(neighbors)
                    visited.add(next_node)
                    path.append(next_node)
                    current = next_node
                if len(path) > len(best_path):
                    best_path = path
            return best_path

        while available:
            if time.time() > deadline:
                break
            path = find_greedy_path(adj, available, rng)
            if not path:
                u, v = next(iter(available))
                path = [u, v]
            for i in range(len(path) - 1):
                e = tuple(sorted((path[i], path[i+1])))
                if e in available:
                    available.remove(e)
            paths.append(path)
        
        if available:
            for u, v in available:
                paths.append([u, v])
        
        return paths

    results = []
    for i, spec in enumerate([(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]):
        if time.time() > deadline:
            edges = instance(*spec)
            results.append([[u, v] for u, v in edges])
            continue
        edges = instance(*spec)
        paths = solve_graph(edges, spec[2])
        results.append(paths)
        
    return results
