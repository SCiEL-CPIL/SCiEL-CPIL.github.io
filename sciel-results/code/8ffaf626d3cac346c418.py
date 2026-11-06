"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict, deque
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # safety margin
    
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def min_cost_max_flow(n_nodes, source, sink, edges):
        """
        Min-cost max-flow using successive shortest paths with Bellman-Ford.
        edges: list of (u, v, capacity, cost)
        Returns (max_flow, min_cost)
        """
        # Build residual graph
        # graph[u] = list of [v, capacity, cost, reverse_index]
        graph = [[] for _ in range(n_nodes)]
        
        def add_edge(u, v, cap, cost):
            graph[u].append([v, cap, cost, len(graph[v])])
            graph[v].append([u, 0, -cost, len(graph[u]) - 1])
        
        for u, v, cap, cost in edges:
            add_edge(u, v, cap, cost)
        
        flow = 0
        cost = 0
        parent_v = [None] * n_nodes
        parent_e = [None] * n_nodes
        
        while True:
            # Bellman-Ford to find shortest path
            dist = [float('inf')] * n_nodes
            dist[source] = 0
            in_queue = [False] * n_nodes
            queue = deque([source])
            in_queue[source] = True
            
            while queue:
                u = queue.popleft()
                in_queue[u] = False
                for i, (v, cap, c, _) in enumerate(graph[u]):
                    if cap > 0 and dist[u] + c < dist[v]:
                        dist[v] = dist[u] + c
                        parent_v[v] = u
                        parent_e[v] = i
                        if not in_queue[v]:
                            queue.append(v)
                            in_queue[v] = True
                            
            if dist[sink] == float('inf'):
                break
                
            # Augment flow
            v = sink
            while v != source:
                u = parent_v[v]
                i = parent_e[v]
                graph[u][i][1] -= 1
                graph[v][graph[u][i][3]][1] += 1
                v = u
                
            flow += 1
            cost += dist[sink]
            
        return flow, cost

    def solve_exact_flow(edges, time_limit=30.0):
        """
        Solve minimum path decomposition using min-cost max-flow.
        
        The key insight: A path decomposition of a graph can be formulated as
        a flow problem where:
        - Each vertex v has an in-node v_in and out-node v_out
        - Each edge (u,v) in G becomes two directed arcs: u_out -> v_in and v_out -> u_in, each with capacity 1, cost 0
        - Add source S connected to each v_in with capacity 1, cost 1 (starting a path)
        - Add each v_out connected to sink T with capacity 1, cost 0 (ending a path)
        - Add zero-cost return arcs v_in -> v_out with capacity 1, cost 0 (continuing a path)
        
        The min-cost max-flow will find the minimum number of paths needed to cover all edges.
        """
        if not edges:
            return []
        
        if time.time() > start_time + time_limit:
            return [[u, v] for u, v in edges]
        
        # Build adjacency
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        vertices = sorted(set().union(*[set(e) for e in edges]))
        if not vertices:
            return []
            
        v_to_idx = {v: i for i, v in enumerate(vertices)}
        n_v = len(vertices)
        
        # Network nodes:
        # S = 0
        # v_in = 1 + i for i in range(n_v)
        # v_out = 1 + n_v + i for i in range(n_v)
        # T = 1 + 2*n_v
        S = 0
        T = 1 + 2 * n_v
        n_nodes = T + 1
        
        def in_node(v):
            return 1 + v_to_idx[v]
        
        def out_node(v):
            return 1 + n_v + v_to_idx[v]
        
        flow_edges = []
        
        # 1. Source to v_in: capacity 1, cost 1 (starting a new path)
        for v in vertices:
            flow_edges.append((S, in_node(v), 1, 1))
        
        # 2. v_in to v_out: capacity 1, cost 0 (continuing a path through vertex)
        for v in vertices:
            flow_edges.append((in_node(v), out_node(v), 1, 0))
        
        # 3. For each edge (u,v) in G: add directed arcs u_out -> v_in and v_out -> u_in
        #    Each with capacity 1, cost 0
        for u, v in edges:
            flow_edges.append((out_node(u), in_node(v), 1, 0))
            flow_edges.append((out_node(v), in_node(u), 1, 0))
        
        # 4. v_out to sink: capacity 1, cost 0 (ending a path)
        for v in vertices:
            flow_edges.append((out_node(v), T, 1, 0))
        
        # Solve min-cost max-flow
        # We want to push as much flow as possible (max flow), which will cover all edges
        # The cost will be the number of paths (each path start costs 1)
        
        flow, min_cost = min_cost_max_flow(n_nodes, S, T, flow_edges)
        
        # Extract the flow to get the path decomposition
        # Rebuild the graph to check which edges have flow
        graph = [[] for _ in range(n_nodes)]
        
        def add_edge_with_tracking(u, v, cap, cost, edge_id):
            graph[u].append([v, cap, cost, len(graph[v]), edge_id])
            graph[v].append([u, 0, -cost, len(graph[u]) - 1, -1])
        
        edge_ids = []
        for u, v, cap, cost in flow_edges:
            edge_ids.append((u, v, cap, cost))
            add_edge_with_tracking(u, v, cap, cost, len(edge_ids) - 1)
        
        # Re-solve to get the actual flow pattern
        flow2, cost2 = min_cost_max_flow(n_nodes, S, T, flow_edges)
        
        # For reconstruction, we need to trace the flow
        # Let's rebuild and solve again, keeping track of the parent pointers
        graph = [[] for _ in range(n_nodes)]
        
        def add_edge(u, v, cap, cost):
            graph[u].append([v, cap, cost, len(graph[v])])
            graph[v].append([u, 0, -cost, len(graph[u]) - 1])
        
        for u, v, cap, cost in flow_edges:
            add_edge(u, v, cap, cost)
        
        # Solve and track the paths
        parent_v = [None] * n_nodes
        parent_e = [None] * n_nodes
        paths = []
        
        while True:
            dist = [float('inf')] * n_nodes
            dist[S] = 0
            in_queue = [False] * n_nodes
            queue = deque([S])
            in_queue[S] = True
            
            while queue:
                u = queue.popleft()
                in_queue[u] = False
                for i, (v, cap, c, _) in enumerate(graph[u]):
                    if cap > 0 and dist[u] + c < dist[v]:
                        dist[v] = dist[u] + c
                        parent_v[v] = u
                        parent_e[v] = i
                        if not in_queue[v]:
                            queue.append(v)
                            in_queue[v] = True
                            
            if dist[T] == float('inf'):
                break
            
            # Trace back from T to S
            path_nodes = []
            v = T
            while v != S:
                path_nodes.append(v)
                u = parent_v[v]
                i = parent_e[v]
                graph[u][i][1] -= 1
                graph[v][graph[u][i][3]][1] += 1
                v = u
            path_nodes.append(S)
            path_nodes.reverse()
            
            # Convert node sequence to vertex path
            # S -> v_in -> v_out -> ... -> T
            # v_in corresponds to vertex vertices[idx] where idx = node - 1
            # v_out corresponds to vertex vertices[idx] where idx = node - 1 - n_v
            
            vertex_path = []
            for node in path_nodes[1:-1]:  # Skip S and T
                if 1 <= node <= n_v:
                    # v_in node
                    idx = node - 1
                    vertex_path.append(vertices[idx])
                elif 1 + n_v <= node <= 1 + 2 * n_v:
                    # v_out node
                    idx = node - 1 - n_v
                    # Don't add duplicate vertex
                    if not vertex_path or vertex_path[-1] != vertices[idx]:
                        vertex_path.append(vertices[idx])
            
            if len(vertex_path) > 1:
                paths.append(vertex_path)
        
        if not paths:
            return [[u, v] for u, v in edges]
        
        # Verify: check that all edges are covered exactly once
        used_edges = set()
        for path in paths:
            for i in range(len(path) - 1):
                e = tuple(sorted((path[i], path[i + 1])))
                if e in used_edges:
                    # Invalid: edge used twice
                    return [[u, v] for u, v in edges]
                used_edges.add(e)
        
        if used_edges != set(edges):
            # Not all edges covered or extra edges
            return [[u, v] for u, v in edges]
        
        return paths

    def solve_heuristic(edges, time_limit=10.0):
        """
        Improved heuristic: Multi-start greedy with local search.
        """
        if not edges:
            return []
        
        edge_set = set(edges)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        best_paths = None
        best_count = float('inf')
        start_time = time.time()
        
        # Try multiple restarts with different random seeds
        for restart in range(200):
            if time.time() > start_time + time_limit:
                break
                
            rng = random.Random(restart * 7 + 13)
            current_edges = set(edge_set)
            paths = []
            
            while current_edges:
                # Choose start vertex
                degs = defaultdict(int)
                for u, v in current_edges:
                    degs[u] += 1
                    degs[v] += 1
                    
                if not degs:
                    break
                    
                max_deg = max(degs.values())
                candidates = [v for v, d in degs.items() if d == max_deg]
                start_v = rng.choice(candidates)
                
                # Build path
                path = [start_v]
                visited = {start_v}
                last_v = start_v
                
                while True:
                    neighbors = []
                    for nb in adj[last_v]:
                        if nb not in visited:
                            e = tuple(sorted((last_v, nb)))
                            if e in current_edges:
                                # Warnsdorff: count onward moves
                                onward = 0
                                for nn in adj[nb]:
                                    if nn not in visited and nn != last_v:
                                        e2 = tuple(sorted((nb, nn)))
                                        if e2 in current_edges:
                                            onward += 1
                                neighbors.append((onward, nb))
                    
                    if not neighbors:
                        break
                        
                    neighbors.sort(key=lambda x: x[0])
                    min_onward = neighbors[0][0]
                    best_nbs = [nb for on, nb in neighbors if on == min_onward]
                    next_v = rng.choice(best_nbs)
                    
                    path.append(next_v)
                    visited.add(next_v)
                    e = tuple(sorted((last_v, next_v)))
                    current_edges.discard(e)
                    last_v = next_v
                    
                if len(path) > 1:
                    paths.append(path)
                else:
                    for e in list(current_edges):
                        paths.append(list(e))
                    current_edges = set()
                    break
                    
            if len(paths) < best_count:
                best_count = len(paths)
                best_paths = [p[:] for p in paths]
                
        if best_paths is None:
            return [[u, v] for u, v in edges]
            
        return best_paths

    results = []
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    
    for i, spec in enumerate(specs):
        g_edges = instance(*spec)
        time_left = deadline - time.time()
        if time_left < 5:
            paths = [[u, v] for u, v in g_edges]
        else:
            # Try exact flow first with time limit
            tl_exact = min(15.0, time_left / 2.0)
            paths = solve_exact_flow(g_edges, time_limit=tl_exact)
            
            # Verify solution quality; if it's just the fallback (each edge as a path), try heuristic
            if len(paths) == len(g_edges):
                tl_heur = min(10.0, time_left - 5.0)
                if tl_heur > 0:
                    heur_paths = solve_heuristic(g_edges, time_limit=tl_heur)
                    if len(heur_paths) < len(paths):
                        paths = heur_paths
        results.append(paths)
        
    return results

# EVOLVE-BLOCK-END
