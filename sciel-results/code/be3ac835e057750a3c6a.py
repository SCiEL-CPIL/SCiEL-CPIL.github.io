import time
import random
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin for 120s limit

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def max_flow_bfs(capacity, source, sink):
        """BFS-based max flow (Edmonds-Karp). Returns flow value and parent pointers."""
        flow = 0
        while True:
            if time.time() > deadline:
                break
            parent = {source: None}
            queue = deque([source])
            found = False
            while queue and not found:
                u = queue.popleft()
                for v in range(len(capacity[u])):
                    if capacity[u][v] > 0 and v not in parent:
                        parent[v] = u
                        if v == sink:
                            found = True
                            break
                        queue.append(v)
            if not found:
                break
            # Find bottleneck
            path_flow = float('inf')
            v = sink
            while v != source:
                u = parent[v]
                path_flow = min(path_flow, capacity[u][v])
                v = u
            # Update residual capacities
            v = sink
            while v != source:
                u = parent[v]
                capacity[u][v] -= path_flow
                capacity[v][u] += path_flow
                v = u
            flow += path_flow
        return flow

    def extract_2factor(adj, n, remaining_edges):
        """
        Find a 2-factor (max set of vertex-disjoint cycles) using flow.
        Returns list of cycles (each cycle is a list of vertices).
        """
        if time.time() > deadline:
            return []
        
        # Build flow network for finding vertex-disjoint cycles
        # Node splitting: each vertex v becomes v_in -> v_out with capacity 1
        # For each edge (u,v), add edges u_out -> v_in and v_out -> u_in with capacity 1
        # Super source S connects to all v_in, super sink T connects from all v_out
        # Actually, for 2-factor: each vertex has degree exactly 2 in the subgraph
        # This is equivalent to finding a perfect matching in the bipartite double cover
        
        # Alternative: use edge-disjoint cycle packing via repeated shortest cycle finding
        # But let's try a different approach: find cycles greedily but smarter
        
        cycles = []
        used_edges = set()
        adj_local = {i: set(adj[i]) for i in range(n)}
        
        # Find cycles in order of increasing length (shorter cycles first)
        # This tends to pack more cycles
        while True:
            if time.time() > deadline:
                break
            
            # Find shortest cycle using BFS from each vertex
            best_cycle = None
            best_len = float('inf')
            
            for start in range(n):
                if not adj_local[start]:
                    continue
                
                # BFS to find shortest cycle through start
                visited = {start: 0}
                parent = {start: None}
                queue = deque([(start, 0)])
                
                while queue:
                    u, dist = queue.popleft()
                    if dist >= best_len:
                        continue
                    for v in sorted(adj_local[u]):
                        e = (min(u, v), max(u, v))
                        if e in used_edges:
                            continue
                        if v == start:
                            if dist + 1 >= 3 and dist + 1 < best_len:
                                # Reconstruct cycle
                                cycle = [start]
                                cur = u
                                while cur != start:
                                    cur = parent[cur]
                                    cycle.append(cur)
                                best_cycle = cycle
                                best_len = dist + 1
                        elif v not in visited:
                            visited[v] = dist + 1
                            parent[v] = u
                            queue.append((v, dist + 1))
            
            if best_cycle is None:
                break
            
            cycles.append(best_cycle)
            for i in range(len(best_cycle)):
                u = best_cycle[i]
                v = best_cycle[(i + 1) % len(best_cycle)]
                e = (min(u, v), max(u, v))
                used_edges.add(e)
                adj_local[u].discard(v)
                adj_local[v].discard(u)
        
        return cycles

    def find_longest_cycle(adj, n, remaining_edges):
        best_len = 0
        best_cycle = []
        
        for start in range(n):
            if not adj[start]:
                continue
            
            visited = {start: 0}
            parent = {start: None}
            in_stack = {start}
            
            def dfs(u):
                nonlocal best_len, best_cycle
                for v in sorted(adj[u]):
                    if (min(u, v), max(u, v)) not in remaining_edges:
                        continue
                    if v not in visited:
                        visited[v] = visited[u] + 1
                        parent[v] = u
                        in_stack.add(v)
                        dfs(v)
                        in_stack.remove(v)
                    elif v in in_stack:
                        length = visited[u] - visited[v] + 1
                        if length >= 3 and length > best_len:
                            cycle = [u]
                            cur = u
                            while cur != v:
                                cur = parent[cur]
                                cycle.append(cur)
                            best_cycle = cycle
                            best_len = length

            dfs(start)
            
            if time.time() > deadline:
                break

        return best_cycle

    def try_merge_into_cycles(pieces, n, adj, remaining_edges):
        """Try to merge single edges into cycles by finding paths between their endpoints."""
        improved = False
        i = 0
        while i < len(pieces) and time.time() < deadline:
            if len(pieces[i]) == 2:
                u, v = pieces[i]
                if (min(u, v), max(u, v)) in remaining_edges:
                    # Try to find a path from u to v using remaining edges (excluding direct edge)
                    path = find_path(u, v, adj, remaining_edges, exclude_edge=(min(u, v), max(u, v)))
                    if path and len(path) >= 3:
                        # Replace single edge with cycle
                        pieces[i] = path
                        for j in range(len(path)):
                            a, b = path[j], path[(j + 1) % len(path)]
                            e = (min(a, b), max(a, b))
                            if e in remaining_edges:
                                remaining_edges.discard(e)
                                adj[a].remove(b)
                                adj[b].remove(a)
                        improved = True
                    else:
                        i += 1
                else:
                    i += 1
            else:
                i += 1
        return improved

    def find_path(start, end, adj, remaining_edges, exclude_edge=None):
        """Find a path from start to end using remaining edges."""
        if start == end:
            return None
        queue = deque([(start, [start])])
        visited = {start}
        
        while queue and time.time() < deadline:
            node, path = queue.popleft()
            for neighbor in sorted(adj[node]):
                e = (min(node, neighbor), max(node, neighbor))
                if e not in remaining_edges:
                    continue
                if e == exclude_edge:
                    continue
                if neighbor == end:
                    return path + [neighbor]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return None

    def decompose_graph(edges, n):
        """Decompose graph into minimum number of cycles and edges."""
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        pieces = []
        remaining_edges = set(edges)
        
        # Phase 1: Extract 2-factors (sets of vertex-disjoint cycles) iteratively
        while remaining_edges and time.time() < deadline:
            # Try to find a 2-factor in the remaining graph
            cycles = extract_2factor(adj, n, remaining_edges)
            
            if not cycles:
                break
            
            for cycle in cycles:
                pieces.append(cycle)
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    e = (min(u, v), max(u, v))
                    if e in remaining_edges:
                        remaining_edges.discard(e)
                        adj[u].discard(v)
                        adj[v].discard(u)
            
            # Remove isolated vertices from consideration
            # Continue extracting 2-factors until no more can be found
        
        # Phase 2: For remaining edges, find longest cycles greedily
        while remaining_edges and time.time() < deadline:
            best_cycle = find_longest_cycle(adj, n, remaining_edges)
            if best_cycle:
                pieces.append(best_cycle)
                for i in range(len(best_cycle)):
                    u = best_cycle[i]
                    v = best_cycle[(i + 1) % len(best_cycle)]
                    e = (min(u, v), max(u, v))
                    if e in remaining_edges:
                        remaining_edges.discard(e)
                        adj[u].discard(v)
                        adj[v].discard(u)
            else:
                for e in remaining_edges:
                    pieces.append(list(e))
                break
        
        # Phase 3: Post-processing - try to merge single edges into cycles
        if time.time() < deadline:
            try_merge_into_cycles(pieces, n, adj, remaining_edges)
        
        # Phase 4: Try to merge two single edges that form a path into a 3-cycle
        # Actually, let's try: if we have two single edges (a,b) and (b,c), check if (a,c) exists
        # and was already used... no, we need to find unused edges.
        # Better: try to find triangles among remaining single edges
        
        return pieces

    results = []
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    
    for spec in specs:
        if time.time() > deadline:
            # Fallback: return single edges for all remaining graphs
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
        
        edges = instance(*spec)
        n = spec[0]
        pieces = decompose_graph(edges, n)
        results.append(pieces)
    
    return results
