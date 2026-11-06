import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    results = []

    def eulerian_paths(edges, n):
        """Decompose edges into minimum number of edge-disjoint paths using Eulerian trail technique."""
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Find odd-degree vertices
        odd_vertices = [v for v in range(n) if len(adj[v]) % 2 == 1]
        
        # Pair up odd vertices
        pairs = []
        for i in range(0, len(odd_vertices), 2):
            if i + 1 < len(odd_vertices):
                pairs.append((odd_vertices[i], odd_vertices[i+1]))
        
        # Add dummy edges to make all degrees even
        # We'll track dummy edges separately
        dummy_edges = set()
        for u, v in pairs:
            dummy_edges.add((min(u, v), max(u, v)))
        
        # Build adjacency with dummy edges included (for Eulerian circuit)
        all_adj = defaultdict(list)
        for u, v in edges:
            all_adj[u].append(v)
            all_adj[v].append(u)
        for u, v in dummy_edges:
            all_adj[u].append(v)
            all_adj[v].append(u)
        
        # Find Eulerian circuit using Hierholzer's algorithm
        # We need to handle multigraphs (parallel edges from original + dummy)
        # Use edge IDs to track which edge we've used
        
        edge_id = 0
        edge_map = {}  # (u,v) -> list of edge_ids
        for u, v in edges:
            key = (min(u, v), max(u, v))
            if key not in edge_map:
                edge_map[key] = []
            edge_map[key].append(edge_id)
            edge_id += 1
        
        dummy_edge_ids = {}
        for u, v in dummy_edges:
            key = (min(u, v), max(u, v))
            if key not in edge_map:
                edge_map[key] = []
            edge_map[key].append(edge_id)
            dummy_edge_ids[edge_id] = (u, v)
            edge_id += 1
        
        # Build adjacency with edge IDs
        adj_edges = defaultdict(list)
        for u, v in edges:
            key = (min(u, v), max(u, v))
            for eid in edge_map[key]:
                adj_edges[u].append((v, eid))
                adj_edges[v].append((u, eid))
        
        for u, v in dummy_edges:
            key = (min(u, v), max(u, v))
            for eid in edge_map[key]:
                adj_edges[u].append((v, eid))
                adj_edges[v].append((u, eid))
        
        used_edges = set()
        
        def find_eulerian_circuit(start):
            """Find Eulerian circuit starting from start vertex."""
            stack = [start]
            circuit = []
            
            while stack:
                v = stack[-1]
                # Find an unused edge from v
                found = False
                for i, (nb, eid) in enumerate(adj_edges[v]):
                    if eid not in used_edges:
                        used_edges.add(eid)
                        stack.append(nb)
                        found = True
                        break
                if not found:
                    circuit.append(stack.pop())
            
            return circuit
        
        # Find Eulerian circuit(s)
        # Start from any vertex with edges
        start_vertex = None
        for v in range(n):
            if adj_edges[v]:
                start_vertex = v
                break
        
        if start_vertex is None:
            return []
        
        # Find all Eulerian circuits (graph might be disconnected after adding dummies)
        all_circuits = []
        remaining = set(range(n))
        
        while remaining:
            # Find a vertex with unused edges
            start = None
            for v in remaining:
                has_unused = any(eid not in used_edges for _, eid in adj_edges[v])
                if has_unused:
                    start = v
                    break
            if start is None:
                break
            
            circuit = find_eulerian_circuit(start)
            all_circuits.append(circuit)
            for v in circuit:
                remaining.discard(v)
        
        # Now break each circuit into paths by removing dummy edges
        paths = []
        for circuit in all_circuits:
            # circuit is a list of vertices forming a cycle
            # Remove dummy edges to break into paths
            # A dummy edge connects two odd vertices
            # We need to identify which consecutive pairs in the circuit are dummy edges
            
            # Build list of edges in the circuit
            circuit_edges = []
            for i in range(len(circuit)):
                u = circuit[i]
                v = circuit[(i+1) % len(circuit)]
                key = (min(u, v), max(u, v))
                if key in dummy_edges:
                    circuit_edges.append((u, v, True))  # dummy
                else:
                    circuit_edges.append((u, v, False))  # original
            
            # Break at dummy edges
            # A path starts after a dummy edge and ends at the next dummy edge
            if not any(d for _, _, d in circuit_edges):
                # No dummy edges, it's a pure cycle of original edges
                # Break it into a path by removing one edge
                if len(circuit) > 1:
                    paths.append(circuit[:-1])  # remove last vertex to make it a path
                continue
            
            # Find positions of dummy edges
            dummy_positions = [i for i, (_, _, d) in enumerate(circuit_edges) if d]
            
            # The circuit is: v0 -e0-> v1 -e1-> v2 ... -e_{k-1}-> v0
            # If e_i is dummy, then the path starts at v_{i+1} and ends at v_i (going forward)
            
            # Actually, let's think differently:
            # The circuit visits vertices in order. When we traverse a dummy edge, we "break" the path.
            # So we have segments between dummy edges.
            
            num_dummies = len(dummy_positions)
            if num_dummies == 0:
                # Pure cycle
                if len(circuit) > 1:
                    paths.append(circuit[:-1])
                continue
            
            # For each dummy edge at position i, the edge goes from circuit[i] to circuit[i+1]
            # The path segment starts at circuit[i+1] (after the dummy) and goes until the next dummy
            
            for idx, pos in enumerate(dummy_positions):
                # This dummy edge connects circuit[pos] to circuit[pos+1]
                # The next dummy is at dummy_positions[(idx+1) % num_dummies]
                next_pos = dummy_positions[(idx + 1) % num_dummies]
                
                # The path goes from circuit[pos+1] through ... to circuit[next_pos]
                # But we need to handle wrap-around
                
                if next_pos > pos + 1:
                    # Path from circuit[pos+1] to circuit[next_pos]
                    path_vertices = circuit[pos+1:next_pos+1]
                elif next_pos < pos + 1:
                    # Wrap around
                    path_vertices = circuit[pos+1:] + circuit[:next_pos+1]
                else:
                    # Adjacent dummies, single vertex path?
                    path_vertices = [circuit[pos+1]]
                
                if len(path_vertices) > 1:
                    # Verify all edges are original
                    valid = True
                    for j in range(len(path_vertices) - 1):
                        u, v = path_vertices[j], path_vertices[j+1]
                        key = (min(u, v), max(u, v))
                        if key in dummy_edges:
                            valid = False
                            break
                    if valid:
                        paths.append(path_vertices)
        
        # If no paths found (shouldn't happen), fallback
        if not paths:
            for e in edges:
                paths.append(list(e))
        
        return paths

    for i, spec in enumerate(specs):
        time_left = deadline - time.time()
        if time_left <= 0:
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue

        edges = instance(*spec)
        n = spec[0]
        
        paths = eulerian_paths(edges, n)
        
        # Verify correctness
        if time.time() < deadline - 5.0:
            used_edges = set()
            valid = True
            for path in paths:
                if len(path) < 2:
                    continue
                for j in range(len(path) - 1):
                    u, v = path[j], path[j+1]
                    e = (min(u, v), max(u, v))
                    if e not in set(edges):
                        valid = False
                        break
                    if e in used_edges:
                        valid = False
                        break
                    used_edges.add(e)
                if len(set(path)) != len(path):
                    valid = False
                    break
                if not valid:
                    break
            
            if not valid or used_edges != set(edges):
                # Fallback: greedy
                adj = defaultdict(set)
                for u, v in edges:
                    adj[u].add(v)
                    adj[v].add(u)
                
                remaining = set(edges)
                paths = []
                while remaining and time.time() < deadline - 2.0:
                    best_path = None
                    best_len = 0
                    active = set()
                    for u, v in remaining:
                        active.add(u)
                        active.add(v)
                    
                    for start in list(active)[:20]:
                        has_edge = any((min(start,nb),max(start,nb)) in remaining for nb in adj[start])
                        if not has_edge:
                            continue
                        path = [start]
                        visited = {start}
                        current = start
                        while True:
                            cands = [nb for nb in adj[current] if nb not in visited and (min(current,nb),max(current,nb)) in remaining]
                            if not cands:
                                break
                            best_nb = min(cands, key=lambda nb: sum(1 for m in adj[nb] if m not in visited and (min(nb,m),max(nb,m)) in remaining))
                            path.append(best_nb)
                            visited.add(best_nb)
                            current = best_nb
                        if len(path) > best_len:
                            best_len = len(path)
                            best_path = path
                    
                    if best_path and len(best_path) > 1:
                        paths.append(best_path)
                        for j in range(len(best_path)-1):
                            remaining.discard((min(best_path[j],best_path[j+1]),max(best_path[j],best_path[j+1])))
                    elif remaining:
                        e = next(iter(remaining))
                        paths.append(list(e))
                        remaining.discard(e)
                
                for e in remaining:
                    paths.append(list(e))
        
        results.append(paths)

    return results
