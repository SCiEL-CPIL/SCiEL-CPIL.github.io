"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def _build_adj(edges):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def _find_shortest_path(adj, start, end, forbidden):
        """BFS to find shortest path from start to end avoiding forbidden vertices (except endpoints)."""
        if start == end:
            return [start]
        visited = {start}
        parent = {start: None}
        queue = deque([start])
        while queue:
            cur = queue.popleft()
            for nb in adj[cur]:
                if nb in visited:
                    continue
                if nb in forbidden and nb != end:
                    continue
                if nb == end:
                    # Reconstruct path
                    path = [end]
                    while parent[nb] is not None:
                        path.append(parent[nb])
                        nb = parent[nb]
                    path.reverse()
                    return path
                visited.add(nb)
                parent[nb] = cur
                queue.append(nb)
        return None

    def _find_vertex_disjoint_paths(odd_verts, adj, edges_set):
        """
        Find vertex-disjoint paths pairing up odd-degree vertices.
        Returns list of (u, v, path) where path connects u to v.
        """
        if not odd_verts:
            return []
        
        # Greedy: repeatedly find shortest path between two unpaired odd vertices
        unpaired = set(odd_verts)
        pairs = []
        
        while len(unpaired) >= 2:
            found = False
            # Try all pairs, find shortest path
            best_pair = None
            best_path = None
            best_len = float('inf')
            
            verts_list = list(unpaired)
            for i in range(len(verts_list)):
                for j in range(i + 1, len(verts_list)):
                    u, v = verts_list[i], verts_list[j]
                    # Forbidden: all other unpaired vertices (to maintain vertex-disjointness)
                    forbidden = unpaired - {u, v}
                    path = _find_shortest_path(adj, u, v, forbidden)
                    if path is not None:
                        # Check all edges exist in graph
                        valid = True
                        for k in range(len(path) - 1):
                            e = tuple(sorted((path[k], path[k+1])))
                            if e not in edges_set:
                                valid = False
                                break
                        if valid:
                            plen = len(path) - 1
                            if plen < best_len:
                                best_len = plen
                                best_pair = (u, v)
                                best_path = path
            
            if best_pair is None:
                # Should not happen for connected graph, but fallback
                break
            
            u, v = best_pair
            pairs.append((u, v, best_path))
            unpaired.remove(u)
            unpaired.remove(v)
            found = True
        
        return pairs

    def _eulerian_circuit(adj, edges_count):
        """
        Find Eulerian circuit in an undirected multigraph.
        adj maps vertex -> list of (neighbor, edge_id)
        Returns list of vertices in circuit order, or None.
        """
        if edges_count == 0:
            return []
        
        # Find a vertex with non-zero degree
        start = None
        for v in adj:
            if adj[v]:
                start = v
                break
        if start is None:
            return []
        
        # Hierholzer's algorithm
        stack = [start]
        circuit = []
        # Use a mutable adjacency structure
        adj_list = defaultdict(list)
        for v in adj:
            adj_list[v] = list(adj[v])
        
        while stack:
            cur = stack[-1]
            if adj_list[cur]:
                # Take an edge
                nb, eid = adj_list[cur].pop()
                stack.append(nb)
            else:
                circuit.append(stack.pop())
        
        circuit.reverse()
        return circuit

    def _path_decomposition_optimal(edges):
        """
        Optimal path decomposition using Gallai's theorem.
        Returns list of paths (vertex lists).
        """
        if not edges:
            return []
        
        edges_set = set(edges)
        adj = _build_adj(edges)
        
        # Compute degrees
        deg = defaultdict(int)
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        
        odd_verts = [v for v, d in deg.items() if d % 2 == 1]
        
        if not odd_verts:
            # Already Eulerian: one circuit
            # Build adjacency with edge IDs
            adj_eid = defaultdict(list)
            for i, (u, v) in enumerate(edges):
                adj_eid[u].append((v, i))
                adj_eid[v].append((u, i))
            circuit = _eulerian_circuit(adj_eid, len(edges))
            if circuit:
                return [circuit]
            else:
                return [[u, v] for u, v in edges]
        
        # Find vertex-disjoint paths pairing odd vertices
        pairs = _find_vertex_disjoint_paths(odd_verts, adj, edges_set)
        
        if len(pairs) != len(odd_verts) // 2:
            # Fallback to greedy
            return _greedy_fallback(edges)
        
        # Build augmented multigraph: original edges + pair paths
        # Edge IDs: 0..len(edges)-1 for original, then pair path edges
        adj_eid = defaultdict(list)
        eid = 0
        for u, v in edges:
            adj_eid[u].append((v, eid))
            adj_eid[v].append((u, eid))
            eid += 1
        
        # Add pair path edges
        pair_edge_ids = []  # (eid, path) for each pair
        for u, v, path in pairs:
            for k in range(len(path) - 1):
                a, b = path[k], path[k + 1]
                adj_eid[a].append((b, eid))
                adj_eid[b].append((a, eid))
                eid += 1
            pair_edge_ids.append((u, v, path, eid - (len(path) - 1)))  # start eid
        
        # Find Eulerian circuit
        circuit = _eulerian_circuit(adj_eid, eid)
        
        if not circuit:
            return _greedy_fallback(edges)
        
        # Cut the circuit at pair path edges to extract paths
        # Mark which edges in the circuit are pair-path edges
        pair_eid_set = set()
        for u, v, path, start_eid in pair_edge_ids:
            for k in range(len(path) - 1):
                pair_eid_set.add(start_eid + k)
        
        # Walk the circuit, cutting at pair edges
        # The circuit is a sequence of vertices; consecutive vertices form edges
        # We need to identify which consecutive pairs are pair edges
        # Rebuild: for each consecutive pair in circuit, determine if it's a pair edge
        
        # Actually, let's track edges as we walk
        # Rebuild adjacency with proper edge tracking
        # Since circuit gives vertices, we need to map consecutive vertices to edge IDs
        # This is tricky with multigraphs. Let me use a different approach.
        
        # Alternative: walk the circuit and whenever we traverse a pair edge, that's a cut point
        # Let me redo with a proper edge-based walk
        
        # Rebuild adjacency properly
        adj_eid2 = defaultdict(list)
        eid = 0
        edge_map = {}  # eid -> (u, v, is_pair)
        for u, v in edges:
            adj_eid2[u].append((v, eid))
            adj_eid2[v].append((u, eid))
            edge_map[eid] = (u, v, False)
            eid += 1
        
        for u, v, path, start_eid in pair_edge_ids:
            for k in range(len(path) - 1):
                a, b = path[k], path[k + 1]
                adj_eid2[a].append((b, eid))
                adj_eid2[b].append((a, eid))
                edge_map[eid] = (a, b, True)
                eid += 1
        
        # Find Eulerian circuit with edge tracking
        adj_list = defaultdict(list)
        for v in adj_eid2:
            adj_list[v] = list(adj_eid2[v])
        
        start = None
        for v in adj_list:
            if adj_list[v]:
                start = v
                break
        if start is None:
            return _greedy_fallback(edges)
        
        stack = [start]
        circuit_edges = []  # list of (u, v, eid) in order
        circuit_vertices = []
        
        while stack:
            cur = stack[-1]
            if adj_list[cur]:
                nb, eid = adj_list[cur].pop()
                stack.append(nb)
                circuit_edges.append((cur, nb, eid))
            else:
                circuit_vertices.append(stack.pop())
                if len(stack) > 0:
                    # The edge just before this backtracking
                    pass
        
        # The circuit_edges is in reverse order of traversal
        circuit_edges.reverse()
        circuit_vertices.reverse()
        
        if not circuit_edges:
            return _greedy_fallback(edges)
        
        # Now walk circuit_edges, cutting at pair edges
        paths = []
        current_path = [circuit_vertices[0]]
        
        for i, (u, v, eid) in enumerate(circuit_edges):
            current_path.append(v)
            is_pair = edge_map[eid][2]
            if is_pair:
                # End current path, start new
                if len(current_path) > 1:
                    paths.append(current_path)
                current_path = [v]
        
        # Close the last path
        if len(current_path) > 1:
            paths.append(current_path)
        
        # Verify: all paths should use only original edges
        # Filter out any path that uses pair edges (shouldn't happen if we cut correctly)
        # Actually, the paths between cuts should only use original edges
        
        # Validate and return
        valid_paths = []
        for path in paths:
            valid = True
            for k in range(len(path) - 1):
                e = tuple(sorted((path[k], path[k+1])))
                if e not in edges_set:
                    valid = False
                    break
            if valid and len(set(path)) == len(path):
                valid_paths.append(path)
        
        if not valid_paths:
            return _greedy_fallback(edges)
        
        return valid_paths

    def _greedy_fallback(edges):
        """Simple greedy fallback."""
        remaining = set(edges)
        adj = _build_adj(edges)
        paths = []
        
        while remaining:
            e = next(iter(remaining))
            u, v = e
            path = [u, v]
            remaining.discard(e)
            # Try to extend
            changed = True
            while changed:
                changed = False
                # Extend at end
                for nb in adj[path[-1]]:
                    if nb not in path:
                        e2 = tuple(sorted((path[-1], nb)))
                        if e2 in remaining:
                            path.append(nb)
                            remaining.discard(e2)
                            changed = True
                            break
                # Extend at start
                if not changed:
                    for nb in adj[path[0]]:
                        if nb not in path:
                            e2 = tuple(sorted((path[0], nb)))
                            if e2 in remaining:
                                path.insert(0, nb)
                                remaining.discard(e2)
                                changed = True
                                break
            paths.append(path)
        
        # Handle remaining
        for e in remaining:
            paths.append(list(e))
        
        return paths

    # Process each graph
    results = []
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    
    for spec in specs:
        if time.time() > deadline:
            results.append([])
            continue
        
        edges = instance(*spec)
        
        if time.time() < deadline - 10:
            paths = _path_decomposition_optimal(edges)
        else:
            paths = [[u, v] for u, v in edges]
        
        results.append(paths)
    
    return results
# EVOLVE-BLOCK-END
