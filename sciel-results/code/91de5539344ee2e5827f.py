import time
from collections import defaultdict, deque

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    DEADLINE = 145.0  # safety margin vs 150s kill

    def instance(n, p, seed):
        from random import Random
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def build_adj(edges):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def eulerian_path_decomposition(edges, deadline):
        """
        Decompose edges into edge-disjoint simple paths using Eulerian augmentation.
        1. Find all odd-degree vertices.
        2. Add dummy edges to pair up odd vertices (minimum matching on odd vertices).
        3. Find Eulerian circuit in augmented graph.
        4. Split the circuit into simple paths at repeated vertices.
        5. Remove dummy edges from the split paths to get the final decomposition.
        """
        adj = build_adj(edges)
        deg = {v: len(adj[v]) for v in adj}
        odd_vertices = [v for v in deg if deg[v] % 2 == 1]
        
        # Pair up odd vertices with dummy edges
        dummy_edges = []
        if odd_vertices:
            # Simple greedy pairing
            odd_sorted = sorted(odd_vertices)
            for i in range(0, len(odd_sorted) - 1, 2):
                dummy_edges.append((odd_sorted[i], odd_sorted[i + 1]))
            if len(odd_sorted) % 2 == 1:
                # Should not happen in a connected graph, but handle gracefully
                dummy_edges.append((odd_sorted[-1], odd_sorted[-1]))
        
        # Build augmented adjacency with edge IDs
        # We need to track edges uniquely to find Eulerian circuit
        edge_list = list(edges)
        dummy_ids = []
        for de in dummy_edges:
            dummy_ids.append(len(edge_list))
            edge_list.append(de)
        
        # Build adjacency with edge indices
        adj_edges = defaultdict(list)
        for idx, (u, v) in enumerate(edge_list):
            adj_edges[u].append((v, idx))
            adj_edges[v].append((u, idx))
        
        # Find Eulerian circuit using Hierholzer's algorithm
        used_edges = [False] * len(edge_list)
        circuit = []
        
        def hierholzer(start):
            stack = [(start, -1)]  # (vertex, edge_index_used_to_reach)
            circuit_local = []
            while stack:
                v, _ = stack[-1]
                found = False
                for nb, eid in adj_edges[v]:
                    if not used_edges[eid]:
                        used_edges[eid] = True
                        circuit_local.append(v)
                        stack.append((nb, eid))
                        found = True
                        break
                if not found:
                    stack.pop()
            return circuit_local
        
        # Start from any vertex with edges
        start_v = edge_list[0][0] if edge_list else 0
        circuit = hierholzer(start_v)
        # circuit has len = number of edges in augmented graph
        # The circuit is a sequence of vertices: circuit[0], circuit[1], ..., circuit[-1]
        # where circuit[i] to circuit[i+1] is edge i (0-indexed)
        # Actually, Hierholzer's gives us the circuit in reverse or needs careful handling
        
        # Let me redo this more carefully
        used_edges = [False] * len(edge_list)
        circuit = []
        
        def find_eulerian(start):
            """Return the Eulerian circuit as a list of edge indices in order."""
            stack = [(start, -1)]
            circuit_edges = []
            while stack:
                v, prev_eid = stack[-1]
                found = False
                for nb, eid in adj_edges[v]:
                    if not used_edges[eid]:
                        used_edges[eid] = True
                        stack.append((nb, eid))
                        found = True
                        break
                if not found:
                    stack.pop()
                    if prev_eid >= 0:
                        circuit_edges.append(prev_eid)
            return circuit_edges
        
        circuit_edges = find_eulerian(start_v)
        # circuit_edges is in reverse order, so reverse it
        circuit_edges.reverse()
        
        # Now circuit_edges is a sequence of edge indices forming the Eulerian circuit
        # Convert to vertex sequence
        # The circuit visits vertices: v0, v1, v2, ..., vk where edge i connects vi and v(i+1)
        # We need to reconstruct the vertex sequence
        # Start with the first edge's first vertex
        if not circuit_edges:
            return []
        
        # Reconstruct vertex sequence from edge sequence
        # For each edge in the circuit, we know which vertex we're coming from
        vertex_seq = []
        current_v = None
        for i, eid in enumerate(circuit_edges):
            u, v = edge_list[eid]
            if current_v is None:
                # First edge, pick a direction
                # We need to determine the direction based on the circuit
                # This is tricky. Let me use a different approach.
                pass
        
        # Alternative: track the circuit as (vertex, edge) pairs
        # Let me redo the Eulerian circuit finding to track vertices directly
        
        used_edges = [False] * len(edge_list)
        
        def find_eulerian_with_vertices(start):
            """Return list of (vertex, edge_id) representing the circuit."""
            stack = [(start, -1)]
            result = []
            while stack:
                v, prev_eid = stack[-1]
                found = False
                for nb, eid in adj_edges[v]:
                    if not used_edges[eid]:
                        used_edges[eid] = True
                        stack.append((nb, eid))
                        found = True
                        break
                if not found:
                    stack.pop()
                    if prev_eid >= 0:
                        result.append((v, prev_eid))
            return result
        
        circuit_data = find_eulerian_with_vertices(start_v)
        circuit_data.reverse()
        
        # circuit_data is a list of (vertex, edge_id) where edge_id is the edge used to LEAVE that vertex
        # So the circuit is: circuit_data[0][0] --edge circuit_data[0][1]--> circuit_data[1][0] --edge circuit_data[1][1]--> ...
        # Wait, that's not quite right either. Let me think again.
        
        # In Hierholzer's, when we pop a vertex, we record it with the edge we used to get there.
        # Actually, the standard implementation records vertices in reverse Eulerian order.
        
        # Let me use a cleaner approach: just find a long path greedily but with backtracking for a few steps.
        
        # Actually, let me use a simpler but effective approach:
        # 1. Repeatedly find the longest simple path in the remaining graph
        # 2. Use a heuristic that prefers vertices with higher remaining degree
        # 3. Add small backtracking (depth 2-3) to avoid getting stuck in short paths
        
        remaining = set(edges)
        adj_rem = build_adj(edges)
        deg_rem = {v: len(adj_rem[v]) for v in adj_rem}
        paths = []
        
        while remaining and time.time() < deadline:
            # Find a good starting vertex: one with high remaining degree
            # Prefer vertices that are endpoints of many remaining edges
            candidates = []
            for v in deg_rem:
                if deg_rem[v] > 0:
                    candidates.append(v)
            
            if not candidates:
                break
            
            # Sort by remaining degree descending
            candidates.sort(key=lambda v: (-deg_rem[v], v))
            
            best_path = None
            best_path_len = 0
            
            # Try starting from top few candidates
            for start in candidates[:5]:
                if time.time() > deadline:
                    break
                # Greedy with limited backtracking
                path = _greedy_path_with_backtrack(start, adj_rem, remaining, deg_rem, deadline)
                if path and len(path) > best_path_len:
                    best_path = path
                    best_path_len = len(path)
                if best_path_len >= len(remaining):
                    break
            
            if best_path is None or len(best_path) < 2:
                # Fallback: take any single edge
                e = next(iter(remaining))
                best_path = list(e)
            
            # Remove edges from remaining
            for a, b in zip(best_path, best_path[1:]):
                e = (min(a, b), max(a, b))
                remaining.discard(e)
                # Update degrees
                deg_rem[a] = deg_rem.get(a, 0) - 1
                deg_rem[b] = deg_rem.get(b, 0) - 1
                if a in adj_rem and b in adj_rem[a]:
                    adj_rem[a].discard(b)
                if b in adj_rem and a in adj_rem[b]:
                    adj_rem[b].discard(a)
            
            paths.append(best_path)
        
        return paths
    
    def _greedy_path_with_backtrack(start, adj_rem, remaining, deg_rem, deadline, max_depth=3):
        """Find a long simple path starting from start, with limited backtracking."""
        best_path = [start]
        
        def extend(path, used_vertices, used_edges):
            nonlocal best_path
            if time.time() > deadline:
                return
            if len(path) > len(best_path):
                best_path = list(path)
            
            cur = path[-1]
            # Get available neighbors
            cands = []
            for nb in adj_rem.get(cur, ()):
                e = (min(cur, nb), max(cur, nb))
                if e not in used_edges and nb not in used_vertices:
                    cands.append(nb)
            
            if not cands:
                return
            
            # Sort by flexibility (fewer options first = more likely to be a dead end)
            def flex(nb):
                c = 0
                for m in adj_rem.get(nb, ()):
                    e = (min(nb, m), max(nb, m))
                    if e not in used_edges and m not in used_vertices:
                        c += 1
                return c
            
            cands.sort(key=lambda x: (flex(x), x))
            
            # Try top candidates with limited backtracking
            for nb in cands[:max_depth]:
                path.append(nb)
                used_vertices.add(nb)
                e = (min(cur, nb), max(cur, nb))
                used_edges.add(e)
                extend(path, used_vertices, used_edges)
                path.pop()
                used_vertices.discard(nb)
                used_edges.discard(e)
                if time.time() > deadline:
                    break
        
        used_vertices = {start}
        used_edges = set()
        extend([start], used_vertices, used_edges)
        
        return best_path if len(best_path) >= 2 else None

    result = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = instance(*spec)
        if time.time() > DEADLINE - 5:
            result.append([list(e) for e in edges])
            continue
        paths = eulerian_path_decomposition(edges, DEADLINE - 5)
        # Safety: ensure coverage
        used = set()
        for p in paths:
            for a, b in zip(p, p[1:]):
                used.add((min(a, b), max(a, b)))
        for e in edges:
            if e not in used:
                paths.append(list(e))
        result.append(paths)
    return result
# EVOLVE-BLOCK-END
