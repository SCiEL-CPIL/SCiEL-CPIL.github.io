import time
from random import Random
from collections import deque
from scipy.optimize import linear_sum_assignment

def construct():
    start_time = time.time()
    DEADLINE = start_time + 115

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def get_odd_vertices(edges):
        degree = {}
        for u, v in edges:
            degree[u] = degree.get(u, 0) + 1
            degree[v] = degree.get(v, 0) + 1
        return [v for v in degree if degree[v] % 2 == 1]

    def find_eulerian_circuit(adj, start):
        """Find Eulerian circuit using Hierholzer's algorithm."""
        if not adj[start]:
            return [start]
        
        circuit = []
        stack = [start]
        visited_edges = set()
        
        while stack:
            v = stack[-1]
            next_v = None
            for u in adj[v]:
                e = tuple(sorted((v, u)))
                if e not in visited_edges:
                    next_v = u
                    visited_edges.add(e)
                    break
            
            if next_v is not None:
                stack.append(next_v)
            else:
                circuit.append(stack.pop())
        
        circuit.reverse()
        return circuit

    def bfs_distance(u, v, adj, n):
        """BFS to find shortest path distance from u to v."""
        if u == v:
            return 0
        visited = {u}
        queue = deque([(u, 0)])
        while queue:
            node, dist = queue.popleft()
            for nb in adj[node]:
                if nb not in visited:
                    if nb == v:
                        return dist + 1
                    visited.add(nb)
                    queue.append((nb, dist + 1))
        return n  # fallback if not connected (shouldn't happen)

    def optimal_pairing(odd_verts, adj, n):
        """Find minimum-weight perfect matching of odd vertices using BFS distances."""
        k = len(odd_verts)
        if k <= 1:
            return []
        
        # Build cost matrix using BFS distances
        cost = [[0] * k for _ in range(k)]
        for i in range(k):
            for j in range(i + 1, k):
                d = bfs_distance(odd_verts[i], odd_verts[j], adj, n)
                cost[i][j] = d
                cost[j][i] = d
        
        # Use linear_sum_assignment for minimum weight perfect matching
        row_ind, col_ind = linear_sum_assignment(cost)
        pairs = [(odd_verts[i], odd_verts[j]) for i, j in zip(row_ind, col_ind) if i < j]
        return pairs

    def decompose_to_paths(edges, n):
        """Decompose edge set into minimum number of simple paths."""
        if not edges:
            return []
        
        edges_set = set(edges)
        odd_verts = get_odd_vertices(edges)
        
        # Build adjacency
        adj = {i: set() for i in range(n)}
        for u, v in edges_set:
            adj[u].add(v)
            adj[v].add(u)
        
        if not odd_verts:
            # Eulerian circuit: pick two vertices, add dummy edge
            # Find two vertices with positive degree
            active = [v for v in range(n) if adj[v]]
            if len(active) < 2:
                # Only one vertex or none - shouldn't happen for connected graph with edges
                return [[v] for v in active] if active else []
            
            u, v = active[0], active[1]
            dummy_edge = tuple(sorted((u, v)))
            
            # Add dummy edge
            adj[u].add(v)
            adj[v].add(u)
            
            # Find Eulerian circuit
            circuit = find_eulerian_circuit(adj, u)
            
            # Remove dummy edge from circuit
            # Find where dummy edge appears
            dummy_idx = -1
            for i in range(len(circuit) - 1):
                e = tuple(sorted((circuit[i], circuit[i+1])))
                if e == dummy_edge:
                    dummy_idx = i
                    break
            if dummy_idx == -1:
                e = tuple(sorted((circuit[-1], circuit[0])))
                if e == dummy_edge:
                    dummy_idx = len(circuit) - 1
            
            if dummy_idx != -1:
                start_idx = (dummy_idx + 1) % len(circuit)
                end_idx = dummy_idx
                path = []
                curr = start_idx
                while True:
                    path.append(circuit[curr])
                    if curr == end_idx:
                        break
                    curr = (curr + 1) % len(circuit)
                return [path] if len(path) > 1 else []
            else:
                # Dummy edge not found? Return full circuit as path (remove last vertex if it's first)
                if circuit[-1] == circuit[0]:
                    circuit = circuit[:-1]
                return [circuit] if len(circuit) > 1 else []
        else:
            # Pair odd vertices optimally
            pairs = optimal_pairing(odd_verts, adj, n)
            
            # Add dummy edges for pairs
            dummy_edges_set = set()
            for u, v in pairs:
                de = tuple(sorted((u, v)))
                dummy_edges_set.add(de)
                adj[u].add(v)
                adj[v].add(u)
            
            # Find Eulerian circuit in augmented graph
            start = odd_verts[0]
            circuit = find_eulerian_circuit(adj, start)
            
            # Split circuit at dummy edges
            segments = []
            current_segment = [circuit[0]]
            
            for i in range(len(circuit) - 1):
                u, v = circuit[i], circuit[i+1]
                e = tuple(sorted((u, v)))
                if e in dummy_edges_set:
                    if len(current_segment) > 1:
                        segments.append(current_segment)
                    current_segment = [v]
                else:
                    current_segment.append(v)
            
            # Check last to first
            u, v = circuit[-1], circuit[0]
            e = tuple(sorted((u, v)))
            if e in dummy_edges_set:
                if len(current_segment) > 1:
                    segments.append(current_segment)
            else:
                if len(current_segment) > 1:
                    segments.append(current_segment)
            
            return segments

    def verify_and_fix(paths, edges_set, n):
        """Verify paths are simple, use valid edges, are edge-disjoint, and cover all edges.
        If not, fall back to greedy."""
        covered = set()
        for p in paths:
            # Check simple
            if len(set(p)) != len(p):
                return None
            for i in range(len(p) - 1):
                e = tuple(sorted((p[i], p[i+1])))
                if e in covered:
                    return None  # edge-disjoint violation
                if e not in edges_set:
                    return None  # invalid edge
                covered.add(e)
        
        if covered != edges_set:
            return None
        
        return paths

    def path_cover_greedy(edges, n):
        """Greedy fallback: extract maximal simple paths."""
        remaining_edges = set(edges)
        paths = []
        
        while remaining_edges:
            sub_adj = {i: set() for i in range(n)}
            for u, v in remaining_edges:
                sub_adj[u].add(v)
                sub_adj[v].add(u)
            
            active_verts = [v for v in range(n) if sub_adj[v]]
            if not active_verts:
                break
            
            start = active_verts[0]
            path = [start]
            visited = {start}
            current = start
            
            while True:
                candidates = [nb for nb in sub_adj[current] if nb not in visited]
                if not candidates:
                    break
                # Choose neighbor with fewest remaining edges (leaf preference)
                best_next = min(candidates, key=lambda nb: len(sub_adj[nb]))
                path.append(best_next)
                visited.add(best_next)
                e = tuple(sorted((current, best_next)))
                remaining_edges.discard(e)
                sub_adj[current].discard(best_next)
                sub_adj[best_next].discard(current)
                current = best_next
            
            if len(path) > 1:
                paths.append(path)
        
        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > DEADLINE:
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
        
        edges = instance(*spec)
        n = spec[0]
        edges_set = set(edges)
        
        # Try Eulerian method with optimal pairing
        try:
            paths = decompose_to_paths(edges, n)
            if paths is None:
                paths = path_cover_greedy(edges, n)
            else:
                verified = verify_and_fix(paths, edges_set, n)
                if verified is None:
                    paths = path_cover_greedy(edges, n)
                else:
                    paths = verified
        except Exception:
            paths = path_cover_greedy(edges, n)
        
        results.append(paths)
    
    return results
