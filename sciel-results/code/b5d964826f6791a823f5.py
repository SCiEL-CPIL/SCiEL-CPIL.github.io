import time
import sys
from random import Random
from collections import deque
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = start_time + 110

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def _greedy_path_cover_fast(edges, n):
        rem_edges = set(tuple(sorted(e)) for e in edges)
        adj = {i: set() for i in range(n)}
        for u, v in rem_edges:
            adj[u].add(v)
            adj[v].add(u)
        paths = []
        while rem_edges:
            min_deg = float('inf')
            start_v = -1
            for v in range(n):
                deg = len(adj[v])
                if deg > 0 and deg < min_deg:
                    min_deg = deg
                    start_v = v
            if start_v == -1:
                break
            path = [start_v]
            visited_in_path = {start_v}
            current_v = start_v
            while True:
                candidates = []
                for nb in adj[current_v]:
                    if nb not in visited_in_path:
                        e_key = tuple(sorted((current_v, nb)))
                        if e_key in rem_edges:
                            candidates.append((len(adj[nb]), nb))
                if not candidates:
                    break
                candidates.sort()
                next_v = candidates[0][1]
                e_key = tuple(sorted((current_v, next_v)))
                rem_edges.discard(e_key)
                adj[current_v].discard(next_v)
                adj[next_v].discard(current_v)
                path.append(next_v)
                visited_in_path.add(next_v)
                current_v = next_v
            if len(path) > 1:
                paths.append(path)
            else:
                break
        return paths

    def _is_valid(paths, edges, n):
        edge_set = set(tuple(sorted(e)) for e in edges)
        used = set()
        for p in paths:
            if len(p) < 2:
                return False
            seen = set()
            for v in p:
                if v in seen:
                    return False
                seen.add(v)
            for i in range(len(p) - 1):
                e = tuple(sorted((p[i], p[i + 1])))
                if e not in edge_set:
                    return False
                if e in used:
                    return False
                used.add(e)
        return used == edge_set

    def _bfs_distances(adj, n, source):
        dist = [-1] * n
        dist[source] = 0
        q = deque([source])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)
        return dist

    def _matching_path_cover(edges, n):
        """Use T-join / matching approach to find optimal path cover."""
        degree = {i: 0 for i in range(n)}
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            degree[u] += 1
            degree[v] += 1
            adj[u].add(v)
            adj[v].add(u)
        
        odd = [v for v in range(n) if degree[v] % 2 == 1]
        
        if not odd:
            # All even: single Eulerian circuit
            # Build adjacency list with edge tracking
            adj_list = {i: [] for i in range(n)}
            for u, v in edges:
                adj_list[u].append(v)
                adj_list[v].append(u)
            
            # Hierholzer's algorithm
            circuit = []
            stack = []
            start = 0
            # Find a vertex with edges
            for i in range(n):
                if adj_list[i]:
                    start = i
                    break
            if not adj_list[start]:
                return []
            
            stack.append(start)
            while stack:
                v = stack[-1]
                if adj_list[v]:
                    nxt = adj_list[v].pop()
                    stack.append(nxt)
                else:
                    circuit.append(stack.pop())
            circuit.reverse()
            return [circuit]
        
        # Find optimal pairing of odd vertices using shortest path distances
        # Compute all-pairs shortest paths from odd vertices
        k = len(odd)
        dist_matrix = np.zeros((k, k), dtype=int)
        for i in range(k):
            dists = _bfs_distances(adj, n, odd[i])
            for j in range(k):
                if dists[odd[j]] == -1:
                    dist_matrix[i][j] = 10**9
                else:
                    dist_matrix[i][j] = dists[odd[j]]
        
        # Solve minimum weight perfect matching on k vertices using Hungarian
        # scipy.optimize.linear_sum_assignment solves min-cost assignment
        try:
            from scipy.optimize import linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(dist_matrix)
            pairing = list(zip(row_ind, col_ind))
        except ImportError:
            return None
        
        # For each pair, find shortest path
        virtual_edges = []
        for i, j in pairing:
            u, v = odd[i], odd[j]
            # BFS to find shortest path from u to v
            parent = {u: None}
            q = deque([u])
            while q:
                node = q.popleft()
                if node == v:
                    break
                for nb in adj[node]:
                    if nb not in parent:
                        parent[nb] = node
                        q.append(nb)
            # Reconstruct path
            path = []
            cur = v
            while cur is not None:
                path.append(cur)
                cur = parent.get(cur)
            path.reverse()
            if len(path) >= 2:
                for idx in range(len(path) - 1):
                    virtual_edges.append((path[idx], path[idx + 1]))
        
        # Build augmented multigraph: original edges + virtual edges
        # For Eulerian circuit, we need to track edges properly
        # Use edge IDs to handle multigraph
        edge_id = 0
        adj_mult = {i: [] for i in range(n)}  # list of (neighbor, edge_id)
        edge_list = []  # (u, v, is_virtual)
        
        for u, v in edges:
            adj_mult[u].append((v, edge_id))
            adj_mult[v].append((u, edge_id))
            edge_list.append((u, v, False))
            edge_id += 1
        
        for u, v in virtual_edges:
            adj_mult[u].append((v, edge_id))
            adj_mult[v].append((u, edge_id))
            edge_list.append((u, v, True))
            edge_id += 1
        
        # Hierholzer's on multigraph
        circuit = []
        stack = []
        start = 0
        for i in range(n):
            if adj_mult[i]:
                start = i
                break
        if not adj_mult[start]:
            return []
        
        stack.append(start)
        while stack:
            v = stack[-1]
            if adj_mult[v]:
                nxt, eid = adj_mult[v].pop()
                stack.append(nxt)
            else:
                circuit.append(stack.pop())
        circuit.reverse()
        
        # Split circuit at virtual edges
        # The circuit is a list of vertices. Virtual edges connect consecutive vertices.
        # Find positions where edge (circuit[i], circuit[i+1]) is virtual
        virtual_edge_set = set()
        for u, v in virtual_edges:
            virtual_edge_set.add(tuple(sorted((u, v))))
        
        paths = []
        current_path = [circuit[0]]
        for i in range(len(circuit) - 1):
            u, v = circuit[i], circuit[i + 1]
            current_path.append(v)
            e_key = tuple(sorted((u, v)))
            if e_key in virtual_edge_set:
                if len(current_path) >= 2:
                    paths.append(current_path)
                current_path = [v]
        if len(current_path) >= 2:
            paths.append(current_path)
        
        # Remove duplicate vertices if circuit wraps around
        # The circuit should start and end at same vertex for Eulerian circuit
        # But our paths are simple, so we need to verify
        # Actually, the circuit is a closed walk. When we split at virtual edges,
        # each segment should be a simple path.
        
        return paths

    def path_cover(edges, n):
        if not edges:
            return []
        
        # Try greedy first (fast)
        greedy_result = _greedy_path_cover_fast(edges, n)
        if not greedy_result or not _is_valid(greedy_result, edges, n):
            # Fallback: each edge as separate path
            return [[e[0], e[1]] for e in edges]
        
        # Try matching-based approach
        matching_result = None
        if time.time() < DEADLINE - 5:
            matching_result = _matching_path_cover(edges, n)
        
        if matching_result and _is_valid(matching_result, edges, n):
            if len(matching_result) < len(greedy_result):
                return matching_result
        
        return greedy_result

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > DEADLINE:
            while len(results) < 3:
                edges = instance(*spec)
                results.append([list(e) for e in edges])
            break
        
        edges = instance(*spec)
        n = spec[0]
        paths = path_cover(edges, n)
        results.append(paths)
    
    return results

if __name__ == "__main__":
    res = construct()
    print("Number of paths:", [len(r) for r in res])
    print("Total paths:", sum(len(r) for r in res))
