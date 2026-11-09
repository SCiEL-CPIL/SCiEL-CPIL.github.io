"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random
import numpy as np
from scipy.optimize import linear_sum_assignment

def construct():
    start_time = time.time()
    DEADLINE = start_time + 140  # 10s safety margin before 150s

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_eulerian_circuit(adj, start):
        """Find Eulerian circuit using Hierholzer's algorithm.
        adj: dict mapping vertex to list of (neighbor, edge_id) pairs
        Returns list of vertices in the circuit.
        """
        used_edges = set()
        stack = [start]
        circuit = []
        
        while stack:
            v = stack[-1]
            found = False
            for nb, eid in adj[v]:
                if eid not in used_edges:
                    used_edges.add(eid)
                    stack.append(nb)
                    found = True
                    break
            if not found:
                circuit.append(stack.pop())
        
        circuit.reverse()
        return circuit

    def bfs_distances(adj, src, n):
        """Compute shortest path distances from src to all other vertices."""
        dist = [float('inf')] * n
        dist[src] = 0
        visited = [False] * n
        visited[src] = True
        queue = [src]
        
        while queue:
            u = queue.pop(0)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    dist[v] = dist[u] + 1
                    queue.append(v)
        
        return dist

    def min_path_cover_matching(edges, n):
        """Find minimum edge-disjoint path cover using minimum-weight perfect matching.
        
        The minimum number of edge-disjoint paths covering all edges equals
        max(1, |odd_vertices| // 2) when the graph is connected.
        We pair odd-degree vertices optimally to minimize the number of paths.
        """
        if not edges:
            return []
        
        # Compute degrees
        degree = [0] * n
        for u, v in edges:
            degree[u] += 1
            degree[v] += 1
        
        odd_verts = [v for v in range(n) if degree[v] % 2 == 1]
        num_paths = max(1, len(odd_verts) // 2)
        
        if num_paths == 1 and len(odd_verts) == 0:
            # All even degrees: single Eulerian circuit
            adj = {i: [] for i in range(n)}
            for eid, (u, v) in enumerate(edges):
                adj[u].append((v, eid))
                adj[v].append((u, eid))
            
            # Find Eulerian circuit starting from any vertex with edges
            start = next((v for v in range(n) if adj[v]), 0)
            
            circuit = find_eulerian_circuit(adj, start)
            if len(circuit) >= 2 and circuit[0] == circuit[-1]:
                return [circuit[:-1]]
            else:
                return [circuit]
        
        if len(odd_verts) == 0:
            # No odd vertices but num_paths > 1 shouldn't happen
            adj = {i: [] for i in range(n)}
            for eid, (u, v) in enumerate(edges):
                adj[u].append((v, eid))
                adj[v].append((u, eid))
            start = next((v for v in range(n) if adj[v]), 0)
            circuit = find_eulerian_circuit(adj, start)
            if len(circuit) >= 2 and circuit[0] == circuit[-1]:
                return [circuit[:-1]]
            else:
                return [circuit]
        
        # Build simple adjacency list for BFS
        simple_adj = {i: set() for i in range(n)}
        for u, v in edges:
            simple_adj[u].add(v)
            simple_adj[v].add(u)
        
        # Compute all-pairs shortest path distances between odd vertices
        k = len(odd_verts)
        dist_matrix = np.zeros((k, k))
        for i in range(k):
            dists = bfs_distances(simple_adj, odd_verts[i], n)
            for j in range(k):
                dist_matrix[i][j] = dists[odd_verts[j]]
        
        # Solve minimum-weight perfect matching on odd vertices
        # The cost of pairing vertices i and j is the shortest path distance between them
        # We want to minimize the total cost of the matching
        # linear_sum_assignment solves min sum C[i, perm[i]]
        # For perfect matching, we need to pair up all k vertices
        # This is equivalent to finding a permutation where we pair (i, perm[i]) 
        # but we need to ensure each vertex appears exactly once
        # Actually, for perfect matching in a complete graph, we can use the fact that
        # a perfect matching corresponds to a permutation where we pair consecutive elements
        # after sorting by some criterion, but that's not optimal.
        
        # Better approach: The minimum weight perfect matching in a complete graph
        # with k vertices (k even) can be solved by reduction to assignment problem.
        # We create a bipartite graph with two copies of the vertices, and find
        # a minimum weight perfect matching.
        
        # Actually, for the path cover problem, the key insight is:
        # Adding edges between pairs of odd vertices to make all degrees even
        # gives an Eulerian graph. The number of paths in the decomposition equals
        # the number of added edges (pairs). Since we need exactly |odd|/2 pairs,
        # the number of paths is always |odd|/2 regardless of how we pair.
        
        # So the minimum number of paths is fixed at |odd|/2.
        # The pairing affects which paths we get, but not the count.
        
        # However, we still need to find a valid decomposition.
        # Let's use the optimal pairing to get a good decomposition.
        
        # For the assignment problem formulation:
        # We want to find a permutation sigma of {0,...,k-1} such that
        # sigma is an involution (sigma(sigma(i)) = i) and sigma(i) != i,
        # minimizing sum over pairs {i, sigma(i)} of dist[i][sigma(i)].
        
        # This is the minimum weight perfect matching problem.
        # We can solve it by trying all pairings if k is small, or use
        # a more sophisticated algorithm.
        
        # For k <= 20, we can use dynamic programming over subsets.
        # But for simplicity, let's use a greedy approach with the distance matrix.
        
        # Actually, let's just use a simple greedy: pair each vertex with its nearest
        # unpaired neighbor.
        
        paired = [False] * k
        pairs = []
        for i in range(k):
            if not paired[i]:
                # Find nearest unpaired vertex
                best_j = -1
                best_dist = float('inf')
                for j in range(i + 1, k):
                    if not paired[j]:
                        if dist_matrix[i][j] < best_dist:
                            best_dist = dist_matrix[i][j]
                            best_j = j
                if best_j >= 0:
                    pairs.append((odd_verts[i], odd_verts[best_j]))
                    paired[i] = True
                    paired[best_j] = True
        
        # Create augmented edge list
        all_edges = list(edges)
        dummy_edges = pairs
        
        total_edges = len(all_edges) + len(dummy_edges)
        
        # Build adjacency with edge IDs
        adj = {i: [] for i in range(n)}
        for eid in range(len(all_edges)):
            u, v = all_edges[eid]
            adj[u].append((v, eid))
            adj[v].append((u, eid))
        for i, (a, b) in enumerate(dummy_edges):
            eid = len(all_edges) + i
            adj[a].append((b, eid))
            adj[b].append((a, eid))
        
        # Find Eulerian circuit in augmented graph
        start = next((v for v in range(n) if adj[v]), 0)
        
        circuit = find_eulerian_circuit(adj, start)
        
        # Build edge map
        edge_map = {}
        for eid, (u, v) in enumerate(all_edges):
            edge_map[(min(u,v), max(u,v))] = eid
        for i, (a, b) in enumerate(dummy_edges):
            eid = len(all_edges) + i
            edge_map[(min(a,b), max(a,b))] = eid
        
        # Find dummy edge positions in circuit
        split_positions = []
        for i in range(len(circuit) - 1):
            u, v = circuit[i], circuit[i+1]
            key = (min(u,v), max(u,v))
            if key in edge_map:
                eid = edge_map[key]
                if eid >= len(all_edges):
                    split_positions.append(i)
        
        # Extract paths
        paths = []
        if not split_positions:
            path = circuit[:-1] if circuit[0] == circuit[-1] else circuit
            if len(path) >= 2:
                paths.append(path)
        else:
            all_splits = sorted(set(split_positions))
            seg_starts = [0] + [s + 1 for s in all_splits]
            seg_ends = all_splits + [len(circuit) - 1]
            
            for s, e in zip(seg_starts, seg_ends):
                segment = circuit[s:e+1]
                if len(segment) >= 2:
                    paths.append(segment)
        
        paths = [p for p in paths if len(p) >= 2]
        
        if len(paths) == 0:
            return [list(e) for e in edges]
        
        return paths

    def greedy_path_cover(edges, n):
        """Fallback greedy path cover."""
        if not edges:
            return []
        
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
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
            
            best_path = None
            best_len = 0
            
            start_candidates = active_verts if len(active_verts) <= 10 else \
                              sorted(active_verts, key=lambda v: len(sub_adj[v]))[:10]
            
            for start in start_candidates:
                path = [start]
                visited = {start}
                current = start
                
                while True:
                    candidates = [nb for nb in sub_adj[current] if nb not in visited]
                    if not candidates:
                        break
                    best_next = min(candidates, key=lambda nb: len(sub_adj[nb]))
                    path.append(best_next)
                    visited.add(best_next)
                    current = best_next
                
                if len(path) > best_len:
                    best_len = len(path)
                    best_path = path
            
            if best_path is None or len(best_path) < 2:
                e = next(iter(remaining_edges))
                best_path = list(e)
            
            for i in range(len(best_path) - 1):
                e1 = tuple(sorted((best_path[i], best_path[i+1])))
                remaining_edges.discard(e1)
            
            paths.append(best_path)
        
        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > DEADLINE:
            while len(results) < 3:
                edges = instance(*spec)
                results.append([list(e) for e in edges])
            break
        
        edges = instance(*spec)
        n = spec[0]
        
        # Try matching-based minimum path cover
        paths = min_path_cover_matching(edges, n)
        
        # Validate: check that paths are valid
        valid = True
        used_edge_set = set()
        edge_set = set(edges)
        for path in paths:
            if len(path) < 2:
                valid = False
                break
            # Check simplicity
            if len(set(path)) != len(path):
                valid = False
                break
            for i in range(len(path) - 1):
                e = tuple(sorted((path[i], path[i+1])))
                if e not in edge_set:
                    valid = False
                    break
                if e in used_edge_set:
                    valid = False
                    break
                used_edge_set.add(e)
            if not valid:
                break
        
        if valid and used_edge_set == edge_set:
            results.append(paths)
        else:
            # Fallback to greedy
            paths = greedy_path_cover(edges, n)
            results.append(paths)
    
    return results
# EVOLVE-BLOCK-END
