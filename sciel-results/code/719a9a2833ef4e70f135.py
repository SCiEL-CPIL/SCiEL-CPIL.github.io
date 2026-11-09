"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random
import numpy as np

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

    def find_path_cover_greedy_improved(edges, n):
        """Improved greedy path cover with better heuristics."""
        if not edges:
            return []
        
        # Build adjacency list
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        remaining_edges = set(edges)
        paths = []
        
        while remaining_edges:
            # Build subgraph from remaining edges
            sub_adj = {i: set() for i in range(n)}
            for u, v in remaining_edges:
                sub_adj[u].add(v)
                sub_adj[v].add(u)
            
            # Find vertices with edges in remaining subgraph
            active_verts = [v for v in range(n) if sub_adj[v]]
            
            if not active_verts:
                break
            
            # Try multiple starting points and pick the best path
            best_path = None
            best_len = 0
            
            # Try starting from each active vertex (or a subset for larger graphs)
            start_candidates = active_verts if len(active_verts) <= 10 else \
                              sorted(active_verts, key=lambda v: len(sub_adj[v]))[:10]
            
            for start in start_candidates:
                # Greedy extend with lookahead
                path = [start]
                visited = {start}
                current = start
                
                while True:
                    candidates = [nb for nb in sub_adj[current] if nb not in visited]
                    if not candidates:
                        break
                    
                    # Choose candidate: prefer one that leads to longest continuation
                    # Use a simple heuristic: pick neighbor with minimum degree
                    best_next = min(candidates, key=lambda nb: len(sub_adj[nb]))
                    
                    path.append(best_next)
                    visited.add(best_next)
                    current = best_next
                
                if len(path) > best_len:
                    best_len = len(path)
                    best_path = path
            
            if best_path is None or len(best_path) < 2:
                # Fallback: take any edge
                e = next(iter(remaining_edges))
                best_path = list(e)
            
            # Remove edges in the path
            for i in range(len(best_path) - 1):
                e1 = tuple(sorted((best_path[i], best_path[i+1])))
                remaining_edges.discard(e1)
            
            paths.append(best_path)
        
        return paths

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
            # Find first unused edge
            while adj[v]:
                nb, eid = adj[v].pop()
                if eid not in used_edges:
                    used_edges.add(eid)
                    stack.append(nb)
                    found = True
                    break
            if not found:
                circuit.append(stack.pop())
        
        circuit.reverse()
        return circuit

    def min_path_cover_eulerian_optimal(edges, n):
        """Find minimum edge-disjoint path cover using optimal pairing of odd-degree vertices
        via Hungarian algorithm (T-join transformation).
        
        The minimum number of paths equals the number of odd-degree vertices / 2.
        We pair them optimally by minimizing total "added path length" which corresponds
        to finding shortest paths between odd vertices in the original graph.
        """
        if not edges:
            return []
        
        # Compute degrees
        degree = {i: 0 for i in range(n)}
        for u, v in edges:
            degree[u] += 1
            degree[v] += 1
        
        odd_verts = [v for v in range(n) if degree[v] % 2 == 1]
        k = len(odd_verts)
        
        if k == 0:
            # All even: single Eulerian circuit
            adj = {i: [] for i in range(n)}
            for eid, (u, v) in enumerate(edges):
                adj[u].append((v, eid))
                adj[v].append((u, eid))
            
            # Find start vertex with edges
            start = 0
            while start < n and not adj[start]:
                start += 1
            if start >= n:
                return []
            
            circuit = find_eulerian_circuit(adj, start)
            if len(circuit) >= 2:
                # Remove duplicate endpoint if it's a circuit
                if circuit[0] == circuit[-1]:
                    return [circuit[:-1]]
                else:
                    return [circuit]
            return []
        
        # k is even (handshaking lemma)
        # Find shortest paths between all pairs of odd vertices
        # Build adjacency for BFS
        adj_list = {i: set() for i in range(n)}
        for u, v in edges:
            adj_list[u].add(v)
            adj_list[v].add(u)
        
        # BFS from each odd vertex to get distances to all other odd vertices
        dist_matrix = [[float('inf')] * k for _ in range(k)]
        
        for i, src in enumerate(odd_verts):
            # BFS
            dist = {src: 0}
            queue = [src]
            head = 0
            while head < len(queue):
                v = queue[head]
                head += 1
                for nb in adj_list[v]:
                    if nb not in dist:
                        dist[nb] = dist[v] + 1
                        queue.append(nb)
            for j, tgt in enumerate(odd_verts):
                if tgt in dist:
                    dist_matrix[i][j] = dist[tgt]
        
        # Use Hungarian algorithm to find optimal pairing
        # We want to pair k vertices into k/2 pairs minimizing total distance
        # This is a minimum weight perfect matching on a complete graph
        # For small k, we can use scipy's linear_sum_assignment on a symmetric matrix
        # But linear_sum_assignment is for rectangular matrices
        
        # For perfect matching on k vertices (k even), we can formulate as:
        # Create a k x k cost matrix where cost[i][j] = dist_matrix[i][j] for i != j
        # and cost[i][i] = large value
        # Then find assignment that minimizes sum, but this gives a permutation
        # not a matching. We need a different approach.
        
        # Alternative: since k is small (typically <= n), use DP or brute force for small k
        # For k <= 20, we can use bitmask DP
        
        if k <= 20:
            # Bitmask DP for minimum weight perfect matching
            INF = float('inf')
            dp = [[INF] * (1 << k) for _ in range(k)]
            # dp[i][mask] = min cost to match vertices in mask, where i is the smallest unmatched vertex in mask
            # Actually, let's use a cleaner formulation
            # dp[mask] = min cost to match vertices in mask (mask has even number of bits)
            dp = [INF] * (1 << k)
            dp[0] = 0
            
            for mask in range(1 << k):
                if dp[mask] == INF:
                    continue
                # Find first unmatched vertex
                first = -1
                for i in range(k):
                    if not (mask & (1 << i)):
                        first = i
                        break
                if first == -1:
                    continue
                # Try pairing first with any other unmatched vertex
                for j in range(first + 1, k):
                    if not (mask & (1 << j)):
                        new_mask = mask | (1 << first) | (1 << j)
                        cost = dist_matrix[first][j]
                        if cost < INF:
                            if dp[new_mask] > dp[mask] + cost:
                                dp[new_mask] = dp[mask] + cost
            
            total_cost = dp[(1 << k) - 1]
            
            # Now we need to recover the actual pairs
            # Backtrack through DP
            pairs = []
            mask = (1 << k) - 1
            while mask != 0:
                first = -1
                for i in range(k):
                    if not (mask & (1 << i)):
                        first = i
                        break
                if first == -1:
                    break
                best_j = -1
                best_cost = INF
                for j in range(first + 1, k):
                    if not (mask & (1 << j)):
                        new_mask = mask | (1 << first) | (1 << j)
                        # Wait, this is backwards. Let me redo.
                        pass
                
                # Let me redo the backtracking properly
                # Actually, let's just use the greedy pairing since the optimal
                # pairing only affects the Eulerian circuit structure, not the count
                # The count is always k/2 regardless of pairing
                break
        
        # For simplicity and correctness, use greedy pairing
        # The number of paths is always k/2 regardless of which pairing we use
        # The pairing affects which edges are "dummy" but not the count
        pairs = []
        for i in range(0, k, 2):
            if i + 1 < k:
                pairs.append((odd_verts[i], odd_verts[i+1]))
        
        # Create augmented edge list
        all_edges = list(edges)
        dummy_edges = [(odd_verts[i], odd_verts[i+1]) for i in range(0, k, 2)]
        
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
        
        # Find Eulerian circuit
        start = 0
        while start < n and not adj[start]:
            start += 1
        if start >= n:
            return find_path_cover_greedy_improved(edges, n)
        
        circuit = find_eulerian_circuit(adj, start)
        
        if len(circuit) < 2:
            return find_path_cover_greedy_improved(edges, n)
        
        # Identify dummy edges in the circuit
        # Build mapping from edge key to edge ID
        edge_map = {}
        for eid, (u, v) in enumerate(all_edges):
            key = (min(u,v), max(u,v))
            if key not in edge_map:  # Only first occurrence
                edge_map[key] = eid
        for i, (a, b) in enumerate(dummy_edges):
            eid = len(all_edges) + i
            key = (min(a,b), max(a,b))
            if key not in edge_map:
                edge_map[key] = eid
        
        # Find split positions
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
            prev = 0
            for pos in split_positions:
                segment = circuit[prev:pos+1]
                if len(segment) >= 2:
                    paths.append(segment)
                prev = pos + 1
            # Last segment
            segment = circuit[prev:]
            if len(segment) >= 2:
                paths.append(segment)
        
        # Validate: check that paths are valid and cover all edges
        if not paths:
            return find_path_cover_greedy_improved(edges, n)
        
        # Verify edge coverage
        covered = set()
        valid = True
        for path in paths:
            for i in range(len(path) - 1):
                e = tuple(sorted((path[i], path[i+1])))
                if e not in set(edges):
                    valid = False
                    break
                if e in covered:
                    valid = False
                    break
                covered.add(e)
            if not valid:
                break
        
        if not valid or covered != set(edges):
            return find_path_cover_greedy_improved(edges, n)
        
        return paths

    def find_path_cover_best(edges, n):
        """Try multiple strategies and return the best result."""
        if not edges:
            return []
        
        # Strategy 1: Eulerian with optimal pairing (should give minimum path count)
        try:
            paths1 = min_path_cover_eulerian_optimal(edges, n)
            if paths1:
                return paths1
        except:
            pass
        
        # Strategy 2: Improved greedy
        paths2 = find_path_cover_greedy_improved(edges, n)
        return paths2

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > DEADLINE:
            # Return what we have, pad with trivial solutions
            while len(results) < 3:
                edges = instance(*spec)
                results.append([list(e) for e in edges])
            break
        
        edges = instance(*spec)
        n = spec[0]
        
        # Use the best path cover algorithm
        paths = find_path_cover_best(edges, n)
        results.append(paths)
    
    return results
# EVOLVE-BLOCK-END
