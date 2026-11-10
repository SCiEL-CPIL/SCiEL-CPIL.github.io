"""Program using MCPM-based Eulerian augmentation for Gallai path cover."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque

def construct():
    _start = time.time()
    _DEADLINE = 145.0

    def instance(n, p, seed):
        import random
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def build_adjacency(edges):
        adj = defaultdict(list)
        for (u, v) in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def bfs_shortest_path(src, dst, adj):
        if src == dst:
            return []
        visited = {src}
        parent = {src: None}
        q = deque([src])
        while q:
            cur = q.popleft()
            for nb in adj[cur]:
                if nb not in visited:
                    visited.add(nb)
                    parent[nb] = cur
                    if nb == dst:
                        path = []
                        node = dst
                        while node is not None:
                            path.append(node)
                            node = parent[node]
                        path.reverse()
                        return path
                    q.append(nb)
        return None

    def find_eulerian_circuit(edge_list, adj_map):
        if not edge_list:
            return []
        
        # Build edge count map
        edge_count = defaultdict(int)
        for (u, v) in edge_list:
            key = (u, v) if u <= v else (v, u)
            edge_count[key] += 1

        # Build adjacency with edge keys
        hier_adj = defaultdict(list)
        for (u, v) in edge_list:
            key = (u, v) if u <= v else (v, u)
            hier_adj[u].append((v, key))
            hier_adj[v].append((u, key))

        used_count = defaultdict(int)
        start = next(iter(adj_map))
        stack = [start]
        circuit = []
        while stack:
            v = stack[-1]
            found = False
            for (nb, key) in hier_adj[v]:
                if used_count[key] < edge_count[key]:
                    used_count[key] += 1
                    stack.append(nb)
                    found = True
                    break
            if not found:
                circuit.append(stack.pop())
        circuit.reverse()
        return circuit

    def minimum_cost_perfect_matching(odd_vertices, adj, edge_weights):
        """
        Find minimum cost perfect matching on odd_vertices using a simple
        Hungarian-like approach or brute force for small n.
        For our sizes (<=30 vertices), brute force matching is feasible.
        """
        n = len(odd_vertices)
        if n == 0:
            return []
        
        # For small n (<= 16), use DP over subsets
        # For larger n, use greedy + local improvement
        if n <= 16:
            # DP over subsets
            from functools import lru_cache
            
            # Map vertices to indices
            idx_map = {v: i for i, v in enumerate(odd_vertices)}
            
            @lru_cache(maxsize=None)
            def dp(mask):
                if mask == (1 << n) - 1:
                    return 0, []
                
                # Find first unmatched vertex
                i = -1
                for j in range(n):
                    if not (mask >> j) & 1:
                        i = j
                        break
                if i == -1:
                    return 0, []
                
                best_cost = float('inf')
                best_pair = -1
                best_rest = None
                
                for j in range(i + 1, n):
                    if not (mask >> j) & 1:
                        cost = edge_weights.get((odd_vertices[i], odd_vertices[j]), float('inf'))
                        if cost < float('inf'):
                            new_mask = mask | (1 << i) | (1 << j)
                            rest_cost, rest_pairs = dp(new_mask)
                            total = cost + rest_cost
                            if total < best_cost:
                                best_cost = total
                                best_pair = j
                                best_rest = rest_pairs
                
                if best_pair == -1:
                    return float('inf'), []
                
                pair = (odd_vertices[i], odd_vertices[best_pair])
                pairs = [pair] + best_rest
                return best_cost, pairs
            
            result_cost, result_pairs = dp(0)
            return result_pairs
        else:
            # Greedy: pair closest odd vertices, then local search
            # For now, use greedy pairing
            pairs = []
            used = set()
            # Sort by distance
            all_pairs = []
            for i in range(n):
                for j in range(i + 1, n):
                    u, v = odd_vertices[i], odd_vertices[j]
                    w = edge_weights.get((u, v), float('inf'))
                    if w < float('inf'):
                        all_pairs.append((w, u, v))
            all_pairs.sort()
            
            for w, u, v in all_pairs:
                if u not in used and v not in used:
                    pairs.append((u, v))
                    used.add(u)
                    used.add(v)
                    if len(used) == n:
                        break
            
            return pairs

    def mcpm_path_cover(edges):
        """
        1. Find odd-degree vertices.
        2. Compute shortest path distances between all odd pairs.
        3. Find minimum cost perfect matching on odd vertices.
        4. Add matched shortest paths as virtual edges.
        5. Find Eulerian circuit in augmented graph.
        6. Break circuit at virtual edges to get path cover.
        """
        if not edges:
            return []

        adj = build_adjacency(edges)
        
        # Find odd-degree vertices
        odd_vertices = [v for v in adj if len(adj[v]) % 2 == 1]
        
        if not odd_vertices:
            # Already Eulerian
            circuit = find_eulerian_circuit(edges, adj)
            if circuit:
                return [circuit]
            return []

        # Compute all-pairs shortest paths between odd vertices
        # Use BFS from each odd vertex
        odd_set = set(odd_vertices)
        dist = {}
        parent_map = {}
        
        for src in odd_vertices:
            # BFS
            d = {src: 0}
            par = {src: None}
            q = deque([src])
            while q:
                cur = q.popleft()
                for nb in adj[cur]:
                    if nb not in d:
                        d[nb] = d[cur] + 1
                        par[nb] = cur
                        q.append(nb)
            for dst in odd_vertices:
                if dst != src and dst in d:
                    key = (src, dst) if src < dst else (dst, src)
                    dist[key] = d[dst]
                    parent_map[(src, dst)] = (d, par)

        # Build edge weights for matching
        edge_weights = {}
        for (u, v), w in dist.items():
            edge_weights[(u, v)] = w

        # Find minimum cost perfect matching
        pairs = minimum_cost_perfect_matching(odd_vertices, adj, edge_weights)
        
        if not pairs:
            # Fallback: greedy
            return []

        # Build virtual edges from matched pairs
        virtual_edges = []
        for (a, b) in pairs:
            sp = bfs_shortest_path(a, b, adj)
            if sp and len(sp) >= 2:
                for i in range(len(sp) - 1):
                    u, v = sp[i], sp[i + 1]
                    key = (u, v) if u <= v else (v, u)
                    virtual_edges.append(key)

        # Build augmented edge list
        all_edges = list(edges) + virtual_edges
        
        # Find Eulerian circuit
        circuit = find_eulerian_circuit(all_edges, adj)
        
        if not circuit:
            return []

        # Break circuit at virtual edges
        virtual_set = set(virtual_edges)
        
        paths = []
        current_path = [circuit[0]]
        for i in range(1, len(circuit)):
            u, v = circuit[i - 1], circuit[i]
            key = (u, v) if u <= v else (v, u)
            current_path.append(v)
            if key in virtual_set:
                if len(current_path) >= 2:
                    paths.append(current_path)
                current_path = [v]

        if len(current_path) >= 2:
            paths.append(current_path)

        paths = [p for p in paths if len(p) >= 2]
        
        # Validate
        orig_edge_set = set()
        for (u, v) in edges:
            orig_edge_set.add((u, v) if u <= v else (v, u))

        covered = set()
        valid = True
        for p in paths:
            if len(set(p)) != len(p):
                valid = False
                break
            for i in range(len(p) - 1):
                u, v = p[i], p[i + 1]
                key = (u, v) if u <= v else (v, u)
                if key not in orig_edge_set:
                    valid = False
                    break
                if key in covered:
                    valid = False
                    break
                covered.add(key)
            if not valid:
                break

        if valid and covered == orig_edge_set:
            return paths
        else:
            return []

    def greedy_path_cover(edges):
        """Fallback greedy path cover."""
        if not edges:
            return []

        edge_set = set()
        for e in edges:
            u, v = e
            edge_set.add((u, v) if u <= v else (v, u))

        adj = defaultdict(list)
        for (u, v) in edge_set:
            adj[u].append(v)
            adj[v].append(u)

        used_edges = set()
        paths = []

        while edge_set - used_edges:
            remaining = edge_set - used_edges
            if not remaining:
                break

            rem_adj = defaultdict(list)
            for (u, v) in remaining:
                rem_adj[u].append(v)
                rem_adj[v].append(u)

            start = max(rem_adj.keys(), key=lambda x: len(rem_adj[x])) if rem_adj else None
            if start is None:
                break

            path = [start]
            visited_in_path = {start}
            current = start
            while True:
                neighbors = [n for n in rem_adj[current] if n not in visited_in_path]
                if not neighbors:
                    break
                best_neighbor = max(neighbors, key=lambda nb: len([x for x in rem_adj[nb] if x not in visited_in_path]))
                path.append(best_neighbor)
                visited_in_path.add(best_neighbor)
                current = best_neighbor

            path_edges = set()
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                key = (u, v) if u <= v else (v, u)
                path_edges.add(key)

            valid = all(e in remaining for e in path_edges)
            if valid and path_edges:
                used_edges |= path_edges
                paths.append(path)
            else:
                e = next(iter(remaining))
                used_edges.add(e)
                paths.append([e[0], e[1]])

        return paths

    def local_search_improve(paths, edges, adj, time_limit=5.0):
        start_time = time.time()
        best_paths = [list(p) for p in paths]
        
        def is_simple_path(path):
            return len(set(path)) == len(path)
        
        while time.time() - start_time < time_limit:
            improved = False
            n_paths = len(best_paths)
            
            for i in range(n_paths):
                if time.time() - start_time > time_limit:
                    return best_paths
                for j in range(i + 1, n_paths):
                    p1 = best_paths[i]
                    p2 = best_paths[j]
                    
                    candidates = []
                    
                    u, v = p1[-1], p2[0]
                    if (u, v) in adj:
                        new_path = p1 + p2[1:]
                        if is_simple_path(new_path):
                            candidates.append(new_path)
                    
                    u, v = p1[0], p2[0]
                    if (u, v) in adj:
                        new_path = list(reversed(p1)) + p2[1:]
                        if is_simple_path(new_path):
                            candidates.append(new_path)
                    
                    u, v = p1[-1], p2[-1]
                    if (u, v) in adj:
                        new_path = p1 + list(reversed(p2[1:]))
                        if is_simple_path(new_path):
                            candidates.append(new_path)
                    
                    u, v = p1[0], p2[-1]
                    if (u, v) in adj:
                        new_path = list(reversed(p1)) + list(reversed(p2[1:]))
                        if is_simple_path(new_path):
                            candidates.append(new_path)
                    
                    if candidates:
                        merged = candidates[0]
                        new_paths = []
                        for k in range(n_paths):
                            if k == i or k == j:
                                continue
                            new_paths.append(best_paths[k])
                        new_paths.append(merged)
                        best_paths = new_paths
                        improved = True
                        break
                if improved:
                    break
                    
            if not improved:
                break

        return best_paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > _DEADLINE - 10:
            edges = instance(*spec)
            results.append(greedy_path_cover(edges))
            continue

        edges = instance(*spec)
        adj = build_adjacency(edges)
        
        # Try MCPM first
        try:
            paths = mcpm_path_cover(edges)
        except Exception:
            paths = []
            
        if not paths:
            paths = greedy_path_cover(edges)

        # Validate
        edge_set = set()
        for (u, v) in edges:
            edge_set.add((u, v) if u <= v else (v, u))

        covered = set()
        valid = True
        for p in paths:
            if len(p) < 2:
                valid = False
                break
            if len(set(p)) != len(p):
                valid = False
                break
            for i in range(len(p) - 1):
                u, v = p[i], p[i + 1]
                key = (u, v) if u <= v else (v, u)
                if key not in edge_set:
                    valid = False
                    break
                if key in covered:
                    valid = False
                    break
                covered.add(key)
            if not valid:
                break

        if not valid or covered != edge_set:
            paths = greedy_path_cover(edges)
            covered2 = set()
            valid2 = True
            for p in paths:
                if len(p) < 2:
                    valid2 = False
                    break
                if len(set(p)) != len(p):
                    valid2 = False
                    break
                for i in range(len(p) - 1):
                    u, v = p[i], p[i + 1]
                    key = (u, v) if u <= v else (v, u)
                    if key not in edge_set:
                        valid2 = False
                        break
                    if key in covered2:
                        valid2 = False
                        break
                    covered2.add(key)
                if not valid2:
                    break
            if not valid2 or covered2 != edge_set:
                paths = [[list(e)] for e in edges]

        # Local search improvement
        remaining_time = _DEADLINE - time.time()
        graphs_left = 3 - len(results)
        time_per_graph = max(1.0, remaining_time / max(1, graphs_left))
        
        try:
            improved_paths = local_search_improve(paths, edges, adj, time_limit=time_per_graph)
            covered3 = set()
            valid3 = True
            for p in improved_paths:
                if len(p) < 2:
                    valid3 = False
                    break
                if len(set(p)) != len(p):
                    valid3 = False
                    break
                for i in range(len(p) - 1):
                    u, v = p[i], p[i + 1]
                    key = (u, v) if u <= v else (v, u)
                    if key not in edge_set:
                        valid3 = False
                        break
                    if key in covered3:
                        valid3 = False
                        break
                    covered3.add(key)
                if not valid3:
                    break
                    
            if valid3 and covered3 == edge_set:
                paths = improved_paths
        except Exception:
            pass

        results.append(paths)

    return results
# EVOLVE-BLOCK-END
