import time
import random
from collections import defaultdict, deque
from random import Random

_DEADLINE = 140.0
_START = time.time()

def _remaining():
    return _DEADLINE - (time.time() - _START)

def construct():
    def instance_graph(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def _validate(paths, edges):
        """Check that paths are simple, use graph edges, are edge-disjoint and cover all edges."""
        if not paths:
            return len(edges) == 0
        used = set()
        for p in paths:
            if len(p) < 2:
                return False
            if len(set(p)) != len(p):
                return False
            for i in range(len(p) - 1):
                e = tuple(sorted((p[i], p[i+1])))
                if e in used:
                    return False
                used.add(e)
        return used == set(edges)

    def _greedy_decompose(edges, n, max_time=3.0):
        """Greedy path decomposition starting from odd vertices."""
        if not edges:
            return []
        
        edge_set = set(tuple(sorted(e)) for e in edges)
        adj = {i: set() for i in range(n)}
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        
        best_paths = None
        best_count = float('inf')
        start_time = time.time()
        
        while time.time() - start_time < max_time and _remaining() > 2.0:
            remaining = set(edge_set)
            paths = []
            
            deg = {i: 0 for i in range(n)}
            for u, v in remaining:
                deg[u] += 1
                deg[v] += 1
            odd_vertices = [v for v in range(n) if deg[v] % 2 == 1]
            
            while remaining:
                rem_deg = {i: 0 for i in range(n)}
                for u, v in remaining:
                    rem_deg[u] += 1
                    rem_deg[v] += 1
                rem_odd = [v for v in rem_deg if rem_deg[v] > 0 and rem_deg[v] % 2 == 1]
                
                if rem_odd:
                    start_v = random.choice(rem_odd)
                else:
                    candidates = [v for v in rem_deg if rem_deg[v] > 0]
                    if not candidates:
                        break
                    start_v = random.choice(candidates)
                
                path = _extend_path(start_v, remaining, adj)
                
                if not path or len(path) < 2:
                    # Find any edge from start_v
                    for u, v in remaining:
                        if u == start_v:
                            path = [start_v, v]
                            remaining.discard((u, v))
                            break
                        elif v == start_v:
                            path = [start_v, u]
                            remaining.discard((u, v))
                            break
                    if not path:
                        break
                else:
                    for i in range(len(path) - 1):
                        u, v = path[i], path[i+1]
                        e = tuple(sorted((u, v)))
                        remaining.discard(e)
                
                paths.append(path)
            
            if len(paths) < best_count:
                best_count = len(paths)
                best_paths = [p[:] for p in paths]
        
        if best_paths is None:
            return [[u, v] for u, v in sorted(edge_set)]
        return best_paths

    def _extend_path(start, remaining, adj):
        """Extend a path from start using DFS with degree heuristic."""
        path = [start]
        visited = {start}
        current = start
        
        while True:
            available = []
            for nb in adj[current]:
                e = tuple(sorted((current, nb)))
                if e in remaining and nb not in visited:
                    # Count available edges from nb
                    deg = 0
                    for x in adj[nb]:
                        e2 = tuple(sorted((nb, x)))
                        if e2 in remaining and x not in visited:
                            deg += 1
                    available.append((deg, nb))
            
            if not available:
                break
            
            # Pick neighbor with minimum degree (dead-end heuristic)
            available.sort()
            min_deg = available[0][0]
            # Among those with min degree, pick randomly
            candidates = [nb for deg, nb in available if deg == min_deg]
            best_nb = random.choice(candidates)
            
            e = tuple(sorted((current, best_nb)))
            remaining.discard(e)
            path.append(best_nb)
            visited.add(best_nb)
            current = best_nb
        
        return path

    def _pair_odd_vertices(adj, odd_vertices, n, max_restarts=100):
        """
        Find vertex-disjoint paths pairing up all odd vertices.
        Returns a list of paths (vertex lists) or None if failed.
        """
        if not odd_vertices:
            return []
        
        for restart in range(max_restarts):
            remaining = odd_vertices[:]
            random.shuffle(remaining)
            used_internal = set()
            pair_paths = []
            success = True
            
            while len(remaining) >= 2:
                best_path = None
                best_len = float('inf')
                best_i, best_j = -1, -1
                
                for i in range(len(remaining)):
                    for j in range(i + 1, len(remaining)):
                        s, t = remaining[i], remaining[j]
                        if s in used_internal or t in used_internal:
                            continue
                        p = _bfs_path(adj, s, t, used_internal, n)
                        if p is not None and len(p) < best_len:
                            best_len = len(p)
                            best_path = p
                            best_i, best_j = i, j
                
                if best_path is None:
                    success = False
                    break
                
                used_internal |= set(best_path[1:-1])
                pair_paths.append(best_path)
                
                new_remaining = []
                for idx, v in enumerate(remaining):
                    if idx == best_i or idx == best_j:
                        continue
                    new_remaining.append(v)
                remaining = new_remaining
            
            if success:
                return pair_paths
        
        return None

    def _bfs_path(adj, s, t, forbidden, n):
        """Shortest simple path from s to t, internal vertices not in forbidden."""
        if s == t:
            return [s]
        if s in forbidden or t in forbidden:
            return None
        prev = {s: None}
        q = deque([s])
        while q:
            u = q.popleft()
            if u == t:
                break
            for v in adj[u]:
                if v not in prev:
                    prev[v] = u
                    q.append(v)
        if t not in prev:
            return None
        path = []
        cur = t
        while cur is not None:
            path.append(cur)
            cur = prev[cur]
        path.reverse()
        if len(path) >= 3 and set(path[1:-1]) & forbidden:
            return None
        return path

    def _solve_one_gallai(edges, n):
        """
        Gallai's conjecture approach:
        1. Find odd-degree vertices
        2. Pair them with vertex-disjoint paths
        3. Add those paths to create an Eulerian multigraph
        4. Find Eulerian circuit
        5. Cut at added edges to get simple paths in original graph
        """
        if not edges:
            return []
        
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        odd_vertices = [v for v in range(n) if len(adj[v]) % 2 == 1]
        
        if not odd_vertices:
            # Eulerian graph - try to find a Hamiltonian path
            # If we can't, fall back to greedy
            paths = _greedy_decompose(edges, n, max_time=2.0)
            return paths
        
        pair_paths = _pair_odd_vertices(adj, odd_vertices, n)
        
        if pair_paths is None:
            return _greedy_decompose(edges, n, max_time=5.0)
        
        # Build multigraph H = G + pair_paths
        edge_list = []
        edge_set = set(tuple(sorted(e)) for e in edges)
        
        for u, v in edges:
            e = tuple(sorted((u, v)))
            edge_list.append((u, v, False, e))
        
        for pp in pair_paths:
            for i in range(len(pp) - 1):
                u, v = pp[i], pp[i+1]
                e = tuple(sorted((u, v)))
                edge_list.append((u, v, True, e))
        
        # Build adjacency for Hierholzer
        adj_h = {i: [] for i in range(n)}
        for idx, (u, v, is_added, e) in enumerate(edge_list):
            adj_h[u].append((v, idx))
            adj_h[v].append((u, idx))
        
        # Hierholzer
        used_edge = [False] * len(edge_list)
        start = 0
        stack = [start]
        circuit_vertices = []
        edge_seq = []
        
        while stack:
            v = stack[-1]
            advanced = False
            while adj_h[v]:
                u, idx = adj_h[v].pop()
                if not used_edge[idx]:
                    used_edge[idx] = True
                    stack.append(u)
                    edge_seq.append(idx)
                    advanced = True
                    break
            if not advanced:
                circuit_vertices.append(stack.pop())
        
        circuit_vertices.reverse()
        edge_seq.reverse()
        
        # Extract paths by cutting at added edges
        paths = []
        current_path = []
        
        for i in range(len(edge_seq)):
            idx = edge_seq[i]
            is_added = edge_list[idx][2]
            u = circuit_vertices[i]
            v = circuit_vertices[i+1]
            
            if is_added:
                if current_path:
                    if len(set(current_path)) == len(current_path):
                        paths.append(current_path)
                    current_path = []
                current_path = [v]
            else:
                if not current_path:
                    current_path = [u]
                current_path.append(v)
        
        if current_path:
            if len(set(current_path)) == len(current_path):
                paths.append(current_path)
        
        paths = [p for p in paths if len(p) >= 2]
        
        if not paths:
            return _greedy_decompose(edges, n, max_time=5.0)
        
        # Validate
        if _validate(paths, edges):
            return paths
        
        return _greedy_decompose(edges, n, max_time=5.0)

    def _solve_with_matching(edges, n):
        """
        Alternative: Use matching on the line graph concept.
        Each edge in G is a vertex in L(G). Two edges in L(G) are adjacent if they share a vertex.
        A matching in L(G) corresponds to pairing edges that are consecutive in a path.
        The number of paths = |E| - |matching|.
        We want to maximize the matching.
        
        For small graphs, we can try to find a large matching greedily with local improvement.
        """
        if not edges:
            return []
        
        edge_set = set(tuple(sorted(e)) for e in edges)
        m = len(edge_set)
        if m == 0:
            return []
        
        # Build line graph adjacency: edge i is adjacent to edge j if they share a vertex
        edges_list = sorted(edge_set)
        edge_to_idx = {e: i for i, e in enumerate(edges_list)}
        
        # For each edge, find which other edges it's adjacent to
        # Two edges (a,b) and (c,d) are adjacent if they share a vertex
        adj_line = [set() for _ in range(m)]
        for i, e1 in enumerate(edges_list):
            for j, e2 in enumerate(edges_list):
                if i >= j:
                    continue
                # Check if they share a vertex
                if e1[0] in e2 or e1[1] in e2:
                    adj_line[i].add(j)
                    adj_line[j].add(i)
        
        # Greedy matching with local improvement
        # Match[i] = j means edge i is matched to edge j, -1 if unmatched
        best_match = None
        best_size = 0
        
        for restart in range(20):
            match = [-1] * m
            matched = [False] * m
            
            # Greedy: process edges in random order
            order = list(range(m))
            random.shuffle(order)
            
            for i in order:
                if matched[i]:
                    continue
                # Find an unmatched neighbor
                candidates = [j for j in adj_line[i] if not matched[j]]
                if candidates:
                    # Pick the one with fewest options (most constrained)
                    best_j = min(candidates, key=lambda j: sum(1 for k in adj_line[j] if not matched[k]))
                    match[i] = best_j
                    match[best_j] = i
                    matched[i] = True
                    matched[best_j] = True
            
            size = sum(1 for i in range(m) if match[i] != -1) // 2
            
            # Local improvement: try swaps
            improved = True
            while improved and _remaining() > 5.0:
                improved = False
                for i in range(m):
                    if match[i] == -1:
                        continue
                    j = match[i]
                    # Try to find k adjacent to i, l adjacent to j, where k and l are unmatched or can be rematched
                    # Simple swap: if i-j is matched, try i-k and j-l if k-l is not needed
                    for k in adj_line[i]:
                        if k == j or match[k] != -1:
                            continue
                        for l in adj_line[j]:
                            if l == i or l == k or match[l] != -1:
                                continue
                            # Swap: break i-j, make i-k and j-l
                            match[i] = k
                            match[k] = i
                            match[j] = l
                            match[l] = j
                            size += 1
                            improved = True
                            break
                        if improved:
                            break
                    if improved:
                        break
            
            if size > best_size:
                best_size = size
                best_match = match[:]
        
        # Reconstruct paths from matching
        # The matching defines which edges are consecutive.
        # Unmatched edges are endpoints.
        # We need to trace the paths.
        
        if best_match is None or best_size == 0:
            return [[u, v] for u, v in edges_list]
        
        # Build the path structure
        # For each edge, we know which other edge it's connected to (if any)
        # We need to determine the direction/order
        # This is tricky. Let's use a different approach.
        
        # Actually, let's just use the greedy decomposition which is simpler and works well.
        return _greedy_decompose(edges, n, max_time=5.0)

    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    results = []
    
    for i, (n, p, seed) in enumerate(specs):
        edges = instance_graph(n, p, seed)
        max_t = 8.0
        if _remaining() < max_t:
            max_t = max(1.0, _remaining() - 2.0)
        
        # Try Gallai's approach first
        paths = _solve_one_gallai(edges, n)
        
        # If it didn't work well, try greedy
        if not _validate(paths, edges) or len(paths) > 20:
            greedy_paths = _greedy_decompose(edges, n, max_time=max_t)
            if _validate(greedy_paths, edges) and len(greedy_paths) < len(paths):
                paths = greedy_paths
        
        results.append(paths)
    
    return results

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
