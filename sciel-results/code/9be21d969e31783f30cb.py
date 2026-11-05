"""Path decomposition via min-cost flow pairing + multi-start greedy + merges."""
import time
from collections import defaultdict, deque
from random import Random

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def build_adjacency(edges, n):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def bfs_path(adj, src, dst, used_edges):
        """Find a simple path from src to dst using only unused edges."""
        if src == dst:
            return [src]
        parent = {src: None}
        queue = deque([src])
        while queue:
            u = queue.popleft()
            if u == dst:
                path = []
                cur = dst
                while cur is not None:
                    path.append(cur)
                    if parent[cur] is None:
                        break
                    cur = parent[cur]
                path.reverse()
                return path
            for v in sorted(adj[u]):
                if v in parent:
                    continue
                edge = (min(u, v), max(u, v))
                if edge in used_edges:
                    continue
                parent[v] = u
                queue.append(v)
        return None

    def min_cost_pairing(odd_vertices, adj, n, deadline):
        """
        Find minimum-cost pairing of odd vertices using shortest paths.
        Returns list of (u, v, cost) pairs.
        """
        if len(odd_vertices) <= 2:
            if len(odd_vertices) == 2:
                return [(odd_vertices[0], odd_vertices[1], 0)]
            return []
        
        # Compute all-pairs shortest paths from odd vertices
        m = len(odd_vertices)
        dist = [[None] * m for _ in range(m)]
        
        for i in range(m):
            src = odd_vertices[i]
            # BFS from src
            d = {src: 0}
            q = deque([src])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if v not in d:
                        d[v] = d[u] + 1
                        q.append(v)
            for j in range(m):
                dst = odd_vertices[j]
                if dst in d:
                    dist[i][j] = d[dst]
        
        # Hungarian algorithm for minimum weight perfect matching
        # For small m (typically < 20), use simple DP or greedy
        # Use greedy with local search for simplicity
        
        # Greedy: repeatedly pick the closest pair
        used = [False] * m
        pairs = []
        
        # Try multiple random orderings and keep best
        best_total = float('inf')
        best_pairs = None
        
        for trial in range(20):
            if time.time() > deadline:
                break
            rng = Random(trial * 1000 + 7)
            order = list(range(m))
            rng.shuffle(order)
            
            used_local = [False] * m
            pairs_local = []
            total_cost = 0
            
            for idx in order:
                if used_local[idx]:
                    continue
                # Find closest unused vertex
                best_j = -1
                best_d = float('inf')
                for j in range(m):
                    if used_local[j] or j == idx:
                        continue
                    d = dist[idx][j]
                    if d is not None and d < best_d:
                        best_d = d
                        best_j = j
                if best_j == -1:
                    # Should not happen for even number of odd vertices
                    continue
                used_local[idx] = True
                used_local[best_j] = True
                pairs_local.append((odd_vertices[idx], odd_vertices[best_j], best_d))
                total_cost += best_d
            
            if total_cost < best_total:
                best_total = total_cost
                best_pairs = pairs_local
        
        return best_pairs if best_pairs else []

    def greedy_build_paths(remaining_edges, adj, n, rng, deadline):
        """Build maximal simple paths greedily from remaining edges."""
        remaining = set(remaining_edges)
        paths = []
        
        rem_adj = defaultdict(set)
        for u, v in remaining:
            rem_adj[u].add(v)
            rem_adj[v].add(u)
        
        def rem_degree(v):
            return len(rem_adj[v]) if v in rem_adj else 0
        
        while remaining and time.time() < deadline:
            # Choose start vertex: prefer odd-degree in remaining graph
            rem_odd = [u for u in rem_adj if rem_degree(u) % 2 == 1]
            if rem_odd:
                candidates = sorted(rem_odd, key=lambda u: -rem_degree(u))[:5]
                start = rng.choice(candidates)
            else:
                candidates = [u for u in rem_adj if rem_degree(u) > 0]
                if not candidates:
                    break
                candidates = sorted(candidates, key=lambda u: -rem_degree(u))[:5]
                start = rng.choice(candidates)
            
            # Build a maximal simple path
            path = [start]
            in_path = {start}
            current = start
            
            while True:
                neighbors = [nb for nb in rem_adj[current] 
                             if (min(current, nb), max(current, nb)) in remaining]
                if not neighbors:
                    break
                
                # Prefer neighbors not in path
                non_cycle = [nb for nb in neighbors if nb not in in_path]
                if non_cycle:
                    non_cycle.sort(key=lambda v: -rem_degree(v))
                    # Add some randomness for diversity
                    if rng.random() < 0.3 and len(non_cycle) > 1:
                        next_v = non_cycle[rng.randrange(1, min(3, len(non_cycle)))]
                    else:
                        next_v = non_cycle[0]
                else:
                    break
                
                key = (min(current, next_v), max(current, next_v))
                remaining.discard(key)
                path.append(next_v)
                in_path.add(next_v)
                
                rem_adj[current].discard(next_v)
                rem_adj[next_v].discard(current)
                if not rem_adj[current]:
                    del rem_adj[current]
                if not rem_adj[next_v]:
                    del rem_adj[next_v]
                
                current = next_v
            
            if len(path) > 1:
                paths.append(path)
        
        # Handle any leftover edges
        while remaining and time.time() < deadline:
            e = next(iter(remaining))
            u, v = e
            paths.append([u, v])
            remaining.discard(e)
            if v in rem_adj.get(u, set()):
                rem_adj[u].discard(v)
            if u in rem_adj.get(v, set()):
                rem_adj[v].discard(u)
            if u in rem_adj and not rem_adj[u]:
                del rem_adj[u]
            if v in rem_adj and not rem_adj[v]:
                del rem_adj[v]
        
        return paths, remaining

    def try_merge_paths(paths, adj, n, deadline):
        """Try to merge paths by reconnecting endpoints to reduce path count."""
        if len(paths) <= 1:
            return paths
        
        # Build edge set for all paths
        all_edges = set()
        for path in paths:
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                all_edges.add((min(u, v), max(u, v)))
        
        # Build adjacency from all edges
        full_adj = defaultdict(set)
        for u, v in all_edges:
            full_adj[u].add(v)
            full_adj[v].add(u)
        
        # Try to merge pairs of paths
        merged = True
        paths_copy = [list(p) for p in paths]
        
        while merged and time.time() < deadline:
            merged = False
            i = 0
            while i < len(paths_copy):
                if time.time() > deadline:
                    return paths_copy
                j = i + 1
                while j < len(paths_copy):
                    if time.time() > deadline:
                        return paths_copy
                    
                    p1 = paths_copy[i]
                    p2 = paths_copy[j]
                    
                    # Try all 4 combinations of endpoint connections
                    # p1: [a, ..., b], p2: [c, ..., d]
                    # Can connect b-c, b-d, a-c, or a-d if edge exists
                    a1, b1 = p1[0], p1[-1]
                    a2, b2 = p2[0], p2[-1]
                    
                    found_merge = False
                    for (x, y, flip1, flip2) in [(b1, a2, False, False),
                                                   (b1, b2, False, True),
                                                   (a1, a2, True, False),
                                                   (a1, b2, True, True)]:
                        if (min(x, y), max(x, y)) in all_edges:
                            # Check that this edge is not already in either path
                            edge_key = (min(x, y), max(x, y))
                            in_p1 = False
                            in_p2 = False
                            for k in range(len(p1) - 1):
                                if (min(p1[k], p1[k+1]), max(p1[k], p1[k+1])) == edge_key:
                                    in_p1 = True
                                    break
                            for k in range(len(p2) - 1):
                                if (min(p2[k], p2[k+1]), max(p2[k], p2[k+1])) == edge_key:
                                    in_p2 = True
                                    break
                            
                            if not in_p1 and not in_p2:
                                # Merge!
                                p1_mod = list(reversed(p1)) if flip1 else list(p1)
                                p2_mod = list(reversed(p2)) if flip2 else list(p2)
                                
                                # Check simplicity: no repeated vertices
                                new_path = p1_mod + p2_mod
                                if len(new_path) == len(set(new_path)):
                                    # Remove the connecting edge from all_edges
                                    all_edges.discard(edge_key)
                                    
                                    # Update paths_copy
                                    paths_copy[i] = new_path
                                    del paths_copy[j]
                                    merged = True
                                    found_merge = True
                                    break
                    
                    if found_merge:
                        break
                    j += 1
                i += 1
        
        return paths_copy

    def decompose_graph(edges, n, deadline):
        """Decompose graph into few paths using min-cost pairing + multi-start greedy."""
        adj = build_adjacency(edges, n)
        edges_set = set(edges)
        
        # Find odd-degree vertices
        odd_vertices = [v for v in range(n) if len(adj[v]) % 2 == 1]
        
        if not odd_vertices:
            # All even: decompose into cycles, convert to paths
            remaining = set(edges_set)
            paths = []
            while remaining and time.time() < deadline:
                start_edge = next(iter(remaining))
                u0, v0 = start_edge
                
                # Find a cycle
                cycle = None
                visited_edges = set()
                stack = [(v0, [u0, v0])]
                
                while stack and cycle is None:
                    cur, path = stack.pop()
                    for nxt in sorted(adj[cur]):
                        e = (min(cur, nxt), max(cur, nxt))
                        if e in remaining and e not in visited_edges:
                            visited_edges.add(e)
                            if nxt == u0 and len(path) >= 2:
                                cycle = path
                                break
                            stack.append((nxt, path + [nxt]))
                
                if cycle is None:
                    e = next(iter(remaining))
                    paths.append([e[0], e[1]])
                    remaining.discard(e)
                    continue
                
                for i in range(len(cycle) - 1):
                    a, b = cycle[i], cycle[i + 1]
                    e = (min(a, b), max(a, b))
                    remaining.discard(e)
                
                path = cycle[:-1]
                paths.append(path)
            
            # Try merges
            paths = try_merge_paths(paths, adj, n, deadline)
            return paths
        
        # Use min-cost pairing to guide initial path construction
        pairing = min_cost_pairing(odd_vertices, adj, n, deadline)
        
        # Multi-start greedy with different random seeds
        best_paths = None
        best_count = float('inf')
        
        max_restarts = 40
        
        for trial in range(max_restarts):
            if time.time() > deadline:
                break
            
            rng = Random(trial * 7919 + 13)
            paths, remaining = greedy_build_paths(edges_set, adj, n, rng, deadline)
            
            # Try merges
            paths = try_merge_paths(paths, adj, n, deadline)
            
            count = len(paths)
            if count < best_count:
                best_count = count
                best_paths = paths
            
            # Early stop if we hit theoretical minimum
            min_paths = max(1, len(odd_vertices) // 2)
            if best_count == min_paths:
                break
        
        if best_paths is None:
            # Fallback: single-edge paths
            best_paths = [[u, v] for u, v in edges_set]
        
        return best_paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > DEADLINE:
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
        edges = instance(*spec)
        n = spec[0]
        paths = decompose_graph(edges, n, DEADLINE)
        results.append(paths)
    
    return results
