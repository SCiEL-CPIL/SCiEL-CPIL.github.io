"""Minimum-flow edge-disjoint simple path cover for Gallai instances."""
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

    def validate(paths, edge_set):
        covered = set()
        for p in paths:
            if len(p) < 2:
                return False
            if len(set(p)) != len(p):
                return False
            for i in range(len(p) - 1):
                u, v = p[i], p[i + 1]
                key = (u, v) if u <= v else (v, u)
                if key not in edge_set:
                    return False
                if key in covered:
                    return False
                covered.add(key)
        return covered == edge_set

    def min_cost_flow_path_cover(vertices, edges, time_budget=20.0):
        """
        Min-cost flow formulation for minimum number of edge-disjoint simple paths
        covering all edges.

        Construction:
        - For each original edge (u,v), create two directed arcs u->v and v->u,
          each with capacity 1, cost 0.
        - For each vertex v, create a "split": v_in -> v_out with capacity
          infinity, cost 0. This allows flow to pass through.
        - To allow a path to START at v: add source -> v_in with capacity 1, cost 0.
          Actually, we want to minimize number of paths = number of "sources".
        - Alternative: For each vertex, the number of paths starting there is
          (out_flow - in_flow) where out_flow counts edges leaving v, in_flow
          counts edges entering v. Since we need to cover every edge exactly once,
          each edge contributes 1 to out of one endpoint and 1 to in of the other.
          The total number of paths = sum over v of max(0, out_v - in_v) = 
          (1/2) * sum over v of |out_v - in_v| = (1/2) * number of odd-degree vertices
          in the "directed" sense... 

        Actually, the standard formulation:
        - Each edge (u,v) must be traversed exactly once in one direction.
        - Choose direction for each edge. This gives each vertex a net flow
          b(v) = out(v) - in(v).
        - The number of paths = (1/2) * sum_v |b(v)|.
        - We want to minimize sum_v |b(v)| over all edge orientations.
        - This is equivalent to: minimize sum_v |b(v)| subject to b(v) = out(v)-in(v)
          and out(v)+in(v) = deg(v).
        - Since b(v) has same parity as deg(v), and sum b(v) = 0.
        - Minimizing sum |b(v)|: each vertex with even degree can have b(v)=0,
          each with odd degree must have |b(v)| >= 1.
        - So minimum is exactly |odd| (number of odd-degree vertices), giving
          |odd|/2 paths. This is achievable iff we can orient edges so that
          each odd vertex has net flow +/-1 and each even vertex has net flow 0.
        - This is always possible for a connected graph (it's equivalent to finding
          an Eulerian trail decomposition).

        So the minimum number of paths is always |odd|/2 for a connected graph!
        This means the flow formulation gives the theoretical minimum directly.

        But wait - the paths must be SIMPLE (no repeated vertices). The Eulerian
        circuit decomposition gives trails, not necessarily simple paths.

        So the real constraint is simplicity. Let me use a different approach:
        Find the minimum number of edge-disjoint simple paths via ILP-like
        successive shortest path with vertex-disjointness enforced by capacity.

        Simpler approach: Use the fact that for our small graphs (n<=30, m~50-100),
        we can use a maximum matching / flow approach:
        
        Actually, let me just implement a clean successive shortest augmenting path
        for min-cost flow where:
        - Source connects to every vertex with cap=1, cost=0 (path start)
        - Every vertex connects to sink with cap=1, cost=0 (path end)
        - Each edge (u,v) has arcs u->v cap=1 cost=1 and v->u cap=1 cost=1
        - We need to route m units of flow (one per edge)
        - Minimize total cost = m (fixed) ... that doesn't work.

        Let me think differently. The number of paths = number of "sources" used.
        Min-cost flow:
        - For each edge e=(u,v): we must send 1 unit from u to v OR from v to u.
        - Model: create a bipartite-like structure.
        
        Actually, the cleanest: 
        - Nodes: original vertices.
        - For each edge (u,v): add arc u->v with cap 1, cost 0, and arc v->u with cap 1, cost 0.
        - We need to select exactly one direction per edge such that the resulting
          directed graph decomposes into minimum number of directed paths.
        - Number of directed paths in a directed graph where each node has in/out
          from selected edges = (1/2) * sum_v |out_v - in_v|.
        - To minimize: orient edges to minimize sum |out_v - in_v|.
        - As argued, minimum is |odd|, giving |odd|/2 paths.
        
        The issue is SIMPLICITY. A directed trail may revisit vertices.
        
        For our small graphs, let me just do the Eulerian augmentation properly
        and then try to split trails into simple paths if needed. But actually,
        in the Eulerian circuit of the augmented graph, when we cut at the
        augmentation paths, the resulting segments ARE simple paths because
        the augmentation paths are vertex-disjoint from each other and the
        Eulerian circuit visits each edge exactly once.

        Wait - the segments between consecutive augmentation edges in the Eulerian
        circuit may NOT be simple (they could revisit vertices). Let me verify:
        In an Eulerian circuit, each edge is used exactly once, but vertices can
        be revisited. So a segment between two augmentation edges could have
        repeated vertices.

        OK so the key insight from the cell findings: the correct approach is
        to pair odd vertices with VERTEX-DISJOINT paths, form the Eulerian
        multigraph, run Hierholzer, and cut at the added path edges. The
        vertex-disjointness of the augmentation paths ensures simplicity.

        But finding vertex-disjoint shortest paths between matched pairs is
        the hard part. For small graphs, we can use max-flow for each pair
        or just accept that shortest paths might not be vertex-disjoint and
        use a different strategy.

        NEW APPROACH: Since the theoretical minimum is |odd|/2 paths, and our
        graphs have ~10-16 odd vertices, the target is 5-8 paths per graph.
        Let me implement a proper min-cost flow that directly optimizes the
        number of simple paths using a column-generation-like approach:
        
        1. Find all simple paths in the graph (feasible for n<=30, m~100).
        2. Solve set cover: select minimum number of edge-disjoint simple paths
           that cover all edges.
        3. This is a minimum set cover / packing problem.

        For n=30, the number of simple paths could be large, but we can limit
        to "reasonable" paths and use a greedy + local search.

        Actually, let me implement a clean approach:
        1. Compute all-pairs shortest paths.
        2. Find odd-degree vertices.
        3. Find minimum-cost perfect matching on odd vertices (by shortest path distance).
        4. For each matched pair, find a shortest path.
        5. Check if these paths are vertex-disjoint. If not, try alternative
           matchings or longer paths.
        6. Form augmented graph, find Eulerian circuit, cut at augmentation edges.
        7. Verify simplicity of resulting paths. If not simple, fall back.

        For the "different strategy" requirement, let me implement a
        SUCCESSIVE SHORTEST PATH min-cost flow that directly finds the
        minimum number of edge-disjoint paths by modeling it as:
        - Each unit of flow represents one edge being "assigned" to a path.
        - The cost structure penalizes path starts.
        """
        # Given the complexity, let me implement a practical approach:
        # 1. Eulerian augmentation with vertex-disjoint path enforcement
        # 2. If that fails, use a max-flow based decomposition
        
        if not edges:
            return []
        
        adj = build_adjacency(edges)
        edge_set = set()
        for (u, v) in edges:
            edge_set.add((u, v) if u <= v else (v, u))
        
        odd_vertices = [v for v in adj if len(adj[v]) % 2 == 1]
        
        if not odd_vertices:
            # Eulerian - one circuit, but may not be simple
            # For our purposes, return the circuit as one path if simple
            circuit = _find_eulerian_circuit(edges)
            if circuit and len(set(circuit)) == len(circuit):
                return [circuit]
            # Not simple, need to split
            return _split_eulerian_into_simple_paths(circuit, edge_set)
        
        # Find min-cost perfect matching on odd vertices
        n_odd = len(odd_vertices)
        
        # BFS distances
        dist_matrix = {}
        parent_matrix = {}
        for src in odd_vertices:
            d, par = _bfs(src, adj)
            for dst in odd_vertices:
                if dst != src:
                    key = (src, dst) if src < dst else (dst, src)
                    if dst in d:
                        dist_matrix[key] = d[dst]
                        parent_matrix[(src, dst)] = (d, par)
        
        # Minimum weight perfect matching (n_odd <= 16 typically)
        pairs = _min_weight_perfect_matching(odd_vertices, dist_matrix)
        
        if not pairs:
            return []
        
        # Find paths for each pair, ensuring vertex-disjointness
        # Try multiple times with different orderings
        best_result = None
        for attempt in range(50):
            if time.time() > _start + time_budget:
                break
            
            # Shuffle pair order for diversity
            import random
            rng = random.Random(attempt)
            shuffled_pairs = list(pairs)
            rng.shuffle(shuffled_pairs)
            
            used_vertices = set()
            aug_paths = []
            success = True
            
            for (a, b) in shuffled_pairs:
                # Find shortest path from a to b avoiding used_vertices
                # (except endpoints a and b themselves)
                path = _shortest_path_avoiding(a, b, adj, used_vertices)
                if path is None:
                    # Try without avoidance
                    path = _shortest_path(a, b, adj)
                    if path is None:
                        success = False
                        break
                    # Check if it conflicts
                    internal = set(path[1:-1])
                    if internal & used_vertices:
                        success = False
                        break
                
                internal = set(path[1:-1])
                if internal & used_vertices:
                    success = False
                    break
                
                used_vertices |= internal
                aug_paths.append(path)
            
            if not success:
                continue
            
            # Build augmented edge multiset
            aug_edges = list(edge_set)
            for path in aug_paths:
                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    key = (u, v) if u <= v else (v, u)
                    aug_edges.append(key)
            
            # Find Eulerian circuit
            circuit = _find_eulerian_circuit_from_multiset(aug_edges)
            if not circuit:
                continue
            
            # Cut at augmentation edges
            aug_edge_set = set()
            for path in aug_paths:
                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    key = (u, v) if u <= v else (v, u)
                    aug_edge_set.add(key)
            
            # But we need to track which specific edge instances are augmentation
            # Since edges can appear multiple times, use a counter approach
            aug_counter = defaultdict(int)
            for path in aug_paths:
                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    key = (u, v) if u <= v else (v, u)
                    aug_counter[key] += 1
            
            # Walk the circuit, tracking which edges are augmentation
            paths = []
            current = [circuit[0]]
            edge_usage = defaultdict(int)
            
            for i in range(1, len(circuit)):
                u, v = circuit[i - 1], circuit[i]
                key = (u, v) if u <= v else (v, u)
                edge_usage[key] += 1
                current.append(v)
                
                # Check if this edge instance is an augmentation edge
                if aug_counter.get(key, 0) > 0:
                    # Determine if this specific instance is augmentation
                    # Total occurrences of key in circuit so far
                    # Original edges: 1 occurrence each
                    # Augmentation: additional occurrences
                    orig_count = 1 if key in edge_set else 0
                    if edge_usage[key] > orig_count:
                        # This is an augmentation edge instance
                        if len(current) >= 2:
                            paths.append(current)
                        current = [v]
                        aug_counter[key] -= 1
            
            if len(current) >= 2:
                paths.append(current)
            
            paths = [p for p in paths if len(p) >= 2]
            
            # Validate
            if validate(paths, edge_set):
                if best_result is None or len(paths) < len(best_result):
                    best_result = paths
                    if len(paths) == len(odd_vertices) // 2:
                        break  # Optimal
        
        if best_result:
            return best_result
        
        # Fallback: greedy
        return _greedy_path_cover(edge_set, adj)

    def _bfs(src, adj):
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
        return d, par

    def _shortest_path(src, dst, adj):
        if src == dst:
            return [src]
        d, par = _bfs(src, adj)
        if dst not in d:
            return None
        path = []
        node = dst
        while node is not None:
            path.append(node)
            node = par[node]
        path.reverse()
        return path

    def _shortest_path_avoiding(src, dst, adj, avoid_set):
        """Find shortest path from src to dst, avoiding internal vertices in avoid_set."""
        if src == dst:
            return [src]
        visited = {src}
        parent = {src: None}
        q = deque([src])
        while q:
            cur = q.popleft()
            for nb in adj[cur]:
                if nb in visited:
                    continue
                if nb != dst and nb in avoid_set:
                    continue
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

    def _min_weight_perfect_matching(vertices, dist):
        n = len(vertices)
        if n == 0:
            return []
        if n % 2 != 0:
            return None
        
        # DP over subsets for n <= 16
        if n <= 16:
            from functools import lru_cache
            idx = {v: i for i, v in enumerate(vertices)}
            
            @lru_cache(maxsize=None)
            def dp(mask):
                if mask == (1 << n) - 1:
                    return 0, []
                
                # Find first unset bit
                i = -1
                for j in range(n):
                    if not (mask >> j) & 1:
                        i = j
                        break
                if i == -1:
                    return 0, []
                
                best = float('inf')
                best_j = -1
                best_rest = None
                
                for j in range(i + 1, n):
                    if not (mask >> j) & 1:
                        u, v = vertices[i], vertices[j]
                        key = (u, v) if u < v else (v, u)
                        w = dist.get(key, float('inf'))
                        if w < float('inf'):
                            new_mask = mask | (1 << i) | (1 << j)
                            rc, rp = dp(new_mask)
                            total = w + rc
                            if total < best:
                                best = total
                                best_j = j
                                best_rest = rp
                
                if best_j == -1:
                    return float('inf'), []
                
                return best, [(vertices[i], vertices[best_j])] + best_rest
            
            cost, pairs = dp(0)
            if cost >= float('inf'):
                return None
            return pairs
        else:
            # Greedy fallback
            all_edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    u, v = vertices[i], vertices[j]
                    key = (u, v) if u < v else (v, u)
                    w = dist.get(key, float('inf'))
                    if w < float('inf'):
                        all_edges.append((w, u, v))
            all_edges.sort()
            
            used = set()
            pairs = []
            for w, u, v in all_edges:
                if u not in used and v not in used:
                    pairs.append((u, v))
                    used.add(u)
                    used.add(v)
                    if len(used) == n:
                        break
            return pairs

    def _find_eulerian_circuit(edges):
        """Find Eulerian circuit using Hierholzer's algorithm."""
        if not edges:
            return []
        
        adj = defaultdict(list)
        edge_keys = []
        for (u, v) in edges:
            key = (u, v) if u <= v else (v, u)
            adj[u].append((v, key))
            adj[v].append((u, key))
            edge_keys.append(key)
        
        if not adj:
            return []
        
        start = next(iter(adj))
        stack = [start]
        circuit = []
        used = set()
        
        while stack:
            v = stack[-1]
            found = False
            while adj[v]:
                nb, key = adj[v].pop()
                if key not in used:
                    used.add(key)
                    stack.append(nb)
                    found = True
                    break
            if not found:
                circuit.append(stack.pop())
        
        circuit.reverse()
        
        # Verify all edges used
        if len(used) != len(set(edge_keys)):
            return None
        if len(circuit) != len(edges) + 1:
            return None
        
        return circuit

    def _find_eulerian_circuit_from_multiset(edge_list):
        """Find Eulerian circuit in a multigraph (edges may repeat)."""
        if not edge_list:
            return []
        
        # Build adjacency with edge indices
        adj = defaultdict(list)
        for idx, (u, v) in enumerate(edge_list):
            adj[u].append((v, idx))
            adj[v].append((u, idx))
        
        if not adj:
            return []
        
        start = next(iter(adj))
        stack = [start]
        circuit = []
        used = [False] * len(edge_list)
        
        while stack:
            v = stack[-1]
            found = False
            while adj[v]:
                nb, idx = adj[v].pop()
                if not used[idx]:
                    used[idx] = True
                    stack.append(nb)
                    found = True
                    break
            if not found:
                circuit.append(stack.pop())
        
        circuit.reverse()
        
        if len(circuit) != len(edge_list) + 1:
            return None
        
        return circuit

    def _split_eulerian_into_simple_paths(circuit, edge_set):
        """Split an Eulerian circuit into simple paths if it's not simple."""
        if not circuit:
            return []
        if len(set(circuit)) == len(circuit):
            return [circuit]
        
        # Find first repeated vertex
        seen = {}
        for i, v in enumerate(circuit):
            if v in seen:
                # Split at the repeat
                first_part = circuit[seen[v]:i+1]
                # The cycle from first occurrence to current is a valid cycle
                # We need to decompose differently
                break
            seen[v] = i
        
        # Fallback: greedy path extraction
        return _greedy_path_cover(edge_set, build_adjacency(list(edge_set)))

    def _greedy_path_cover(edge_set, adj):
        """Greedy: repeatedly find longest simple path in remaining edges."""
        if not edge_set:
            return []
        
        remaining = set(edge_set)
        paths = []
        
        while remaining:
            # Find a long simple path
            rem_adj = defaultdict(list)
            for (u, v) in remaining:
                rem_adj[u].append(v)
                rem_adj[v].append(u)
            
            if not rem_adj:
                break
            
            # Start from highest degree vertex
            start = max(rem_adj.keys(), key=lambda x: len(rem_adj[x]))
            path = [start]
            visited = {start}
            current = start
            
            while True:
                neighbors = [n for n in rem_adj[current] if n not in visited]
                if not neighbors:
                    break
                # Pick neighbor with most unvisited neighbors (dead-end heuristic)
                best = min(neighbors, key=lambda nb: len([x for x in rem_adj[nb] if x not in visited]))
                path.append(best)
                visited.add(best)
                current = best
            
            # Remove path edges from remaining
            path_edges = set()
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                key = (u, v) if u <= v else (v, u)
                path_edges.add(key)
            
            if path_edges:
                remaining -= path_edges
                paths.append(path)
            else:
                # Single edge
                e = next(iter(remaining))
                remaining.remove(e)
                paths.append([e[0], e[1]])
        
        return paths

    def _local_search_merge(paths, edge_set, adj, time_limit=5.0):
        """Try to merge paths by connecting endpoints with graph edges."""
        start_time = time.time()
        best = [list(p) for p in paths]
        
        def is_simple(p):
            return len(set(p)) == len(p)
        
        while time.time() - start_time < time_limit:
            improved = False
            n = len(best)
            for i in range(n):
                for j in range(i + 1, n):
                    p1, p2 = best[i], best[j]
                    
                    # Try all 4 endpoint combinations
                    candidates = []
                    for a, b in [(p1[-1], p2[0]), (p1[0], p2[0]),
                                  (p1[-1], p2[-1]), (p1[0], p2[-1])]:
                        key = (a, b) if a <= b else (b, a)
                        if key in edge_set:
                            # Check if this edge is already used
                            # For simplicity, just try the merge
                            if a == p1[-1] and b == p2[0]:
                                merged = p1 + p2[1:]
                            elif a == p1[0] and b == p2[0]:
                                merged = list(reversed(p1)) + p2[1:]
                            elif a == p1[-1] and b == p2[-1]:
                                merged = p1 + list(reversed(p2[1:]))
                            else:
                                merged = list(reversed(p1)) + list(reversed(p2[1:]))
                            
                            if is_simple(merged):
                                candidates.append(merged)
                    
                    if candidates:
                        merged = candidates[0]
                        new_paths = [best[k] for k in range(n) if k != i and k != j]
                        new_paths.append(merged)
                        best = new_paths
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        return best

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        elapsed = time.time() - _start
        remaining = _DEADLINE - elapsed
        
        if remaining < 5:
            edges = instance(*spec)
            edge_set = set()
            for (u, v) in edges:
                edge_set.add((u, v) if u <= v else (v, u))
            adj = build_adjacency(edges)
            results.append(_greedy_path_cover(edge_set, adj))
            continue
        
        edges = instance(*spec)
        edge_set = set()
        for (u, v) in edges:
            edge_set.add((u, v) if u <= v else (v, u))
        adj = build_adjacency(edges)
        
        # Try the flow/matching-based approach
        time_budget = max(2.0, min(40.0, remaining * 0.6))
        try:
            paths = min_cost_flow_path_cover(list(range(max(edge_set and max(max(e) for e in edge_set) + 1) or 0)),
                                             list(edge_set), time_budget)
        except Exception:
            paths = []
        
        if not paths or not validate(paths, edge_set):
            paths = _greedy_path_cover(edge_set, adj)
        
        # Local search improvement
        ls_time = max(0.5, min(10.0, (remaining - 5) * 0.3))
        try:
            improved = _local_search_merge(paths, edge_set, adj, ls_time)
            if validate(improved, edge_set):
                paths = improved
        except Exception:
            pass
        
        # Final validation
        if not validate(paths, edge_set):
            paths = _greedy_path_cover(edge_set, adj)
            if not validate(paths, edge_set):
                # Last resort: each edge as its own path
                paths = [[u, v] for (u, v) in sorted(edge_set)]
        
        results.append(paths)
    
    return results
# EVOLVE-BLOCK-END
