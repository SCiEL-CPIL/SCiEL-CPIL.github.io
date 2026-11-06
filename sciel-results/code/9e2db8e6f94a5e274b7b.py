import time
import random
from collections import defaultdict
import heapq

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    graphs = [instance(*spec) for spec in specs]

    def min_cost_circulation(nodes, arcs, time_limit):
        """
        Find min-cost circulation in a directed graph.
        nodes: list of node ids
        arcs: list of (u, v, cap, cost)
        Returns (flow_dict, total_cost) where flow_dict[(u,v)] = flow
        """
        if not arcs:
            return {}, 0

        # Build residual network
        # For each arc (u,v,cap,cost), we have forward (cap, cost) and backward (0, -cost)
        # We'll use indexed arcs
        arc_index = {}
        for i, (u, v, cap, cost) in enumerate(arcs):
            arc_index[(u, v, i)] = i

        # Residual graph: adj[u] = list of (v, residual_cap, cost, arc_id)
        # arc_id identifies the original arc; for reverse, we track separately
        # Let's use a simpler representation:
        # residual[u] = list of [v, cap, cost, reverse_idx]
        # and we maintain a mapping from (u,v,arc_idx) to position in residual[u]

        # Simpler: just use a dict-based residual network
        # res[u][v] = (cap, cost) for forward, but we need to handle multiple arcs
        # Since our arcs are unique per (u,v) pair in the construction, let's verify.
        # Actually, in our formulation, for each undirected edge (u,v), we have:
        #   arc1: u->v with cap=1, cost=0
        #   arc2: v->u with cap=1, cost=0
        # These are distinct directed arcs. So (u,v) appears at most once as a directed arc.
        # Good, so we can use a simple dict.

        # But wait - we also need to track which original arc each residual edge corresponds to
        # for extracting the flow. Let me use a different approach.

        # Let's build the network with explicit arc tracking.
        # Each original arc gets an id. The residual graph has forward and backward edges.

        n_arcs = len(arcs)
        # For each original arc i: (u_i, v_i, cap_i, cost_i)
        # Forward residual: u_i -> v_i, cap = cap_i - flow_i, cost = cost_i
        # Backward residual: v_i -> u_i, cap = flow_i, cost = -cost_i

        # We'll maintain:
        # flow[i] for each original arc
        # And use successive shortest path (Bellman-Ford or Dijkstra with potentials)

        # Initialize flow to 0
        flow = [0] * n_arcs

        # Build initial residual graph
        # adj[u] = list of (v, cap, cost, arc_id, is_forward)
        adj = defaultdict(list)
        for i, (u, v, cap, cost) in enumerate(arcs):
            adj[u].append([v, cap, cost, i, True])   # forward
            adj[v].append([u, 0, -cost, i, False])   # backward

        total_cost = 0

        # Successive shortest path: repeatedly find shortest path from any node to any node
        # that can form a cycle? No - for circulation, we need to find negative-cost cycles
        # in the residual graph and augment along them.

        # Actually, for min-cost circulation, the standard approach is:
        # 1. Find any feasible circulation (e.g., all zero flow is feasible if no demands)
        # 2. Repeatedly find negative-cost cycles in residual graph and augment

        # Since all costs are 0 in our formulation, the min-cost circulation is just
        # any circulation with minimum... wait, all costs are 0, so any circulation has cost 0.
        # That's not helpful.

        # Let me reconsider the formulation. The cost should reflect what we're optimizing.
        # We want to MAXIMIZE the number of edges in cycles, equivalently MINIMIZE edges not in cycles.
        # 
        # Alternative formulation:
        # For each undirected edge (u,v), create two directed arcs u->v and v->u, each with cap=1.
        # Assign cost 0 to all arcs.
        # We want a circulation (flow conservation at every node).
        # A circulation decomposes into cycles. The number of edges used in the circulation
        # is the total flow. We want to MAXIMIZE total flow.
        #
        # This is a max-flow problem! We want the maximum flow circulation.
        # But a circulation with flow conservation at every node... 
        # Actually, if we just want max flow through the network with no source/sink,
        # that's not standard. 
        #
        # Let me think differently. The problem of finding a maximum subset of edges
        # that can be decomposed into cycles is equivalent to finding a maximum-size
        # subgraph where every vertex has even degree (since such a subgraph decomposes
        # into cycles). 
        #
        # This is the "maximum even subgraph" problem, which can be solved via:
        # - For each connected component, we can include all edges except those in a
        #   minimum T-join where T is the set of odd-degree vertices.
        # - Actually, it's simpler: remove the minimum number of edges to make all degrees even.
        #   This is a minimum-weight T-join problem where T = set of odd-degree vertices.
        #
        # Wait, that's for making a given graph have all even degrees by removing edges.
        # The minimum number of edges to remove = (number of odd-degree vertices) / 2 in the
        # best case, but it's actually a matching problem on the odd-degree vertices.
        #
        # Specifically: In each connected component, the set of odd-degree vertices has even size.
        # We need to find a minimum-cardinality set of edges whose removal makes all degrees even.
        # This is equivalent to finding a minimum T-join where T is the set of odd-degree vertices.
        # For unweighted graphs, this can be solved by finding a minimum-weight perfect matching
        # on the odd-degree vertices, where the weight between two odd vertices is the shortest
        # path distance between them.
        #
        # Then the remaining edges form a subgraph with all even degrees, which decomposes into cycles.
        # The number of edges in cycles = |E| - (number of edges removed).
        #
        # Let me implement this approach!

        return None  # Won't use this function, implementing different approach

    def decompose(edges, time_limit):
        """Decompose edges into cycles and single edges, minimizing piece count."""
        if not edges:
            return []

        n = max(max(u, v) for u, v in edges) + 1 if edges else 0

        # Build adjacency
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # Find connected components
        visited = set()
        components = []
        for node in range(n):
            if node in visited or node not in adj:
                continue
            comp = set()
            stack = [node]
            visited.add(node)
            while stack:
                u = stack.pop()
                comp.add(u)
                for v in adj[u]:
                    if v not in visited:
                        visited.add(v)
                        stack.append(v)
            components.append(comp)

        # For each component, find odd-degree vertices
        # and compute minimum T-join (minimum edges to remove to make all degrees even)
        removed_edges = set()

        for comp in components:
            if len(comp) < 3:
                # Can't form a cycle with < 3 vertices
                for u in comp:
                    for v in adj[u]:
                        if v in comp:
                            removed_edges.add((min(u, v), max(u, v)))
                continue

            # Find odd-degree vertices within this component
            odd_vertices = []
            for u in comp:
                deg = sum(1 for v in adj[u] if v in comp)
                if deg % 2 == 1:
                    odd_vertices.append(u)

            if not odd_vertices:
                # Already all even degrees, no edges to remove
                continue

            # Compute shortest path distances between all pairs of odd vertices
            # Using BFS from each odd vertex
            k = len(odd_vertices)
            if k == 0:
                continue

            # dist[i][j] = shortest path distance between odd_vertices[i] and odd_vertices[j]
            # Also track parent for path reconstruction
            dist = [[float('inf')] * k for _ in range(k)]
            parent = [[None] * k for _ in range(k)]  # parent[i][j] = previous node on path from odd[i] to odd[j]

            for i, src in enumerate(odd_vertices):
                # BFS from src
                bfs_dist = {src: 0}
                bfs_parent = {src: None}
                queue = [src]
                qi = 0
                while qi < len(queue):
                    u = queue[qi]
                    qi += 1
                    for v in adj[u]:
                        if v in comp and v not in bfs_dist:
                            bfs_dist[v] = bfs_dist[u] + 1
                            bfs_parent[v] = u
                            queue.append(v)

                for j, dst in enumerate(odd_vertices):
                    if j == i:
                        continue
                    if dst in bfs_dist:
                        dist[i][j] = bfs_dist[dst]
                        # Reconstruct path
                        path = []
                        cur = dst
                        while cur != src:
                            path.append(cur)
                            cur = bfs_parent[cur]
                        path.append(src)
                        path.reverse()
                        parent[i][j] = path

            # Now find minimum-weight perfect matching on k odd vertices
            # k is even (handshaking lemma)
            # Use DP over subsets: dp[mask] = min cost to match vertices in mask
            # mask has k bits, set bit i means vertex i is unmatched
            # We want dp[0] = 0, and dp[mask] = min over pairs (i,j) in mask of dist[i][j] + dp[mask ^ (1<<i) ^ (1<<j)]

            if k > 20:
                # Too many odd vertices for DP, use greedy matching
                # Greedy: repeatedly match the closest pair
                matched = [False] * k
                match_pairs = []
                while True:
                    best_i, best_j, best_d = -1, -1, float('inf')
                    for i in range(k):
                        if matched[i]:
                            continue
                        for j in range(i + 1, k):
                            if matched[j]:
                                continue
                            if dist[i][j] < best_d:
                                best_d = dist[i][j]
                                best_i, best_j = i, j
                    if best_i == -1:
                        break
                    matched[best_i] = matched[best_j] = True
                    match_pairs.append((best_i, best_j))
            else:
                # DP over subsets
                size = 1 << k
                dp = [float('inf')] * size
                dp[0] = 0
                match_choice = [None] * size  # match_choice[mask] = (i, j) chosen

                for mask in range(size):
                    if dp[mask] == float('inf'):
                        continue
                    # Find first unmatched vertex
                    first = -1
                    for i in range(k):
                        if not (mask >> i) & 1:
                            first = i
                            break
                    if first == -1:
                        continue
                    for j in range(first + 1, k):
                        if not (mask >> j) & 1:
                            new_mask = mask | (1 << first) | (1 << j)
                            new_cost = dp[mask] + dist[first][j]
                            if new_cost < dp[new_mask]:
                                dp[new_mask] = new_cost
                                match_choice[new_mask] = (first, j)

                # Extract matching from dp
                match_pairs = []
                mask = (1 << k) - 1
                while mask:
                    i, j = match_choice[mask]
                    match_pairs.append((i, j))
                    mask ^= (1 << i)
                    mask ^= (1 << j)

            # Remove edges along the paths for each matched pair
            for i, j in match_pairs:
                if parent[i][j]:
                    path = parent[i][j]
                    for idx in range(len(path) - 1):
                        u, v = path[idx], path[idx + 1]
                        removed_edges.add((min(u, v), max(u, v)))
                elif dist[i][j] == float('inf'):
                    # Disconnected within component? Shouldn't happen since it's a connected component
                    # But just in case, remove an arbitrary edge incident to odd_vertices[i]
                    if adj[odd_vertices[i]]:
                        v = next(iter(adj[odd_vertices[i]]))
                        removed_edges.add((min(odd_vertices[i], v), max(odd_vertices[i], v)))

        # Remaining edges form a subgraph with all even degrees
        remaining = set((min(u, v), max(u, v)) for u, v in edges) - removed_edges

        # Decompose remaining edges into cycles
        pieces = []
        rem_adj = defaultdict(set)
        for u, v in remaining:
            rem_adj[u].add(v)
            rem_adj[v].add(u)

        def extract_cycle(start):
            """Extract a cycle starting from start using DFS."""
            visited = {start}
            path = [start]

            def dfs(node):
                for neighbor in sorted(rem_adj[node]):
                    if neighbor == start and len(path) >= 3:
                        return path[:]
                    if neighbor not in visited:
                        visited.add(neighbor)
                        path.append(neighbor)
                        result = dfs(neighbor)
                        if result:
                            return result
                        path.pop()
                        visited.discard(neighbor)
                return None

            return dfs(start)

        while remaining:
            # Find a vertex with degree >= 2
            start = None
            for v in list(rem_adj.keys()):
                if len(rem_adj[v]) >= 2:
                    start = v
                    break

            if start is None:
                # No more cycles possible
                break

            cycle = extract_cycle(start)
            if cycle is None:
                # Shouldn't happen if all degrees are even and >= 2
                # But just in case, take a single edge
                u, v = next(iter(remaining))
                pieces.append([u, v])
                remaining.discard((u, v))
                rem_adj[u].discard(v)
                rem_adj[v].discard(u)
                continue

            pieces.append(cycle)
            # Remove cycle edges
            n_c = len(cycle)
            for i in range(n_c):
                u = cycle[i]
                v = cycle[(i + 1) % n_c]
                e = (min(u, v), max(u, v))
                remaining.discard(e)
                rem_adj[u].discard(v)
                rem_adj[v].discard(u)

        # Any remaining edges (shouldn't be any if all degrees were even) become single edges
        for u, v in sorted(remaining):
            pieces.append([u, v])

        # Add removed edges as single-edge pieces
        for u, v in sorted(removed_edges):
            pieces.append([u, v])

        return pieces

    results = []
    for i, edges in enumerate(graphs):
        remaining_time = deadline - time.time()
        time_per_graph = remaining_time / (len(graphs) - i)
        pieces = decompose(edges, time.time() + time_per_graph)
        results.append(pieces)

    return results
