"""Program that decomposes graphs into few cycles and edges."""
# EVOLVE-BLOCK-START
import time
from random import Random
from collections import deque


def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def hopcroft_karp(bip_adj, left_nodes, right_nodes):
        """Maximum bipartite matching via Hopcroft-Karp.
        bip_adj: dict mapping left node -> list of right neighbors.
        Returns (pair_u, pair_v) dicts."""
        pair_u = {u: -1 for u in left_nodes}
        pair_v = {v: -1 for v in right_nodes}
        dist = {}

        def bfs():
            queue = deque()
            found = False
            for u in left_nodes:
                if pair_u[u] == -1:
                    dist[u] = 0
                    queue.append(u)
                else:
                    dist[u] = float('inf')
            while queue:
                u = queue.popleft()
                for v in bip_adj.get(u, []):
                    pu = pair_v[v]
                    if pu == -1:
                        found = True
                    elif dist.get(pu, float('inf')) == float('inf'):
                        dist[pu] = dist[u] + 1
                        queue.append(pu)
            return found

        def dfs(u):
            for v in bip_adj.get(u, []):
                pu = pair_v[v]
                if pu == -1 or (dist.get(pu, float('inf')) == dist[u] + 1 and dfs(pu)):
                    pair_u[u] = v
                    pair_v[v] = u
                    return True
            dist[u] = float('inf')
            return False

        while bfs():
            for u in left_nodes:
                if pair_u[u] == -1:
                    dfs(u)
        return pair_u, pair_v

    def decompose(edges, n):
        # Build live adjacency (will be updated as edges are removed)
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        remaining = set((min(u, v), max(u, v)) for u, v in edges)
        pieces = []

        def remove_cycle(cycle):
            """Remove edges of cycle from remaining and adj."""
            for i in range(len(cycle)):
                u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                e = (min(u, v), max(u, v))
                if e in remaining:
                    remaining.discard(e)
                    adj[u].discard(v)
                    adj[v].discard(u)

        def extract_max_2factor():
            """Extract a maximum 2-factor using Hopcroft-Karp on remaining edges.
            Returns list of cycles (each a list of vertices)."""
            # Candidates: vertices with at least 2 remaining incident edges
            candidates = [v for v in range(n) if len(adj[v]) >= 2]
            if len(candidates) < 3:
                return []
            cand_set = set(candidates)

            # Build bipartite adjacency: u_left -> v_right if edge (u,v) in remaining
            bip_adj = {u: [] for u in candidates}
            for u in candidates:
                for v in adj[u]:
                    if v in cand_set and u != v:
                        bip_adj[u].append(v)

            pair_u, pair_v = hopcroft_karp(bip_adj, candidates, candidates)

            # Build directed edges: u -> pair_u[u]
            # Also need reverse: for each v, which u has pair_u[u] == v?
            # Actually we need: out[u] = pair_u[u], and in[v] = {u : pair_u[u] == v}
            # But for cycle extraction in a functional graph where each node has at most
            # one outgoing edge (pair_u) and at most one incoming edge (pair_v),
            # we trace: u -> pair_u[u] -> pair_u[pair_u[u]] -> ...
            
            # Build successor map
            succ = {}
            for u in candidates:
                if pair_u[u] != -1:
                    succ[u] = pair_u[u]

            # Find cycles of length >= 3
            visited = set()
            cycles = []
            for start in candidates:
                if start in visited:
                    continue
                # Trace the path
                path = [start]
                cur = start
                while cur in succ and cur not in visited:
                    cur = succ[cur]
                    if cur in visited:
                        # Found a cycle - check if it's in our path
                        if cur in path:
                            idx = path.index(cur)
                            cycle = path[idx:]
                            if len(cycle) >= 3:
                                cycles.append(cycle)
                                visited.update(cycle)
                        break
                    if cur in path:
                        idx = path.index(cur)
                        cycle = path[idx:]
                        if len(cycle) >= 3:
                            cycles.append(cycle)
                            visited.update(cycle)
                        break
                    path.append(cur)
                else:
                    # cur not in succ or cur in visited
                    if cur in visited:
                        if cur in path:
                            idx = path.index(cur)
                            cycle = path[idx:]
                            if len(cycle) >= 3:
                                cycles.append(cycle)
                                visited.update(cycle)
                    elif cur in path:
                        idx = path.index(cur)
                        cycle = path[idx:]
                        if len(cycle) >= 3:
                            cycles.append(cycle)
                            visited.update(cycle)
                # Mark non-cycle nodes as visited to avoid re-tracing
                for node in path:
                    if node not in visited:
                        visited.add(node)

            return cycles

        def find_cycle_dfs(start):
            """Find a simple cycle through start using DFS on remaining edges."""
            visited = set()
            parent = {}
            cycle = None

            def dfs(node):
                nonlocal cycle
                if cycle is not None:
                    return
                visited.add(node)
                for neighbor in sorted(adj[node]):
                    if neighbor == start and len(visited) >= 3:
                        # Reconstruct cycle
                        cycle = [start]
                        cur = node
                        while cur != start:
                            cycle.append(cur)
                            cur = parent[cur]
                        return
                    if neighbor not in visited:
                        parent[neighbor] = node
                        dfs(neighbor)
                        if cycle is not None:
                            return

            dfs(start)
            return cycle

        # Main loop: extract 2-factors, then individual cycles
        max_iterations = 1000
        iter_count = 0
        while iter_count < max_iterations:
            iter_count += 1
            if time.time() > deadline - 5:
                break
            if not remaining:
                break

            # Try to extract a 2-factor (multiple cycles at once)
            cycles_2f = extract_max_2factor()
            if cycles_2f:
                for cyc in cycles_2f:
                    remove_cycle(cyc)
                    pieces.append(cyc)
                continue

            # Fallback: find a single cycle via DFS
            cycle = None
            for start in range(n):
                if len(adj[start]) < 2:
                    continue
                cycle = find_cycle_dfs(start)
                if cycle is not None:
                    break
            if cycle is None:
                break

            remove_cycle(cycle)
            pieces.append(cycle)

        # Remaining edges become individual pieces
        for u, v in sorted(remaining):
            pieces.append([u, v])

        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        n, p, seed = spec
        edges = instance(n, p, seed)
        pieces = decompose(edges, n)
        results.append(pieces)

    return results
# EVOLVE-BLOCK-END
