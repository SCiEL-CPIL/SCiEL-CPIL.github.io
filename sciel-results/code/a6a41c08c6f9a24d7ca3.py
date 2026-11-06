"""Program that decomposes graphs into few cycles and edges."""
# EVOLVE-BLOCK-START
import time
from random import Random


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

    def decompose(edges, n):
        # Build adjacency
        adj = {i: set() for i in range(n)}
        edge_set = set()
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((min(u, v), max(u, v)))

        remaining = set(edge_set)
        pieces = []

        def find_cycle_dfs(start):
            """Find a cycle starting from start using DFS."""
            visited = set()
            parent = {}
            cycle = None

            def dfs(node):
                nonlocal cycle
                if cycle is not None:
                    return
                visited.add(node)
                for neighbor in sorted(adj[node]):
                    if (min(node, neighbor), max(node, neighbor)) not in remaining:
                        continue
                    if neighbor == start and len(visited) >= 3:
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

        def find_longest_cycle():
            """Find a longer cycle by trying multiple starts."""
            best = None
            for start in range(n):
                if len(adj[start]) < 2:
                    continue
                visited = set()
                parent = {}
                cycle = None

                def dfs(node, start_node, depth):
                    nonlocal cycle, best
                    if cycle is not None:
                        return
                    if depth > 50:  # limit depth for speed
                        return
                    visited.add(node)
                    for neighbor in sorted(adj[node]):
                        if (min(node, neighbor), max(node, neighbor)) not in remaining:
                            continue
                        if neighbor == start_node and len(visited) >= 3:
                            new_cycle = [start_node]
                            cur = node
                            while cur != start_node:
                                new_cycle.append(cur)
                                cur = parent[cur]
                            if best is None or len(new_cycle) > len(best):
                                best = new_cycle
                            cycle = new_cycle
                            return
                        if neighbor not in visited:
                            parent[neighbor] = node
                            dfs(neighbor, start_node, depth + 1)
                            if cycle is not None:
                                return

                dfs(start, start, 0)
                if cycle is not None:
                    return cycle
            return best

        def find_2_factor():
            """Find a 2-factor (collection of edge-disjoint cycles covering all vertices with degree >= 2)
            using max matching on a bipartite graph representation."""
            # Only consider vertices with degree >= 2 in remaining
            candidates = [v for v in range(n) if len(adj[v]) >= 2 and 
                         all((min(v, u), max(v, u)) in remaining for u in adj[v])]
            if len(candidates) < 3:
                return []

            # Build bipartite graph: left = vertices, right = vertices
            # Edge (u_left, v_right) exists if (u,v) in remaining and u != v
            from collections import deque

            # Hopcroft-Karp for maximum bipartite matching
            left_nodes = candidates
            right_nodes = candidates
            
            # adjacency for bipartite graph
            bip_adj = {u: [] for u in left_nodes}
            for u in left_nodes:
                for v in adj[u]:
                    if v in set(candidates) and (min(u, v), max(u, v)) in remaining and u != v:
                        bip_adj[u].append(v)

            # Hopcroft-Karp
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
                    for v in bip_adj[u]:
                        pu = pair_v[v]
                        if pu == -1:
                            found = True
                        elif dist[pu] == float('inf'):
                            dist[pu] = dist[u] + 1
                            queue.append(pu)
                return found

            def dfs(u):
                for v in bip_adj[u]:
                    pu = pair_v[v]
                    if pu == -1 or (dist[pu] == dist[u] + 1 and dfs(pu)):
                        pair_u[u] = v
                        pair_v[v] = u
                        return True
                dist[u] = float('inf')
                return False

            matching_size = 0
            while bfs():
                for u in left_nodes:
                    if pair_u[u] == -1:
                        if dfs(u):
                            matching_size += 1

            # Extract cycles from the matching
            # Each matched pair (u, v) means edge u->v in the 2-factor
            # We need to find cycles in the directed graph where each node has out-degree 1
            matched_edges = []
            for u in left_nodes:
                if pair_u[u] != -1:
                    matched_edges.append((u, pair_u[u]))

            # Find cycles in the functional graph
            visited_nodes = set()
            cycles = []
            for u, v in matched_edges:
                if u in visited_nodes:
                    continue
                # Trace the cycle
                path = [u]
                cur = v
                while cur != u and cur not in visited_nodes:
                    path.append(cur)
                    visited_nodes.add(cur)
                    # Find next: what node points to cur?
                    next_node = None
                    for uu, vv in matched_edges:
                        if vv == cur and uu not in visited_nodes:
                            next_node = uu
                            break
                    if next_node is None:
                        break
                    cur = next_node
                if cur == u and len(path) >= 3:
                    cycles.append(path)
                    visited_nodes.update(path)

            return cycles

        # Greedily extract cycles
        max_iterations = 500
        iter_count = 0
        while iter_count < max_iterations:
            iter_count += 1
            if time.time() > deadline - 10:
                break

            # First try to find a 2-factor to extract multiple cycles at once
            two_factor = find_2_factor()
            if two_factor:
                for cycle in two_factor:
                    # Remove cycle edges from remaining
                    for i in range(len(cycle)):
                        u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                        e = (min(u, v), max(u, v))
                        if e in remaining:
                            remaining.discard(e)
                    pieces.append(cycle)
                continue

            # Fallback to single cycle extraction
            cycle = find_longest_cycle()
            if cycle is None:
                # Try DFS from each vertex
                for start in range(n):
                    if len(adj[start]) < 2:
                        continue
                    cycle = find_cycle_dfs(start)
                    if cycle is not None:
                        break
            if cycle is None:
                break

            # Remove cycle edges from remaining
            for i in range(len(cycle)):
                u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                e = (min(u, v), max(u, v))
                if e in remaining:
                    remaining.discard(e)

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
