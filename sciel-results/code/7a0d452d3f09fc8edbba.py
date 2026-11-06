"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict, deque

def construct():
    t_start = time.time()
    DEADLINE = 140.0

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))

    def hopcroft_karp(adj, n):
        """
        Find a maximum matching in a bipartite graph represented as adj[u] = list of v.
        Returns matching as dict u->v.
        """
        if n == 0:
            return {}
        pair_u = [-1] * (n + 1)
        pair_v = [-1] * (n + 1)
        dist = [0] * (n + 1)

        def bfs():
            queue = deque()
            for u in range(1, n + 1):
                if pair_u[u] == -1:
                    dist[u] = 0
                    queue.append(u)
                else:
                    dist[u] = -1
            found = False
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    pu = pair_v[v]
                    if pu == -1:
                        found = True
                    elif dist[pu] == -1:
                        dist[pu] = dist[u] + 1
                        queue.append(pu)
            return found

        def dfs(u):
            for v in adj[u]:
                pu = pair_v[v]
                if pu == -1 or (dist[pu] == dist[u] + 1 and dfs(pu)):
                    pair_u[u] = v
                    pair_v[v] = u
                    return True
            dist[u] = -1
            return False

        matching = 0
        while bfs():
            for u in range(1, n + 1):
                if pair_u[u] == -1:
                    if dfs(u):
                        matching += 1
        return {u: pair_u[u] for u in range(1, n + 1) if pair_u[u] != -1}

    def decompose(edges, n):
        """
        Decompose edges into cycles and single edges.
        Strategy:
          1. Find a 2-factor (collection of edge-disjoint cycles covering all matched vertices)
             via bipartite matching on the double cover.
          2. Extract cycles from the 2-factor.
          3. On the remaining edges, repeatedly find long cycles via DFS.
          4. Leftover edges become single-edge pieces.
        """
        edge_set = set()
        for e in edges:
            edge_set.add(tuple(sorted(e)))

        pieces = []
        used_edges = set()

        # Step 1: Find a 2-factor using bipartite matching on the double cover
        # Build adjacency for the double cover
        adj_double = {i: [] for i in range(1, n + 1)}
        for (u, v) in edge_set:
            adj_double[u].append(v)
            adj_double[v].append(u)

        # Find maximum matching in the bipartite graph
        # The bipartite graph has left nodes 1..n and right nodes 1..n
        # An edge u-L to v-R exists if (u,v) is in the original graph
        matching = hopcroft_karp(adj_double, n)

        # Build the 2-factor from the matching
        # Each vertex u is matched to some v, and v is matched to some u'
        # This gives us a set of cycles
        visited = set()
        for start in range(1, n + 1):
            if start in visited:
                continue
            if start not in matching:
                continue
            # Follow the matching to form a cycle
            cycle = [start]
            visited.add(start)
            cur = start
            while True:
                # cur is matched to some vertex
                if cur not in matching:
                    break
                nxt = matching[cur]
                if nxt in visited:
                    if nxt == start:
                        break
                    else:
                        # Should not happen in a proper matching
                        break
                cycle.append(nxt)
                visited.add(nxt)
                cur = nxt

            if len(cycle) >= 3:
                # Verify all edges exist and are not used
                valid = True
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    edge = tuple(sorted((u, v)))
                    if edge not in edge_set or edge in used_edges:
                        valid = False
                        break
                if valid:
                    # Mark edges as used
                    for i in range(len(cycle)):
                        u = cycle[i]
                        v = cycle[(i + 1) % len(cycle)]
                        edge = tuple(sorted((u, v)))
                        used_edges.add(edge)
                    pieces.append(cycle)

        # Step 2: On remaining edges, repeatedly find cycles
        while time.time() < DEADLINE:
            # Build current adjacency without used edges
            cur_adj = defaultdict(list)
            for (u, v) in edge_set:
                if (u, v) in used_edges:
                    continue
                cur_adj[u].append(v)
                cur_adj[v].append(u)

            if not cur_adj:
                break

            # Find a cycle using DFS
            n_max = max(cur_adj.keys()) + 1 if cur_adj else 0
            if n_max == 0:
                break

            color = [0] * (n_max + 1)
            parent = [-1] * (n_max + 1)
            found_cycle = None

            def dfs2(u, start):
                nonlocal found_cycle
                color[u] = 1
                for v in cur_adj[u]:
                    if color[v] == 0:
                        parent[v] = u
                        if dfs2(v, start):
                            return True
                    elif color[v] == 1:
                        if v == start:
                            cyc = [u]
                            cur = u
                            while cur != start:
                                cur = parent[cur]
                                if cur == -1:
                                    return False
                                cyc.append(cur)
                            cyc.reverse()
                            if len(cyc) >= 3:
                                found_cycle = cyc
                                return True
                        else:
                            cyc = [u]
                            cur = u
                            while cur != v:
                                cur = parent[cur]
                                if cur == -1:
                                    return False
                                cyc.append(cur)
                            cyc.append(v)
                            if len(cyc) >= 3:
                                found_cycle = cyc
                                return True
                color[u] = 2
                return False

            for s in range(1, n_max + 1):
                if color[s] == 0 and cur_adj[s]:
                    if dfs2(s, s):
                        break

            if found_cycle is None:
                break

            # Remove edges in the cycle
            for i in range(len(found_cycle)):
                u = found_cycle[i]
                v = found_cycle[(i + 1) % len(found_cycle)]
                edge = tuple(sorted((u, v)))
                used_edges.add(edge)

            pieces.append(found_cycle)

        # Step 3: Remaining edges
        for e in edge_set:
            if e not in used_edges:
                pieces.append(list(e))

        return pieces

    results = []
    for spec in specs:
        if time.time() > DEADLINE:
            # Fallback: just return single edges for remaining
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
        edges = instance(*spec)
        n = spec[0]
        result = decompose(edges, n)
        results.append(result)

    return results
# EVOLVE-BLOCK-END
