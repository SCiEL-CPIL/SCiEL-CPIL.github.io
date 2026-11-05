"""Path cover via odd-vertex pairing + Eulerian trail extraction."""
import time
import heapq
from collections import deque
from random import Random


def construct():
    t0 = time.time()
    DEADLINE = 140.0  # safety margin under 150s kill

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def build_adj(edges, n):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def odd_vertices(adj, n):
        deg = [len(a) for a in adj]
        return [v for v in range(n) if deg[v] % 2 == 1]

    def all_pairs_dists(adj, n, sources):
        # BFS from each source; return dict source -> list of (dist, prev)
        res = {}
        for s in sources:
            dist = [-1] * n
            prev = [-1] * n
            dist[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        prev[v] = u
                        q.append(v)
            res[s] = (dist, prev)
        return res

    def min_cost_matching(costs, k):
        # Hungarian algorithm for min cost perfect matching, k even, k small
        # costs: k x k symmetric with inf on diagonal
        # Returns list of k//2 pairs (i, j) with i < j
        if k == 0:
            return []
        INF = 10**9
        # Standard Hungarian for min assignment
        n = k
        u = [0] * (n + 1)
        v = [0] * (n + 1)
        p = [0] * (n + 1)
        way = [0] * (n + 1)
        for i in range(1, n + 1):
            p[0] = i
            j0 = 0
            minv = [INF] * (n + 1)
            used = [False] * (n + 1)
            while True:
                used[j0] = True
                i0 = p[j0]
                delta = INF
                j1 = 0
                for j in range(1, n + 1):
                    if not used[j]:
                        cur = costs[i0 - 1][j - 1] - u[i0] - v[j]
                        if cur < minv[j]:
                            minv[j] = cur
                            way[j] = j0
                        if minv[j] < delta:
                            delta = minv[j]
                            j1 = j
                for j in range(0, n + 1):
                    if used[j]:
                        u[p[j]] += delta
                        v[j] -= delta
                    else:
                        minv[j] -= delta
                j0 = j1
                if p[j0] == 0:
                    break
            while True:
                j1 = way[j0]
                p[j0] = p[j1]
                j0 = j1
                if j0 == 0:
                    break
        # p[j] = row assigned to column j
        pairs = []
        for j in range(1, n + 1):
            i = p[j]
            if i > 0 and i < j:
                pairs.append((i - 1, j - 1))
        return pairs

    def extract_paths(adj, n, extra_edges):
        # adj: original adjacency (list of lists)
        # extra_edges: list of (u, v) to add (will be treated as used once)
        # Build multigraph edge list with unique ids
        edge_list = []  # (u, v)
        edge_idx = {}
        for u, v in extra_edges:
            edge_list.append((u, v))
            edge_list.append((v, u))
            edge_idx[(u, v)] = len(edge_list) - 2
        # We'll track used edge ids
        used = [False] * len(edge_list)
        # Build adjacency with edge ids
        adj2 = [[] for _ in range(n)]
        for idx, (u, v) in enumerate(edge_list):
            adj2[u].append((v, idx))
        # Find all vertices with odd degree in the multigraph (should be even count)
        deg = [0] * n
        for u, v in extra_edges:
            deg[u] += 1
            deg[v] += 1
        # Start from an odd-degree vertex, do Euler trail
        paths = []
        start = None
        for v in range(n):
            if deg[v] % 2 == 1:
                start = v
                break
        if start is None:
            # All even; pick any vertex with edges
            for v in range(n):
                if deg[v] > 0:
                    start = v
                    break
        if start is None:
            return paths
        # Hierholzer for Euler trail/circuit
        stack = [start]
        path = []
        while stack:
            u = stack[-1]
            # find an unused edge
            found = None
            for (v, idx) in adj2[u]:
                if not used[idx]:
                    found = (v, idx)
                    break
            if found is None:
                path.append(u)
                stack.pop()
            else:
                v, idx = found
                used[idx] = True
                # also mark the reverse as used
                used[idx ^ 1] = True
                stack.append(v)
        path.reverse()
        paths.append(path)
        return paths

    def solve_one(n, p, seed):
        edges = instance(n, p, seed)
        adj = build_adj(edges, n)
        odds = odd_vertices(adj, n)
        k = len(odds)
        if k == 0:
            # All even: single Euler circuit
            return extract_paths(adj, n, [])
        # Compute all-pairs distances from each odd vertex
        dists = all_pairs_dists(adj, n, odds)
        # Build cost matrix
        INF = 10**9
        costs = [[INF] * k for _ in range(k)]
        for i in range(k):
            for j in range(k):
                if i == j:
                    continue
                d = dists[odds[i]][0][odds[j]]
                if d >= 0:
                    costs[i][j] = d
                else:
                    costs[i][j] = INF
        # Min cost perfect matching
        pairs = min_cost_matching(costs, k)
        # Build extra edges: for each pair, add shortest path
        extra_edges = []
        for (i, j) in pairs:
            u, v = odds[i], odds[j]
            # Reconstruct shortest path from u to v
            dist, prev = dists[u]
            path = []
            cur = v
            while cur != -1:
                path.append(cur)
                cur = prev[cur]
            path.reverse()
            # path is u -> ... -> v
            for a, b in zip(path, path[1:]):
                extra_edges.append((a, b))
        paths = extract_paths(adj, n, extra_edges)
        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > DEADLINE:
            # Fallback: trivial edge-as-path cover
            results.append([[u, v] for u, v in instance(*spec)])
            continue
        paths = solve_one(*spec)
        results.append(paths)
    return results


if __name__ == "__main__":
    r = construct()
    for i, lst in enumerate(r):
        print(i, len(lst))
