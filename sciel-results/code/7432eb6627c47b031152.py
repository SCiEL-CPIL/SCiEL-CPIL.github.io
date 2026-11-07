import time
import sys
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 82  # 82s safety margin (90s kill)

    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)

    # Precompute all-pairs shortest paths on the full graph (for ordering hints)
    all_pairs_dist = [[n+1]*n for _ in range(n)]
    for i in range(n):
        all_pairs_dist[i][i] = 0
        for j in adj[i]:
            all_pairs_dist[i][j] = 1
    for k in range(n):
        for i in range(n):
            if all_pairs_dist[i][k] > n:
                continue
            for j in range(n):
                if all_pairs_dist[k][j] > n:
                    continue
                nd = all_pairs_dist[i][k] + all_pairs_dist[k][j]
                if nd < all_pairs_dist[i][j]:
                    all_pairs_dist[i][j] = nd

    # ---------- Max-flow oracle for vertex-disjoint paths ----------
    def flow_feasible(branches):
        """Check if vertex-disjoint interior paths exist for all pairs in branches.
        Returns True if feasible (necessary condition via max flow)."""
        if len(branches) < 2:
            return True
        b_set = set(branches)
        k = len(branches)
        required_flow = k * (k - 1) // 2

        # Build split graph for max flow
        S = 2 * n
        T = 2 * n + 1
        total_nodes = 2 * n + 2

        flow_adj = [[] for _ in range(total_nodes)]

        def add_flow_edge(u, v, cap):
            flow_adj[u].append([v, cap, len(flow_adj[v])])
            flow_adj[v].append([u, 0, len(flow_adj[u]) - 1])

        for v in range(n):
            if v in b_set:
                add_flow_edge(2 * v, 2 * v + 1, 10**9)
            else:
                add_flow_edge(2 * v, 2 * v + 1, 1)

        for u in range(n):
            for v in adj[u]:
                if u < v:
                    add_flow_edge(2 * u + 1, 2 * v, 10**9)
                    add_flow_edge(2 * v + 1, 2 * u, 10**9)

        for b in branches:
            add_flow_edge(S, 2 * b, 10**9)
            add_flow_edge(2 * b + 1, T, 10**9)

        # Edmonds-Karp (BFS-based max flow)
        def bfs_max_flow():
            total_flow = 0
            while True:
                parent = [-1] * total_nodes
                parent_edge = [None] * total_nodes
                queue = deque([S])
                parent[S] = S
                found = False
                while queue and not found:
                    cur = queue.popleft()
                    for i, (to, cap, rev) in enumerate(flow_adj[cur]):
                        if cap > 0 and parent[to] == -1:
                            parent[to] = cur
                            parent_edge[to] = (i, rev)
                            if to == T:
                                found = True
                                break
                            queue.append(to)
                if not found:
                    break
                bottleneck = 10**18
                node = T
                while node != S:
                    i, rev = parent_edge[node]
                    bottleneck = min(bottleneck, flow_adj[parent[node]][i][1])
                    node = parent[node]
                node = T
                while node != S:
                    i, rev = parent_edge[node]
                    flow_adj[parent[node]][i][1] -= bottleneck
                    flow_adj[node][rev][1] += bottleneck
                    node = parent[node]
                total_flow += bottleneck
            return total_flow

        maxflow = bfs_max_flow()
        return maxflow >= required_flow

    # ---------- Path finding with backtracking ----------
    def find_paths(branches, max_backtrack=5000):
        """Given a feasible branch set, find actual paths using BFS with
        careful ordering and limited backtracking. Returns dict of (u,v) -> path, or None."""
        b_set = set(branches)
        pairs = list(combinations(sorted(branches), 2))
        
        # Sort pairs: direct edges first, then by shortest path length
        pairs.sort(key=lambda uv: (0 if uv[1] in adj[uv[0]] else all_pairs_dist[uv[0]][uv[1]]))
        
        paths = {}
        used_interior = set()

        def solve(idx, paths_so_far, used_so_far):
            if idx == len(pairs):
                return True
            if time.time() > deadline:
                return False
            u, v = pairs[idx]
            if v in adj[u]:
                paths_so_far[(u, v)] = [u, v]
                return solve(idx + 1, paths_so_far, used_so_far)
            
            forbidden = b_set - {u, v}
            forbidden = forbidden | used_so_far
            visited = {u}
            parent = {u: None}
            queue = deque([u])
            found = False
            while queue:
                if time.time() > deadline:
                    return False
                cur = queue.popleft()
                if cur == v:
                    path = []
                    node = v
                    while node is not None:
                        path.append(node)
                        node = parent[node]
                    path.reverse()
                    paths_so_far[(u, v)] = path
                    new_used = used_so_far | set(path[1:-1])
                    if solve(idx + 1, paths_so_far, new_used):
                        return True
                    del paths_so_far[(u, v)]
                    return False
                for nb in sorted(adj[cur], key=lambda x: all_pairs_dist[x][v]):
                    if nb not in visited and nb not in forbidden:
                        visited.add(nb)
                        parent[nb] = cur
                        queue.append(nb)
            return False

        # Try with backtracking (limited)
        if solve(0, paths, used_interior):
            return paths
        
        # Fallback: greedy without backtracking
        paths = {}
        used_interior = set()
        for u, v in pairs:
            if v in adj[u]:
                paths[(u, v)] = [u, v]
                continue
            forbidden = b_set - {u, v}
            forbidden |= used_interior
            visited = {u}
            parent = {u: None}
            queue = deque([u])
            found = False
            while queue:
                if time.time() > deadline:
                    return None
                cur = queue.popleft()
                if cur == v:
                    path = []
                    node = v
                    while node is not None:
                        path.append(node)
                        node = parent[node]
                    path.reverse()
                    paths[(u, v)] = path
                    for mid in path[1:-1]:
                        used_interior.add(mid)
                    found = True
                    break
                for nb in sorted(adj[cur], key=lambda x: all_pairs_dist[x][v]):
                    if nb not in visited and nb not in forbidden:
                        visited.add(nb)
                        parent[nb] = cur
                        queue.append(nb)
            if not found:
                return None
        return paths

    # ---------- Find maximum clique (Bron-Kerbosch with pivot) ----------
    best_clique = []

    def bronk(R, P_set, X):
        nonlocal best_clique
        if time.time() > deadline:
            return
        if not P_set and not X:
            if len(R) > len(best_clique):
                best_clique = list(R)
            return
        candidates = list(P_set | X)
        if not candidates:
            return
        pivot = max(candidates, key=lambda u: len(adj[u] & P_set))
        for v in list(P_set - adj[pivot]):
            bronk(R | {v}, P_set & adj[v], X & adj[v])
            P_set = P_set - {v}
            X = X | {v}
            if time.time() > deadline:
                return

    bronk(set(), set(range(n)), set())

    # ---------- Local search over branch sets ----------
    current_branches = list(best_clique)
    current_paths = find_paths(current_branches)
    if current_paths is None:
        current_paths = {}
        for u, v in combinations(sorted(current_branches), 2):
            current_paths[(u, v)] = [u, v]

    best_size = len(current_branches)
    best_branches = list(current_branches)

    # Try to add vertices
    remaining = [v for v in range(n) if v not in set(current_branches)]
    improved = True
    while improved:
        improved = False
        for v in remaining[:]:
            if time.time() > deadline:
                break
            test_branches = current_branches + [v]
            if flow_feasible(test_branches):
                test_paths = find_paths(test_branches)
                if test_paths is not None:
                    current_branches = test_branches
                    current_paths = test_paths
                    remaining.remove(v)
                    improved = True
                    best_size = len(current_branches)
                    best_branches = list(current_branches)

    # Try swap moves
    if time.time() < deadline - 10:
        for _ in range(3):
            swap_improved = False
            for i in range(len(current_branches)):
                if time.time() > deadline:
                    break
                for v in remaining:
                    if time.time() > deadline:
                        break
                    test_branches = current_branches[:i] + current_branches[i+1:] + [v]
                    if flow_feasible(test_branches):
                        test_paths = find_paths(test_branches)
                        if test_paths is not None:
                            current_branches = test_branches
                            current_paths = test_paths
                            remaining.remove(v)
                            remaining.append(current_branches[i])
                            swap_improved = True
                            best_size = len(current_branches)
                            best_branches = list(current_branches)
                            break
                if swap_improved:
                    break
            if not swap_improved:
                break

    # Use the best found
    if len(best_branches) > len(current_branches):
        current_branches = best_branches
        current_paths = find_paths(current_branches)

    # Build result
    if current_paths is None:
        current_branches = list(best_clique)
        current_paths = {}
        for u, v in combinations(sorted(current_branches), 2):
            current_paths[(u, v)] = [u, v]

    branches = [int(b) for b in sorted(current_branches)]
    paths_list = []
    for u, v in combinations(sorted(current_branches), 2):
        p = current_paths.get((u, v), [u, v])
        paths_list.append([int(x) for x in p])

    return {'branches': branches, 'paths': paths_list}
