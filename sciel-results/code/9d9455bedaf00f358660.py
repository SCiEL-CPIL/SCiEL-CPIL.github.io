import time
import sys
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 82

    n = P['n']
    adj_list = [[] for _ in range(n)]
    adj_set = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj_list[u].append(v)
        adj_list[v].append(u)
        adj_set[u].add(v)
        adj_set[v].add(u)

    def time_check():
        return time.time() > deadline

    def hopcroft_karp(n_left, n_right, left_adj):
        """Bipartite matching. Returns (matchL, max_matching_size)."""
        INF = float('inf')
        matchL = [-1] * n_left
        matchR = [-1] * n_right
        dist = [0] * n_left

        def bfs():
            q = deque()
            for u in range(n_left):
                if matchL[u] == -1:
                    dist[u] = 0
                    q.append(u)
                else:
                    dist[u] = INF
            found = False
            while q:
                u = q.popleft()
                for v in left_adj[u]:
                    mu = matchR[v]
                    if mu == -1:
                        found = True
                    elif dist[mu] == INF:
                        dist[mu] = dist[u] + 1
                        q.append(mu)
            return found

        def dfs(u):
            for v in left_adj[u]:
                mu = matchR[v]
                if mu == -1 or (dist[mu] == dist[u] + 1 and dfs(mu)):
                    matchL[u] = v
                    matchR[v] = u
                    return True
            dist[u] = INF
            return False

        flow = 0
        while bfs():
            for u in range(n_left):
                if matchL[u] == -1:
                    if dfs(u):
                        flow += 1
        return matchL, flow

    def find_paths_for_branches(branches):
        """Try to find disjoint internal paths for all pairs of branches.
        Returns dict of (u,v) -> path, or None if infeasible."""
        branch_set = set(branches)
        pairs = list(combinations(branches, 2))

        pair_info = {}
        non_direct_pairs = []
        for u, v in pairs:
            if u in adj_set[v]:
                pair_info[(u, v)] = ('direct', None)
            else:
                common = set()
                for w in adj_list[u]:
                    if w not in branch_set and w in adj_set[v]:
                        common.add(w)
                pair_info[(u, v)] = ('need', common)
                non_direct_pairs.append((u, v))

        for pair in non_direct_pairs:
            if len(pair_info[pair][1]) == 0:
                return None

        if non_direct_pairs:
            # Use Hopcroft-Karp for bipartite matching
            pair_indices = {pair: i for i, pair in enumerate(non_direct_pairs)}
            all_internal = set()
            for u, v in non_direct_pairs:
                all_internal.update(pair_info[(u, v)][1])
            internal_list = list(all_internal)
            vertex_indices = {v: i for i, v in enumerate(internal_list)}

            left_adj = [[] for _ in range(len(non_direct_pairs))]
            for u, v in non_direct_pairs:
                pi = pair_indices[(u, v)]
                for w in pair_info[(u, v)][1]:
                    left_adj[pi].append(vertex_indices[w])

            matchL, flow = hopcroft_karp(len(non_direct_pairs), len(internal_list), left_adj)

            if flow < len(non_direct_pairs):
                # Matching failed; try BFS-based fallback
                return _bfs_disjoint_paths(branches, pairs, adj_list, adj_set, n, branch_set)

            assignment = {}
            for i, (u, v) in enumerate(non_direct_pairs):
                if matchL[i] != -1:
                    w = internal_list[matchL[i]]
                    assignment[(u, v)] = [w]
        else:
            assignment = {}

        result_paths = {}
        for pair in pairs:
            u, v = pair
            if pair_info[pair][0] == 'direct':
                result_paths[pair] = [u, v]
            else:
                internal = assignment[pair]
                result_paths[pair] = [u] + internal + [v]

        return result_paths

    def _bfs_disjoint_paths(branches, pairs, adj_list, adj_set, n, branch_set):
        """Fallback: greedy BFS for vertex-disjoint paths."""
        used_internals = set()
        pair_paths = {}

        def find_shortest_path(u, v, blocked):
            if u == v:
                return [u, v]
            parent = {u: -1}
            q = deque([u])
            while q:
                node = q.popleft()
                if node == v:
                    break
                for nb in adj_list[node]:
                    if nb not in parent and nb not in blocked:
                        parent[nb] = node
                        if nb == v:
                            q = deque()
                            break
                        q.append(nb)
            if v not in parent:
                return None
            path = []
            cur = v
            while cur != -1:
                path.append(cur)
                cur = parent[cur]
            path.reverse()
            return path

        def pair_score(pair):
            u, v = pair
            common = 0
            for w in adj_list[u]:
                if w not in branch_set and w in adj_set[v]:
                    common += 1
            return common

        sorted_pairs = sorted(pairs, key=pair_score)

        for u, v in sorted_pairs:
            if u in adj_set[v]:
                pair_paths[(u, v)] = [u, v]
                continue
            blocked = branch_set | used_internals
            blocked.discard(u)
            blocked.discard(v)
            path = find_shortest_path(u, v, blocked)
            if path is None:
                return None
            internal = path[1:-1]
            for w in internal:
                if w in used_internals or w in branch_set:
                    return None
            for w in internal:
                used_internals.add(w)
            pair_paths[(u, v)] = path

        return pair_paths

    def verify_solution(branches, paths):
        """Verify validity."""
        branch_set = set(branches)
        if len(branch_set) != len(branches) or len(branches) < 2:
            return False
        pairs = set((min(u, v), max(u, v)) for u, v in combinations(branches, 2))
        used_internals = set()
        for (u, v), path in paths.items():
            key = (min(u, v), max(u, v))
            if key not in pairs or len(path) < 2:
                return False
            if path[0] not in branch_set or path[-1] not in branch_set:
                return False
            if set([path[0], path[-1]]) != set([u, v]):
                return False
            for i in range(len(path) - 1):
                if path[i + 1] not in adj_set[path[i]]:
                    return False
            for w in path[1:-1]:
                if w in branch_set or w in used_internals:
                    return False
                used_internals.add(w)
        if len(paths) != len(pairs):
            return False
        return True

    def solve():
        best_result = None
        best_k = 0

        degree = [len(adj_list[i]) for i in range(n)]
        sorted_vertices = sorted(range(n), key=lambda x: -degree[x])

        # Find max clique using Bron-Kerbosch with pivot
        def find_max_clique():
            best_clique = []
            def bronk(R, P, X):
                nonlocal best_clique
                if time_check():
                    return
                if not P and not X:
                    if len(R) > len(best_clique):
                        best_clique = list(R)
                    return
                if P:
                    pivot = max(list(P) + list(X), key=lambda v: len(adj_set[v] & P))
                else:
                    return
                for v in list(P - adj_set[pivot]):
                    bronk(R | {v}, P & adj_set[v], X & adj_set[v])
                    P.remove(v)
                    X.add(v)
            bronk(set(), set(range(n)), set())
            return best_clique

        clique = find_max_clique()
        if clique:
            branches = sorted(clique)
            paths = {}
            for u, v in combinations(branches, 2):
                if u in adj_set[v]:
                    paths[(u, v)] = [u, v]
            if verify_solution(branches, paths):
                best_result = {
                    'branches': [int(b) for b in branches],
                    'paths': [[int(x) for x in paths[p]] for p in sorted(paths.keys())]
                }
                best_k = len(branches)

        # Greedy extension
        current_branches = list(clique) if clique else []
        for i in range(n):
            if time_check():
                break
            candidate = sorted_vertices[i]
            if candidate in current_branches:
                continue
            new_branches = current_branches + [candidate]
            paths = find_paths_for_branches(new_branches)
            if paths is not None and verify_solution(new_branches, paths):
                current_branches = new_branches
                if len(current_branches) > best_k:
                    best_result = {
                        'branches': [int(b) for b in sorted(current_branches)],
                        'paths': [[int(x) for x in paths[p]] for p in sorted(paths.keys())]
                    }
                    best_k = len(current_branches)

        # Local search: 1-swap improvements
        for _ in range(30):
            if time_check():
                break
            improved = False
            current_set = set(current_branches)
            for i in range(n):
                if time_check():
                    break
                candidate = sorted_vertices[i]
                if candidate in current_set:
                    continue
                for j in range(len(current_branches)):
                    if time_check():
                        break
                    trial = current_branches[:j] + current_branches[j+1:] + [candidate]
                    paths = find_paths_for_branches(trial)
                    if paths is not None and verify_solution(trial, paths):
                        current_branches = trial
                        current_set = set(trial)
                        improved = True
                        if len(current_branches) > best_k:
                            best_result = {
                                'branches': [int(b) for b in sorted(current_branches)],
                                'paths': [[int(x) for x in paths[p]] for p in sorted(paths.keys())]
                            }
                            best_k = len(current_branches)
                        break
                if improved:
                    break
            if not improved:
                break

        if best_result is not None:
            return best_result

        for u, v in P['edges']:
            return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
        return {'branches': [0, 1], 'paths': [[0, 1]]}

    return solve()
