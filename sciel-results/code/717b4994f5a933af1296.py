"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
from collections import deque
import numpy as np

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit

    n = P['n']
    edges = P['edges']

    adj_list = [[] for _ in range(n)]
    adj_set = [set() for _ in range(n)]
    for u, v in edges:
        adj_list[u].append(v)
        adj_list[v].append(u)
        adj_set[u].add(v)
        adj_set[v].add(u)

    # NumPy adjacency matrix + common-neighbor bitmasks (fast vectorized checks)
    adj_mat = np.zeros((n, n), dtype=np.int8)
    for u, v in edges:
        adj_mat[u, v] = 1
        adj_mat[v, u] = 1
    # common[u,v] = bitmask of vertices adjacent to both u and v
    common_mask = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        for j in range(i + 1, n):
            cm = adj_mat[i] & adj_mat[j]
            mask = 0
            for k in range(n):
                if cm[k]:
                    mask |= (1 << k)
            common_mask[i, j] = mask
            common_mask[j, i] = mask

    degrees = [len(adj_list[i]) for i in range(n)]

    def time_check():
        return time.time() > deadline

    def bfs_path(src, dst, blocked):
        if src == dst:
            return [src]
        parent = {src: None}
        queue = deque([src])
        while queue:
            node = queue.popleft()
            if node == dst:
                path = []
                cur = dst
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                path.reverse()
                return path
            for neighbor in adj_list[node]:
                if neighbor not in parent and neighbor not in blocked:
                    parent[neighbor] = node
                    queue.append(neighbor)
        return None

    def hopcroft_karp(feasible_pairs):
        """feasible_pairs: list of (pair, [candidate internal vertices]).
        Returns dict pair -> internal_vertex, or None if no perfect matching."""
        if not feasible_pairs:
            return {}
        all_internals = set()
        for _, candidates in feasible_pairs:
            all_internals.update(candidates)
        left_nodes = [p[0] for p in feasible_pairs]
        right_nodes = list(all_internals)
        left_to_idx = {node: i for i, node in enumerate(left_nodes)}
        right_to_idx = {node: i for i, node in enumerate(right_nodes)}
        adj = [[] for _ in range(len(left_nodes))]
        for pair_idx, candidates in feasible_pairs:
            li = left_to_idx[pair_idx]
            for w in candidates:
                if w in right_to_idx:
                    adj[li].append(right_to_idx[w])
        pair_u = [-1] * len(left_nodes)
        pair_v = [-1] * len(right_nodes)
        dist = [0] * len(left_nodes)

        def bfs():
            q = deque()
            found = False
            for u in range(len(left_nodes)):
                if pair_u[u] == -1:
                    dist[u] = 0
                    q.append(u)
                else:
                    dist[u] = -1
            while q:
                u = q.popleft()
                for v in adj[u]:
                    pu = pair_v[v]
                    if pu == -1:
                        found = True
                    elif dist[pu] == -1:
                        dist[pu] = dist[u] + 1
                        q.append(pu)
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

        matching_size = 0
        while bfs():
            for u in range(len(left_nodes)):
                if pair_u[u] == -1:
                    if dfs(u):
                        matching_size += 1
        if matching_size != len(left_nodes):
            return None
        result = {}
        for u in range(len(left_nodes)):
            if pair_u[u] != -1:
                result[left_nodes[u]] = right_nodes[pair_u[u]]
        return result

    def find_paths_for_branches(branches):
        """Returns dict pair -> path list, or None if infeasible."""
        branch_set = set(branches)
        pairs = list(itertools.combinations(branches, 2))

        pair_info = {}
        non_direct_pairs = []
        for u, v in pairs:
            if u in adj_set[v]:
                pair_info[(u, v)] = ('direct', None)
            else:
                # common-neighbor bitmask (fast NumPy-backed) minus branches
                cm = common_mask[u, v]
                common = set()
                for w in range(n):
                    if (cm >> w) & 1 and w not in branch_set:
                        common.add(w)
                pair_info[(u, v)] = ('need', common)
                non_direct_pairs.append((u, v))

        for pair in non_direct_pairs:
            if len(pair_info[pair][1]) == 0:
                return None

        assignment = {}
        if non_direct_pairs:
            # Try fast matching first (1-hop internal vertices)
            feasible_list = [(p, list(pair_info[p][1])) for p in non_direct_pairs]
            matching = hopcroft_karp(feasible_list)
            if matching is not None:
                for pair, w in matching.items():
                    assignment[pair] = [w]
            else:
                # Backtracking with BFS for longer paths, sorted by tightness
                used_internals = set()
                sorted_non_direct = sorted(non_direct_pairs,
                                           key=lambda p: len(pair_info[p][1]))

                def backtrack_paths(idx):
                    if time_check():
                        return False
                    if idx == len(sorted_non_direct):
                        return True
                    u, v = sorted_non_direct[idx]
                    blocked = (branch_set - {u, v}) | used_internals
                    path = bfs_path(u, v, blocked)
                    if path is None:
                        return False
                    internal = path[1:-1]
                    if len(internal) != len(set(internal)):
                        return False
                    for w in internal:
                        used_internals.add(w)
                    assignment[(u, v)] = internal
                    if backtrack_paths(idx + 1):
                        return True
                    for w in internal:
                        used_internals.discard(w)
                    del assignment[(u, v)]
                    return False

                if not backtrack_paths(0):
                    return None

        result_paths = {}
        for pair in pairs:
            u, v = pair
            if pair_info[pair][0] == 'direct':
                result_paths[pair] = [u, v]
            else:
                internal = assignment[pair]
                result_paths[pair] = [u] + list(internal) + [v]
        return result_paths

    # --- Step 1: Max clique via Bron-Kerbosch for a strong starting set ---
    max_clique = []

    def bron_kerbosch(R, P, X):
        nonlocal max_clique
        if time.time() > deadline - 20.0:
            return
        if len(P) == 0 and len(X) == 0:
            if len(R) > len(max_clique):
                max_clique = list(R)
            return
        if len(P) == 0:
            if len(R) > len(max_clique):
                max_clique = list(R)
            return
        union_p_x = P.union(X)
        if not union_p_x:
            return
        best_pivot = None
        max_uncovered = -1
        for u in union_p_x:
            uncovered = len(P.difference(adj_set[u]))
            if uncovered > max_uncovered:
                max_uncovered = uncovered
                best_pivot = u
        if best_pivot is None:
            return
        for v in list(P.difference(adj_set[best_pivot])):
            bron_kerbosch(R.union([v]), P.intersection(adj_set[v]), X.intersection(adj_set[v]))
            P.remove(v)
            X.add(v)

    all_vertices = set(range(n))
    bron_kerbosch(set(), all_vertices, set())
    if not max_clique:
        max_clique = [0, 1]

    # Greedy initial build: start from clique, try to add high-degree vertices
    current_B = set(max_clique)
    best_B = set(current_B)
    best_paths = find_paths_for_branches(list(current_B))
    best_score = len(best_B)

    # --- Step 2: Local search (add + swap) ---
    vertices_by_degree = sorted(range(n), key=lambda x: degrees[x], reverse=True)

    improved = True
    while improved and time.time() < deadline - 8.0:
        improved = False
        candidates = [v for v in vertices_by_degree if v not in current_B]
        added_any = False
        for v in candidates:
            if time.time() > deadline - 8.0:
                break
            new_B = current_B | {v}
            paths = find_paths_for_branches(list(new_B))
            if paths is not None:
                current_B = new_B
                best_B = set(current_B)
                best_paths = paths
                best_score = len(current_B)
                improved = True
                added_any = True
                break

        if not added_any and len(current_B) > 2:
            b_sorted = sorted(current_B, key=lambda x: degrees[x])
            candidates_not_in_b = [v for v in vertices_by_degree if v not in current_B]
            for remove_v in b_sorted[:3]:
                for add_v in candidates_not_in_b[:6]:
                    if time.time() > deadline - 8.0:
                        break
                    swap_B = (current_B - {remove_v}) | {add_v}
                    paths = find_paths_for_branches(list(swap_B))
                    if paths is not None:
                        current_B = swap_B
                        best_paths = paths
                        improved = True
                        break
                if improved:
                    break

    # --- Step 3: Final verification / fallback ---
    if best_paths is None:
        best_paths = find_paths_for_branches(list(best_B))
    if best_paths is None:
        # Fallback to clique with direct edges
        best_B = set(max_clique)
        best_paths = {}
        for u, v in itertools.combinations(list(best_B), 2):
            best_paths[(u, v)] = [u, v]

    branches = sorted([int(v) for v in best_B])
    paths_list = []
    for u, v in itertools.combinations(branches, 2):
        key = (u, v)
        if key not in best_paths:
            key = (v, u)
        p = best_paths.get(key, [u, v])
        paths_list.append([int(x) for x in p])

    return {'branches': branches, 'paths': paths_list}

# EVOLVE-BLOCK-END
