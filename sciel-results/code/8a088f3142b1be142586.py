"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
from itertools import combinations
from collections import deque
import random

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = _time.time()
    deadline = start_time + 82
    n = P['n']
    edges = P['edges']
    adj = [set() for _ in range(n)]
    adj_list = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        adj_list[u].append(v)
        adj_list[v].append(u)

    def time_check():
        return _time.time() > deadline

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

    def find_paths_for_branches(branches):
        branch_set = set(branches)
        pairs = list(combinations(branches, 2))
        pair_info = {}
        non_direct_pairs = []
        for u, v in pairs:
            if u in adj[v]:
                pair_info[(u, v)] = ('direct', None)
            else:
                common = set()
                for w in adj_list[u]:
                    if w not in branch_set and w in adj[v]:
                        common.add(w)
                pair_info[(u, v)] = ('need', common)
                non_direct_pairs.append((u, v))
        for pair in non_direct_pairs:
            if len(pair_info[pair][1]) == 0:
                return None

        if non_direct_pairs:
            pair_to_idx = {p: i for i, p in enumerate(non_direct_pairs)}
            feasible_list = [(pair_to_idx[p], list(pair_info[p][1])) for p in non_direct_pairs]
            sorted_pairs = sorted(feasible_list, key=lambda x: len(x[1]))
            assignment = {}
            used = set()

            def bt(idx):
                if time_check():
                    return False
                if idx == len(sorted_pairs):
                    return True
                pair_idx, candidates = sorted_pairs[idx]
                for w in candidates:
                    if w not in used:
                        used.add(w)
                        assignment[pair_idx] = w
                        if bt(idx + 1):
                            return True
                        used.remove(w)
                        del assignment[pair_idx]
                return False

            if not bt(0):
                # Fallback to longer paths
                used_internals = set()
                assignment2 = {}
                sorted_non_direct = sorted(non_direct_pairs, key=lambda p: len(pair_info[p][1]))

                def backtrack_paths(idx):
                    if time_check():
                        return False
                    if idx == len(sorted_non_direct):
                        return True
                    u, v = sorted_non_direct[idx]
                    blocked = branch_set | used_internals
                    blocked.discard(u)
                    blocked.discard(v)
                    path = bfs_path(u, v, blocked)
                    if path is None:
                        return False
                    internal = path[1:-1]
                    if len(internal) != len(set(internal)):
                        return False
                    for w in internal:
                        if w in branch_set:
                            return False
                    for w in internal:
                        used_internals.add(w)
                    assignment2[(u, v)] = internal
                    if backtrack_paths(idx + 1):
                        return True
                    for w in internal:
                        used_internals.discard(w)
                    del assignment2[(u, v)]
                    return False

                if not backtrack_paths(0):
                    return None
                # Convert to assignment format
                for pair in non_direct_pairs:
                    if pair in assignment2:
                        assignment[pair_to_idx[pair]] = assignment2[pair][0] if len(assignment2[pair]) == 1 else None
                        # If longer path, we need to store the full path
                # Actually, let's just return the paths directly from assignment2
                result_paths = {}
                for pair in pairs:
                    u, v = pair
                    if pair_info[pair][0] == 'direct':
                        result_paths[pair] = [u, v]
                    else:
                        if pair in assignment2:
                            result_paths[pair] = [u] + assignment2[pair] + [v]
                        else:
                            return None
                return result_paths
            else:
                result_paths = {}
                for pair in pairs:
                    u, v = pair
                    if pair_info[pair][0] == 'direct':
                        result_paths[pair] = [u, v]
                    else:
                        w = assignment[pair_to_idx[pair]]
                        result_paths[pair] = [u, w, v]
                return result_paths
        else:
            result_paths = {}
            for pair in pairs:
                u, v = pair
                result_paths[pair] = [u, v]
            return result_paths

    def max_clique():
        best = []

        def expand(clique, candidates):
            nonlocal best
            if time_check():
                return
            if not candidates:
                if len(clique) > len(best):
                    best = list(clique)
                return

            colors = {}
            for v in candidates:
                used_colors = set()
                for u in candidates:
                    if u in adj[v] and u in colors:
                        used_colors.add(colors[u])
                c = 0
                while c in used_colors:
                    c += 1
                colors[v] = c

            max_color = max(colors.values()) + 1
            if len(clique) + max_color <= len(best):
                return

            ordered = sorted(candidates, key=lambda v: -colors[v])
            candidates_set = set(candidates)

            for v in ordered:
                if v not in candidates_set:
                    continue
                new_clique = clique + [v]
                new_candidates = [u for u in candidates_set if u in adj[v] and u != v]
                expand(new_clique, new_candidates)
                candidates_set.discard(v)
                remaining_max = 0
                for u in candidates_set:
                    if all(u in adj[c] for c in clique):
                        remaining_max = max(remaining_max, colors.get(u, 0) + 1)
                if len(clique) + remaining_max <= len(best):
                    break

        expand([], list(range(n)))
        return best

    clique = max_clique()

    best_branches = None
    best_paths = None

    if len(clique) >= 2:
        paths = find_paths_for_branches(clique)
        if paths is not None:
            best_branches = list(clique)
            best_paths = paths

    if best_branches:
        current = list(best_branches)
        improved = True
        while improved and not time_check():
            improved = False
            for candidate in range(n):
                if candidate in current:
                    continue
                new_set = current + [candidate]
                paths = find_paths_for_branches(new_set)
                if paths is not None:
                    current = new_set
                    best_branches = current
                    best_paths = paths
                    improved = True
                    break

    deg = [len(adj[i]) for i in range(n)]
    top = sorted(range(n), key=lambda x: -deg[x])[:12]
    for size in range(min(10, len(top)), 1, -1):
        if time_check():
            break
        found = False
        for combo in combinations(top, size):
            if time_check():
                break
            paths = find_paths_for_branches(list(combo))
            if paths is not None and (best_branches is None or size > len(best_branches)):
                best_branches = list(combo)
                best_paths = paths
                found = True
                break
        if found:
            break

    random.seed(42)
    for _ in range(300):
        if time_check():
            break
        size = random.randint(4, 14)
        combo = random.sample(range(n), size)
        paths = find_paths_for_branches(combo)
        if paths is not None and (best_branches is None or size > len(best_branches)):
            best_branches = combo
            best_paths = paths

    if best_branches is None or len(best_branches) < 2:
        u, v = edges[0]
        best_branches = [u, v]
        best_paths = {(u, v): [u, v]}

    branches = [int(b) for b in best_branches]
    paths_list = []
    for (u, v) in combinations(branches, 2):
        path = best_paths.get((u, v)) or best_paths.get((v, u))
        if path is None:
            path = [u, v]
        paths_list.append([int(x) for x in path])

    return {'branches': branches, 'paths': paths_list}
# EVOLVE-BLOCK-END
