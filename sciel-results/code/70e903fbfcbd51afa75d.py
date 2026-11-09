import time
import random
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

DEADLINE = time.time() + 55.0

def construct():
    t0 = time.time()
    n = P['n']
    edges = P['edges']

    adj = [set() for _ in range(n)]
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)

    def time_up():
        return time.time() - t0 > DEADLINE - 3.0

    def fallback():
        u, v = edges[0]
        return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}

    best = fallback()
    best_size = 2

    def find_disjoint_paths(branches, max_path_len=4):
        """
        Find pairwise internally disjoint paths for all pairs of branch vertices.
        Uses DFS with path length limit. Returns list of paths or None.
        """
        bs = set(branches)
        sorted_bs = sorted(branches)
        pairs = list(combinations(sorted_bs, 2))
        m = len(pairs)
        if m == 0:
            return []

        # Track used internal vertices
        used_internal = set()

        paths_found = []

        for u, v in pairs:
            if time_up():
                return None

            # Find a path from u to v avoiding used_internal and other branches
            # BFS first for shortest path
            found_path = None

            # BFS
            parent = {u: None}
            q = deque([u])
            while q:
                curr = q.popleft()
                if curr == v:
                    break
                if len(parent) > max_path_len:
                    break
                for nb in adj[curr]:
                    if nb not in parent and nb not in bs and nb not in used_internal:
                        parent[nb] = curr
                        if nb == v:
                            break
                        q.append(nb)

            if v in parent:
                # Reconstruct path
                path = [v]
                node = v
                while node != u:
                    node = parent[node]
                    path.append(node)
                path.reverse()
                found_path = path
            else:
                # Try DFS with limited depth
                def dfs_path(curr, path, visited):
                    nonlocal found_path
                    if found_path is not None:
                        return
                    if curr == v:
                        found_path = list(path)
                        return
                    if len(path) > max_path_len:
                        return
                    for nb in adj[curr]:
                        if nb not in visited and nb not in bs and nb not in used_internal:
                            visited.add(nb)
                            path.append(nb)
                            dfs_path(nb, path, visited)
                            path.pop()
                            visited.discard(nb)
                            if found_path is not None:
                                return

                dfs_path(u, [u], {u})

            if found_path is None:
                return None

            # Verify path
            if found_path[0] != u or found_path[-1] != v:
                return None
            internal = set(found_path[1:-1])
            if internal & bs or internal & used_internal:
                return None
            for i in range(len(found_path) - 1):
                if found_path[i+1] not in adj[found_path[i]]:
                    return None

            used_internal |= internal
            paths_found.append(found_path)

        return paths_found

    # Phase 1: Find good initial solution using Bron-Kerbosch
    def bron_kerbosch(R, P_set, X, result, time_limit):
        if time.time() > time_limit:
            return
        if not P_set and not X:
            if len(R) > best_size:
                result.append(list(R))
            return
        candidates = P_set | X
        if candidates:
            pivot = max(candidates, key=lambda v: len(P_set & adj[v]))
            candidates_to_try = P_set - adj[pivot]
        else:
            candidates_to_try = P_set
        for v in list(candidates_to_try):
            bron_kerbosch(R | {v}, P_set & adj[v], X & adj[v], result, time_limit)
            P_set.discard(v)
            X.add(v)

    cliques = []
    initial_P = set(range(n))
    bron_kerbosch(set(), initial_P, set(), cliques, time.time() + 8.0)
    cliques.sort(key=len, reverse=True)

    for clique in cliques:
        if time_up():
            break
        if len(clique) <= best_size:
            break
        paths = find_disjoint_paths(clique)
        if paths is not None:
            best = {'branches': [int(x) for x in sorted(clique)], 'paths': [[int(x) for x in p] for p in paths]}
            best_size = len(clique)

    # Phase 2: Local search with add, remove, and swap moves
    current_branches = set(best['branches'])
    no_improve_count = 0
    max_no_improve = 50

    while not time_up() and no_improve_count < max_no_improve:
        improved = False

        # Try adding a vertex
        candidates_add = [v for v in range(n) if v not in current_branches]
        candidates_add.sort(key=lambda v: sum(1 for b in current_branches if v in adj[b]), reverse=True)

        for v in candidates_add[:10]:
            if time_up():
                break
            new_set = current_branches | {v}
            paths = find_disjoint_paths(new_set)
            if paths is not None:
                best = {'branches': [int(x) for x in sorted(new_set)], 'paths': [[int(x) for x in p] for p in paths]}
                best_size = len(new_set)
                current_branches = new_set
                improved = True
                no_improve_count = 0
                break
        if improved:
            continue

        # Try swapping a branch with a non-branch
        non_branches = [v for v in range(n) if v not in current_branches]
        non_branches.sort(key=lambda v: sum(1 for b in current_branches if v in adj[b]), reverse=True)

        for v_out in list(current_branches):
            if time_up():
                break
            for v_in in non_branches[:8]:
                if v_in == v_out:
                    continue
                new_set = (current_branches - {v_out}) | {v_in}
                paths = find_disjoint_paths(new_set)
                if paths is not None:
                    # Accept swap (maintains same size but may enable future additions)
                    current_branches = new_set
                    best = {'branches': [int(x) for x in sorted(new_set)], 'paths': [[int(x) for x in p] for p in paths]}
                    improved = True
                    no_improve_count = 0
                    break
            if improved:
                break
        if improved:
            continue

        # Try removing a vertex (to potentially re-add a different one)
        if len(current_branches) > 2:
            for v_out in list(current_branches):
                if time_up():
                    break
                new_set = current_branches - {v_out}
                paths = find_disjoint_paths(new_set)
                if paths is not None:
                    # Only accept removal if we can immediately re-add something better
                    current_branches = new_set
                    improved = True
                    no_improve_count = 0
                    break
        if improved:
            continue

        no_improve_count += 1

    # Phase 3: Random restarts with local search
    if not time_up():
        deg_sorted = sorted(range(n), key=lambda x: len(adj[x]), reverse=True)
        for restart in range(30):
            if time_up():
                break
            # Random subset of size best_size + 1 or best_size + 2
            target_size = min(best_size + random.randint(1, 2), n)
            subset = set(random.sample(deg_sorted[:min(18, n)], min(target_size, 18)))
            paths = find_disjoint_paths(subset)
            if paths is not None:
                # Do local search from here
                current_branches = subset
                for _ in range(20):
                    if time_up():
                        break
                    local_improved = False
                    cands = [v for v in range(n) if v not in current_branches]
                    cands.sort(key=lambda v: sum(1 for b in current_branches if v in adj[b]), reverse=True)
                    for v in cands[:6]:
                        if time_up():
                            break
                        new_set = current_branches | {v}
                        p = find_disjoint_paths(new_set)
                        if p is not None:
                            current_branches = new_set
                            best = {'branches': [int(x) for x in sorted(new_set)], 'paths': [[int(x) for x in p] for p in p]}
                            best_size = len(new_set)
                            local_improved = True
                            break
                    if not local_improved:
                        break

    return best
