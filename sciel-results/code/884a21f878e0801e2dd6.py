"""Combined solver: greedy growth with DFS multi-path backtracking."""
import time
import random
from itertools import combinations
from collections import deque

# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    t_start = time.time()
    DEADLINE = 80.0

    n = P['n']
    edges = P['edges']
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((min(u, v), max(u, v)))

    def time_up():
        return time.time() - t_start > DEADLINE

    def find_paths_dfs(u, v, banned, limit=8):
        """Find up to 'limit' simple paths from u to v. Interior vertices must avoid banned."""
        paths = []
        if u in banned or v in banned:
            return paths
        if u == v:
            return [[u]]

        def dfs(cur, path, visited):
            if time_up():
                return
            if len(paths) >= limit:
                return
            if cur == v:
                paths.append(list(path))
                return
            for nb in sorted(adj[cur]):
                if nb in visited:
                    continue
                # Interior vertices must not be in banned
                if nb in banned and nb != v:
                    continue
                visited.add(nb)
                path.append(nb)
                dfs(nb, path, visited)
                path.pop()
                visited.remove(nb)

        dfs(u, [u], {u})
        paths.sort(key=len)
        return paths

    def try_add_vertex(v, branches):
        """Try to add vertex v to existing branch set. Returns list of paths (v to each branch) or None."""
        if time_up():
            return None

        targets = list(branches)
        banned = set(branches) | {v}

        # Sort targets: those with fewer available neighbors first (harder first)
        def target_key(t):
            avail = len(adj[t] - banned - {v})
            return avail
        targets.sort(key=target_key)

        solution_paths = []

        def search_paths(k, used_interiors):
            if time_up():
                return False
            if k == len(targets):
                return True

            target = targets[k]
            effective_banned = banned | used_interiors
            candidates = find_paths_dfs(v, target, effective_banned, limit=6)

            for path in candidates:
                new_used = set(used_interiors)
                for node in path[1:-1]:
                    new_used.add(node)

                solution_paths.append(path)
                if search_paths(k + 1, new_used):
                    return True
                solution_paths.pop()

            return False

        if search_paths(0, set()):
            return solution_paths
        return None

    def try_build_from_seed(branches, max_grow=30):
        """Try to grow the branch set greedily. Returns (branches, all_paths) or None."""
        if len(branches) < 2:
            return None
        banned = set(branches)
        all_paths = {}

        # Find paths for the initial pair
        p = find_paths_dfs(branches[0], branches[1], banned, limit=1)
        if not p:
            return None
        all_paths[(branches[0], branches[1])] = p[0]
        for node in p[0][1:-1]:
            banned.add(node)

        current_branches = list(branches)

        for _ in range(max_grow):
            if time_up():
                break
            added = False
            # Try vertices in order of degree (lower degree first - easier to route around)
            candidates = [w for w in range(n) if w not in current_branches]
            candidates.sort(key=lambda w: len(adj[w]))

            for w in candidates:
                if time_up():
                    break
                new_paths = try_add_vertex(w, current_branches)
                if new_paths is not None:
                    # Success: add w
                    current_branches.append(w)
                    for i, target in enumerate(current_branches[:-1]):
                        key = tuple(sorted((w, target)))
                        all_paths[key] = new_paths[i]
                        for node in new_paths[i][1:-1]:
                            banned.add(node)
                    added = True
                    break

            if not added:
                break

        return current_branches, all_paths

    def get_all_paths(branches, all_paths):
        """Return list of paths in consistent order for the output."""
        result = []
        for u, v in combinations(branches, 2):
            key = (u, v)
            if key in all_paths:
                result.append(list(all_paths[key]))
            else:
                key2 = (v, u)
                if key2 in all_paths:
                    result.append(list(all_paths[key2]))
                else:
                    return None
        return result

    # Strategy: try multiple seeds and grow
    best_result = None
    best_size = 0

    # Seed with edges (pairs of adjacent vertices)
    seed_edges = list(edges)
    random.shuffle(seed_edges)

    for u, v in seed_edges:
        if time_up():
            break
        res = try_build_from_seed([u, v])
        if res is not None:
            branches, paths_dict = res
            if len(branches) > best_size:
                best_size = len(branches)
                all_p = get_all_paths(branches, paths_dict)
                if all_p is not None:
                    best_result = (branches, all_p)

    # Also try seed with non-edges (pairs that are not directly connected)
    # This can sometimes find larger structures
    all_pairs = list(combinations(range(n), 2))
    random.shuffle(all_pairs)

    for u, v in all_pairs:
        if time_up():
            break
        if (min(u, v), max(u, v)) in edge_set:
            continue  # Already tried as edge
        res = try_build_from_seed([u, v])
        if res is not None:
            branches, paths_dict = res
            if len(branches) > best_size:
                best_size = len(branches)
                all_p = get_all_paths(branches, paths_dict)
                if all_p is not None:
                    best_result = (branches, all_p)

    # Random restarts: pick random 2-vertex seeds
    for _ in range(50):
        if time_up():
            break
        u, v = random.sample(range(n), 2)
        res = try_build_from_seed([u, v])
        if res is not None:
            branches, paths_dict = res
            if len(branches) > best_size:
                best_size = len(branches)
                all_p = get_all_paths(branches, paths_dict)
                if all_p is not None:
                    best_result = (branches, all_p)

    if best_result is not None:
        branches, paths = best_result
        return {'branches': [int(b) for b in branches], 'paths': [list(p) for p in paths]}

    # Fallback: single edge
    u, v = edges[0]
    return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
# EVOLVE-BLOCK-END
