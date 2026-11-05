import time
import random
from collections import deque

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}


def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)

    # Find largest connected component
    visited = [False] * n
    components = []
    for i in range(n):
        if not visited[i]:
            stack = [i]
            visited[i] = True
            comp = []
            while stack:
                node = stack.pop()
                comp.append(node)
                for nei in adj[node]:
                    if not visited[nei]:
                        visited[nei] = True
                        stack.append(nei)
            components.append(comp)

    if not components:
        return [0]

    largest_comp = max(components, key=len)
    comp_set = set(largest_comp)
    comp_size = len(largest_comp)

    start_time = time.time()
    time_limit = 80.0  # Safer margin

    def is_valid_path(path):
        if not path:
            return False
        s = set(path)
        if len(s) != len(path):
            return False
        for i in range(len(path) - 1):
            if path[i + 1] not in adj[path[i]]:
                return False
        return True

    def greedy_extend(path, seen, strategy='min'):
        """Extend path from both ends greedily."""
        path = list(path)
        seen = set(seen)
        improved = True
        while improved:
            improved = False
            # Try right end
            cur = path[-1]
            neighbors = [x for x in adj[cur] if x not in seen and x in comp_set]
            if neighbors:
                if strategy == 'min':
                    v = min(neighbors, key=lambda x: (len([y for y in adj[x] if y not in seen and y in comp_set]), x))
                elif strategy == 'max':
                    v = max(neighbors, key=lambda x: (len([y for y in adj[x] if y not in seen and y in comp_set]), x))
                else:
                    v = random.choice(neighbors)
                path.append(v)
                seen.add(v)
                improved = True
            # Try left end
            cur = path[0]
            neighbors = [x for x in adj[cur] if x not in seen and x in comp_set]
            if neighbors:
                if strategy == 'min':
                    v = min(neighbors, key=lambda x: (len([y for y in adj[x] if y not in seen and y in comp_set]), x))
                elif strategy == 'max':
                    v = max(neighbors, key=lambda x: (len([y for y in adj[x] if y not in seen and y in comp_set]), x))
                else:
                    v = random.choice(neighbors)
                path.insert(0, v)
                seen.add(v)
                improved = True
        return path, seen

    def try_insert_node(path, seen):
        """Try to insert an unvisited node into the path."""
        path_set = set(path)
        candidates = [x for x in comp_set if x not in path_set]
        for u in candidates:
            adj_u = adj[u]
            if not (adj_u & path_set):
                continue
            for i in range(len(path) - 1):
                if path[i] in adj_u and path[i + 1] in adj_u:
                    new_path = path[:i + 1] + [u] + path[i + 1:]
                    return new_path
        return None

    def two_opt_improve(path):
        """
        2-opt: try reversing a segment of the path.
        Returns a list of candidate improved paths (same length, different order).
        """
        results = []
        n_path = len(path)
        if n_path < 3:
            return results
        seen_in_path = set(path)
        
        def endpoint_score(p):
            s = set(p)
            return (len(adj[p[0]] - s) + len(adj[p[-1]] - s), p[0], p[-1])
        
        base_score = endpoint_score(path)
        
        for i in range(0, n_path - 1):
            for j in range(i + 1, n_path + 1):
                if i == 0 and j == n_path:
                    new_path = path[::-1]
                    if endpoint_score(new_path) > base_score:
                        results.append(new_path)
                    continue
                if i == 0:
                    if path[0] not in adj[path[j - 1]]:
                        continue
                    new_path = path[:j][::-1] + path[j:]
                elif j == n_path:
                    if path[i - 1] not in adj[path[n_path - 1]]:
                        continue
                    new_path = path[:i] + path[i:][::-1]
                else:
                    if path[i - 1] not in adj[path[j - 1]]:
                        continue
                    if path[i] not in adj[path[j]]:
                        continue
                    new_path = path[:i] + path[i:j][::-1] + path[j:]
                
                ok = True
                for k in range(len(new_path) - 1):
                    if new_path[k + 1] not in adj[new_path[k]]:
                        ok = False
                        break
                if not ok:
                    continue
                if endpoint_score(new_path) > base_score:
                    results.append(new_path)
        return results

    def swap_improve(path):
        """
        Try 2-opt style segment reversals to create new extendable endpoints.
        Returns list of improved paths (same length, different endpoint structure).
        """
        results = []
        n_path = len(path)
        if n_path < 3:
            return results

        for i in range(1, n_path):
            for j in range(i + 1, n_path + 1):
                if j == n_path:
                    if path[i - 1] in adj[path[n_path - 1]]:
                        new_path = path[:i] + path[i:][::-1]
                        if is_valid_path(new_path):
                            results.append(new_path)
                else:
                    if path[i - 1] in adj[path[j - 1]] and path[i] in adj[path[j]]:
                        new_path = path[:i] + path[i:j][::-1] + path[j:]
                        if is_valid_path(new_path):
                            results.append(new_path)

        for i in range(1, n_path - 1):
            if path[i] in adj[path[-1]]:
                if i + 1 < n_path and path[i + 1] in adj[path[i - 1]]:
                    new_path = path[:i] + path[i + 1:] + [path[i]]
                    if is_valid_path(new_path):
                        results.append(new_path)
            if path[i] in adj[path[0]]:
                if i + 1 < n_path and path[i + 1] in adj[path[i - 1]]:
                    new_path = [path[i]] + path[:i] + path[i + 1:]
                    if is_valid_path(new_path):
                        results.append(new_path)

        return results

    def splice_paths(path1, path2):
        """
        Try to splice two paths together.
        """
        for a in [0, -1]:
            for b in [0, -1]:
                if path1[a] in adj[path2[b]]:
                    if b == -1:
                        path2_reversed = path2[::-1]
                    else:
                        path2_reversed = path2
                    if not (set(path1) & set(path2_reversed)):
                        merged = path1 + path2_reversed
                        if is_valid_path(merged):
                            return merged
        return None

    def dfs_path(start, max_depth, seen, adj_list):
        """
        DFS to find a long path starting from start with backtracking.
        """
        best_path = [start]
        seen = set(seen)
        seen.add(start)

        def dfs(current, path, depth):
            nonlocal best_path
            if depth > len(best_path):
                best_path = list(path)

            if depth >= max_depth:
                return

            neighbors = [x for x in adj_list[current] if x not in seen and x in comp_set]
            neighbors.sort(key=lambda x: (len([y for y in adj_list[x] if y not in seen and y in comp_set]), x))

            for v in neighbors:
                seen.add(v)
                path.append(v)
                dfs(v, path, depth + 1)
                path.pop()
                seen.remove(v)

                if time.time() - start_time > time_limit:
                    return

        dfs(start, [start], 1)
        return best_path

    def full_refine(path):
        """
        Iteratively refine: extend, insert, swap. Repeat until no improvement.
        """
        path = list(path)
        seen = set(path)
        best = list(path)
        improved = True
        iteration = 0
        while improved and iteration < 50:
            improved = False
            iteration += 1

            extended, new_seen = greedy_extend(path, seen, 'min')
            if len(extended) > len(path):
                path = extended
                seen = new_seen
                improved = True

            inserted = True
            while inserted:
                inserted = try_insert_node(path, seen)
                if inserted is not None:
                    path = inserted
                    seen = set(path)
                    improved = True

            swapped = swap_improve(path)
            for sp in swapped:
                ext_sp, seen_sp = greedy_extend(sp, set(sp), 'min')
                if len(ext_sp) > len(path):
                    path = ext_sp
                    seen = seen_sp
                    improved = True
                    break

            if not improved:
                two_opt_candidates = two_opt_improve(path)
                for sp in two_opt_candidates[:5]:
                    ext_sp, seen_sp = greedy_extend(sp, set(sp), 'min')
                    if len(ext_sp) > len(path):
                        path = ext_sp
                        seen = seen_sp
                        improved = True
                        break

            if len(path) > len(best):
                best = list(path)

        return best

    def final_polish(path):
        """
        Final high-precision polish: aggressive local search with multiple strategies.
        """
        best = list(path)
        seen = set(path)
        for _ in range(20):
            if time.time() - start_time > time_limit - 1:
                break
            for strat in ['min', 'max', 'random']:
                ext, new_seen = greedy_extend(list(best), set(best), strat)
                if len(ext) > len(best):
                    best = ext
                    seen = new_seen
            inserted = True
            while inserted:
                ins = try_insert_node(best, seen)
                if ins is not None:
                    best = ins
                    seen = set(best)
                else:
                    inserted = False
            opts = two_opt_improve(best)
            for op in opts[:10]:
                ext, _ = greedy_extend(op, set(op), 'min')
                if len(ext) > len(best):
                    best = ext
                    seen = set(best)
                    break
            swaps = swap_improve(best)
            for sw in swaps[:10]:
                ext, _ = greedy_extend(sw, set(sw), 'min')
                if len(ext) > len(best):
                    best = ext
                    seen = set(best)
                    break
        return best

    best_path = []

    # Phase 1: Greedy from each start node
    for start in largest_comp:
        if time.time() - start_time > time_limit:
            break
        for strategy in ['min', 'max']:
            p, _ = greedy_extend([start], {start}, strategy)
            if len(p) > len(best_path):
                best_path = list(p)

    # Phase 2: DFS from promising starts
    if time.time() - start_time < time_limit - 10:
        rng = random.Random(748)
        dfs_starts = rng.sample(largest_comp, min(10, len(largest_comp)))
        for start in dfs_starts:
            if time.time() - start_time > time_limit - 5:
                break
            max_depth = min(comp_size, 30)
            p = dfs_path(start, max_depth, set(), adj)
            if len(p) > len(best_path):
                best_path = list(p)

    # Phase 3: Full refinement of best path
    if best_path:
        refined = full_refine(best_path)
        if len(refined) > len(best_path):
            best_path = refined

    # Phase 4: Randomized restarts with refinement
    rng2 = random.Random(12345)
    for _ in range(100):
        if time.time() - start_time > time_limit - 3:
            break
        start = rng2.choice(largest_comp)
        strategy = rng2.choice(['min', 'max', 'random'])
        p, _ = greedy_extend([start], {start}, strategy)
        p = full_refine(p)
        if len(p) > len(best_path):
            best_path = list(p)

    # Phase 5: Try swap + 2-opt improvements on best path and refine
    if best_path and time.time() - start_time < time_limit - 5:
        candidates = list(swap_improve(best_path))
        candidates.extend(two_opt_improve(best_path))
        seen_sp = set()
        for sp in candidates:
            if time.time() - start_time > time_limit - 1:
                break
            key = tuple(sp)
            if key in seen_sp:
                continue
            seen_sp.add(key)
            ext_sp = full_refine(sp)
            if len(ext_sp) > len(best_path):
                best_path = list(ext_sp)

    # Phase 6: One more DFS attempt with higher depth if time remains
    if time.time() - start_time < time_limit - 8 and best_path:
        start = best_path[-1]
        seen_init = set(best_path)
        max_depth = min(comp_size, 40)
        p = dfs_path(start, max_depth, seen_init, adj)
        if len(p) > len(best_path):
            best_path = list(p)
            best_path = full_refine(best_path)

    # Phase 7: Path splicing
    if time.time() - start_time < time_limit - 10 and best_path:
        partial_paths = []
        rng3 = random.Random(999)
        for _ in range(20):
            if time.time() - start_time > time_limit - 5:
                break
            start = rng3.choice(largest_comp)
            p, _ = greedy_extend([start], {start}, 'min')
            if len(p) >= 2:
                partial_paths.append(p)
        
        for i in range(len(partial_paths)):
            for j in range(i + 1, len(partial_paths)):
                if time.time() - start_time > time_limit - 2:
                    break
                merged = splice_paths(partial_paths[i], partial_paths[j])
                if merged is not None:
                    merged = full_refine(merged)
                    if len(merged) > len(best_path):
                        best_path = list(merged)

    # Phase 8: Final polish
    if best_path and time.time() - start_time < time_limit - 2:
        polished = final_polish(best_path)
        if len(polished) > len(best_path):
            best_path = polished

    if not best_path:
        return [0]

    if not is_valid_path(best_path):
        for u, v in P['edges']:
            if u in comp_set and v in comp_set:
                return [u, v]
        return [largest_comp[0]]

    return best_path
