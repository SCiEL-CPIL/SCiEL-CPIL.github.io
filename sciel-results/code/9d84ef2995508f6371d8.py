import time
import random

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}


def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)

    # Largest connected component
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

    start_time = time.time()
    time_limit = 85.0  # safety margin before 90s kill

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

    def greedy_two_end(start, rng, strategy='min'):
        """Extend path from both ends simultaneously."""
        if start not in comp_set:
            return []
        path = [start]
        seen = {start}
        left_ok = True
        right_ok = True
        while left_ok or right_ok:
            # Decide which end to extend
            if left_ok and right_ok:
                go_left = rng.random() < 0.5
            elif left_ok:
                go_left = True
            elif right_ok:
                go_left = False
            else:
                break

            if go_left:
                cur = path[0]
                neighbors = [x for x in adj[cur] if x not in seen and x in comp_set]
                if neighbors:
                    if strategy == 'min':
                        v = min(neighbors, key=lambda x: (len([y for y in adj[x] if y not in seen and y in comp_set]), x))
                    elif strategy == 'max':
                        v = max(neighbors, key=lambda x: (len([y for y in adj[x] if y not in seen and y in comp_set]), x))
                    else:
                        v = rng.choice(neighbors)
                    path.insert(0, v)
                    seen.add(v)
                else:
                    left_ok = False
            else:
                cur = path[-1]
                neighbors = [x for x in adj[cur] if x not in seen and x in comp_set]
                if neighbors:
                    if strategy == 'min':
                        v = min(neighbors, key=lambda x: (len([y for y in adj[x] if y not in seen and y in comp_set]), x))
                    elif strategy == 'max':
                        v = max(neighbors, key=lambda x: (len([y for y in adj[x] if y not in seen and y in comp_set]), x))
                    else:
                        v = rng.choice(neighbors)
                    path.append(v)
                    seen.add(v)
                else:
                    right_ok = False
        return path

    def greedy_one_end(start, strategy='min'):
        """Extend path from one end only (right)."""
        if start not in comp_set:
            return []
        path = [start]
        seen = {start}
        while True:
            cur = path[-1]
            neighbors = [x for x in adj[cur] if x not in seen and x in comp_set]
            if not neighbors:
                break
            if strategy == 'min':
                v = min(neighbors, key=lambda x: (len([y for y in adj[x] if y not in seen and y in comp_set]), x))
            elif strategy == 'max':
                v = max(neighbors, key=lambda x: (len([y for y in adj[x] if y not in seen and y in comp_set]), x))
            else:
                v = random.choice(neighbors)
            path.append(v)
            seen.add(v)
        return path

    def try_insert(path):
        """Try to insert an unvisited node into the path. Returns (new_path, inserted) or (path, None)."""
        path_set = set(path)
        candidates = [x for x in comp_set if x not in path_set]
        for u in candidates:
            if not (adj[u] & path_set):
                continue
            for i in range(len(path) - 1):
                if u in adj[path[i]] and u in adj[path[i + 1]]:
                    new_path = path[:i + 1] + [u] + path[i + 1:]
                    return new_path, u
        return path, None

    def rearrange_path(path):
        """
        Try to rearrange the path by moving an interior vertex to an endpoint.
        This changes the shape of the path and may enable further extension.
        Returns a list of candidate rearranged paths (same length, different shape).
        """
        results = []
        n = len(path)
        if n < 3:
            return results
        seen = set(path)

        # Try moving path[i] to the right end
        for i in range(1, n - 1):
            # path[i] must connect to path[-1]
            if path[i] in adj[path[-1]]:
                # path[i-1] must connect to path[i+1]
                if i + 1 < n and path[i + 1] in adj[path[i - 1]]:
                    new_path = path[:i] + path[i + 1:] + [path[i]]
                    # Verify
                    if is_valid_path(new_path):
                        results.append(new_path)

        # Try moving path[i] to the left end
        for i in range(1, n - 1):
            # path[i] must connect to path[0]
            if path[i] in adj[path[0]]:
                # path[i-1] must connect to path[i+1]
                if i + 1 < n and path[i + 1] in adj[path[i - 1]]:
                    new_path = [path[i]] + path[:i] + path[i + 1:]
                    if is_valid_path(new_path):
                        results.append(new_path)

        # Try 2-opt reversal: reverse a segment
        # Reverse path[i:j] (i.e., path[i..j-1] becomes reversed)
        # This requires path[i-1] connected to path[j-1] and path[i] connected to path[j]
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                # Reverse path[i:j]
                # Before: ... path[i-1] path[i] ... path[j-1] path[j] ...
                # After:  ... path[i-1] path[j-1] ... path[i] path[j] ...
                # Need: path[i-1] - path[j-1] edge and path[i] - path[j] edge
                if j < n:
                    if path[i - 1] in adj[path[j - 1]] and path[i] in adj[path[j]]:
                        new_path = path[:i] + path[i:j][::-1] + path[j:]
                        if is_valid_path(new_path):
                            results.append(new_path)
                else:
                    # j == n, reverse path[i:n]
                    if path[i - 1] in adj[path[n - 1]]:
                        new_path = path[:i] + path[i:j][::-1]
                        if is_valid_path(new_path):
                            results.append(new_path)

        return results

    def extend_both_ends(path):
        """Greedily extend path from both ends as much as possible."""
        path = list(path)
        seen = set(path)
        improved = True
        while improved:
            improved = False
            # Try right end
            cur = path[-1]
            neighbors = [x for x in adj[cur] if x not in seen and x in comp_set]
            if neighbors:
                v = min(neighbors, key=lambda x: (len([y for y in adj[x] if y not in seen and y in comp_set]), x))
                path.append(v)
                seen.add(v)
                improved = True
            # Try left end
            cur = path[0]
            neighbors = [x for x in adj[cur] if x not in seen and x in comp_set]
            if neighbors:
                v = min(neighbors, key=lambda x: (len([y for y in adj[x] if y not in seen and y in comp_set]), x))
                path.insert(0, v)
                seen.add(v)
                improved = True
        return path

    def full_refine(path):
        """
        Iteratively: extend both ends, try insertion, try rearrangement.
        Repeat until no improvement.
        """
        path = list(path)
        best = list(path)
        improved = True
        while improved:
            improved = False
            # Extend both ends
            extended = extend_both_ends(path)
            if len(extended) > len(path):
                path = extended
                improved = True
            # Try insertion
            inserted, node = try_insert(path)
            if node is not None:
                path = inserted
                improved = True
            # Try rearrangement
            rearranged = rearrange_path(path)
            for rp in rearranged:
                # Extend the rearranged path
                ext_rp = extend_both_ends(rp)
                if len(ext_rp) > len(path):
                    path = ext_rp
                    improved = True
                    break
            # Update best
            if len(path) > len(best):
                best = list(path)
        return best

    best_path = []

    # Phase 1: Greedy from each start node, both strategies
    for start in largest_comp:
        if time.time() - start_time > time_limit:
            break
        for strategy in ['min', 'max']:
            p = greedy_one_end(start, strategy)
            if len(p) > len(best_path):
                best_path = p

    # Phase 2: Two-ended greedy from each start
    rng = random.Random(748)
    for start in largest_comp:
        if time.time() - start_time > time_limit:
            break
        for strategy in ['min', 'max', 'random']:
            p = greedy_two_end(start, rng, strategy)
            if len(p) > len(best_path):
                best_path = p

    # Phase 3: Full refinement of the best path
    if best_path:
        refined = full_refine(best_path)
        if len(refined) > len(best_path):
            best_path = refined

    # Phase 4: Randomized restarts with full refinement
    rng2 = random.Random(12345)
    restarts_done = 0
    max_restarts = 200
    for _ in range(max_restarts):
        if time.time() - start_time > time_limit - 2:
            break
        start = rng2.choice(largest_comp)
        strategy = rng2.choice(['min', 'max', 'random'])
        p = greedy_two_end(start, rng2, strategy)
        p = full_refine(p)
        if len(p) > len(best_path):
            best_path = p
        restarts_done += 1

    # Phase 5: If we have time, try more aggressive search
    # Try rearranging the best path and extending
    if best_path and time.time() - start_time < time_limit - 5:
        rearranged = rearrange_path(best_path)
        for rp in rearranged:
            if time.time() - start_time > time_limit:
                break
            ext_rp = full_refine(rp)
            if len(ext_rp) > len(best_path):
                best_path = ext_rp

    if not best_path:
        return [0]

    # Final validation
    if not is_valid_path(best_path):
        # Fallback: return a valid single edge or vertex
        for u, v in P['edges']:
            if u in comp_set and v in comp_set:
                return [u, v]
        return [largest_comp[0]]

    return best_path
