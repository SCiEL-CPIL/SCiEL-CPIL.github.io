import time
import sys

# EVOLVE-BLOCK-START
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)

    # Find largest connected component
    visited_comp = [False] * n
    largest_comp = []
    for i in range(n):
        if not visited_comp[i]:
            stack = [i]
            comp = []
            visited_comp[i] = True
            while stack:
                node = stack.pop()
                comp.append(node)
                for nei in adj[node]:
                    if not visited_comp[nei]:
                        visited_comp[nei] = True
                        stack.append(nei)
            if len(comp) > len(largest_comp):
                largest_comp = comp

    if not largest_comp:
        return [0]

    comp_set = set(largest_comp)

    # Multi-strategy greedy initialization
    def greedy_warnsdorff(start_node):
        visited = {start_node}
        path = [start_node]
        while True:
            last = path[-1]
            neighbors = [x for x in adj[last] if x not in visited]
            if not neighbors:
                break
            next_node = min(neighbors, key=lambda x: (len(adj[x] - visited), x))
            path.append(next_node)
            visited.add(next_node)
        return path

    def greedy_random(start_node, seed_offset):
        import random
        rng = random.Random(seed_offset)
        visited = {start_node}
        path = [start_node]
        while True:
            last = path[-1]
            neighbors = [x for x in adj[last] if x not in visited]
            if not neighbors:
                break
            next_node = rng.choice(neighbors)
            path.append(next_node)
            visited.add(next_node)
        return path

    # Generate diverse initial paths
    candidates = []
    seen = set()

    # Warnsdorff from each node in component
    for i in largest_comp:
        path = greedy_warnsdorff(i)
        key = tuple(path)
        if key not in seen:
            seen.add(key)
            candidates.append(path)

    # Random greedy from several starts
    for i in largest_comp[:20]:
        for seed_off in range(3):
            path = greedy_random(i, 1000 + i * 10 + seed_off)
            key = tuple(path)
            if key not in seen:
                seen.add(key)
                candidates.append(path)

    if not candidates:
        return [0]

    candidates.sort(key=len, reverse=True)
    best_path = candidates[0]
    best_len = len(best_path)

    # Bidirectional extension: try to extend both ends of a path
    def try_extend_ends(path):
        if time.time() > deadline - 5.0:
            return path

        path = list(path)
        visited = set(path)

        # Try extending the front
        extended = False
        while True:
            first = path[0]
            neighbors = [x for x in adj[first] if x not in visited]
            if not neighbors:
                break
            # Pick neighbor with fewest unvisited neighbors (Warnsdorff)
            next_node = min(neighbors, key=lambda x: (len(adj[x] - visited), x))
            path.insert(0, next_node)
            visited.add(next_node)
            extended = True
        if not extended:
            # Try from the back
            while True:
                last = path[-1]
                neighbors = [x for x in adj[last] if x not in visited]
                if not neighbors:
                    break
                next_node = min(neighbors, key=lambda x: (len(adj[x] - visited), x))
                path.append(next_node)
                visited.add(next_node)
            # Try front again
            while True:
                first = path[0]
                neighbors = [x for x in adj[first] if x not in visited]
                if not neighbors:
                    break
                next_node = min(neighbors, key=lambda x: (len(adj[x] - visited), x))
                path.insert(0, next_node)
                visited.add(next_node)

        return path

    # 2-opt style local search for paths
    def local_search(path):
        if len(path) < 2:
            return path

        current_path = list(path)
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            L = len(current_path)

            # Try reversing subpaths
            for i in range(1, L - 1):
                if time.time() > deadline - 5.0:
                    break
                for j in range(i + 1, L):
                    valid = True
                    # Check edge before i
                    if i > 0:
                        if current_path[i - 1] not in adj[current_path[j - 1]]:
                            valid = False
                    # Check edge at j
                    if valid and j < L:
                        if current_path[i] not in adj[current_path[j]]:
                            valid = False
                    if not valid:
                        continue

                    # Perform the reversal
                    current_path[i:j] = current_path[i:j][::-1]
                    improved = True
                    break
                if improved:
                    break

        return current_path

    # Apply bidirectional extension and local search to top candidates
    for c in candidates[:10]:
        if time.time() > deadline - 5.0:
            break

        # Bidirectional extension
        extended = try_extend_ends(c)
        if len(extended) > best_len:
            best_len = len(extended)
            best_path = extended

        # Local search
        improved = local_search(extended)
        if len(improved) > best_len:
            best_len = len(improved)
            best_path = improved

    # Iterative deepening DFS to try to find longer paths
    def iddfs_search(target_len):
        if time.time() > deadline - 5.0:
            return None

        def dfs(node, path, visited):
            if time.time() > deadline - 5.0:
                return None
            if len(path) == target_len:
                return list(path)

            neighbors = [x for x in adj[node] if x not in visited]
            if not neighbors:
                return None

            # Warnsdorff ordering
            neighbors.sort(key=lambda x: len(adj[x] - visited))

            for x in neighbors:
                path.append(x)
                visited.add(x)
                res = dfs(x, path, visited)
                if res is not None:
                    return res
                path.pop()
                visited.remove(x)

            return None

        # Try starting from low-degree nodes first
        start_nodes = sorted(largest_comp, key=lambda x: len(adj[x]))

        for start in start_nodes:
            if time.time() > deadline - 5.0:
                return None
            path = [start]
            visited = {start}
            res = dfs(start, path, visited)
            if res is not None:
                return res

        return None

    # Try to find paths longer than current best
    current_target = best_len + 1
    max_possible = len(largest_comp)

    while current_target <= max_possible and time.time() < deadline - 10.0:
        found_path = iddfs_search(current_target)
        if found_path:
            best_len = current_target
            best_path = found_path
            current_target += 1
        else:
            break

    return best_path
# EVOLVE-BLOCK-END
