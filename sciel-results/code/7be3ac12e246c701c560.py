import time
from collections import deque
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # safety margin before 60s

    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)

    best_branches = None
    best_paths = None
    best_count = 0

    def update_best(branches, paths):
        nonlocal best_branches, best_paths, best_count
        if branches is not None and len(branches) > best_count:
            best_count = len(branches)
            best_branches = branches
            best_paths = paths

    def bfs_path(start, end, forbidden, adj):
        """Find shortest path from start to end avoiding forbidden vertices (except start/end)."""
        if start == end:
            return [start]
        visited = {start}
        parent = {start: None}
        queue = deque([start])
        while queue:
            u = queue.popleft()
            if u == end:
                path = []
                node = end
                while node is not None:
                    path.append(node)
                    node = parent[node]
                path.reverse()
                return path
            for v in adj[u]:
                if v in visited:
                    continue
                if v in forbidden and v != end:
                    continue
                visited.add(v)
                parent[v] = u
                queue.append(v)
        return None

    def bfs_path_length(start, end, forbidden, adj, max_len):
        """Find a path of length <= max_len from start to end avoiding forbidden (except start/end).
        Returns None if no such path exists. Uses BFS up to depth max_len."""
        if start == end:
            return [start]
        if max_len <= 1:
            if end in adj[start]:
                return [start, end]
            return None
        visited = {start}
        parent = {start: None}
        depth = {start: 0}
        queue = deque([start])
        while queue:
            u = queue.popleft()
            if depth[u] >= max_len:
                continue
            if u == end:
                path = []
                node = end
                while node is not None:
                    path.append(node)
                    node = parent[node]
                path.reverse()
                return path
            for v in adj[u]:
                if v in visited:
                    continue
                if v in forbidden and v != end:
                    continue
                visited.add(v)
                parent[v] = u
                depth[v] = depth[u] + 1
                queue.append(v)
        return None

    def find_paths_short(remaining_pairs, used_int, current_paths, branch_set):
        """Greedy: for each pair, find the shortest path avoiding used interiors and other branches."""
        if time.time() > deadline:
            return None
        if not remaining_pairs:
            return current_paths
        for idx, (a, b) in enumerate(remaining_pairs):
            rest = remaining_pairs[:idx] + remaining_pairs[idx+1:]
            forbidden = used_int | (branch_set - {a, b})
            path = bfs_path(a, b, forbidden, adj)
            if path is not None:
                interior = set(path[1:-1])
                if not (interior & used_int):
                    new_used = used_int | interior
                    result = find_paths_short(rest, new_used, current_paths + [path], branch_set)
                    if result is not None:
                        return result
        return None

    def find_paths_long(remaining_pairs, used_int, current_paths, branch_set, max_search_len):
        """Greedy: for each pair, try paths of increasing length to find one that fits."""
        if time.time() > deadline:
            return None
        if not remaining_pairs:
            return current_paths
        for idx, (a, b) in enumerate(remaining_pairs):
            rest = remaining_pairs[:idx] + remaining_pairs[idx+1:]
            forbidden = used_int | (branch_set - {a, b})
            # Try increasing path lengths
            found = False
            for plen in range(2, max_search_len + 1):
                if time.time() > deadline:
                    return None
                path = bfs_path_length(a, b, forbidden, adj, plen)
                if path is not None:
                    interior = set(path[1:-1])
                    if not (interior & used_int):
                        new_used = used_int | interior
                        result = find_paths_long(rest, new_used, current_paths + [path], branch_set, max_search_len)
                        if result is not None:
                            return result
                    # Found a path at this length but couldn't complete - try longer
                    found = True
            # If no path found at any length, this ordering fails
            if not found:
                continue
            # If we found paths at all lengths but none completed, try next pair ordering
        return None

    def try_branch_set(branches):
        """Try to find internally disjoint paths for all pairs in branches."""
        if len(branches) < 2:
            return None
        k = len(branches)
        branch_set = set(branches)
        pairs = list(combinations(branches, 2))
        import random
        rng = random.Random(hash(tuple(sorted(branches))) % 100000)

        # Strategy 1: Greedy with shortest path, natural order
        result = find_paths_short(pairs, set(), [], branch_set)
        if result is not None:
            return result

        # Strategy 2: Greedy with shortest path, sorted by difficulty
        def pair_difficulty(pair):
            a, b = pair
            forbidden = branch_set - {a, b}
            path = bfs_path(a, b, forbidden, adj)
            return len(path) if path else 999
        sorted_pairs = sorted(pairs, key=pair_difficulty)
        result = find_paths_short(sorted_pairs, set(), [], branch_set)
        if result is not None:
            return result

        # Strategy 3: Randomized orderings with shortest path
        for attempt in range(15):
            if time.time() > deadline:
                return None
            shuffled = pairs[:]
            rng.shuffle(shuffled)
            result = find_paths_short(shuffled, set(), [], branch_set)
            if result is not None:
                return result

        # Strategy 4: Long path search with natural order
        max_len = min(n, 2 * k + 2)
        result = find_paths_long(pairs, set(), [], branch_set, max_len)
        if result is not None:
            return result

        # Strategy 5: Long path search with randomized orderings
        for attempt in range(10):
            if time.time() > deadline:
                return None
            shuffled = pairs[:]
            rng.shuffle(shuffled)
            result = find_paths_long(shuffled, set(), [], branch_set, max_len)
            if result is not None:
                return result

        return None

    # Build candidate branch sets

    # 1. Find a good clique (many edges = short paths between pairs)
    def find_max_clique_greedy(adj, n):
        best = []
        for start in range(n):
            if time.time() > deadline:
                break
            clique = [start]
            candidates = set(adj[start])
            while candidates:
                best_v = max(candidates, key=lambda v: len(adj[v] & set(clique)))
                if len(adj[best_v] & set(clique)) < len(clique):
                    break
                clique.append(best_v)
                candidates = candidates & adj[best_v]
                candidates = candidates - set(clique)
            if len(clique) > len(best):
                best = clique[:]
        return best

    initial_clique = find_max_clique_greedy(adj, n)
    if initial_clique:
        paths_result = try_branch_set(initial_clique)
        if paths_result is not None:
            update_best(initial_clique[:], paths_result)

    # 2. Try to expand the clique by adding high-degree vertices
    if best_branches is not None:
        current_branches = best_branches[:]
        expanded = True
        while expanded and time.time() < deadline:
            expanded = False
            current_set = set(current_branches)
            candidates = [v for v in range(n) if v not in current_set]
            candidates.sort(key=lambda v: len(adj[v]), reverse=True)
            for v in candidates:
                if time.time() > deadline:
                    break
                new_branches = current_branches + [v]
                paths_result = try_branch_set(new_branches)
                if paths_result is not None:
                    current_branches = new_branches
                    update_best(current_branches[:], paths_result)
                    expanded = True
                    break

    # 3. Try sets combining clique vertices with high-degree vertices
    high_degree = sorted(range(n), key=lambda v: len(adj[v]), reverse=True)
    clique_set = set(initial_clique) if initial_clique else set()

    # Build several candidate sets of increasing size
    candidate_sets = []
    # Clique + top-degree non-clique vertices
    non_clique_high = [v for v in high_degree if v not in clique_set]
    for size in range(2, n + 1):
        if time.time() > deadline:
            break
        # Candidate: clique vertices + best non-clique vertices
        c = list(clique_set)
        for v in non_clique_high:
            if len(c) >= size:
                break
            c.append(v)
        if len(c) >= 2:
            candidate_sets.append(c[:size])
        # Also try pure high-degree sets
        if size <= len(high_degree):
            candidate_sets.append(high_degree[:size])

    # Try candidate sets from largest to smallest
    for cset in sorted(candidate_sets, key=len, reverse=True):
        if time.time() > deadline:
            break
        if len(cset) <= best_count:
            continue
        paths_result = try_branch_set(cset)
        if paths_result is not None:
            update_best(cset[:], paths_result)

    # 4. Randomized search for larger sets
    import random
    rng2 = random.Random(12345)
    for size in range(n, best_count, -1):
        if time.time() > deadline:
            break
        for attempt in range(200):
            if time.time() > deadline:
                break
            random_set = rng2.sample(range(n), size)
            paths_result = try_branch_set(random_set)
            if paths_result is not None:
                update_best(random_set[:], paths_result)
                break

    # Ensure we have at least 2 branches
    if best_branches is None or len(best_branches) < 2:
        u, v = P['edges'][0]
        best_branches = [int(u), int(v)]
        best_paths = [[int(u), int(v)]]
    else:
        best_branches = [int(v) for v in best_branches]
        best_paths = [[int(v) for v in path] for path in best_paths]

    return {'branches': best_branches, 'paths': best_paths}
