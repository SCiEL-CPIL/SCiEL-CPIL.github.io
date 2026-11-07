"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    n = P['n']
    edges = P['edges']

    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    def check_time():
        return time.time() < deadline

    def bfs_path(u, v, forbidden, max_len=None):
        """Find a path from u to v avoiding forbidden. If max_len, find path of exactly that length (or shorter)."""
        if u in forbidden or v in forbidden:
            return None
        if u == v:
            return [u, v]

        # BFS to find shortest path
        visited = {u: None}
        queue = deque([u])

        while queue:
            node = queue.popleft()
            if node == v:
                path = [v]
                while path[-1] != u:
                    path.append(visited[path[-1]])
                path.reverse()
                if max_len is None or len(path) - 1 <= max_len:
                    return path
                # Found shortest but too short? Try longer paths via DFS
                break
            for neighbor in adj[node]:
                if neighbor not in visited and neighbor not in forbidden:
                    visited[neighbor] = node
                    queue.append(neighbor)

        # Try to find a longer path via DFS with length constraint
        if max_len is not None:
            def dfs(node, path, forbidden):
                if node == v:
                    if len(path) - 1 == max_len:
                        return path
                    return None
                for neighbor in adj[node]:
                    if neighbor not in forbidden and neighbor not in path:
                        if len(path) < max_len:
                            result = dfs(neighbor, path + [neighbor], forbidden)
                            if result:
                                return result
                return None

            result = dfs(u, [u], forbidden)
            if result:
                return result

        # If no max_len constraint, return shortest found
        if max_len is None:
            visited = {u: None}
            queue = deque([u])
            while queue:
                node = queue.popleft()
                if node == v:
                    path = [v]
                    while path[-1] != u:
                        path.append(visited[path[-1]])
                    path.reverse()
                    return path
                for neighbor in adj[node]:
                    if neighbor not in visited and neighbor not in forbidden:
                        visited[neighbor] = node
                        queue.append(neighbor)

        return None

    def is_valid_solution(branches, paths_dict):
        """Validate that the solution satisfies all constraints."""
        if len(branches) < 2:
            return False
        if len(set(branches)) != len(branches):
            return False

        for u, v in combinations(branches, 2):
            if (u, v) not in paths_dict and (v, u) not in paths_dict:
                return False

        used_interiors = set()
        for key, path in paths_dict.items():
            if len(path) < 2:
                return False
            u, v = key
            if path[0] not in (u, v) or path[-1] not in (u, v):
                return False
            if path[0] == path[-1]:
                return False
            if len(set(path)) != len(path):
                return False

            for i in range(len(path) - 1):
                if path[i + 1] not in adj[path[i]]:
                    return False

            interiors = path[1:-1]
            for node in interiors:
                if node in branches:
                    return False
                if node in used_interiors:
                    return False
            used_interiors.update(interiors)

        return True

    def find_disjoint_paths(branches, num_attempts=50):
        """
        Given a set of branch vertices, find vertex-disjoint paths between all pairs.
        Uses randomized ordering and tries multiple path lengths.
        """
        if len(branches) < 2:
            return {}

        pairs = list(combinations(branches, 2))

        for attempt in range(num_attempts):
            if not check_time():
                return None

            random.shuffle(pairs)
            candidate_paths = {}
            used_interiors = set()

            success = True
            for u, v in pairs:
                path_forbidden = set(used_interiors)
                for b in branches:
                    if b != u and b != v:
                        path_forbidden.add(b)

                # Try shortest path first, then slightly longer
                p = bfs_path(u, v, path_forbidden)
                if p is None:
                    # Try a few longer paths
                    found = False
                    for extra in range(1, 4):
                        p = bfs_path(u, v, path_forbidden, max_len=len(p) + extra if p else extra + 1)
                        if p is not None:
                            found = True
                            break
                    if not found:
                        success = False
                        break

                candidate_paths[(u, v)] = p
                used_interiors.update(p[1:-1])

            if success:
                return candidate_paths

        return None

    def max_clique():
        """Find maximum clique using Bron-Kerbosch with pivot."""
        best_clique = []

        def bron_kerbosch(R, P, X):
            nonlocal best_clique
            if not check_time():
                return

            if not P and not X:
                if len(R) > len(best_clique):
                    best_clique = list(R)
                return

            if P:
                pivot = max(P, key=lambda v: len(adj[v] & (P | X)))
            else:
                pivot = None

            candidates = set(P)
            if pivot is not None:
                candidates -= adj[pivot]

            for v in list(candidates):
                if not check_time():
                    return
                bron_kerbosch(R | {v}, P & adj[v], X & adj[v])
                P.remove(v)
                X.add(v)

        bron_kerbosch(set(), set(range(n)), set())
        return best_clique

    def greedy_expand(start_branches):
        """
        Start with initial branches and greedily add more.
        Returns (branches, paths_dict).
        """
        current_branches = list(start_branches)

        current_paths = find_disjoint_paths(current_branches, num_attempts=50)
        if current_paths is None:
            return [], {}

        for _ in range(300):
            if not check_time():
                break

            candidates = [i for i in range(n) if i not in current_branches]
            if not candidates:
                break

            # Sort by number of edges to existing branches (more connected first)
            def score_candidate(c):
                return sum(1 for b in current_branches if c in adj[b])
            candidates.sort(key=score_candidate, reverse=True)
            # Tie-break by degree
            candidates.sort(key=lambda x: len(adj[x]), reverse=True)

            added = False
            for new_b in candidates[:15]:
                new_branches = current_branches + [new_b]
                new_paths = find_disjoint_paths(new_branches, num_attempts=30)

                if new_paths is not None:
                    if is_valid_solution(new_branches, new_paths):
                        current_branches = new_branches
                        current_paths = new_paths
                        added = True
                        break

            if not added:
                break

        return current_branches, current_paths

    def local_search(branches, paths_dict):
        """
        Try to improve the solution by swapping branches.
        """
        best = (list(branches), dict(paths_dict))
        best_size = len(branches)

        for _ in range(200):
            if not check_time():
                break

            if len(branches) < 3:
                break

            remove_idx = random.randint(0, len(branches) - 1)
            removed = branches[remove_idx]
            remaining = [b for i, b in enumerate(branches) if i != remove_idx]

            candidates = [i for i in range(n) if i not in remaining]

            # Sort candidates by connectivity to remaining branches
            def score_candidate(c):
                return sum(1 for b in remaining if c in adj[b])
            candidates.sort(key=score_candidate, reverse=True)
            random.shuffle(candidates[:10])  # Add some randomness

            for new_b in candidates[:20]:
                new_branches = remaining + [new_b]
                new_paths = find_disjoint_paths(new_branches, num_attempts=30)

                if new_paths is not None and is_valid_solution(new_branches, new_paths):
                    if len(new_branches) > best_size:
                        best = (new_branches, new_paths)
                        best_size = len(new_branches)
                        branches, paths_dict = new_branches, new_paths
                    break

        return best

    # Strategy 1: Find maximum clique, then try to find disjoint paths
    clique = max_clique()

    best_branches = []
    best_paths = {}

    if len(clique) >= 2:
        clique_sorted = sorted(clique, key=lambda x: len(adj[x]), reverse=True)

        for size in range(min(len(clique_sorted), 12), 1, -1):
            if not check_time():
                break

            subset = clique_sorted[:size]
            paths = find_disjoint_paths(subset, num_attempts=100)
            if paths is not None:
                if is_valid_solution(subset, paths):
                    best_branches = subset
                    best_paths = paths
                    break

    # Strategy 2: Greedy expansion from high-degree edges
    if len(best_branches) < 2:
        sorted_edges = sorted(edges, key=lambda e: len(adj[e[0]]) + len(adj[e[1]]), reverse=True)

        for u, v in sorted_edges[:50]:
            if not check_time():
                break

            current_branches, current_paths = greedy_expand([u, v])

            if len(current_branches) > len(best_branches):
                best_branches = current_branches
                best_paths = current_paths

    # Strategy 3: Multiple random restarts
    for restart in range(50):
        if not check_time():
            break

        if edges:
            u, v = random.choice(edges)
            current_branches, current_paths = greedy_expand([u, v])
        else:
            break

        if len(current_branches) > len(best_branches):
            best_branches = current_branches
            best_paths = current_paths

    # Final local search
    if len(best_branches) >= 2:
        best_branches, best_paths = local_search(best_branches, best_paths)

    # Build final output
    if len(best_branches) < 2:
        if edges:
            u, v = edges[0]
            return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
        else:
            return {'branches': [0, 1], 'paths': [[0, 1]]}

    final_paths = []
    for u, v in combinations(best_branches, 2):
        if (u, v) in best_paths:
            final_paths.append([int(x) for x in best_paths[(u, v)]])
        elif (v, u) in best_paths:
            final_paths.append([int(x) for x in best_paths[(v, u)]])

    return {
        'branches': [int(x) for x in best_branches],
        'paths': final_paths
    }

# EVOLVE-BLOCK-END
