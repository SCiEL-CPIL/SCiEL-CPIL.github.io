import time
import random
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

DEADLINE = time.time() + 50.0

def construct():
    t0 = time.time()
    n = P['n']
    edges = P['edges']

    adj = [set() for _ in range(n)]
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)

    def time_up():
        return time.time() - t0 > DEADLINE - 2.0

    def fallback():
        u, v = edges[0]
        return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}

    best = fallback()
    best_size = 2

    def find_paths(branches, max_backtrack=5000):
        """
        Try to find pairwise internally disjoint paths for all pairs in branches.
        Returns list of paths (one per pair) or None.
        Uses backtracking with BFS shortest path, trying different orderings.
        """
        bs = set(branches)
        sorted_bs = sorted(branches)
        pairs = list(combinations(sorted_bs, 2))
        if not pairs:
            return []

        rng = random.Random(12345)

        def bfs_path(s, t, blocked):
            """Find shortest path from s to t avoiding blocked vertices (except s,t)."""
            if s == t:
                return [s]
            parent = {s: None}
            q = deque([s])
            while q:
                curr = q.popleft()
                if curr == t:
                    break
                for nb in sorted(adj[curr]):
                    if nb == t or (nb not in blocked and nb not in parent):
                        if nb in parent:
                            continue
                        parent[nb] = curr
                        if nb == t:
                            break
                        q.append(nb)
            if t not in parent:
                return None
            path = []
            node = t
            while node is not None:
                path.append(node)
                node = parent[node]
            path.reverse()
            return path

        def solve(ordering):
            chosen = [None] * len(ordering)
            backtracks = [0]

            def backtrack(idx, used_internal):
                if backtracks[0] > max_backtrack:
                    return False
                if idx == len(ordering):
                    return True
                u, v = ordering[idx]
                blocked = used_internal | bs
                path = bfs_path(u, v, blocked)
                if path is None:
                    return False
                internal = set(path[1:-1])
                if internal & used_internal or internal & bs:
                    return False
                chosen[idx] = path
                for vn in internal:
                    used_internal.add(vn)
                if backtrack(idx + 1, used_internal):
                    return True
                for vn in internal:
                    used_internal.discard(vn)
                chosen[idx] = None
                backtracks[0] += 1
                return False

            if backtrack(0, set()):
                std_pairs = list(combinations(sorted_bs, 2))
                path_map = {}
                for i, pair in enumerate(ordering):
                    path_map[pair] = chosen[i]
                result = []
                for p in std_pairs:
                    if p in path_map:
                        result.append(path_map[p])
                    else:
                        return None
                return result
            return None

        # Ordering 1: most constrained first
        def pair_tightness(p):
            u, v = p
            common = (adj[u] & adj[v]) - bs
            return len(common)

        orderings = [
            sorted(pairs, key=pair_tightness),
            sorted(pairs, key=lambda p: -pair_tightness(p)),
            list(pairs),
        ]
        # Add a couple random orderings
        for _ in range(2):
            sh = list(pairs)
            rng.shuffle(sh)
            orderings.append(sh)

        for ordering in orderings:
            if time_up():
                break
            res = solve(ordering)
            if res is not None:
                return res
        return None

    # Bron-Kerbosch to find maximal cliques
    def bron_kerbosch(R, P_set, X, result, time_limit):
        if time.time() > time_limit:
            return
        if not P_set and not X:
            if len(R) > best_size:
                result.append(list(R))
            return
        # Choose pivot from P_set union X
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

    # Find all maximal cliques
    cliques = []
    initial_P = set(range(n))
    bron_kerbosch(set(), initial_P, set(), cliques, time.time() + 10.0)

    # Sort cliques by size descending
    cliques.sort(key=len, reverse=True)

    # Try each clique as a candidate branch set
    for clique in cliques:
        if time_up():
            break
        if len(clique) <= best_size:
            break
        paths = find_paths(clique)
        if paths is not None:
            best = {'branches': [int(x) for x in sorted(clique)], 'paths': [[int(x) for x in p] for p in paths]}
            best_size = len(clique)

    # Now try to extend the best clique by adding vertices
    if best_size < n and not time_up():
        current_branches = set(best['branches'])
        # Score non-branch vertices by how many current branches they connect to
        def score_vertex(v):
            return sum(1 for b in current_branches if v in adj[b])

        candidates = [v for v in range(n) if v not in current_branches]
        candidates.sort(key=score_vertex, reverse=True)

        # Try adding one vertex at a time, greedily
        for _ in range(10):  # up to 10 additions
            if time_up():
                break
            improved = False
            for v in candidates:
                if time_up():
                    break
                new_set = current_branches | {v}
                paths = find_paths(new_set)
                if paths is not None:
                    best = {'branches': [int(x) for x in sorted(new_set)], 'paths': [[int(x) for x in p] for p in paths]}
                    best_size = len(new_set)
                    current_branches = new_set
                    candidates = [x for x in range(n) if x not in current_branches]
                    candidates.sort(key=score_vertex, reverse=True)
                    improved = True
                    break
            if not improved:
                break

    # Try random subsets larger than best_size
    if not time_up():
        deg_sorted = sorted(range(n), key=lambda x: len(adj[x]), reverse=True)
        for k in range(best_size + 1, min(best_size + 5, n + 1)):
            if time_up():
                break
            for _ in range(50):
                if time_up():
                    break
                subset = random.sample(deg_sorted[:min(18, n)], k)
                paths = find_paths(subset)
                if paths is not None:
                    best = {'branches': [int(x) for x in sorted(subset)], 'paths': [[int(x) for x in p] for p in paths]}
                    best_size = k
                    break

    return best
