"""Maximum pinned-graph clique subdivision via path-conflict max-clique with branch-and-bound."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import deque
from itertools import combinations

sys.setrecursionlimit(200000)

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    t0 = time.time()
    DEADLINE = t0 + 58.0
    n = P['n']
    adj = [set() for _ in range(n)]
    adj_list = [[] for _ in range(n)]
    for a, b in P['edges']:
        adj[a].add(b)
        adj[b].add(a)
        adj_list[a].append(b)
        adj_list[b].append(a)

    # --- Max clique via Bron-Kerbosch (trivial subdivision) ---
    best_clique = []
    def bron_kerbosch(R, P_set, X):
        nonlocal best_clique
        if not P_set and not X:
            if len(R) > len(best_clique):
                best_clique = list(R)
            return
        if P_set:
            pivot = max(P_set, key=lambda v: len(adj[v] & P_set))
            candidates = P_set - adj[pivot]
        else:
            candidates = P_set
        for v in list(candidates):
            bron_kerbosch(R | {v}, P_set & adj[v], X & adj[v])
            P_set.remove(v)
            X.add(v)
    bron_kerbosch(set(), set(range(n)), set())
    max_clique_size = len(best_clique)

    # --- All-pairs shortest path for fallback ---
    INF = float('inf')
    dist = [[INF]*n for _ in range(n)]
    pred = [[-1]*n for _ in range(n)]
    for start in range(n):
        dist[start][start] = 0
        pred[start][start] = -1
        q = deque([start])
        while q:
            u = q.popleft()
            for v in adj_list[u]:
                if dist[start][v] == INF:
                    dist[start][v] = dist[start][u] + 1
                    pred[start][v] = u
                    q.append(v)

    def get_shortest_path(start, end):
        if dist[start][end] == INF:
            return None
        path = []
        cur = end
        while cur != -1:
            path.append(cur)
            cur = pred[start][cur]
        path.reverse()
        return path

    # --- Fallback: trivial clique ---
    def make_trivial(clique):
        if len(clique) < 2:
            clique = [int(P['edges'][0][0]), int(P['edges'][0][1])]
        branches = [int(b) for b in clique]
        paths = []
        for i in range(len(clique)):
            for j in range(i + 1, len(clique)):
                sp = get_shortest_path(clique[i], clique[j])
                if sp:
                    paths.append([int(x) for x in sp])
                else:
                    paths.append([int(clique[i]), int(clique[j])])
        return {'branches': branches, 'paths': paths}

    # --- Path enumeration for a pair (u,v) avoiding branch set B ---
    def enumerate_paths(u, v, B_set, max_len=4):
        """All simple paths from u to v with interior vertices not in B_set, up to max_len edges."""
        if u == v:
            return []
        if v not in adj[u]:
            # direct edge: path is [u, v]
            if u not in B_set and v not in B_set:
                pass
            return [[u, v]] if dist[u][v] == 1 else []
        paths = []
        visited = {u}
        def dfs(curr, path):
            if len(path) - 1 > max_len:
                return
            if curr == v:
                # check interior not in B_set
                interior = path[1:-1]
                if not any(x in B_set for x in interior):
                    paths.append(list(path))
                return
            for nxt in adj_list[curr]:
                if nxt in visited:
                    continue
                if nxt in B_set and nxt != v:
                    continue
                visited.add(nxt)
                path.append(nxt)
                dfs(nxt, path)
                path.pop()
                visited.remove(nxt)
        dfs(u, [u])
        return paths

    # --- Check feasibility of a branch set using path-conflict max-clique ---
    def check_feasible(B, time_limit=3.0):
        """Return dict {(i,j): path} for all pairs, or None."""
        start_t = time.time()
        B_list = list(B)
        B_set = set(B)
        k = len(B_list)
        pairs = list(combinations(range(k), 2))
        if not pairs:
            return {}

        # For each pair, enumerate candidate paths
        pair_paths = {}
        for (i, j) in pairs:
            u, v = B_list[i], B_list[j]
            cands = enumerate_paths(u, v, B_set, max_len=4)
            if not cands:
                return None
            pair_paths[(i, j)] = cands

        # Build conflict graph: nodes are (pair_idx, path_idx)
        # Two nodes are compatible if their paths are internally disjoint
        node_pairs = []  # (pair_idx, path_idx)
        node_interior = []  # frozenset of interior vertices
        node_path = []  # the actual path
        for pi, (i, j) in enumerate(pairs):
            for pidx, path in enumerate(pair_paths[(i, j)]):
                interior = frozenset(path[1:-1])
                node_pairs.append((pi, pidx))
                node_interior.append(interior)
                node_path.append(path)

        num_nodes = len(node_pairs)
        if num_nodes == 0:
            return None

        # Precompute compatibility: compatible[a][b] = True if interiors disjoint
        # Also need: for each pair, at least one path must be chosen
        # Build a "conflict" graph where edge means INCOMPATIBLE
        # We want a max independent set in the conflict graph that covers all pairs
        # Equivalently: max clique in the compatibility graph with the constraint of covering all pairs

        # For efficiency with small num_nodes, use branch and bound on compatibility
        # compatible[a][b] = interiors are disjoint
        compatible = [set() for _ in range(num_nodes)]
        for a in range(num_nodes):
            for b in range(a + 1, num_nodes):
                if node_interior[a] & node_interior[b]:
                    continue
                # Also check they don't share the same pair (we'll handle via constraint)
                compatible[a].add(b)
                compatible[b].add(a)

        # Branch and bound: find a clique in compatibility graph that includes at least one node per pair
        best_size = 0
        best_clique_nodes = []
        # Greedy lower bound
        greedy = set()
        for a in range(num_nodes):
            if all(node_interior[a] & node_interior[b] == frozenset() for b in greedy):
                greedy.add(a)
        # Check if greedy covers all pairs
        greedy_pairs = set(node_pairs[a][0] for a in greedy)
        if len(greedy_pairs) == len(pairs):
            best_size = len(greedy)
            best_clique_nodes = list(greedy)

        # Branch and bound
        def bb(candidates, current, current_pairs):
            nonlocal best_size, best_clique_nodes
            if time.time() - start_t > time_limit:
                return
            # Upper bound: current size + number of distinct pairs still coverable
            remaining_pairs = set(range(len(pairs))) - current_pairs
            ub = len(current) + len(remaining_pairs)
            if ub <= best_size:
                return
            if not candidates:
                if len(current_pairs) == len(pairs) and len(current) > best_size:
                    best_size = len(current)
                    best_clique_nodes = list(current)
                return
            # Pick a node
            v = candidates[0]
            rest = candidates[1:]
            vi_pair = node_pairs[v][0]
            # Include v
            new_candidates = [w for w in rest if w in compatible[v]]
            new_current = current + [v]
            new_pairs = current_pairs | {vi_pair}
            bb(new_candidates, new_current, new_pairs)
            # Exclude v
            bb(rest, current, current_pairs)

        all_nodes = list(range(num_nodes))
        bb(all_nodes, [], set())

        if len(best_clique_nodes) == 0:
            return None
        # Verify coverage
        chosen_pairs = set(node_pairs[a][0] for a in best_clique_nodes)
        if len(chosen_pairs) != len(pairs):
            return None
        # Build result
        result = {}
        for a in best_clique_nodes:
            pi, pidx = node_pairs[a]
            i, j = pairs[pi]
            result[(i, j)] = node_path[a]
        return result

    # --- Main search ---
    # Start from max clique and try to extend
    current_B = list(best_clique)
    ok, paths_dict = check_feasible(current_B, time_limit=5.0)
    best_result = None
    if ok:
        best_result = make_from_dict(current_B, paths_dict)

    # Try to find larger feasible sets
    # Strategy: iteratively add vertices, using the path-conflict formulation
    # For k from max_clique_size+1 to n, try random/greedy subsets
    for k in range(max_clique_size + 1, min(n, 12) + 1):
        if time.time() > DEADLINE:
            break
        # Try several random subsets of size k
        for attempt in range(30):
            if time.time() > DEADLINE:
                break
            # Greedy: start from a clique, add random vertices
            base = list(best_clique)
            others = [v for v in range(n) if v not in set(base)]
            random.shuffle(others)
            candidate = base + others[:k - len(base)]
            if len(candidate) != k:
                candidate = others[:k]
            if len(candidate) < 2:
                continue
            res = check_feasible(candidate, time_limit=2.0)
            if res is not None:
                best_result = make_from_dict(candidate, res)
                break
        if best_result is not None and len(best_result['branches']) >= k:
            break

    if best_result is not None:
        return best_result

    # Final fallback
    return make_trivial(best_clique)

def make_from_dict(B, paths_dict):
    branches = [int(b) for b in B]
    paths = []
    for (i, j), seq in sorted(paths_dict.items()):
        paths.append([int(x) for x in seq])
    return {'branches': branches, 'paths': paths}
# EVOLVE-BLOCK-END
