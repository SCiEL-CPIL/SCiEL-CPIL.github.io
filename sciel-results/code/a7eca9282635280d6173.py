import time
import sys
import random
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 82  # safety margin before 90s limit

    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)

    def time_ok():
        return time.time() < deadline

    # ---------- Max-flow feasibility check ----------
    def flow_feasible(branches):
        """Check if vertex-disjoint paths exist for all pairs of branches."""
        if not time_ok():
            return False
        bset = set(branches)
        k = len(branches)
        pairs = list(combinations(sorted(branches), 2))
        if not pairs:
            return True

        # Build flow network
        # Node indices: 0=source, 1=sink, then 2..(2n+1) for vertex splits
        # For vertex v: in-node = 2+v, out-node = 2+n+v
        SRC = 0
        SNK = 1
        def vin(v): return 2 + v
        def vout(v): return 2 + n + v
        num_nodes = 2 + 2 * n

        # Edge list for Dinic: adj_flow[u] = list of [v, cap, rev_index]
        adj_flow = [[] for _ in range(num_nodes)]
        def add_edge(u, v, cap):
            adj_flow[u].append([v, cap, len(adj_flow[v])])
            adj_flow[v].append([u, 0, len(adj_flow[u]) - 1])

        # Vertex-split edges: in -> out with capacity 1 for non-branch vertices
        for v in range(n):
            if v not in bset:
                add_edge(vin(v), vout(v), 1)
            else:
                # Branch vertices: unlimited capacity through them
                add_edge(vin(v), vout(v), k * k)  # large enough

        # For each pair, add a source->branch_in edge and branch_out->sink edge
        # Actually, we need to route flow for each pair. Let's use a different approach:
        # For each pair (u,v), we need a path from u to v. We can model this as:
        # source -> u_in (cap 1), v_out -> sink (cap 1), and check if max flow >= 1
        # But we need ALL pairs simultaneously with vertex-disjoint interiors.
        # Better: for each pair, add source->u_in and v_out->sink, and check total flow = num_pairs

        # Clear and rebuild
        adj_flow = [[] for _ in range(num_nodes)]
        def add_edge2(u, v, cap):
            adj_flow[u].append([v, cap, len(adj_flow[v])])
            adj_flow[v].append([u, 0, len(adj_flow[u]) - 1])

        # Vertex split edges
        for v in range(n):
            if v not in bset:
                add_edge2(vin(v), vout(v), 1)
            else:
                add_edge2(vin(v), vout(v), 10**6)

        # Graph edges: for each edge (a,b) in original graph, add a_out -> b_in and b_out -> a_in
        for a in range(n):
            for b in adj[a]:
                if a < b:
                    add_edge2(vout(a), vin(b), 10**6)
                    add_edge2(vout(b), vin(a), 10**6)

        # For each pair (u,v): source -> u_in, v_out -> sink
        # But this doesn't work directly because we need to distinguish which pair uses which path.
        # Alternative: check each pair independently won't work (interiors must be disjoint).
        # Correct approach: multi-commodity is hard. Instead, use a single flow where
        # each unit of flow represents one pair's path.
        # We add: for pair i=(u_i, v_i): SRC -> u_i_in (cap 1), v_i_out -> SNK (cap 1)
        # Then max flow should equal number of pairs.
        for (u, v) in pairs:
            add_edge2(SRC, vin(u), 1)
            add_edge2(vout(v), SNK, 1)

        # Dinic's algorithm
        def dinic():
            if not time_ok():
                return -1
            flow = 0
            INF = 10**9
            while time_ok():
                # BFS level graph
                level = [-1] * num_nodes
                level[SRC] = 0
                q = deque([SRC])
                while q:
                    cur = q.popleft()
                    for e in adj_flow[cur]:
                        if e[1] > 0 and level[e[0]] < 0:
                            level[e[0]] = level[cur] + 1
                            q.append(e[0])
                if level[SNK] < 0:
                    break
                # DFS
                it = [0] * num_nodes
                def dfs(v, f):
                    if v == SNK:
                        return f
                    while it[v] < len(adj_flow[v]):
                        e = adj_flow[v][it[v]]
                        if e[1] > 0 and level[v] < level[e[0]]:
                            d = dfs(e[0], min(f, e[1]))
                            if d > 0:
                                e[1] -= d
                                adj_flow[e[0]][e[2]][1] += d
                                return d
                        it[v] += 1
                    return 0
                while time_ok():
                    f = dfs(SRC, INF)
                    if f == 0:
                        break
                    flow += f
            return flow

        max_flow = dinic()
        return max_flow == len(pairs)

    # ---------- Path extraction using randomized BFS ----------
    def extract_paths(branches):
        """Extract vertex-disjoint paths for all pairs. Returns dict or None."""
        if not time_ok():
            return None
        bset = set(branches)
        k = len(branches)
        pairs = list(combinations(sorted(branches), 2))
        if not pairs:
            return {}

        # Order pairs: direct edges first, then by BFS distance
        def bfs_dist(src, dst, forbidden):
            if src == dst:
                return 0
            if dst in adj[src]:
                return 1
            visited = {src}
            queue = deque([(src, 0)])
            while queue:
                cur, d = queue.popleft()
                for nb in adj[cur]:
                    if nb not in visited and (nb == dst or nb not in forbidden):
                        if nb == dst:
                            return d + 1
                        visited.add(nb)
                        queue.append((nb, d + 1))
            return 10**9

        # Sort pairs by estimated difficulty
        def pair_key(p):
            u, v = p
            if v in adj[u]:
                return (0, 0, u, v)
            return (1, 0, u, v)
        pairs_sorted = sorted(pairs, key=pair_key)

        paths = {}
        used_interior = set()

        def find_path_bfs(src, dst, forbidden, rng):
            """Find a simple path from src to dst avoiding forbidden vertices."""
            if src == dst:
                return [src]
            if dst in adj[src]:
                return [src, dst]
            # BFS with random tie-breaking
            visited = {src}
            parent = {src: None}
            queue = deque([src])
            while queue:
                cur = queue.popleft()
                neighbors = [nb for nb in adj[cur] if nb not in visited and (nb == dst or nb not in forbidden)]
                rng.shuffle(neighbors)
                for nb in neighbors:
                    visited.add(nb)
                    parent[nb] = cur
                    if nb == dst:
                        path = []
                        node = dst
                        while node is not None:
                            path.append(node)
                            node = parent[node]
                        path.reverse()
                        return path
                    queue.append(nb)
            return None

        # Try multiple random orderings and path-finding strategies
        best_paths = None
        num_attempts = min(100, max(20, 300 // k))

        for attempt in range(num_attempts):
            if not time_ok():
                break
            rng = random.Random(attempt * 12345 + 7)
            # Shuffle pairs (but keep direct edges first for efficiency)
            direct_pairs = [p for p in pairs if p[1] in adj[p[0]]]
            non_direct_pairs = [p for p in pairs if p[1] not in adj[p[0]]]
            rng.shuffle(non_direct_pairs)
            order = direct_pairs + non_direct_pairs

            paths_try = {}
            used_try = set()
            success = True

            for u, v in order:
                if v in adj[u]:
                    paths_try[(u, v)] = [u, v]
                else:
                    forbidden = bset - {u, v}
                    forbidden |= used_try
                    path = find_path_bfs(u, v, forbidden, rng)
                    if path is None:
                        success = False
                        break
                    paths_try[(u, v)] = path
                    for mid in path[1:-1]:
                        used_try.add(mid)

            if success:
                best_paths = paths_try
                break

        return best_paths

    # ---------- Local search over branch sets ----------
    def local_search(initial_branches, max_iters=500):
        """Local search: add, remove, swap vertices to maximize branch count."""
        current = list(initial_branches)
        current_paths = extract_paths(current)
        if current_paths is None:
            return current, None

        best = list(current)
        best_paths = current_paths

        for iteration in range(max_iters):
            if not time_ok():
                break
            improved = False
            bset = set(current)

            # Try adding a vertex
            candidates = [v for v in range(n) if v not in bset]
            rng = random.Random(iteration * 7919 + 3)
            rng.shuffle(candidates)
            for v in candidates:
                if not time_ok():
                    break
                test = current + [v]
                test_paths = extract_paths(test)
                if test_paths is not None:
                    current = test
                    current_paths = test_paths
                    if len(current) > len(best):
                        best = list(current)
                        best_paths = test_paths
                    improved = True
                    break  # restart candidate list after improvement

            if not improved:
                # Try swapping: remove one, add one
                if len(current) >= 2:
                    for i in range(len(current)):
                        if not time_ok():
                            break
                        for v in [x for x in range(n) if x not in bset]:
                            if not time_ok():
                                break
                            test = [x for j, x in enumerate(current) if j != i] + [v]
                            test_paths = extract_paths(test)
                            if test_paths is not None:
                                current = test
                                current_paths = test_paths
                                if len(current) > len(best):
                                    best = list(current)
                                    best_paths = test_paths
                                improved = True
                                break
                        if improved:
                            break

            if not improved and len(current) > 2:
                # Try removing a vertex that's causing issues
                for i in range(len(current)):
                    if not time_ok():
                        break
                    test = [x for j, x in enumerate(current) if j != i]
                    test_paths = extract_paths(test)
                    if test_paths is not None:
                        # Only accept if we can then add something back
                        current = test
                        current_paths = test_paths
                        improved = True
                        break

            if not improved:
                break

        return best, best_paths

    # ---------- Main algorithm ----------
    # Start with a greedy construction
    # Sort vertices by degree (high degree first)
    deg = [len(adj[v]) for v in range(n)]
    vertices_by_deg = sorted(range(n), key=lambda v: -deg[v])

    current_branches = []
    for v in vertices_by_deg:
        if not time_ok():
            break
        test = current_branches + [v]
        if len(test) < 2:
            current_branches = test
            continue
        test_paths = extract_paths(test)
        if test_paths is not None:
            current_branches = test

    if len(current_branches) < 2:
        # Fallback: take first two vertices with an edge
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    current_branches = [u, v]
                    break
            if len(current_branches) >= 2:
                break

    # Local search
    best_branches, best_paths = local_search(current_branches, max_iters=300)

    # Try multiple random restarts
    for restart in range(20):
        if not time_ok():
            break
        rng = random.Random(restart * 104729 + 13)
        # Random initial set
        size = rng.randint(3, min(8, n))
        init = rng.sample(range(n), size)
        init_paths = extract_paths(init)
        if init_paths is not None:
            b, p = local_search(init, max_iters=100)
            if len(b) > len(best_branches):
                best_branches = b
                best_paths = p

    # Final extraction
    if best_paths is None:
        best_paths = extract_paths(best_branches)
    if best_paths is None:
        # Last resort: just use direct edges
        best_paths = {}
        for u, v in combinations(sorted(best_branches), 2):
            if v in adj[u]:
                best_paths[(u, v)] = [u, v]
            else:
                # Find any path
                forbidden = set(best_branches) - {u, v}
                path = find_path_bfs(u, v, forbidden, random.Random(42))
                if path:
                    best_paths[(u, v)] = path
                else:
                    best_paths[(u, v)] = [u, v]  # fallback (may be invalid)

    # Build result
    branches = [int(b) for b in sorted(best_branches)]
    paths_list = []
    for u, v in combinations(sorted(branches), 2):
        p = best_paths.get((u, v), [u, v])
        paths_list.append([int(x) for x in p])

    return {'branches': branches, 'paths': paths_list}
