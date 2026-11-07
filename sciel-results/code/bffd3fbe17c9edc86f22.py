import time
import random
import itertools
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin (limit 90s)

    n = P['n']
    # Path edges
    base_edges = set()
    for e in P['edges']:
        u, v = e
        base_edges.add((min(u,v), max(u,v)))

    # Build base adjacency (list of lists for speed)
    base_adj = [set() for _ in range(n)]
    for u, v in base_edges:
        base_adj[u].add(v)
        base_adj[v].add(u)

    # Precompute all valid "long" edges: even-odd pairs (not adjacent on path)
    # Triangle-free constraint: adding (u,v) to a set S is valid iff
    #   no w exists such that w is adjacent to both u and v in (base + S)
    # Since the graph is bipartite (even/odd), triangles can only form via
    # a path u-w-v where w has the opposite parity to u,v.
    # For even u and odd v: common neighbors must be odd (adjacent to u) AND even (adjacent to v).
    # Wait: u is even, its neighbors are odd. v is odd, its neighbors are even.
    # A common neighbor w would need to be both odd (neighbor of even u) and even (neighbor of odd v).
    # That's impossible! So NO triangle can be formed by adding a single even-odd edge to a bipartite graph.
    #
    # But wait - if we add multiple edges, we need to check:
    # Adding edge (u,v) where u even, v odd.
    # Triangle u-w-v requires w adjacent to u and w adjacent to v.
    # w adjacent to u: w is odd (since u is even in bipartite graph)
    # w adjacent to v: w is even (since v is odd in bipartite graph)
    # Contradiction. So any set of even-odd edges added to a path remains triangle-free!
    #
    # This means the triangle-free constraint is automatically satisfied for any subset of even-odd non-path edges.
    # We only need to minimize the number of such edges to achieve diameter <= 4.

    # Generate all candidate even-odd edges (excluding path edges)
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) in base_edges:
                continue
            # Must be even-odd pair
            if (u % 2 == v % 2):
                continue
            # Normalize: smaller first
            if u > v:
                u, v = v, u
            candidates.append((u, v))

    # Sort candidates by "usefulness" - edges that connect far-apart vertices first
    # Distance on path = |u - v|
    candidates.sort(key=lambda e: -(abs(e[0] - e[1])))

    def build_adj(added_indices):
        """Build adjacency list from base + selected candidate edges."""
        adj = [list(s) for s in base_adj]
        for idx in added_indices:
            u, v = candidates[idx]
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def bfs_diameter(adj):
        """Compute diameter of graph. Returns -1 if disconnected."""
        n_local = len(adj)
        max_dist = 0
        for s in range(n_local):
            dist = [-1] * n_local
            dist[s] = 0
            q = deque([s])
            while q:
                node = q.popleft()
                for nb in adj[node]:
                    if dist[nb] == -1:
                        dist[nb] = dist[node] + 1
                        if dist[nb] > max_dist:
                            max_dist = dist[nb]
                        q.append(nb)
            if any(d == -1 for d in dist):
                return -1  # disconnected
        return max_dist

    def diameter_le_4_fast(adj):
        """Check if diameter <= 4 using early termination BFS."""
        n_local = len(adj)
        for s in range(n_local):
            dist = [-1] * n_local
            dist[s] = 0
            q = deque([s])
            count = 1
            while q:
                node = q.popleft()
                if dist[node] == 4:
                    continue
                for nb in adj[node]:
                    if dist[nb] == -1:
                        dist[nb] = dist[node] + 1
                        count += 1
                        q.append(nb)
            if count < n_local:
                return False
        return True

    # Strategy: Try small subsets of candidates, starting from size 0, 1, 2, ...
    # Use a smart ordering and pruning.

    best_solution = None

    # First, try to find a solution with very few edges
    # The path has diameter 23. We need to reduce it to <= 4.
    # Key insight: we need edges that "shortcut" the path significantly.

    # Try greedy first to get an upper bound
    def greedy_solve():
        adj = [list(s) for s in base_adj]
        added = []
        added_set = set()
        while time.time() < deadline:
            if diameter_le_4_fast(adj):
                break
            best_idx = -1
            best_diam = 999
            for i, (u, v) in enumerate(candidates):
                if i in added_set:
                    continue
                # Add edge
                adj[u].append(v)
                adj[v].append(u)
                # Quick check: is it still connected and diameter improving?
                # Just check diameter <= 4
                if diameter_le_4_fast(adj):
                    d = 4
                else:
                    d = bfs_diameter(adj)
                    if d == -1:
                        d = 999
                # Remove edge
                adj[u].pop()
                adj[v].pop()
                if d < best_diam:
                    best_diam = d
                    best_idx = i
            if best_idx == -1:
                break
            u, v = candidates[best_idx]
            adj[u].append(v)
            adj[v].append(u)
            added.append(best_idx)
            added_set.add(best_idx)
        if diameter_le_4_fast(adj):
            return added
        return None

    greedy_result = greedy_solve()
    if greedy_result is not None:
        best_solution = greedy_result

    # Now try to find smaller solutions using systematic search
    # For each k from 1 to len(best_solution) if best exists, else up to some limit
    max_k = len(best_solution) if best_solution is not None else 15

    # Pre-filter: only consider "long" edges that could help reduce diameter
    # An edge (u,v) helps if |u-v| is large
    # For diameter 4 on 24 vertices, we need edges spanning at least ~5-6 positions

    # Use a more targeted search: focus on edges that connect distant parts
    # Partition candidates into "very useful" (span >= 10), "useful" (span >= 6), etc.
    very_useful = [i for i, (u,v) in enumerate(candidates) if abs(u-v) >= 12]
    useful = [i for i, (u,v) in enumerate(candidates) if 8 <= abs(u-v) < 12]
    moderate = [i for i, (u,v) in enumerate(candidates) if 5 <= abs(u-v) < 8]

    # Try combinations from most useful first
    all_priority = very_useful + useful + moderate

    for k in range(1, min(max_k, 10) + 1):
        if time.time() > deadline - 5:
            break
        if best_solution is not None and k >= len(best_solution):
            break

        # Try combinations from the priority list
        # For small k, enumerate; for larger k, use randomized search
        if k <= 4:
            for combo in itertools.combinations(all_priority, k):
                if time.time() > deadline - 5:
                    break
                adj = build_adj(combo)
                if diameter_le_4_fast(adj):
                    best_solution = list(combo)
                    break
        else:
            # Randomized search for larger k
            random.seed(42 + k)
            for _ in range(5000):
                if time.time() > deadline - 5:
                    break
                combo = random.sample(all_priority, k)
                adj = build_adj(combo)
                if diameter_le_4_fast(adj):
                    best_solution = combo
                    break

    # Final verification
    if best_solution is not None:
        adj = build_adj(best_solution)
        if not diameter_le_4_fast(adj):
            best_solution = None

    if best_solution is None:
        # Fallback: return empty (shouldn't happen, but safety)
        return []

    return [[int(candidates[i][0]), int(candidates[i][1])] for i in sorted(best_solution)]
