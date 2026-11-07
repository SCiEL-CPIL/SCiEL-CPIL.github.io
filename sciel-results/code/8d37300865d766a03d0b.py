"""Rewritten program for triangle-free diameter-four augmentation."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    start_time = time.time()
    deadline = 88.0

    n = P['n']

    # Build initial adjacency from path
    base_adj = [[False] * n for _ in range(n)]
    for u, v in P['edges']:
        base_adj[u][v] = True
        base_adj[v][u] = True

    # Precompute all possible non-path edges
    all_non_path_edges = [(u, v) for u in range(n) for v in range(u + 1, n) if not base_adj[u][v]]

    def bfs_from(src, adj):
        """BFS from src, return distance array."""
        dist = [-1] * n
        dist[src] = 0
        queue = [src]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in range(n):
                if adj[u][v] and dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        return dist

    def compute_diameter(adj):
        """Compute diameter of graph."""
        max_diam = 0
        for src in range(n):
            dist = bfs_from(src, adj)
            for d in dist:
                if d > max_diam:
                    max_diam = d
        return max_diam

    def is_triangle_free_edge(adj, u, v):
        """Check if adding edge (u,v) keeps graph triangle-free."""
        for w in range(n):
            if w != u and w != v and adj[u][w] and adj[v][w]:
                return False
        return True

    def check_validity(added_edges):
        """Full validation."""
        edge_set = set()
        for u, v in added_edges:
            if u == v:
                return False
            a, b = (u, v) if u < v else (v, u)
            if (a, b) in edge_set:
                return False
            edge_set.add((a, b))
            if base_adj[a][b]:
                return False

        adj = [row[:] for row in base_adj]
        for u, v in added_edges:
            adj[u][v] = True
            adj[v][u] = True

        # Check triangle-free
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u][v]:
                    for w in range(v + 1, n):
                        if adj[u][w] and adj[v][w]:
                            return False

        return compute_diameter(adj) <= 4

    # Hardcoded 8-edge solution
    hardcoded = [[12, 1], [12, 3], [12, 6], [12, 9], [12, 15], [12, 18], [12, 21], [12, 23]]
    best_solution = hardcoded if check_validity(hardcoded) else None

    # Precompute triangle-free candidates relative to base graph
    # An edge (u,v) is triangle-free w.r.t. base if no w is adjacent to both u and v in base
    base_candidates = []
    for u, v in all_non_path_edges:
        tf = True
        for w in range(n):
            if w != u and w != v and base_adj[u][w] and base_adj[v][w]:
                tf = False
                break
        if tf:
            base_candidates.append((u, v))

    # For the path, neighbors of u are {u-1, u+1} (with bounds)
    # So (u,v) is triangle-free iff |u-v| >= 3 (no shared neighbor in path)
    # Let's verify: u and v share a neighbor w iff w is adjacent to both
    # In a path, adj(u) = {u-1, u+1} intersect valid range
    # So u and v share neighbor iff |u-v| == 2 (they share the middle vertex)
    # Thus triangle-free edges are those with |u-v| >= 3

    # Efficient greedy: for a given "coverage" function, pick edge with max gain
    def greedy_cover(coverage_fn, max_edges=None):
        """
        Greedy set cover. coverage_fn(adj) returns set of "covered" vertices.
        Returns list of added edges.
        """
        adj = [row[:] for row in base_adj]
        selected = []
        if max_edges is None:
            max_edges = 20

        for _ in range(max_edges):
            if time.time() > deadline - 5:
                break
            covered = coverage_fn(adj)
            if len(covered) == n:
                break

            best_edge = None
            best_gain = 0

            for u, v in base_candidates:
                if adj[u][v]:
                    continue
                # Check triangle-free with current adj
                if not is_triangle_free_edge(adj, u, v):
                    continue
                # Simulate
                adj[u][v] = True
                adj[v][u] = True
                new_covered = coverage_fn(adj)
                adj[u][v] = False
                adj[v][u] = False

                gain = len(new_covered - covered)
                if gain > best_gain:
                    best_gain = gain
                    best_edge = (u, v)

            if best_edge is None or best_gain == 0:
                break

            u, v = best_edge
            selected.append([u, v])
            adj[u][v] = True
            adj[v][u] = True

        # Check if all covered
        covered = coverage_fn(adj)
        if len(covered) == n:
            return selected
        return None

    # Coverage: all vertices within dist 2 of root
    def make_root_coverage(root):
        def cover(adj):
            dist = bfs_from(root, adj)
            return set(v for v in range(n) if dist[v] <= 2)
        return cover

    # Coverage: all vertices within dist 2 of root1 OR root2
    def make_two_root_coverage(r1, r2):
        def cover(adj):
            d1 = bfs_from(r1, adj)
            d2 = bfs_from(r2, adj)
            return set(v for v in range(n) if d1[v] <= 2 or d2[v] <= 2)
        return cover

    # Coverage: diameter <= 4 (all pairs within 4)
    # This is harder to express as a set cover. Use: for each vertex, it must be
    # within dist 2 of some "center". But we don't know centers in advance.
    # Alternative: all vertices within dist 2 of a chosen set of centers.
    # For now, use rooted and two-rooted approaches.

    # Try single-root greedy
    for root in range(8, 16):
        if time.time() > deadline - 15:
            break
        result = greedy_cover(make_root_coverage(root))
        if result is not None and check_validity(result):
            if best_solution is None or len(result) < len(best_solution):
                best_solution = [list(e) for e in result]

    # Try two-root greedy
    for r1 in range(6, 14):
        for r2 in range(r1 + 1, 18):
            if time.time() > deadline - 12:
                break
            result = greedy_cover(make_two_root_coverage(r1, r2))
            if result is not None and check_validity(result):
                if best_solution is None or len(result) < len(best_solution):
                    best_solution = [list(e) for e in result]

    # Branch-and-bound: try to find solutions with fewer edges than current best
    if best_solution is not None:
        target = len(best_solution)
        # Try to find a solution with target-1 edges
        for trial_size in range(max(1, target - 2), target):
            if time.time() > deadline - 8:
                break
            found = _branch_and_bound(trial_size, deadline)
            if found is not None and check_validity(found):
                best_solution = [list(e) for e in found]
                target = len(found)
                if target <= 1:
                    break

    # Edge removal polish
    if best_solution:
        current = [list(e) for e in best_solution]
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            for idx in range(len(current)):
                trial = current[:idx] + current[idx+1:]
                if check_validity(trial):
                    current = trial
                    improved = True
                    break
        if len(current) < len(best_solution):
            best_solution = current

    if best_solution is None:
        return []

    return [[int(u), int(v)] for u, v in best_solution]

def _branch_and_bound(max_edges, deadline):
    """Try to find a valid solution with at most max_edges edges using B&B."""
    n = P['n']
    base_adj = [[False] * n for _ in range(n)]
    for u, v in P['edges']:
        base_adj[u][v] = True
        base_adj[v][u] = True

    # Triangle-free candidates: |u-v| >= 3
    candidates = [(u, v) for u in range(n) for v in range(u + 1, n) if v - u >= 3]

    def bfs_from(src, adj):
        dist = [-1] * n
        dist[src] = 0
        queue = [src]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in range(n):
                if adj[u][v] and dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        return dist

    def is_tf_edge(adj, u, v):
        for w in range(n):
            if w != u and w != v and adj[u][w] and adj[v][w]:
                return False
        return True

    def check_diam4(adj):
        for src in range(n):
            dist = bfs_from(src, adj)
            for d in dist:
                if d > 4:
                    return False
        # Also check triangle-free
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u][v]:
                    for w in range(v + 1, n):
                        if adj[u][w] and adj[v][w]:
                            return False
        return True

    # Sort candidates by "centrality" - prefer edges connecting central to peripheral
    def centrality_score(u, v):
        # Prefer edges that reduce diameter: connect far-apart vertices
        # But also need triangle-free
        return abs(u - v)  # larger = more "spanning"

    candidates.sort(key=lambda e: centrality_score(e[0], e[1]), reverse=True)

    # Limit candidates for search
    top_cands = candidates[:80]

    adj = [row[:] for row in base_adj]
    selected = []

    def dfs(idx, remaining):
        if time.time() > deadline - 3:
            return None
        if remaining == 0:
            if check_diam4(adj):
                return list(selected)
            return None

        # Pruning: if even adding all remaining can't fix diameter, return None
        # (weak pruning, but helps)
        for i in range(idx, len(top_cands)):
            u, v = top_cands[i]
            if adj[u][v]:
                continue
            if not is_tf_edge(adj, u, v):
                continue

            adj[u][v] = True
            adj[v][u] = True
            selected.append([u, v])

            result = dfs(i + 1, remaining - 1)
            if result is not None:
                return result

            selected.pop()
            adj[u][v] = False
            adj[v][u] = False

        return None

    return dfs(0, max_edges)

# EVOLVE-BLOCK-END
