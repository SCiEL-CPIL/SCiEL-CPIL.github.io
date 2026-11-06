import time
import random
import sys

def construct():
    start_time = time.time()
    DEADLINE = 125

    def elapsed():
        return time.time() - start_time

    def deadline_exceeded():
        return elapsed() > DEADLINE

    def build_adj(n, edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def has_triangle(n, adj):
        for u in range(n):
            au = adj[u]
            v_mask = au & ~((1 << (u + 1)) - 1)
            while v_mask:
                v = (v_mask & -v_mask).bit_length() - 1
                v_mask &= v_mask - 1
                common = au & adj[v]
                if common:
                    return True
        return False

    def find_independent_set(n, adj, limit):
        """Find an independent set of size >= limit, or return None."""
        result = []

        def dfs(candidates, size):
            if deadline_exceeded():
                return False
            if size >= limit:
                return True
            if not candidates:
                return False
            if size + candidates.bit_count() < limit:
                return False

            v = (candidates & -candidates).bit_length() - 1
            candidates_without_v = candidates & ~(1 << v)

            # Try including v
            new_cands = candidates_without_v & ~adj[v]
            result.append(v)
            if dfs(new_cands, size + 1):
                return True
            result.pop()

            # Try excluding v
            if dfs(candidates_without_v, size):
                return True

            return False

        if dfs((1 << n) - 1, 0):
            return result
        return None

    def is_valid(n, edges):
        if deadline_exceeded():
            return False
        adj = build_adj(n, edges)
        if has_triangle(n, adj):
            return False
        ind_set = find_independent_set(n, adj, 10)
        if ind_set is not None:
            return False
        return True

    def andrasfai_9():
        n = 26
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                diff = (j - i) % n
                if diff % 3 == 1:
                    edges.append([i, j])
        return n, edges

    # Start with the known valid baseline
    best_n, best_edges = andrasfai_9()
    if is_valid(best_n, best_edges):
        pass

    # Strategy: Build a maximal triangle-free graph on n vertices using
    # a greedy/branch approach, then try to minimize alpha by removing vertices.
    # We use the fact that a maximal triangle-free graph can be built by
    # ensuring every non-edge is part of no triangle (i.e., for each non-edge (u,v),
    # N(u) and N(v) are disjoint).

    def build_maximal_triangle_free(n, seed=42):
        """Build a maximal triangle-free graph on n vertices."""
        rng = random.Random(seed)
        adj = [0] * n
        edges = []

        # Initialize with a random triangle-free graph
        # Use a structured approach: partition vertices and connect across partitions
        # in a way that avoids triangles

        # Simple approach: random edges, rejecting those that form triangles
        order = list(range(n))
        rng.shuffle(order)

        for idx_u in range(n - 1):
            u = order[idx_u]
            for idx_v in range(idx_u + 1, n):
                v = order[idx_v]
                common = adj[u] & adj[v]
                if not common:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edges.append([u, v])

        # Now make it maximal: for each non-edge (u,v), if N(u) and N(v) are disjoint,
        # we can add the edge. But we want to ADD edges to make it maximal.
        # Actually, a graph is maximal triangle-free if every non-edge (u,v) has
        # N(u) ∩ N(v) ≠ ∅ (otherwise we could add edge (u,v) without creating a triangle).
        # So we need to add edges where possible.

        improved = True
        while improved and not deadline_exceeded():
            improved = False
            for u in range(n):
                if deadline_exceeded():
                    return None
                for v in range(u + 1, n):
                    if (adj[u] >> v) & 1:
                        continue
                    common = adj[u] & adj[v]
                    if not common:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        edges.append([u, v])
                        improved = True

        return n, edges, adj

    # Try to find a valid graph on 40 vertices
    for target_n in [40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30]:
        if deadline_exceeded():
            break

        found = False
        for seed in range(200):
            if deadline_exceeded():
                break

            result = build_maximal_triangle_free(target_n, seed)
            if result is None:
                continue
            n, edges, adj = result

            # Check if this maximal triangle-free graph has alpha < 10
            ind_set = find_independent_set(target_n, adj, 10)
            if ind_set is None:
                # Valid!
                best_n = target_n
                best_edges = edges
                return {'n': best_n, 'edges': best_edges}

            # If alpha >= 10, try to remove vertices from the independent set
            # to break it, while maintaining triangle-freeness
            # This is complex; instead, try different seeds

            # Alternative: try to add edges to reduce alpha
            # But the graph is already maximal triangle-free, so we can't add edges.

            # Try a different approach: start with fewer edges, then add strategically
            # to reduce alpha while keeping triangle-free

            # For now, just try more seeds
            pass

        # If we couldn't find a valid maximal triangle-free graph,
        # try a different approach: start with a sparse graph and add edges
        # to reduce alpha

        for seed in range(100):
            if deadline_exceeded():
                break

            rng = random.Random(seed)
            adj = [0] * target_n
            edges = []

            # Start with a random sparse triangle-free graph
            for u in range(target_n):
                for v in range(u + 1, target_n):
                    if rng.random() < 0.2:
                        common = adj[u] & adj[v]
                        if not common:
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            edges.append([u, v])

            # Iteratively add edges to reduce alpha
            for iteration in range(50):
                if deadline_exceeded():
                    break

                # Find an independent set of size 10
                ind_set = find_independent_set(target_n, adj, 10)
                if ind_set is None:
                    # Valid!
                    best_n = target_n
                    best_edges = edges
                    return {'n': best_n, 'edges': best_edges}

                # Try to add an edge within the independent set that doesn't create a triangle
                added = False
                for i in range(len(ind_set)):
                    for j in range(i + 1, len(ind_set)):
                        u, v = ind_set[i], ind_set[j]
                        if (adj[u] >> v) & 1:
                            continue
                        common = adj[u] & adj[v]
                        if not common:
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            edges.append([u, v])
                            added = True
                            break
                    if added:
                        break

                if not added:
                    # Can't add any edge within the independent set
                    # Try random edge additions elsewhere
                    for _ in range(10):
                        u = rng.randint(0, target_n - 1)
                        v = rng.randint(0, target_n - 1)
                        if u == v or (adj[u] >> v) & 1:
                            continue
                        common = adj[u] & adj[v]
                        if not common:
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            edges.append([u, v])
                            break

    return {'n': best_n, 'edges': best_edges}
