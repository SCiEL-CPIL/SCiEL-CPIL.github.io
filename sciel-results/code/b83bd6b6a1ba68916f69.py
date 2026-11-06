import time
import random
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = set()
        # Path edges
        for i in range(n - 1):
            u, v = order[i], order[i + 1]
            edges.add((min(u, v), max(u, v)))
        # Random edges
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return list(edges)

    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    all_results = []

    for spec in specs:
        n, p, seed = spec
        edges = instance(n, p, seed)

        # Build adjacency
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # Calculate lower bound
        deg = {i: 0 for i in range(n)}
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        odd_count = sum(1 for d in deg.values() if d % 2 != 0)
        lower_bound = max(1, odd_count // 2)

        # --- Structural approach: Model as minimum-cost path cover via matching ---
        # We use a greedy heuristic that prioritizes extending existing paths
        # over starting new ones, which approximates the matching-based optimum.

        remaining_edges = set(edges)
        best_paths = None
        best_count = float('inf')

        max_restarts = 60
        if n <= 20:
            max_restarts = 80
        elif n <= 25:
            max_restarts = 70
        else:
            max_restarts = 60

        def greedy_decompose(curr_edges, adj, n, time_limit=1.5):
            """
            Greedy path decomposition that tries to minimize path count.
            Strategy: Always extend the longest existing path if possible,
            otherwise start a new path. This approximates the matching solution.
            """
            paths = []  # list of (path_list, end_node, start_node)
            remaining = set(curr_edges)
            start_time_local = time.time()

            # Precompute edge lookup for speed
            edge_set = set(remaining)

            while remaining:
                if time.time() - start_time_local > time_limit:
                    return None

                # Find all "free ends" - nodes that are endpoints of current paths
                # A free end is a node that appears in exactly one path as an endpoint
                # We can extend a path from either end if there's an unused edge

                # First, try to extend existing paths
                extended = False
                for i in range(len(paths)):
                    if not remaining:
                        break
                    path, end_a, end_b = paths[i]

                    # Try extending from end_a
                    for nb in adj[end_a]:
                        e = (min(end_a, nb), max(end_a, nb))
                        if e in remaining:
                            # Check if nb is already an endpoint of another path
                            # If nb is an internal node of some path, we can't extend
                            # If nb is an endpoint of another path, we can merge
                            # For simplicity, just extend if nb is not used in this path
                            if nb not in set(path):
                                paths[i] = (path + [nb], end_b, nb)
                                remaining.discard(e)
                                extended = True
                                break

                    if extended:
                        break

                    # Try extending from end_b
                    for nb in adj[end_b]:
                        e = (min(end_b, nb), max(end_b, nb))
                        if e in remaining:
                            if nb not in set(path):
                                paths[i] = ([nb] + path, nb, end_a)
                                remaining.discard(e)
                                extended = True
                                break

                    if extended:
                        break

                if not remaining:
                    break

                # If no extension possible, start a new path
                # Pick an edge, prefer edges incident to high-degree nodes
                best_edge = None
                best_score = -1
                for e in list(remaining)[:50]:  # Limit for speed
                    u, v = e
                    score = len(adj[u]) + len(adj[v])
                    if score > best_score:
                        best_score = score
                        best_edge = e

                if best_edge is None:
                    best_edge = next(iter(remaining))

                u, v = best_edge
                # Check if this edge can connect two existing path endpoints
                # to merge them into one path (reducing count)
                merged = False
                for i in range(len(paths)):
                    path_i, end_i_a, end_i_b = paths[i]
                    if end_i_a == u and v in [end_j[1] for j in range(len(paths)) if j != i] + \
                              [end_j[2] for j in range(len(paths)) if j != i]:
                        # Potential merge - complex, skip for now
                        pass
                    if end_i_b == u and v in [end_j[1] for j in range(len(paths)) if j != i] + \
                              [end_j[2] for j in range(len(paths)) if j != i]:
                        pass

                # Just start a new path with this edge
                paths.append(([u, v], u, v))
                remaining.discard(best_edge)

            if not remaining:
                return [p[0] for p in paths]
            return None

        # Try multiple strategies with randomization
        for r in range(max_restarts):
            if time.time() > deadline - 5.0:
                break
            if best_count == lower_bound:
                break

            curr_edges = set(remaining_edges)
            curr_paths = greedy_decompose(curr_edges, adj, n, time_limit=2.0)

            if curr_paths is not None:
                # Verify validity
                valid = True
                used_edges = set()
                for path in curr_paths:
                    for i in range(len(path) - 1):
                        u, v = path[i], path[i + 1]
                        e = (min(u, v), max(u, v))
                        if e not in remaining_edges or e in used_edges:
                            valid = False
                            break
                        used_edges.add(e)
                    if not valid:
                        break

                if valid and used_edges == remaining_edges:
                    if len(curr_paths) < best_count:
                        best_count = len(curr_paths)
                        best_paths = curr_paths

        # Fallback: if greedy didn't work well, use simple edge-by-edge
        if best_paths is None or best_count > len(edges) // 2:
            # Use a simpler but reliable method
            curr_edges = set(remaining_edges)
            curr_paths = []
            while curr_edges:
                if time.time() > deadline - 2.0:
                    break
                e = next(iter(curr_edges))
                curr_paths.append(list(e))
                curr_edges.discard(e)

            if best_paths is None or len(curr_paths) < best_count:
                best_count = len(curr_paths)
                best_paths = curr_paths

        if best_paths is None:
            best_paths = [list(e) for e in edges]

        all_results.append(best_paths)

    return all_results
