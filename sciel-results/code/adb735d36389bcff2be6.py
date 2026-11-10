"""Path decomposition for Gallai's conjecture instances."""
import time
import random
import sys

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    from random import Random

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose_greedy(edges_list):
        """Greedy edge-disjoint path decomposition."""
        edges_set = set(edges_list)
        if not edges_set:
            return []

        # Build adjacency
        adj = {}
        for u, v in edges_set:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)

        remaining = set(edges_set)
        paths = []

        # Try to build long paths greedily
        while remaining:
            # Pick a random starting edge
            edge = next(iter(remaining))
            u, v = edge
            remaining.remove(edge)

            # Extend path from both ends
            path = [u, v]
            current_end = v
            other_end = u

            # Extend from the right end
            while True:
                extended = False
                for w in sorted(adj.get(current_end, set())):
                    e = tuple(sorted((current_end, w)))
                    if e in remaining:
                        remaining.remove(e)
                        path.append(w)
                        current_end = w
                        extended = True
                        break
                if not extended:
                    break

            # Extend from the left end
            while True:
                extended = False
                for w in sorted(adj.get(other_end, set())):
                    e = tuple(sorted((other_end, w)))
                    if e in remaining:
                        remaining.remove(e)
                        path.insert(0, w)
                        other_end = w
                        extended = True
                        break
                if not extended:
                    break

            paths.append(path)

        return paths

    def merge_paths(paths, edges_set):
        """Try to merge adjacent paths to reduce count."""
        if len(paths) <= 1:
            return paths

        # Build edge lookup: for each edge, which path contains it
        # and its position in the path
        path_edges = []  # list of lists of (edge, index_in_path)
        for p in paths:
            pe = []
            for i in range(len(p) - 1):
                e = tuple(sorted((p[i], p[i + 1])))
                pe.append((e, i))
            path_edges.append(pe)

        # For merging: path A ending at vertex x, path B starting at vertex x
        # We can merge if the last vertex of A == first vertex of B (or vice versa)
        # and they share no edges (already guaranteed by edge-disjointness)
        # Actually for simple paths, we need: last vertex of path1 == first vertex of path2
        # Then merged path = path1 + path2[1:]

        # Check if merge produces a simple path (no repeated vertices)
        def try_merge(p1, p2):
            """Try to merge p1 and p2. Returns merged path or None."""
            # p1 ends at p1[-1], p2 starts at p2[0]
            if p1[-1] == p2[0]:
                merged = p1 + p2[1:]
                if len(set(merged)) == len(merged):
                    return merged
            # p2 ends at p2[-1], p1 starts at p1[0]
            if p2[-1] == p1[0]:
                merged = p2 + p1[1:]
                if len(set(merged)) == len(merged):
                    return merged
            return None

        changed = True
        while changed:
            changed = False
            new_paths = []
            used = [False] * len(paths)
            for i in range(len(paths)):
                if used[i]:
                    continue
                for j in range(i + 1, len(paths)):
                    if used[j]:
                        continue
                    merged = try_merge(paths[i], paths[j])
                    if merged is not None:
                        new_paths.append(merged)
                        used[i] = True
                        used[j] = True
                        changed = True
                        break
                if not used[i]:
                    new_paths.append(paths[i])
            paths = new_paths

        return paths

    def decompose_with_local_search(edges_list, max_time=30.0):
        """Decompose edges into paths, then try local search to improve."""
        edges_set = set(edges_list)
        if not edges_set:
            return []

        # Initial greedy decomposition
        paths = decompose_greedy(list(edges_set))
        # Verify all edges covered
        covered = set()
        for p in paths:
            for i in range(len(p) - 1):
                covered.add(tuple(sorted((p[i], p[i + 1]))))
        if covered != edges_set:
            # Fallback: each edge as its own path
            return [[list(e)] for e in edges_list]

        # Merge pass
        paths = merge_paths(paths, edges_set)

        # Local search: try to swap segments between paths
        deadline = time.time() + max_time
        while time.time() < deadline and len(paths) > 1:
            improved = False
            # Try merging any pair
            for i in range(len(paths)):
                for j in range(i + 1, len(paths)):
                    merged = try_merge(paths[i], paths[j])
                    if merged is not None:
                        paths.pop(j)
                        paths[i] = merged
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break

        return paths

    results = []
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))

    for spec in specs:
        if time.time() > DEADLINE:
            # Emergency: return trivial solution for remaining
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue

        edges = instance(*spec)
        time_per_graph = min(40.0, max(5.0, DEADLINE - (time.time() - start_time) - 5.0))
        paths = decompose_with_local_search(edges, max_time=time_per_graph)

        # Validate
        edges_set = set(edges)
        covered = set()
        valid = True
        for p in paths:
            # Check simple path
            if len(set(p)) != len(p):
                valid = False
                break
            for i in range(len(p) - 1):
                e = tuple(sorted((p[i], p[i + 1])))
                if e not in edges_set:
                    valid = False
                    break
                if e in covered:
                    valid = False
                    break
                covered.add(e)
            if not valid:
                break

        if not valid or covered != edges_set:
            # Fallback
            paths = [list(e) for e in edges]

        results.append(paths)

    return results
