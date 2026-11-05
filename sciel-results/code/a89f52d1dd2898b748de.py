import time
import random
from collections import defaultdict

def construct():
    """
    Decomposes the edges of three fixed instance graphs into the minimum number of
    edge-disjoint simple paths.

    Returns:
        A list of three lists of paths. Each path is a list of vertices.
    """
    # Deadline guard: 150s limit, leave 10s safety margin
    start_time = time.time()
    deadline = start_time + 140.0

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def _valid_solution(paths, edges_set):
        used = set()
        for path in paths:
            if len(path) < 2:
                return False
            if len(set(path)) != len(path):
                return False
            for i in range(len(path) - 1):
                e = tuple(sorted((path[i], path[i + 1])))
                if e not in edges_set:
                    return False
                if e in used:
                    return False
                used.add(e)
        return len(used) == len(edges_set)

    def _merge_polish(paths, edges_set, rng, max_iters=2000):
        # Fast local search: try to merge two paths using a single unused edge
        # that connects their endpoints and whose internal vertices are disjoint.
        # This is a safe move: it preserves simplicity, edge-disjointness, and coverage,
        # and strictly reduces the number of paths.
        #
        # We do this in a limited number of iterations and with randomized orderings
        # to escape local plateaus.
        paths = [list(p) for p in paths]
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            n_paths = len(paths)
            if n_paths <= 1:
                break

            # Build endpoint lookup for quick access
            # For each path, endpoints are paths[i][0] and paths[i][-1]
            # We'll try random pairs (i, j) and random orientation choices.

            # Precompute vertex set for each path for fast disjointness checks
            path_sets = [set(p) for p in paths]

            improved = False
            # Try a few random pair attempts per iteration to keep it cheap
            for _ in range(8):
                i = rng.randrange(n_paths)
                j = rng.randrange(n_paths)
                if i == j:
                    continue

                pi = paths[i]
                pj = paths[j]
                if len(pi) < 2 or len(pj) < 2:
                    continue

                # Try all 4 endpoint combinations: (i_front, j_front), (i_front, j_back),
                # (i_back, j_front), (i_back, j_back)
                candidates = []
                # i front + j front: edge (pi[0], pj[0])
                candidates.append((pi[0], pj[0], 'i_front', 'j_front'))
                # i front + j back: edge (pi[0], pj[-1])
                candidates.append((pi[0], pj[-1], 'i_front', 'j_back'))
                # i back + j front: edge (pi[-1], pj[0])
                candidates.append((pi[-1], pj[0], 'i_back', 'j_front'))
                # i back + j back: edge (pi[-1], pj[-1])
                candidates.append((pi[-1], pj[-1], 'i_back', 'j_back'))

                rng.shuffle(candidates)

                for (u, v, side_i, side_j) in candidates:
                    if u == v:
                        continue
                    e = tuple(sorted((u, v)))
                    if e not in edges_set:
                        continue

                    # The edge must not be currently used by either path (or any path).
                    # Since we maintain used edges implicitly by paths, we can check:
                    # - e is not an edge in pi
                    # - e is not an edge in pj
                    # For efficiency, we can check by scanning the two paths (they're short).
                    # But for correctness, we should ensure e is not used anywhere.
                    # Since the number of paths is small and paths are short, a full scan
                    # is acceptable for a polish pass.
                    used_e = False
                    for p in paths:
                        for k in range(len(p) - 1):
                            ee = tuple(sorted((p[k], p[k + 1])))
                            if ee == e:
                                used_e = True
                                break
                        if used_e:
                            break
                    if used_e:
                        continue

                    # Vertex disjointness: the two paths must share no vertices.
                    # This ensures the merged path is simple.
                    if path_sets[i] & path_sets[j]:
                        continue

                    # Now construct the merged path based on which endpoints are used.
                    # We need to orient each path so that the used endpoint is at the
                    # junction.
                    # For path i:
                    #   if side_i == 'i_front', we keep pi as is, and u = pi[0]
                    #   if side_i == 'i_back', we reverse pi, and u = pi[-1] (which becomes front)
                    # For path j:
                    #   if side_j == 'j_front', we keep pj as is, and v = pj[0]
                    #   if side_j == 'j_back', we reverse pj, and v = pj[-1] (which becomes front)

                    # However, the edge connects u (from i) to v (from j).
                    # The merged path will be: [oriented_i] + [oriented_j]
                    # but we must ensure that the last vertex of oriented_i is u,
                    # and the first vertex of oriented_j is v, and u-v is an edge.

                    # Let's build oriented_i:
                    if side_i == 'i_front':
                        # We want u to be at the end of oriented_i, so that we can append v.
                        # Currently u = pi[0]. So we need to reverse pi so that u is at the end.
                        oriented_i = pi[::-1]
                    else:  # 'i_back'
                        # u = pi[-1], which is already at the end.
                        oriented_i = pi[:]

                    # Build oriented_j:
                    if side_j == 'j_front':
                        # v = pj[0], which is already at the front.
                        oriented_j = pj[:]
                    else:  # 'j_back'
                        # v = pj[-1], so we need to reverse pj so that v is at the front.
                        oriented_j = pj[::-1]

                    # Now check that oriented_i ends with u and oriented_j starts with v.
                    if oriented_i[-1] != u or oriented_j[0] != v:
                        # This should not happen given our construction, but be safe.
                        continue

                    # The merged path is oriented_i + oriented_j
                    # But we must ensure that the edge (u, v) is valid and that
                    # the concatenation is a simple path.
                    # Since the vertex sets are disjoint and the edge exists, this is fine.
                    merged = oriented_i + oriented_j

                    # Validate simplicity (should be true by construction)
                    if len(set(merged)) != len(merged):
                        continue

                    # Perform the merge
                    # Remove paths i and j, add merged
                    # We need to be careful with indices if i > j
                    if i < j:
                        paths[i] = merged
                        paths.pop(j)
                    else:
                        paths[j] = merged
                        paths.pop(i)

                    # Update path_sets
                    path_sets = [set(p) for p in paths]
                    n_paths = len(paths)
                    improved = True
                    break

                if improved:
                    break

            if not improved:
                # No merge found in this iteration; try a different random ordering
                # by shuffling the paths themselves.
                rng.shuffle(paths)
                path_sets = [set(p) for p in paths]

        return paths

    def decompose_greedy(edges, rng, strategy='random'):
        """
        Runs a greedy path decomposition algorithm.
        Tries to build long paths by extending from both ends.
        Uses randomization in tie-breaking to explore different solutions.
        """
        if not edges:
            return []

        # Copy edges to a mutable set for removal
        remaining = set(edges)

        paths = []

        # Build adjacency for faster neighbor lookups
        adj = defaultdict(set)
        for u, v in remaining:
            adj[u].add(v)
            adj[v].add(u)

        def get_neighbors(u):
            return list(adj[u])

        def remove_edge(u, v):
            e = (u, v) if (u, v) in remaining else (v, u)
            remaining.remove(e)
            adj[u].discard(v)
            adj[v].discard(u)

        def get_degree(u):
            return len(adj[u])

        def extend_path(path, strategy):
            changed = True
            while changed:
                changed = False

                # Try extending the back end
                if len(path) > 0:
                    back = path[-1]
                    neighbors = get_neighbors(back)
                    valid_neighbors = [n for n in neighbors if n not in path]
                    if valid_neighbors:
                        if strategy == 'random':
                            next_node = rng.choice(valid_neighbors)
                        elif strategy == 'low_deg':
                            # Prefer neighbors with lower remaining degree
                            degs = {n: get_degree(n) for n in valid_neighbors}
                            min_deg = min(degs.values())
                            candidates = [n for n, d in degs.items() if d == min_deg]
                            next_node = rng.choice(candidates)
                        else:
                            next_node = rng.choice(valid_neighbors)
                        path.append(next_node)
                        remove_edge(back, next_node)
                        changed = True

                # Try extending the front end
                if len(path) > 0:
                    front = path[0]
                    neighbors = get_neighbors(front)
                    valid_neighbors = [n for n in neighbors if n not in path]
                    if valid_neighbors:
                        if strategy == 'random':
                            next_node = rng.choice(valid_neighbors)
                        elif strategy == 'low_deg':
                            degs = {n: get_degree(n) for n in valid_neighbors}
                            min_deg = min(degs.values())
                            candidates = [n for n, d in degs.items() if d == min_deg]
                            next_node = rng.choice(candidates)
                        else:
                            next_node = rng.choice(valid_neighbors)
                        path.insert(0, next_node)
                        remove_edge(front, next_node)
                        changed = True

        # Main loop: construct paths
        while remaining:
            # Pick a starting edge
            # Heuristic: Pick an edge incident to a vertex with the highest degree
            degrees = {u: len(adj[u]) for u in adj}
            max_deg = max(degrees.values())
            candidates = [v for v, d in degrees.items() if d == max_deg]
            start_u = rng.choice(candidates)

            neighbors = get_neighbors(start_u)
            if not neighbors:
                continue

            start_v = rng.choice(neighbors)

            # Initialize path
            path = [start_u, start_v]
            remove_edge(start_u, start_v)

            # Extend path
            extend_path(path, strategy)

            paths.append(path)

            # Check deadline
            if time.time() > deadline:
                break

        return paths

    def decompose_greedy_deterministic(edges):
        """
        Deterministic version that always picks the neighbor with the highest degree
        to encourage long paths.
        """
        if not edges:
            return []

        remaining = set(edges)
        paths = []

        adj = defaultdict(set)
        for u, v in remaining:
            adj[u].add(v)
            adj[v].add(u)

        def get_neighbors(u):
            return sorted(adj[u])

        def remove_edge(u, v):
            e = (u, v) if (u, v) in remaining else (v, u)
            remaining.remove(e)
            adj[u].discard(v)
            adj[v].discard(u)

        def get_degree(u):
            return len(adj[u])

        def extend_path(path):
            changed = True
            while changed:
                changed = False
                if len(path) > 0:
                    back = path[-1]
                    neighbors = get_neighbors(back)
                    valid_neighbors = [n for n in neighbors if n not in path]
                    if valid_neighbors:
                        # Pick the one with highest degree
                        next_node = max(valid_neighbors, key=lambda x: get_degree(x))
                        path.append(next_node)
                        remove_edge(back, next_node)
                        changed = True

                if len(path) > 0:
                    front = path[0]
                    neighbors = get_neighbors(front)
                    valid_neighbors = [n for n in neighbors if n not in path]
                    if valid_neighbors:
                        next_node = max(valid_neighbors, key=lambda x: get_degree(x))
                        path.insert(0, next_node)
                        remove_edge(front, next_node)
                        changed = True

        while remaining:
            degrees = {u: len(adj[u]) for u in adj}
            max_deg = max(degrees.values())
            candidates = [v for v, d in degrees.items() if d == max_deg]
            start_u = min(candidates)  # Deterministic tie-break

            neighbors = get_neighbors(start_u)
            if not neighbors:
                continue

            # Pick neighbor with highest degree
            start_v = max(neighbors, key=lambda x: get_degree(x))

            path = [start_u, start_v]
            remove_edge(start_u, start_v)
            extend_path(path)
            paths.append(path)

        return paths

    def solve_instance(n, p, seed):
        edges = instance(n, p, seed)
        edges_set = set(edges)
        if not edges:
            return []

        best_paths = None
        best_count = float('inf')

        # Run multiple iterations with different random seeds for the greedy choices
        # We have ~140s. Graphs are small (N=20-30, E~100-200). One iteration is very fast.
        # We can run thousands of iterations.

        # We'll alternate between 'random' and 'low_deg' strategies to explore
        # different solution landscapes.
        iterations = 2000
        for i in range(iterations):
            if time.time() > deadline:
                break
            rng = random.Random(seed * 1000 + i)
            # Alternate strategies
            strategy = 'random' if i % 2 == 0 else 'low_deg'
            paths = decompose_greedy(edges, rng, strategy=strategy)

            # Verify validity
            if _valid_solution(paths, edges_set):
                # Apply local polish to try to merge paths
                polished = _merge_polish(paths, edges_set, rng, max_iters=500)
                if _valid_solution(polished, edges_set):
                    if len(polished) < best_count:
                        best_count = len(polished)
                        best_paths = polished
                else:
                    # If polish broke validity (shouldn't happen), use original
                    if len(paths) < best_count:
                        best_count = len(paths)
                        best_paths = paths

            # If we reached the theoretical lower bound, we can stop early
            # Lower bound: at least 1 if graph is connected and has an Eulerian path,
            # but for general graphs, a simple lower bound is 1.
            if best_count == 1:
                break

        # Also try deterministic version
        det_paths = decompose_greedy_deterministic(edges)
        if _valid_solution(det_paths, edges_set):
            polished = _merge_polish(det_paths, edges_set, random.Random(seed + 99999), max_iters=500)
            if _valid_solution(polished, edges_set):
                if len(polished) < best_count:
                    best_count = len(polished)
                    best_paths = polished
            else:
                if len(det_paths) < best_count:
                    best_count = len(det_paths)
                    best_paths = det_paths

        if best_paths is None:
            # Fallback: return single edges if greedy failed completely (shouldn't happen)
            return [[list(e)] for e in edges]

        return best_paths

    specs = [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]
    results = []

    for spec in specs:
        n, p, seed = spec
        paths = solve_instance(n, p, seed)
        results.append(paths)

        if time.time() > deadline:
            # If we ran out of time, fill remaining with trivial solution?
            # But we should have finished.
            break

    # If we didn't finish all specs, ensure we return a valid structure
    while len(results) < 3:
        # This should not happen if time management is correct
        results.append([])

    return results

if __name__ == "__main__":
    # Quick test
    res = construct()
    total = sum(len(r) for r in res)
    print(f"Total paths: {total}")
    for i, r in enumerate(res):
        print(f"Instance {i}: {len(r)} paths")
