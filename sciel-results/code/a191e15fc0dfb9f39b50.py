import time
import random
from collections import defaultdict
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def get_adjacency(edges):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def lower_bound(edges):
        if not edges:
            return 0
        degree = defaultdict(int)
        for u, v in edges:
            degree[u] += 1
            degree[v] += 1
        odd_count = sum(1 for d in degree.values() if d % 2 == 1)
        return max(1, (odd_count + 1) // 2)

    def extend_path(start, adj, visited, rng, max_steps=100):
        """Extend a path from start using Warnsdorff-like heuristic."""
        path = [start]
        current = start
        for _ in range(max_steps):
            neighbors = [nb for nb in adj.get(current, []) if nb not in visited]
            if not neighbors:
                break
            # Warnsdorff: prefer neighbors with fewer unvisited options
            neighbors.sort(key=lambda nb: len([x for x in adj.get(nb, []) if x not in visited]))
            # Mix of greedy and random
            if rng.random() < 0.15 and len(neighbors) > 1:
                next_v = rng.choice(neighbors)
            else:
                next_v = neighbors[0]
            path.append(next_v)
            visited.add(next_v)
            current = next_v
        return path

    def greedy_decompose(edges, rng, deadline, max_paths=200):
        """Greedy decomposition with randomized restarts."""
        remaining = set(edges)
        if not remaining:
            return []

        adj = defaultdict(set)
        for u, v in remaining:
            adj[u].add(v)
            adj[v].add(u)

        degree = defaultdict(int)
        for u, v in remaining:
            degree[u] += 1
            degree[v] += 1

        paths = []
        path_count = 0

        while remaining and path_count < max_paths:
            if time.time() > deadline:
                return None

            # Pick start: prefer odd-degree vertices
            odd_vertices = [v for v in adj if degree.get(v, 0) % 2 == 1 and adj[v]]
            if odd_vertices:
                start = rng.choice(odd_vertices)
            else:
                candidates = [v for v in adj if adj[v]]
                if not candidates:
                    break
                # Weight by degree
                weights = [max(1, degree.get(v, 0)) for v in candidates]
                total = sum(weights)
                r = rng.random() * total
                cum = 0
                start = candidates[-1]
                for v, w in zip(candidates, weights):
                    cum += w
                    if r <= cum:
                        start = v
                        break

            # Find long path
            visited = {start}
            best_path = None
            best_len = 0

            # Try multiple extensions
            for _ in range(5):
                path = extend_path(start, adj, set(visited), rng, max_steps=80)
                if len(path) > best_len:
                    best_len = len(path)
                    best_path = path

            if best_path is None or len(best_path) < 2:
                # Take a single edge
                if adj[start]:
                    next_v = rng.choice(list(adj[start]))
                    best_path = [start, next_v]
                else:
                    e = next(iter(remaining))
                    best_path = [e[0], e[1]]

            # Remove edges from path
            path_edges = set()
            for i in range(len(best_path) - 1):
                u, v = best_path[i], best_path[i + 1]
                edge = tuple(sorted((u, v)))
                if edge in remaining:
                    path_edges.add(edge)

            if not path_edges:
                break

            for e in path_edges:
                remaining.discard(e)
                u, v = e
                adj[u].discard(v)
                adj[v].discard(u)
                degree[u] -= 1
                degree[v] -= 1

            # Clean up empty adj
            for v in list(adj.keys()):
                if not adj[v]:
                    del adj[v]

            paths.append(best_path)
            path_count += 1

        # Add remaining edges as single-edge paths
        for e in remaining:
            paths.append(list(e))

        return paths

    def merge_paths(paths, edges, deadline):
        """Merge paths that share endpoints to reduce count."""
        edge_set = set(tuple(sorted(e)) for e in edges)
        changed = True
        max_iterations = 100

        while changed and max_iterations > 0:
            if time.time() > deadline:
                break
            changed = False
            max_iterations -= 1

            # Build endpoint map
            endpoint_map = defaultdict(list)
            for i, path in enumerate(paths):
                endpoint_map[path[0]].append((i, 0))
                endpoint_map[path[-1]].append((i, 1))

            # Try to merge pairs
            for vertex, entries in list(endpoint_map.items()):
                if len(entries) < 2:
                    continue

                for a in range(len(entries)):
                    for b in range(a + 1, len(entries)):
                        i, ei = entries[a]
                        j, ej = entries[b]
                        if i == j:
                            continue

                        pi = paths[i]
                        pj = paths[j]
                        merged = None

                        if ei == 1 and ej == 0:
                            merged = pi + pj[1:]
                        elif ei == 0 and ej == 1:
                            merged = pj + pi[1:]
                        elif ei == 1 and ej == 1:
                            merged = pi + pj[::-1][1:]
                        else:  # ei == 0 and ej == 0
                            merged = pi + pj[::-1][1:]

                        if merged and len(merged) == len(set(merged)):
                            valid = True
                            for k in range(len(merged) - 1):
                                if tuple(sorted((merged[k], merged[k + 1]))) not in edge_set:
                                    valid = False
                                    break
                            if valid:
                                new_paths = []
                                for idx, p in enumerate(paths):
                                    if idx != i and idx != j:
                                        new_paths.append(p)
                                new_paths.append(merged)
                                paths = new_paths
                                changed = True
                                break
                    if changed:
                        break

        return paths

    def ilp_refine(paths, edges, deadline):
        """Use ILP to find optimal subset of candidate paths."""
        edge_set = set(tuple(sorted(e)) for e in edges)
        if not paths:
            return paths

        # Build candidate list (include reversed versions)
        candidates = []
        seen = set()
        for path in paths:
            key = tuple(path)
            key_rev = tuple(reversed(path))
            if key not in seen and key_rev not in seen:
                candidates.append(path)
                seen.add(key)
                seen.add(key_rev)

        num_candidates = len(candidates)
        num_edges = len(edge_set)

        if num_candidates == 0 or num_edges == 0:
            return paths

        # Check time
        if time.time() > deadline - 5:
            return paths

        edge_index = {e: i for i, e in enumerate(sorted(edge_set))}

        # Build constraint matrix
        rows = []
        cols = []
        data = []
        for i, path in enumerate(candidates):
            for k in range(len(path) - 1):
                u, v = path[k], path[k + 1]
                e = tuple(sorted((u, v)))
                if e in edge_index:
                    rows.append(i)
                    cols.append(edge_index[e])
                    data.append(1)

        if not rows:
            return paths

        A = csr_matrix((data, (rows, cols)), shape=(num_candidates, num_edges))

        try:
            c = np.ones(num_candidates)
            lower = np.ones(num_edges)
            upper = np.ones(num_edges)
            constraint = LinearConstraint(A, lb=lower, ub=upper)
            bounds = Bounds(lb=0, ub=1)

            result = milp(
                c=c,
                constraints=constraint,
                bounds=bounds,
                integrality=np.ones(num_candidates)
            )

            if result.status == 0 and result.x is not None:
                selected = [i for i in range(num_candidates) if result.x[i] > 0.5]
                ilp_paths = [candidates[i] for i in selected]

                # Verify coverage
                covered = set()
                for path in ilp_paths:
                    for k in range(len(path) - 1):
                        covered.add(tuple(sorted((path[k], path[k + 1]))))

                if covered == edge_set:
                    final_paths = merge_paths(ilp_paths, edges, deadline)
                    return final_paths
        except Exception:
            pass

        return paths

    def solve_graph(n, p, seed, deadline):
        edges = instance(n, p, seed)
        if not edges:
            return []

        lb = lower_bound(edges)
        best_paths = None
        best_count = len(edges)

        # Multi-restart greedy
        num_restarts = min(50, max(10, int((deadline - time.time()) * 10)))

        for attempt in range(num_restarts):
            if time.time() > deadline - 10:
                break

            rng = random.Random(seed * 1000 + attempt * 7919 + 13)
            paths = greedy_decompose(edges, rng, deadline)

            if paths is None:
                continue

            # Merge
            paths = merge_paths(paths, edges, deadline)
            count = len(paths)

            if count < best_count:
                best_count = count
                best_paths = paths

                if count <= lb:
                    break

        if best_paths is None:
            # Fallback
            best_paths = [list(e) for e in edges]
            best_count = len(best_paths)

        # ILP refinement if time permits
        if time.time() < deadline - 15 and best_count > lb:
            refined = ilp_refine(best_paths, edges, deadline)
            if len(refined) < best_count:
                best_paths = refined
                best_count = len(best_paths)

        return best_paths

    results = []
    specs = [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]

    for n, p, seed in specs:
        if time.time() > deadline - 20:
            edges = instance(n, p, seed)
            results.append([list(e) for e in edges])
            continue

        time_limit = deadline - (time.time() - start_time) - 5
        paths = solve_graph(n, p, seed, time_limit)
        results.append(paths)

    return results
