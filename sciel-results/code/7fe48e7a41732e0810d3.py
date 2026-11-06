"""Path decomposition: greedy longest-path + merge pass + ILP refinement."""
import time
import random
from collections import defaultdict
import numpy as np

def construct():
    start_time = time.time()
    deadline = 130.0  # safety margin before 150s kill

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

    def greedy_longest_path_decomp(edges, adj, time_limit=60.0):
        """Repeatedly extract longest path from remaining edges."""
        remaining_edges = set(tuple(sorted(e)) for e in edges)
        paths_found = []

        while remaining_edges:
            if time.time() - start_time > time_limit:
                break

            # Find a good starting vertex: prefer high degree in remaining graph
            remaining_adj = defaultdict(set)
            for u, v in remaining_edges:
                remaining_adj[u].add(v)
                remaining_adj[v].add(u)

            if not remaining_adj:
                break

            # Try multiple starting vertices, pick the longest path
            candidates = sorted(remaining_adj.keys(),
                                key=lambda x: len(remaining_adj[x]),
                                reverse=True)
            # Limit candidates for speed
            candidates = candidates[:min(20, len(candidates))]

            best_path = None
            best_len = 0

            for start in candidates:
                # Warnsdorff-style greedy: at each step, prefer neighbor with
                # fewest remaining options (to avoid dead ends)
                path = [start]
                visited = {start}
                current = start

                while True:
                    neighbors = [nb for nb in remaining_adj.get(current, [])
                                 if nb not in visited]
                    if not neighbors:
                        break

                    # Warnsdorff: prefer neighbor with fewest unvisited neighbors
                    def score(nb):
                        return len([x for x in remaining_adj.get(nb, []) if x not in visited])

                    neighbors.sort(key=score)
                    next_v = neighbors[0]
                    path.append(next_v)
                    visited.add(next_v)
                    current = next_v

                if len(path) > best_len:
                    best_len = len(path)
                    best_path = list(path)

            if best_path is None or best_len < 2:
                # Single edge fallback
                e = next(iter(remaining_edges))
                best_path = list(e)

            paths_found.append(best_path)

            # Remove used edges
            for k in range(len(best_path) - 1):
                u, v = best_path[k], best_path[k + 1]
                remaining_edges.discard(tuple(sorted((u, v))))

        # Add any remaining edges as single-edge paths
        for e in remaining_edges:
            paths_found.append(list(e))

        return paths_found

    def merge_paths(paths, edges, adj, time_limit=20.0):
        """Post-processing: merge paths that share endpoints to reduce count."""
        edge_set = set(tuple(sorted(e)) for e in edges)
        changed = True
        iterations = 0
        max_iterations = 50

        while changed and iterations < max_iterations:
            changed = False
            iterations += 1
            if time.time() - start_time > time_limit:
                break

            # Build endpoint index: vertex -> list of (path_index, endpoint)
            endpoint_map = defaultdict(list)
            for i, path in enumerate(paths):
                endpoint_map[path[0]].append((i, 0))  # start
                endpoint_map[path[-1]].append((i, 1))  # end

            # Try to merge pairs of paths that share an endpoint
            merged = False
            new_paths = list(paths)

            for vertex in list(endpoint_map.keys()):
                if time.time() - start_time > time_limit:
                    break
                entries = endpoint_map[vertex]
                if len(entries) < 2:
                    continue

                # Find two paths that can be merged at this vertex
                for a in range(len(entries)):
                    for b in range(a + 1, len(entries)):
                        i, ei = entries[a]
                        j, ej = entries[b]
                        if i == j:
                            continue
                        if i >= len(new_paths) or j >= len(new_paths):
                            continue

                        pi = new_paths[i]
                        pj = new_paths[j]

                        # Determine how to merge
                        merged_path = None
                        # Case 1: pi ends at vertex, pj starts at vertex
                        if ei == 1 and ej == 0 and pi[-1] == vertex and pj[0] == vertex:
                            merged_path = pi + pj[1:]
                        # Case 2: pi starts at vertex, pj ends at vertex
                        elif ei == 0 and ej == 1 and pi[0] == vertex and pj[-1] == vertex:
                            merged_path = pj + pi[1:]
                        # Case 3: both end at vertex -> reverse one
                        elif ei == 1 and ej == 1 and pi[-1] == vertex and pj[-1] == vertex:
                            merged_path = pi + pj[::-1][1:]
                        # Case 4: both start at vertex -> reverse one
                        elif ei == 0 and ej == 0 and pi[0] == vertex and pj[0] == vertex:
                            merged_path = pi + pj[::-1][1:]

                        if merged_path is None:
                            continue

                        # Verify merged path is simple and uses valid edges
                        if len(merged_path) != len(set(merged_path)):
                            continue

                        valid = True
                        for k in range(len(merged_path) - 1):
                            u, v = merged_path[k], merged_path[k + 1]
                            if tuple(sorted((u, v))) not in edge_set:
                                valid = False
                                break

                        if not valid:
                            continue

                        # Perform merge: remove i and j, add merged_path
                        # Remove in reverse order to preserve indices
                        remove_idx = sorted([i, j], reverse=True)
                        for idx in remove_idx:
                            new_paths.pop(idx)
                        new_paths.append(merged_path)
                        merged = True
                        changed = True
                        break
                    if merged:
                        break
                if merged:
                    break

            paths = new_paths

        return paths

    def try_ilp_refinement(edges, adj, initial_paths, time_limit=15.0):
        """Use ILP to refine path decomposition if tractable."""
        if time.time() - start_time > time_limit:
            return initial_paths

        edge_set = set(tuple(sorted(e)) for e in edges)
        num_edges = len(edge_set)

        # Generate candidate paths: all simple paths up to length 6
        candidate_paths = []
        nodes = sorted(adj.keys())

        def find_paths_from(start, max_len=6):
            found = []

            def dfs(path, visited):
                if len(path) >= 2:
                    found.append(list(path))
                if len(path) >= max_len:
                    return
                last = path[-1]
                for nb in sorted(adj.get(last, [])):
                    if nb not in visited:
                        visited.add(nb)
                        path.append(nb)
                        dfs(path, visited)
                        path.pop()
                        visited.remove(nb)

            dfs([start], {start})
            return found

        for start in nodes:
            candidate_paths.extend(find_paths_from(start))

        # Filter valid paths
        valid_paths = []
        for path in candidate_paths:
            valid = True
            for k in range(len(path) - 1):
                u, v = path[k], path[k + 1]
                if tuple(sorted((u, v))) not in edge_set:
                    valid = False
                    break
            if valid and len(path) >= 2:
                valid_paths.append(path)

        # Deduplicate (consider reverse as same)
        seen = set()
        unique_paths = []
        for p in valid_paths:
            key = tuple(p)
            key_rev = tuple(reversed(p))
            if key not in seen and key_rev not in seen:
                seen.add(key)
                seen.add(key_rev)
                unique_paths.append(p)

        if len(unique_paths) > 2000:
            # Too many candidates, skip ILP
            return initial_paths

        if not unique_paths:
            return initial_paths

        num_paths = len(unique_paths)
        edge_index = {e: i for i, e in enumerate(edge_set)}

        # Incidence matrix
        A = np.zeros((num_paths, num_edges), dtype=np.int8)
        for i, path in enumerate(unique_paths):
            for k in range(len(path) - 1):
                u, v = path[k], path[k + 1]
                e = tuple(sorted((u, v)))
                A[i, edge_index[e]] = 1

        try:
            from scipy.optimize import milp, LinearConstraint, Bounds

            c = np.ones(num_paths)
            lower = np.ones(num_edges)
            upper = np.ones(num_edges)
            constraint = LinearConstraint(A.T, lb=lower, ub=upper)
            bounds = Bounds(lb=0, ub=1)

            result = milp(
                c=c,
                constraints=constraint,
                bounds=bounds,
                integrality=np.ones(num_paths)
            )

            if result.status == 0 and result.x is not None:
                selected = [i for i in range(num_paths) if result.x[i] > 0.5]
                if len(selected) > 0:
                    covered_edges = set()
                    for i in selected:
                        path = unique_paths[i]
                        for k in range(len(path) - 1):
                            u, v = path[k], path[k + 1]
                            covered_edges.add(tuple(sorted((u, v))))
                    if covered_edges == edge_set:
                        ilp_paths = [unique_paths[i] for i in selected]
                        # ILP is better if fewer paths
                        if len(ilp_paths) < len(initial_paths):
                            return ilp_paths
        except Exception:
            pass

        return initial_paths

    results = []
    specs = [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]

    for n, p, seed in specs:
        if time.time() - start_time > deadline - 10:
            # Emergency fallback
            edges = instance(n, p, seed)
            results.append([list(e) for e in edges])
            continue

        edges = instance(n, p, seed)
        adj = get_adjacency(edges)

        # Greedy longest path decomposition
        greedy_paths = greedy_longest_path_decomp(edges, adj, time_limit=deadline - 20)

        # Merge pass to reduce path count
        merged_paths = merge_paths(greedy_paths, edges, adj, time_limit=deadline - 15)

        # Try ILP refinement if we have time
        if time.time() - start_time < deadline - 20:
            refined_paths = try_ilp_refinement(edges, adj, merged_paths,
                                               time_limit=deadline - 10)
        else:
            refined_paths = merged_paths

        results.append(refined_paths)

    return results
