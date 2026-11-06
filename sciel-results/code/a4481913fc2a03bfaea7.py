"""Path decomposition: multi-start greedy + local search with split/remerge + ILP polish."""
import time
import random
from collections import defaultdict
import numpy as np


def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill

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

    def _extend_path(start, remaining_adj, rng, max_steps=60):
        """Extend a path from start using Warnsdorff-like heuristic with randomization."""
        path = [start]
        visited = {start}
        current = start

        for _ in range(max_steps):
            neighbors = [nb for nb in remaining_adj.get(current, []) if nb not in visited]
            if not neighbors:
                break

            # Score neighbors: prefer those with fewer unvisited neighbors (Warnsdorff)
            def score(nb):
                return len([x for x in remaining_adj.get(nb, []) if x not in visited])

            neighbors.sort(key=score)

            # With some probability, pick a random neighbor instead of the best
            if rng.random() < 0.25 and len(neighbors) > 1:
                next_v = rng.choice(neighbors)
            else:
                next_v = neighbors[0]

            path.append(next_v)
            visited.add(next_v)
            current = next_v

        return path

    def greedy_decompose(edges, adj, time_limit=60.0, rng_seed=0):
        """Repeatedly extract a good path from remaining edges using randomized heuristics."""
        rng = random.Random(rng_seed)
        remaining_edges = set(tuple(sorted(e)) for e in edges)
        paths_found = []

        while remaining_edges:
            if time.time() - start_time > time_limit:
                break

            # Build remaining adjacency
            remaining_adj = defaultdict(set)
            for u, v in remaining_edges:
                remaining_adj[u].add(v)
                remaining_adj[v].add(u)

            if not remaining_adj:
                break

            # Try multiple strategies to find a good path
            best_path = None
            best_len = 0

            # Strategy 1: Start from high-degree vertices
            candidates = sorted(remaining_adj.keys(),
                                key=lambda x: len(remaining_adj[x]),
                                reverse=True)
            candidates = candidates[:min(25, len(candidates))]

            for start in candidates:
                if time.time() - start_time > time_limit:
                    break
                path = _extend_path(start, remaining_adj, rng, max_steps=60)
                if len(path) > best_len:
                    best_len = len(path)
                    best_path = path

            # Strategy 2: Random starts for diversification
            all_vertices = list(remaining_adj.keys())
            num_random = min(10, len(all_vertices))
            random_starts = rng.sample(all_vertices, num_random)
            for start in random_starts:
                if time.time() - start_time > time_limit:
                    break
                path = _extend_path(start, remaining_adj, rng, max_steps=60)
                if len(path) > best_len:
                    best_len = len(path)
                    best_path = path

            if best_path is None or best_len < 2:
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

    def merge_paths(paths, edge_set, time_limit=20.0):
        """Aggressively merge paths that share endpoints to reduce count."""
        changed = True
        iterations = 0
        max_iterations = 300

        while changed and iterations < max_iterations:
            changed = False
            iterations += 1
            if time.time() - start_time > time_limit:
                break

            # Build endpoint index
            endpoint_map = defaultdict(list)
            for i, path in enumerate(paths):
                endpoint_map[path[0]].append((i, 0))
                if path[-1] != path[0]:  # avoid double-counting for length-1 paths
                    endpoint_map[path[-1]].append((i, 1))

            merged_any = False
            new_paths = list(paths)

            for vertex in sorted(endpoint_map.keys()):
                if time.time() - start_time > time_limit:
                    break
                entries = endpoint_map[vertex]
                if len(entries) < 2:
                    continue

                # Try all pairs at this vertex
                for a in range(len(entries)):
                    if merged_any:
                        break
                    for b in range(a + 1, len(entries)):
                        i, ei = entries[a]
                        j, ej = entries[b]
                        if i == j or i >= len(new_paths) or j >= len(new_paths):
                            continue

                        pi = new_paths[i]
                        pj = new_paths[j]

                        merged_path = None
                        if ei == 1 and ej == 0 and pi[-1] == vertex and pj[0] == vertex:
                            merged_path = pi + pj[1:]
                        elif ei == 0 and ej == 1 and pi[0] == vertex and pj[-1] == vertex:
                            merged_path = pj + pi[1:]
                        elif ei == 1 and ej == 1 and pi[-1] == vertex and pj[-1] == vertex:
                            merged_path = pi + pj[::-1][1:]
                        elif ei == 0 and ej == 0 and pi[0] == vertex and pj[0] == vertex:
                            merged_path = pi + pj[::-1][1:]

                        if merged_path is None:
                            continue

                        # Verify simplicity and edge validity
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

                        # Perform merge
                        remove_idx = sorted([i, j], reverse=True)
                        for idx in remove_idx:
                            new_paths.pop(idx)
                        new_paths.append(merged_path)
                        merged_any = True
                        changed = True
                        break
                if merged_any:
                    break

            paths = new_paths

        return paths

    def local_search_split_remerge(paths, edge_set, adj, time_limit=25.0, rng_seed=42):
        """Local search: try splitting paths and remerging to reduce total count.
        
        This escapes local optima by breaking a path into pieces and recombining
        them differently with other paths.
        """
        rng = random.Random(rng_seed)
        current_paths = list(paths)
        current_count = len(current_paths)
        best_paths = list(current_paths)
        best_count = current_count

        max_iterations = 200
        for it in range(max_iterations):
            if time.time() - start_time > time_limit:
                break
            if len(current_paths) <= 2:
                break

            # Pick a path to split (prefer longer paths for more potential)
            path_indices = list(range(len(current_paths)))
            path_indices.sort(key=lambda i: len(current_paths[i]), reverse=True)
            
            # Try splitting one of the longer paths
            split_candidates = path_indices[:min(5, len(path_indices))]
            chosen_idx = rng.choice(split_candidates)
            path_to_split = current_paths[chosen_idx]

            if len(path_to_split) < 4:
                continue

            # Choose a split point
            split_pos = rng.randint(1, len(path_to_split) - 2)
            part1 = path_to_split[:split_pos + 1]
            part2 = path_to_split[split_pos:]

            # Create new path set with the split
            new_paths = list(current_paths)
            new_paths.pop(chosen_idx)
            new_paths.append(part1)
            new_paths.append(part2)

            # Now try to merge the new pieces with other paths
            new_paths = merge_paths(new_paths, edge_set, time_limit=min(
                time_limit, time.time() - start_time + 5))

            if len(new_paths) < len(current_paths):
                current_paths = new_paths
                current_count = len(current_paths)
                if current_count < best_count:
                    best_count = current_count
                    best_paths = list(current_paths)

        return best_paths

    def try_ilp_refinement(edges, adj, initial_paths, time_limit=15.0):
        """Use ILP to refine path decomposition if tractable."""
        if time.time() - start_time > time_limit:
            return initial_paths

        edge_set = set(tuple(sorted(e)) for e in edges)
        num_edges = len(edge_set)
        nodes = sorted(adj.keys())

        # Generate candidate paths up to length 7
        candidate_paths = []

        def find_paths_from(start, max_len=7):
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

        # Deduplicate
        seen = set()
        unique_paths = []
        for p in valid_paths:
            key = tuple(p)
            key_rev = tuple(reversed(p))
            if key not in seen and key_rev not in seen:
                seen.add(key)
                seen.add(key_rev)
                unique_paths.append(p)

        if len(unique_paths) > 3000:
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
                        if len(ilp_paths) < len(initial_paths):
                            return ilp_paths
        except Exception:
            pass

        return initial_paths

    results = []
    specs = [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]

    for n, p, seed in specs:
        if time.time() - start_time > deadline - 15:
            # Emergency fallback
            edges = instance(n, p, seed)
            results.append([list(e) for e in edges])
            continue

        edges = instance(n, p, seed)
        adj = get_adjacency(edges)
        edge_set = set(tuple(sorted(e)) for e in edges)

        # Multi-start greedy: try several random seeds, keep best
        best_greedy = None
        best_greedy_count = float('inf')
        num_starts = min(5, max(2, int((deadline - 25 - (time.time() - start_time)) / 10)))
        
        for attempt in range(num_starts):
            if time.time() - start_time > deadline - 25:
                break
            greedy_paths = greedy_decompose(edges, adj, time_limit=deadline - 30, rng_seed=attempt * 100 + 7)
            if len(greedy_paths) < best_greedy_count:
                best_greedy_count = len(greedy_paths)
                best_greedy = greedy_paths

        if best_greedy is None:
            best_greedy = greedy_decompose(edges, adj, time_limit=deadline - 30, rng_seed=0)

        # Merge pass
        merged_paths = merge_paths(best_greedy, edge_set, time_limit=deadline - 20)

        # Local search with split/remerge
        if time.time() - start_time < deadline - 15:
            ls_paths = local_search_split_remerge(merged_paths, edge_set, adj,
                                                   time_limit=deadline - 12,
                                                   rng_seed=seed * 1000 + 42)
            if len(ls_paths) < len(merged_paths):
                merged_paths = ls_paths

        # Try ILP refinement if we have time
        if time.time() - start_time < deadline - 10:
            refined_paths = try_ilp_refinement(edges, adj, merged_paths,
                                               time_limit=deadline - 5)
        else:
            refined_paths = merged_paths

        results.append(refined_paths)

    return results
