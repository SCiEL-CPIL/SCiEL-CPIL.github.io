"""Path decomposition: ILP over all simple paths + local search refinement."""
import time
import random
from collections import defaultdict
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill

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

    def generate_all_simple_paths(edges, adj, max_path_len=12, time_limit=30.0):
        """Generate all simple paths up to max_path_len using DFS."""
        edge_set = set(tuple(sorted(e)) for e in edges)
        nodes = sorted(adj.keys())
        paths = []
        seen = set()

        def dfs(path, visited):
            if time.time() - start_time > time_limit:
                return
            if len(path) >= 2:
                key = tuple(path)
                key_rev = tuple(reversed(path))
                if key not in seen and key_rev not in seen:
                    seen.add(key)
                    seen.add(key_rev)
                    paths.append(list(path))
            if len(path) >= max_path_len:
                return
            last = path[-1]
            for nb in sorted(adj.get(last, [])):
                if nb not in visited:
                    visited.add(nb)
                    path.append(nb)
                    dfs(path, visited)
                    path.pop()
                    visited.remove(nb)

        for start in nodes:
            dfs([start], {start})

        return paths

    def greedy_decompose(edges, adj, time_limit=10.0):
        """Greedy longest-path decomposition as fallback."""
        remaining_edges = set(tuple(sorted(e)) for e in edges)
        paths_found = []

        while remaining_edges:
            if time.time() - start_time > time_limit:
                break

            remaining_adj = defaultdict(set)
            for u, v in remaining_edges:
                remaining_adj[u].add(v)
                remaining_adj[v].add(u)

            if not remaining_adj:
                break

            best_path = None
            best_len = 0

            candidates = sorted(remaining_adj.keys(),
                                key=lambda x: len(remaining_adj[x]),
                                reverse=True)
            candidates = candidates[:min(20, len(candidates))]

            for start in candidates:
                if time.time() - start_time > time_limit:
                    break
                path = _extend_path_greedy(start, remaining_adj, remaining_edges)
                if len(path) > best_len:
                    best_len = len(path)
                    best_path = path

            if best_path is None or best_len < 2:
                e = next(iter(remaining_edges))
                best_path = list(e)

            paths_found.append(best_path)

            for k in range(len(best_path) - 1):
                u, v = best_path[k], best_path[k + 1]
                remaining_edges.discard(tuple(sorted((u, v))))

        for e in remaining_edges:
            paths_found.append(list(e))

        return paths_found

    def _extend_path_greedy(start, remaining_adj, remaining_edges):
        """Extend a path from start using Warnsdorff heuristic."""
        path = [start]
        visited = {start}
        current = start

        while True:
            neighbors = [nb for nb in remaining_adj.get(current, []) if nb not in visited]
            if not neighbors:
                break

            def score(nb):
                return len([x for x in remaining_adj.get(nb, []) if x not in visited])

            neighbors.sort(key=score)
            next_v = neighbors[0]

            path.append(next_v)
            visited.add(next_v)
            current = next_v

        return path

    def solve_ilp(edges, adj, time_limit=60.0):
        """Solve path decomposition via ILP over all simple paths."""
        edge_set = set(tuple(sorted(e)) for e in edges)
        num_edges = len(edge_set)

        if num_edges == 0:
            return []

        # Generate candidate paths
        candidate_paths = generate_all_simple_paths(edges, adj, max_path_len=15,
                                                     time_limit=time_limit - 5)

        if not candidate_paths:
            return greedy_decompose(edges, adj, time_limit=time_limit)

        # Filter to only valid paths
        valid_paths = []
        for path in candidate_paths:
            valid = True
            for k in range(len(path) - 1):
                u, v = path[k], path[k + 1]
                if tuple(sorted((u, v))) not in edge_set:
                    valid = False
                    break
            if valid:
                valid_paths.append(path)

        if not valid_paths:
            return greedy_decompose(edges, adj, time_limit=time_limit)

        # Build incidence matrix
        num_paths = len(valid_paths)
        edge_index = {e: i for i, e in enumerate(sorted(edge_set))}

        A = np.zeros((num_paths, num_edges), dtype=np.int8)
        for i, path in enumerate(valid_paths):
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
                        path = valid_paths[i]
                        for k in range(len(path) - 1):
                            u, v = path[k], path[k + 1]
                            covered_edges.add(tuple(sorted((u, v))))
                    if covered_edges == edge_set:
                        return [valid_paths[i] for i in selected]
        except Exception:
            pass

        # Fallback to greedy
        return greedy_decompose(edges, adj, time_limit=time_limit)

    def local_search_refinement(paths, edges, adj, time_limit=20.0):
        """Try to improve solution by replacing subsets of paths with fewer paths."""
        edge_set = set(tuple(sorted(e)) for e in edges)
        current_paths = list(paths)
        best_count = len(current_paths)

        # Try merging pairs
        changed = True
        while changed:
            changed = False
            if time.time() - start_time > time_limit:
                break

            for i in range(len(current_paths)):
                if time.time() - start_time > time_limit:
                    break
                for j in range(i + 1, len(current_paths)):
                    pi = current_paths[i]
                    pj = current_paths[j]

                    # Try all 4 ways to merge
                    candidates = []
                    if pi[-1] == pj[0]:
                        candidates.append(pi + pj[1:])
                    if pi[0] == pj[-1]:
                        candidates.append(pj + pi[1:])
                    if pi[-1] == pj[-1]:
                        candidates.append(pi + pj[::-1][1:])
                    if pi[0] == pj[0]:
                        candidates.append(pj + pi[::-1][1:])

                    for merged in candidates:
                        if len(merged) != len(set(merged)):
                            continue
                        valid = True
                        for k in range(len(merged) - 1):
                            u, v = merged[k], merged[k + 1]
                            if tuple(sorted((u, v))) not in edge_set:
                                valid = False
                                break
                        if valid:
                            new_paths = [p for idx, p in enumerate(current_paths)
                                         if idx != i and idx != j]
                            new_paths.append(merged)
                            if len(new_paths) < best_count:
                                current_paths = new_paths
                                best_count = len(new_paths)
                                changed = True
                            break
                    if changed:
                        break
                if changed:
                    break

        return current_paths

    results = []
    specs = [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]

    for n, p, seed in specs:
        remaining_time = deadline - (time.time() - start_time)
        if remaining_time < 5:
            edges = instance(n, p, seed)
            results.append([list(e) for e in edges])
            continue

        edges = instance(n, p, seed)
        adj = get_adjacency(edges)

        # Try ILP first
        ilp_time = min(40.0, remaining_time * 0.5)
        ilp_paths = solve_ilp(edges, adj, time_limit=ilp_time)

        # If ILP succeeded, try local search refinement
        if len(ilp_paths) < len(edges):
            ls_time = min(15.0, remaining_time * 0.3)
            refined_paths = local_search_refinement(ilp_paths, edges, adj, time_limit=ls_time)
            results.append(refined_paths)
        else:
            # ILP failed or didn't improve; use greedy
            greedy_time = min(20.0, remaining_time * 0.5)
            greedy_paths = greedy_decompose(edges, adj, time_limit=greedy_time)
            ls_time = min(10.0, remaining_time * 0.3)
            refined_paths = local_search_refinement(greedy_paths, edges, adj, time_limit=ls_time)
            results.append(refined_paths)

    return results
