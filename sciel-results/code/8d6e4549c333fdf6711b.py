import time
import random
from collections import defaultdict
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

def construct():
    start_time = time.time()
    deadline = 135.0

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

    def greedy_decompose(edges, adj, time_limit=20.0, rng_seed=0):
        rng = random.Random(rng_seed)
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

            # Try starting from all vertices, but prioritize high degree
            candidates = sorted(remaining_adj.keys(), key=lambda x: len(remaining_adj[x]), reverse=True)
            
            # Limit candidates for speed
            for start in candidates[:10]:
                path = _extend_path(start, remaining_adj, rng, max_steps=100)
                if len(path) > best_len:
                    best_len = len(path)
                    best_path = path
            
            # Random starts
            all_vertices = list(remaining_adj.keys())
            for start in rng.sample(all_vertices, min(10, len(all_vertices))):
                path = _extend_path(start, remaining_adj, rng, max_steps=100)
                if len(path) > best_len:
                    best_len = len(path)
                    best_path = path

            if best_path is None or len(best_path) < 2:
                e = next(iter(remaining_edges))
                best_path = list(e)

            paths_found.append(best_path)
            for k in range(len(best_path) - 1):
                u, v = best_path[k], best_path[k + 1]
                remaining_edges.discard(tuple(sorted((u, v))))

        for e in remaining_edges:
            paths_found.append(list(e))
        return paths_found

    def _extend_path(start, remaining_adj, rng, max_steps=100):
        path = [start]
        visited = {start}
        current = start
        for _ in range(max_steps):
            neighbors = [nb for nb in remaining_adj.get(current, []) if nb not in visited]
            if not neighbors:
                break
            # Warnsdorff rule
            neighbors.sort(key=lambda nb: len([x for x in remaining_adj.get(nb, []) if x not in visited]))
            if rng.random() < 0.1 and len(neighbors) > 1:
                next_v = rng.choice(neighbors)
            else:
                next_v = neighbors[0]
            path.append(next_v)
            visited.add(next_v)
            current = next_v
        return path

    def merge_paths(paths, edges, time_limit=10.0):
        edge_set = set(tuple(sorted(e)) for e in edges)
        changed = True
        while changed:
            if time.time() - start_time > time_limit:
                break
            changed = False
            endpoint_map = defaultdict(list)
            for i, path in enumerate(paths):
                endpoint_map[path[0]].append((i, 0))
                endpoint_map[path[-1]].append((i, 1))
            
            for vertex, entries in endpoint_map.items():
                if len(entries) < 2:
                    continue
                # Try to merge any pair
                for a in range(len(entries)):
                    for b in range(a + 1, len(entries)):
                        i, ei = entries[a]
                        j, ej = entries[b]
                        if i == j: continue
                        pi = paths[i]
                        pj = paths[j]
                        merged = None
                        if ei == 1 and ej == 0:
                            merged = pi + pj[1:]
                        elif ei == 0 and ej == 1:
                            merged = pj + pi[1:]
                        elif ei == 1 and ej == 1:
                            merged = pi + pj[::-1][1:]
                        elif ei == 0 and ej == 0:
                            merged = pi + pj[::-1][1:]
                        
                        if merged:
                            if len(merged) == len(set(merged)):
                                valid = True
                                for k in range(len(merged) - 1):
                                    if tuple(sorted((merged[k], merged[k+1]))) not in edge_set:
                                        valid = False
                                        break
                                if valid:
                                    # Replace i, j with merged
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

    def solve_graph(n, p, seed, time_limit):
        edges = instance(n, p, seed)
        adj = get_adjacency(edges)
        edge_set = set(tuple(sorted(e)) for e in edges)
        
        # 1. Greedy Init
        greedy_paths = greedy_decompose(edges, adj, time_limit=10.0, rng_seed=seed)
        greedy_paths = merge_paths(greedy_paths, edges, time_limit=5.0)
        
        # 2. ILP Refinement
        # Generate candidate paths
        candidates = []
        rng = random.Random(seed + 100)
        
        # Add greedy paths
        candidates.extend(greedy_paths)
        
        # Generate random long paths
        for _ in range(200):
            if time.time() - start_time > time_limit - 10:
                break
            start = rng.choice(list(adj.keys()))
            path = _extend_path(start, adj, rng, max_steps=n)
            if len(path) >= 2:
                candidates.append(path)
        
        # Deduplicate candidates
        unique_candidates = []
        seen = set()
        for path in candidates:
            key = tuple(path)
            key_rev = tuple(reversed(path))
            if key not in seen and key_rev not in seen:
                # Verify validity
                valid = True
                for k in range(len(path) - 1):
                    if tuple(sorted((path[k], path[k+1]))) not in edge_set:
                        valid = False
                        break
                if valid and len(set(path)) == len(path):
                    unique_candidates.append(path)
                    seen.add(key)
                    seen.add(key_rev)
        
        if not unique_candidates:
            return greedy_paths
        
        num_edges = len(edge_set)
        num_candidates = len(unique_candidates)
        
        # Build constraint matrix
        # A[i, j] = 1 if candidate i uses edge j
        edge_index = {e: i for i, e in enumerate(edge_set)}
        
        # Use sparse matrix for efficiency
        rows = []
        cols = []
        data = []
        for i, path in enumerate(unique_candidates):
            for k in range(len(path) - 1):
                u, v = path[k], path[k+1]
                e = tuple(sorted((u, v)))
                rows.append(i)
                cols.append(edge_index[e])
                data.append(1)
        
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
                ilp_paths = [unique_candidates[i] for i in selected]
                
                # Verify coverage
                covered = set()
                for path in ilp_paths:
                    for k in range(len(path) - 1):
                        covered.add(tuple(sorted((path[k], path[k+1]))))
                        
                if covered == edge_set:
                    # Final merge pass
                    final_paths = merge_paths(ilp_paths, edges, time_limit=5.0)
                    if len(final_paths) < len(greedy_paths):
                        return final_paths
        except Exception:
            pass
        
        return greedy_paths

    results = []
    specs = [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]
    
    for n, p, seed in specs:
        if time.time() - start_time > deadline - 15:
            edges = instance(n, p, seed)
            results.append([list(e) for e in edges])
            continue
        
        time_limit_for_graph = deadline - (time.time() - start_time) - 5
        paths = solve_graph(n, p, seed, time_limit_for_graph)
        results.append(paths)
        
    return results
