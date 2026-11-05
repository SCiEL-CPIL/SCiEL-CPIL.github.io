import time
import random
from collections import defaultdict

def construct():
    """
    Decomposes the edges of three fixed instance graphs into the minimum number of
    edge-disjoint simple paths.
    """
    start_time = time.time()
    deadline = start_time + 140.0

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def validate_paths(paths, edges_set):
        if not paths:
            return len(edges_set) == 0
        used_edges = set()
        for path in paths:
            if len(path) < 2:
                return False
            if len(set(path)) != len(path):
                return False
            for j in range(len(path) - 1):
                u, v = path[j], path[j+1]
                e = tuple(sorted((u, v)))
                if e in used_edges:
                    return False
                if e not in edges_set:
                    return False
                used_edges.add(e)
        return len(used_edges) == len(edges_set)

    def try_merge_paths(paths, edges_set, rng):
        if len(paths) <= 1:
            return paths
        paths = [list(p) for p in paths]
        improved = True
        while improved:
            improved = False
            for i in range(len(paths)):
                if i >= len(paths):
                    break
                for j in range(i + 1, len(paths)):
                    if j >= len(paths):
                        break
                    p1 = paths[i]
                    p2 = paths[j]
                    candidates = []
                    # Option 1: p1 + p2
                    if p1[-1] != p2[0]:
                        e = tuple(sorted((p1[-1], p2[0])))
                        if e in edges_set:
                            merged = p1 + p2[1:]
                            if len(set(merged)) == len(merged):
                                candidates.append(merged)
                    # Option 2: p1 + reversed(p2)
                    if p1[-1] != p2[-1]:
                        e = tuple(sorted((p1[-1], p2[-1])))
                        if e in edges_set:
                            merged = p1 + p2[-2::-1]
                            if len(set(merged)) == len(merged):
                                candidates.append(merged)
                    # Option 3: reversed(p1) + p2
                    if p1[0] != p2[0]:
                        e = tuple(sorted((p1[0], p2[0])))
                        if e in edges_set:
                            merged = p1[-2::-1] + p2
                            if len(set(merged)) == len(merged):
                                candidates.append(merged)
                    # Option 4: reversed(p1) + reversed(p2)
                    if p1[0] != p2[-1]:
                        e = tuple(sorted((p1[0], p2[-1])))
                        if e in edges_set:
                            merged = p1[-2::-1] + p2[1:][::-1]
                            if len(set(merged)) == len(merged):
                                candidates.append(merged)
                    if candidates:
                        merged = rng.choice(candidates)
                        paths[i] = merged
                        paths.pop(j)
                        improved = True
                        break
                if improved:
                    break
        return paths

    def decompose_greedy(edges, rng, strategy='random'):
        if not edges:
            return []
        remaining = set(edges)
        paths = []
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        def get_neighbors(u):
            neighbors = []
            for v in adj[u]:
                e = tuple(sorted((u, v)))
                if e in remaining:
                    neighbors.append(v)
            return neighbors

        def remove_edge(u, v):
            e = tuple(sorted((u, v)))
            if e in remaining:
                remaining.remove(e)

        def extend_path(path):
            changed = True
            while changed:
                changed = False
                if len(path) > 0:
                    back = path[-1]
                    neighbors = get_neighbors(back)
                    valid_neighbors = [n for n in neighbors if n not in path]
                    if valid_neighbors:
                        if strategy == 'random':
                            next_node = rng.choice(valid_neighbors)
                        elif strategy == 'low_deg':
                            degs = {n: len(get_neighbors(n)) for n in valid_neighbors}
                            min_deg = min(degs.values())
                            candidates = [n for n in valid_neighbors if degs[n] == min_deg]
                            next_node = rng.choice(candidates)
                        else:
                            next_node = rng.choice(valid_neighbors)
                        path.append(next_node)
                        remove_edge(back, next_node)
                        changed = True
                if len(path) > 0:
                    front = path[0]
                    neighbors = get_neighbors(front)
                    valid_neighbors = [n for n in neighbors if n not in path]
                    if valid_neighbors:
                        if strategy == 'random':
                            next_node = rng.choice(valid_neighbors)
                        elif strategy == 'low_deg':
                            degs = {n: len(get_neighbors(n)) for n in valid_neighbors}
                            min_deg = min(degs.values())
                            candidates = [n for n in valid_neighbors if degs[n] == min_deg]
                            next_node = rng.choice(candidates)
                        else:
                            next_node = rng.choice(valid_neighbors)
                        path.insert(0, next_node)
                        remove_edge(front, next_node)
                        changed = True

        while remaining:
            degrees = defaultdict(int)
            for u, v in remaining:
                degrees[u] += 1
                degrees[v] += 1
            max_deg = max(degrees.values())
            candidates = [v for v, d in degrees.items() if d == max_deg]
            start_u = rng.choice(candidates)
            neighbors = get_neighbors(start_u)
            if not neighbors:
                continue
            start_v = rng.choice(neighbors)
            path = [start_u, start_v]
            remove_edge(start_u, start_v)
            extend_path(path)
            paths.append(path)
            if time.time() > deadline:
                break
        return paths

    def solve_instance(n, p, seed):
        edges = instance(n, p, seed)
        edges_set = set(tuple(sorted(e)) for e in edges)
        if not edges:
            return []
        best_paths = None
        best_count = float('inf')
        iterations = 2000
        for i in range(iterations):
            if time.time() > deadline:
                break
            strategy = 'random' if i % 2 == 0 else 'low_deg'
            rng = random.Random(seed * 1000 + i)
            paths = decompose_greedy(edges, rng, strategy)
            valid = validate_paths(paths, edges_set)
            if valid:
                merge_rng = random.Random(seed * 10000 + i)
                merged_paths = try_merge_paths(paths, edges_set, merge_rng)
                if validate_paths(merged_paths, edges_set):
                    if len(merged_paths) < best_count:
                        best_count = len(merged_paths)
                        best_paths = merged_paths
                else:
                    if len(paths) < best_count:
                        best_count = len(paths)
                        best_paths = paths
            if best_count == 1:
                break
        if best_paths is None:
            return [[list(e)] for e in edges]
        return best_paths

    specs = [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]
    results = []
    for spec in specs:
        n, p, seed = spec
        paths = solve_instance(n, p, seed)
        results.append(paths)
        if time.time() > deadline:
            break
    while len(results) < 3:
        results.append([])
    return results

if __name__ == "__main__":
    res = construct()
    total = sum(len(r) for r in res)
    print(f"Total paths: {total}")
    for i, r in enumerate(res):
        print(f"Instance {i}: {len(r)} paths")
