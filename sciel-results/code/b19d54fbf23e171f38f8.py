import time
import random
from collections import defaultdict

def construct():
    deadline = time.time() + 140
    results = []

    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > deadline - 2:
            edges = _instance(*spec)
            results.append([list(e) for e in edges])
            continue

        edges = _instance(*spec)
        paths = _solve_instance(edges, deadline)
        results.append(paths)

    return results

def _instance(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _solve_instance(edges, deadline):
    if not edges:
        return []

    adj = defaultdict(set)
    degree = defaultdict(int)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        degree[u] += 1
        degree[v] += 1

    odd_vertices = [v for v in adj if degree[v] % 2 == 1]
    k = len(odd_vertices) // 2

    if k == 0:
        paths = _greedy_decompose(edges, deadline)
        paths = _merge_paths(paths, edges, deadline)
        return paths

    # Structural change: enumerate permutations of odd vertices with better pairing strategies
    best_paths = None
    best_count = float('inf')

    # Generate several candidate pairings
    candidate_pairings = _generate_pairings(odd_vertices, deadline)

    for pairs in candidate_pairings:
        if time.time() > deadline - 3:
            break

        paths = _solve_with_pairs(edges, adj, pairs, deadline)
        if paths is not None:
            if len(paths) < best_count:
                best_count = len(paths)
                best_paths = paths
                if best_count == k:
                    break

    if best_paths is not None:
        merged_paths = _merge_paths(best_paths, edges, deadline)
        return merged_paths
    else:
        paths = _greedy_decompose(edges, deadline)
        paths = _merge_paths(paths, edges, deadline)
        return paths

def _generate_pairings(odd_vertices, deadline):
    """Generate diverse pairings of odd vertices."""
    n = len(odd_vertices)
    pairings = []

    # Strategy 1: Random pairings
    rng = random.Random(42)
    for _ in range(30):
        if time.time() > deadline - 3:
            break
        perm = odd_vertices[:]
        rng.shuffle(perm)
        pairs = [(perm[i], perm[i + 1]) for i in range(0, n, 2)]
        pairings.append(pairs)

    # Strategy 2: Sort by degree or other heuristic, then pair adjacent
    for sort_key in ['random', 'index']:
        perm = odd_vertices[:]
        if sort_key == 'index':
            perm.sort()
        else:
            random.shuffle(perm)
        pairs = [(perm[i], perm[i + 1]) for i in range(0, n, 2)]
        pairings.append(pairs)

    return pairings

def _solve_with_pairs(edges, adj, pairs, deadline):
    # Add virtual edges
    virtual_edges = set(tuple(sorted(p)) for p in pairs)

    # Build full graph including virtual edges
    full_edges = set(tuple(sorted(e)) for e in edges)
    full_edges.update(virtual_edges)

    # Find Eulerian Circuit
    circuit = _find_eulerian_circuit(full_edges)
    if circuit is None:
        return None

    # Split circuit by virtual edges
    paths = []
    current_path = []
    for i in range(len(circuit) - 1):
        u, v = circuit[i], circuit[i + 1]
        edge = tuple(sorted((u, v)))
        current_path.append(u)
        if edge in virtual_edges:
            current_path.append(v)
            # Check if current_path is simple
            if len(set(current_path)) == len(current_path):
                paths.append(current_path)
            else:
                return None  # Failed simplicity check
            current_path = []

    if current_path:
        current_path.append(circuit[-1])
        if len(set(current_path)) == len(current_path):
            paths.append(current_path)
        else:
            return None

    if len(paths) != len(pairs):
        return None

    return paths

def _find_eulerian_circuit(edges):
    if not edges:
        return []

    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    start = next(iter(adj))
    stack = [start]
    circuit = []

    while stack:
        u = stack[-1]
        while adj[u] and tuple(sorted((u, adj[u][-1]))) in set(tuple(sorted(e)) for e in edges if e[0] == u or e[1] == u):
            v = adj[u].pop()
            edge = tuple(sorted((u, v)))
            stack.append(v)
            # Remove edge
            break
        if not adj[u]:
            circuit.append(stack.pop())

    return circuit[::-1]

def _greedy_decompose(edges, deadline):
    remaining = set(tuple(sorted(e)) for e in edges)
    adj = defaultdict(set)
    degree = defaultdict(int)
    for u, v in remaining:
        adj[u].add(v)
        adj[v].add(u)
        degree[u] += 1
        degree[v] += 1

    paths = []

    while remaining:
        if time.time() > deadline:
            break
        # Pick start
        odd_vertices = [v for v in adj if degree.get(v, 0) % 2 == 1 and adj[v]]
        if odd_vertices:
            start = random.choice(odd_vertices)
        else:
            start = max(adj.keys(), key=lambda x: degree.get(x, 0))

        # Find long path
        path = [start]
        visited = {start}
        current = start
        while True:
            neighbors = [n for n in adj[current] if n not in visited]
            if not neighbors:
                break
            # Warnsdorff
            neighbors.sort(key=lambda x: sum(1 for y in adj[x] if y not in visited))
            next_v = neighbors[0]
            path.append(next_v)
            visited.add(next_v)

            # Remove edge
            edge = tuple(sorted((current, next_v)))
            remaining.discard(edge)
            adj[current].discard(next_v)
            adj[next_v].discard(current)
            degree[current] -= 1
            degree[next_v] -= 1

            if not adj[current]:
                del adj[current]
            if not adj[next_v]:
                del adj[next_v]

            current = next_v

        paths.append(path)

    return paths

def _merge_paths(paths, edges, deadline):
    merged = True
    while merged:
        merged = False
        new_paths = []
        used = [False] * len(paths)

        for i in range(len(paths)):
            if used[i]:
                continue

            current_path = paths[i]
            start_v = current_path[0]
            end_v = current_path[-1]

            best_merge = None
            best_j = -1

            for j in range(len(paths)):
                if i == j or used[j]:
                    continue

                other_path = paths[j]
                other_start = other_path[0]
                other_end = other_path[-1]

                # Case 1: current_path ends at u, other_path starts at u
                if end_v == other_start:
                    set_a = set(current_path[:-1])
                    set_b = set(other_path)
                    if not set_a & set_b:
                        merged_path = current_path + other_path[1:]
                        if best_merge is None or len(merged_path) > len(best_merge):
                            best_merge = merged_path
                            best_j = j

                # Case 2: current_path ends at u, other_path ends at u
                if end_v == other_end:
                    reversed_other = other_path[::-1]
                    set_a = set(current_path[:-1])
                    set_b = set(reversed_other)
                    if not set_a & set_b:
                        merged_path = current_path + reversed_other[1:]
                        if best_merge is None or len(merged_path) > len(best_merge):
                            best_merge = merged_path
                            best_j = j

                # Case 3: current_path starts at u, other_path starts at u
                if start_v == other_start:
                    reversed_current = current_path[::-1]
                    set_a = set(reversed_current[:-1])
                    set_b = set(other_path)
                    if not set_a & set_b:
                        merged_path = reversed_current + other_path[1:]
                        if best_merge is None or len(merged_path) > len(best_merge):
                            best_merge = merged_path
                            best_j = j

                # Case 4: current_path starts at u, other_path ends at u
                if start_v == other_end:
                    reversed_other = other_path[::-1]
                    set_a = set(current_path)
                    set_b = set(reversed_other[:-1])
                    if not set_a & set_b:
                        merged_path = reversed_other + current_path[1:]
                        if best_merge is None or len(merged_path) > len(best_merge):
                            best_merge = merged_path
                            best_j = j

            if best_merge is not None:
                new_paths.append(best_merge)
                used[i] = True
                used[best_j] = True
                merged = True
            else:
                new_paths.append(current_path)
                used[i] = True

        paths = new_paths
        if time.time() > deadline:
            break

    return paths
