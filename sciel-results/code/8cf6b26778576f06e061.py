import time
import random
from collections import defaultdict
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def validate_paths(paths, edges):
        if not paths:
            return len(edges) == 0
        edge_set = set(edges)
        used_edges = set()
        for path in paths:
            if len(path) < 2:
                continue
            if len(set(path)) != len(path):
                return False
            for j in range(len(path) - 1):
                u, v = path[j], path[j+1]
                e = tuple(sorted((u, v)))
                if e not in edge_set:
                    return False
                if e in used_edges:
                    return False
                used_edges.add(e)
        return used_edges == edge_set

    def greedy_random(edges, seed):
        rng = random.Random(seed)
        remaining = set(edges)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        paths = []
        while remaining:
            edge = rng.choice(list(remaining))
            remaining.remove(edge)
            u, v = edge
            path = [u, v]
            changed = True
            while changed:
                changed = False
                for end in [0, 1]:
                    if end == 0:
                        current = path[0]
                        cands = [nb for nb in adj[current]
                                  if nb not in path and ((current, nb) in remaining or (nb, current) in remaining)]
                        if cands:
                            nb = rng.choice(cands)
                            if (current, nb) in remaining:
                                remaining.remove((current, nb))
                            else:
                                remaining.remove((nb, current))
                            path.insert(0, nb)
                            changed = True
                    else:
                        current = path[-1]
                        cands = [nb for nb in adj[current]
                                  if nb not in path and ((current, nb) in remaining or (nb, current) in remaining)]
                        if cands:
                            nb = rng.choice(cands)
                            if (current, nb) in remaining:
                                remaining.remove((current, nb))
                            else:
                                remaining.remove((nb, current))
                            path.append(nb)
                            changed = True
                if not changed:
                    break
            paths.append(path)
        return paths

    def greedy_target(edges, k, seed_val):
        """Try to cover all edges with exactly k simple paths."""
        if k <= 0 and edges:
            return None
        if len(edges) < k:
            return None
        rng = random.Random(seed_val)
        remaining = set(edges)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        paths = [[] for _ in range(k)]
        edge_list = list(remaining)
        rng.shuffle(edge_list)
        start_edges = edge_list[:k]
        for i in range(k):
            u, v = start_edges[i]
            if (u, v) in remaining:
                remaining.remove((u, v))
            else:
                remaining.remove((v, u))
            paths[i] = [u, v]
        changed = True
        while changed:
            changed = False
            idxs = list(range(k))
            rng.shuffle(idxs)
            for i in idxs:
                if not paths[i]:
                    continue
                back = paths[i][-1]
                cands = [nb for nb in adj[back] if nb not in paths[i]
                         and ((back, nb) in remaining or (nb, back) in remaining)]
                if cands:
                    nb = rng.choice(cands)
                    if (back, nb) in remaining:
                        remaining.remove((back, nb))
                    else:
                        remaining.remove((nb, back))
                    paths[i].append(nb)
                    changed = True
                    continue
                front = paths[i][0]
                cands = [nb for nb in adj[front] if nb not in paths[i]
                         and ((front, nb) in remaining or (nb, front) in remaining)]
                if cands:
                    nb = rng.choice(cands)
                    if (front, nb) in remaining:
                        remaining.remove((front, nb))
                    else:
                        remaining.remove((nb, front))
                    paths[i].insert(0, nb)
                    changed = True
                    continue
        if remaining:
            return None
        return paths

    def max_matching_bipartite(left_nodes, right_nodes, adjacency):
        if not left_nodes or not right_nodes:
            return []
        left_index = {node: i for i, node in enumerate(left_nodes)}
        right_index = {node: i for i, node in enumerate(right_nodes)}
        n_left = len(left_nodes)
        n_right = len(right_nodes)
        adj = [[] for _ in range(n_left)]
        for u in left_nodes:
            ui = left_index[u]
            for v in adjacency.get(u, set()):
                if v in right_index:
                    adj[ui].append(right_index[v])
        match_left = [-1] * n_left
        match_right = [-1] * n_right
        dist = [0] * n_left

        def bfs():
            queue = []
            for u in range(n_left):
                if match_left[u] == -1:
                    dist[u] = 0
                    queue.append(u)
                else:
                    dist[u] = float('inf')
            found = False
            while queue:
                u = queue.pop(0)
                for v in adj[u]:
                    if match_right[v] == -1:
                        found = True
                    elif dist[match_right[v]] == float('inf'):
                        dist[match_right[v]] = dist[u] + 1
                        queue.append(match_right[v])
            return found

        def dfs(u):
            for v in adj[u]:
                if match_right[v] == -1 or (dist[match_right[v]] == dist[u] + 1 and dfs(match_right[v])):
                    match_left[u] = v
                    match_right[v] = u
                    return True
            dist[u] = float('inf')
            return False

        while bfs():
            for u in range(n_left):
                if match_left[u] == -1:
                    dfs(u)
        matching = []
        for u in range(n_left):
            if match_left[u] != -1:
                matching.append((left_nodes[u], right_nodes[match_left[u]]))
        return matching

    def merge_endpoints(paths, edges, adj, used_edges, max_iter=50):
        """Merge paths at their endpoints using bipartite matching."""
        if not paths:
            return []
        current = [list(p) for p in paths]
        current_used = set(used_edges)
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            endpoints = []
            for i, path in enumerate(current):
                if len(path) >= 2:
                    endpoints.append((i, 0, path[0]))
                    endpoints.append((i, 1, path[-1]))
            if len(endpoints) < 4:
                break
            left_nodes = []
            right_nodes = []
            for i, path in enumerate(current):
                if len(path) < 2:
                    continue
                for pos in [0, 1]:
                    if i % 2 == 0:
                        left_nodes.append((i, pos))
                    else:
                        right_nodes.append((i, pos))
            if not left_nodes or not right_nodes:
                break
            adjacency = defaultdict(set)
            for (i1, pos1) in left_nodes:
                v1 = current[i1][0] if pos1 == 0 else current[i1][-1]
                for (i2, pos2) in right_nodes:
                    if i1 == i2:
                        continue
                    v2 = current[i2][0] if pos2 == 0 else current[i2][-1]
                    e = tuple(sorted((v1, v2)))
                    if e not in edges or e in current_used:
                        continue
                    if set(current[i1]) & set(current[i2]):
                        continue
                    adjacency[(i1, pos1)].add((i2, pos2))
            if not adjacency:
                break
            matching = max_matching_bipartite(left_nodes, right_nodes, adjacency)
            if not matching:
                break
            new_paths = {i: list(p) for i, p in enumerate(current)}
            removed = set()
            applied = 0
            for (lep, rep) in matching:
                i1, pos1 = lep
                i2, pos2 = rep
                if i1 in removed or i2 in removed:
                    continue
                p1 = new_paths[i1]
                p2 = new_paths[i2]
                v1 = p1[0] if pos1 == 0 else p1[-1]
                v2 = p2[0] if pos2 == 0 else p2[-1]
                e = tuple(sorted((v1, v2)))
                if e not in edges or e in current_used:
                    continue
                q1 = list(reversed(p1)) if pos1 == 1 else list(p1)
                q2 = list(reversed(p2)) if pos2 == 1 else list(p2)
                if q1[-1] != v1 or q2[0] != v2:
                    continue
                merged = q1 + q2[1:]
                if len(set(merged)) != len(merged):
                    continue
                ok = True
                for j in range(len(merged) - 1):
                    ej = tuple(sorted((merged[j], merged[j+1])))
                    if ej not in edges:
                        ok = False
                        break
                    if ej in current_used and ej != e:
                        ok = False
                        break
                if not ok:
                    continue
                new_paths[i1] = merged
                del new_paths[i2]
                removed.add(i2)
                current_used.add(e)
                applied += 1
            if applied == 0:
                break
            current = [new_paths[i] for i in sorted(new_paths.keys())]
            if len(current) >= len(paths):
                break
        return current

    def internal_reconnect(paths, edges, adj, used_edges, max_iter=30):
        """
        Split paths at internal vertices and recombine to reduce path count.
        This allows more aggressive merging than endpoint-only approaches.
        """
        if not paths or len(paths) <= 1:
            return [list(p) for p in paths]
        current = [list(p) for p in paths]
        current_used = set(used_edges)

        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            improved = False

            # Try splitting each path at each internal vertex and see if we can
            # recombine the pieces to reduce total path count
            for pi in range(len(current)):
                path = current[pi]
                if len(path) < 3:
                    continue

                # Try each internal split point
                for split_pos in range(1, len(path) - 1):
                    # Split into [0..split_pos] and [split_pos..end]
                    # The shared vertex is path[split_pos]
                    left_part = path[:split_pos + 1]
                    right_part = path[split_pos:]

                    # Remove the edge that was at the split point from used
                    # Actually, we're not removing edges, we're reorganizing
                    # Let's think differently: try to find if any other path
                    # can absorb one of these pieces

                    # For each other path, check if we can merge one piece
                    for pj in range(len(current)):
                        if pj == pi:
                            continue
                        other = current[pj]
                        if len(other) < 2:
                            continue

                        # Try merging left_part with other
                        # Check if left_part's endpoints can connect to other's endpoints
                        for lpos in [0, 1]:
                            for opos in [0, 1]:
                                lv = left_part[0] if lpos == 0 else left_part[-1]
                                ov = other[0] if opos == 0 else other[-1]
                                e = tuple(sorted((lv, ov)))
                                if e not in edges or e in current_used:
                                    continue
                                # Check simplicity
                                if set(left_part) & set(other):
                                    continue
                                # Build merged
                                lp = list(reversed(left_part)) if lpos == 1 else list(left_part)
                                op = list(reversed(other)) if opos == 1 else list(other)
                                if lp[-1] != lv or op[0] != ov:
                                    continue
                                merged = lp + op[1:]
                                if len(set(merged)) != len(merged):
                                    continue
                                # Validate edges
                                ok = True
                                for j in range(len(merged) - 1):
                                    ej = tuple(sorted((merged[j], merged[j+1])))
                                    if ej not in edges:
                                        ok = False
                                        break
                                if not ok:
                                    continue
                                # This merge is valid. Now we have:
                                # merged (from left_part + other) and right_part
                                # Total paths: same count (we split pi into two, merged one with pj)
                                # But we might be able to further merge right_part
                                # For now, just do the swap
                                new_current = [list(p) for p in current]
                                new_current[pj] = merged
                                new_current[pi] = right_part
                                # Check if this is an improvement (it's neutral in count,
                                # but may enable further merges)
                                # Only apply if it doesn't increase used edges incorrectly
                                # The edge e is now used in merged, so add it
                                current_used_new = set(current_used)
                                current_used_new.add(e)
                                # Validate the new configuration
                                if validate_paths(new_current, edges):
                                    current = new_current
                                    current_used = current_used_new
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break

            # Also try: take two paths, and see if we can recombine them by
            # finding a common internal vertex structure
            if not improved:
                for pi in range(len(current)):
                    for pj in range(pi + 1, len(current)):
                        if time.time() > deadline - 5:
                            break
                        p1 = current[pi]
                        p2 = current[pj]
                        # Try all endpoint combinations
                        for pos1 in [0, 1]:
                            for pos2 in [0, 1]:
                                v1 = p1[0] if pos1 == 0 else p1[-1]
                                v2 = p2[0] if pos2 == 0 else p2[-1]
                                e = tuple(sorted((v1, v2)))
                                if e not in edges or e in current_used:
                                    continue
                                if set(p1) & set(p2):
                                    continue
                                q1 = list(reversed(p1)) if pos1 == 1 else list(p1)
                                q2 = list(reversed(p2)) if pos2 == 1 else list(p2)
                                if q1[-1] != v1 or q2[0] != v2:
                                    continue
                                merged = q1 + q2[1:]
                                if len(set(merged)) != len(merged):
                                    continue
                                ok = True
                                for j in range(len(merged) - 1):
                                    ej = tuple(sorted((merged[j], merged[j+1])))
                                    if ej not in edges:
                                        ok = False
                                        break
                                    if ej in current_used and ej != e:
                                        ok = False
                                        break
                                if not ok:
                                    continue
                                new_current = [list(p) for p in current]
                                new_current[pi] = merged
                                del new_current[pj]
                                current_used.add(e)
                                current = new_current
                                improved = True
                                break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break

            if not improved:
                break

        return current

    def solve_instance(n, p, seed):
        edges = instance(n, p, seed)
        if not edges:
            return []
        adj = defaultdict(set)
        degrees = defaultdict(int)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            degrees[u] += 1
            degrees[v] += 1
        odd_vertices = [v for v in degrees if degrees[v] % 2 != 0]
        num_odd = len(odd_vertices)
        if num_odd == 0:
            lb = 1 if len(edges) == 1 else 2
        else:
            lb = num_odd // 2

        best_paths = None
        best_count = float('inf')
        max_attempts = 300

        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            seed_val = seed * 10000 + attempt

            # Try both random greedy and target greedy
            if attempt % 3 == 0:
                # Target greedy with lb
                init = greedy_target(edges, lb, seed_val)
                if init is None:
                    init = greedy_target(edges, lb + 1, seed_val)
                if init is None:
                    init = greedy_random(edges, seed_val)
            elif attempt % 3 == 1:
                init = greedy_random(edges, seed_val)
            else:
                # Target with lb+1
                init = greedy_target(edges, lb + 1, seed_val)
                if init is None:
                    init = greedy_random(edges, seed_val)

            if not init:
                continue

            used_edges = set()
            for path in init:
                for j in range(len(path) - 1):
                    u, v = path[j], path[j+1]
                    used_edges.add(tuple(sorted((u, v))))

            # Merge endpoints
            merged = merge_endpoints(init, edges, adj, used_edges)

            # Internal reconnection
            merged = internal_reconnect(merged, edges, adj, used_edges)

            if validate_paths(merged, edges):
                if len(merged) < best_count:
                    best_count = len(merged)
                    best_paths = [list(p) for p in merged]
                    if best_count <= lb:
                        break

        if best_paths is None:
            for i in range(50):
                if time.time() > deadline - 5:
                    break
                paths = greedy_random(edges, seed * 10000 + 50000 + i)
                if validate_paths(paths, edges):
                    if len(paths) < best_count:
                        best_count = len(paths)
                        best_paths = [list(p) for p in paths]

        if best_paths is None:
            best_paths = [[u, v] for u, v in edges]

        return best_paths

    specs = [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]
    results = []
    for spec in specs:
        n, p, seed = spec
        if time.time() > deadline - 10:
            edges = instance(n, p, seed)
            results.append([[u, v] for u, v in edges])
            continue
        paths = solve_instance(n, p, seed)
        results.append(paths)
    while len(results) < 3:
        n, p, seed = specs[len(results)]
        edges = instance(n, p, seed)
        results.append([[u, v] for u, v in edges])
    return results

if __name__ == "__main__":
    res = construct()
    total = sum(len(r) for r in res)
    print(f"Total paths: {total}")
    for i, r in enumerate(res):
        print(f"Instance {i}: {len(r)} paths")
