"""Path decomposition of three fixed connected graphs (Gallai's conjecture)."""
import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START
def _instance(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _build_adj(edges):
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _validate(paths, edge_set):
    """Check paths are simple, use valid edges, edge-disjoint, cover all edges."""
    used = set()
    for p in paths:
        seen = set()
        for v in p:
            if v in seen:
                return False  # not simple
            seen.add(v)
        for i in range(len(p) - 1):
            e = tuple(sorted((p[i], p[i + 1])))
            if e in used:
                return False  # edge-disjoint violated
            if e not in edge_set:
                return False  # edge not in graph
            used.add(e)
    return used == edge_set


def _greedy_decomp(edges, adj, deadline):
    """Greedy: build paths by extending one edge at a time."""
    if not edges:
        return []
    edge_set = set(edges)
    deg = defaultdict(int)
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1

    # Priority: edges with high-degree endpoints first
    sorted_edges = sorted(edges, key=lambda e: -(deg[e[0]] + deg[e[1]]))

    paths = []
    used = set()
    # endpoint_map: vertex -> list of (path_idx, 0=start, 1=end)
    endpoint_map = defaultdict(list)

    for e in sorted_edges:
        if time.time() > deadline:
            break
        if e in used:
            continue
        u, v = e
        # Try to attach to an existing path endpoint
        attached = False
        for vert in (u, v):
            other = v if vert == u else u
            if vert in endpoint_map:
                # Find a path where vert is an endpoint
                for idx, pos in endpoint_map[vert]:
                    p = paths[idx]
                    if not p:
                        continue
                    if pos == 0:  # start
                        # p starts with vert, prepend other
                        new_p = [other] + p
                        # Check simplicity: other not in p
                        if other not in p:
                            # Update endpoint_map
                            endpoint_map[vert].remove((idx, pos))
                            endpoint_map[other].append((idx, 0))
                            # The end of p is still the same
                            paths[idx] = new_p
                            used.add(e)
                            attached = True
                            break
                    else:  # end
                        # p ends with vert, append other
                        if other not in p:
                            endpoint_map[vert].remove((idx, pos))
                            endpoint_map[other].append((idx, 1))
                            paths[idx] = p + [other]
                            used.add(e)
                            attached = True
                            break
            if attached:
                break
        if not attached:
            # Create new path
            idx = len(paths)
            paths.append([u, v])
            endpoint_map[u].append((idx, 0))
            endpoint_map[v].append((idx, 1))
            used.add(e)
    return paths


def _merge_pass(paths, edges, adj, edge_set, deadline):
    """Try to merge pairs of paths using unused edges."""
    best = [list(p) for p in paths]
    best_count = len(best)
    improved_any = True
    while improved_any and time.time() < deadline:
        improved_any = False
        # Compute used edges
        used = set()
        for p in best:
            for i in range(len(p) - 1):
                e = tuple(sorted((p[i], p[i + 1])))
                used.add(e)
        unused = [e for e in edges if e not in used]
        if not unused:
            break

        # Build endpoint info
        endpoints = {}  # vertex -> list of (path_idx, 0=start, 1=end)
        for idx, p in enumerate(best):
            if p:
                endpoints.setdefault(p[0], []).append((idx, 0))
                endpoints.setdefault(p[-1], []).append((idx, 1))

        # Try each unused edge to merge two paths
        merged_any = False
        for e in unused:
            if time.time() > deadline:
                break
            u, v = e
            # Get all endpoint occurrences for u and v
            occ_u = endpoints.get(u, [])
            occ_v = endpoints.get(v, [])
            # Try all combinations
            for idx1, pos1 in occ_u:
                for idx2, pos2 in occ_v:
                    if idx1 == idx2:
                        continue
                    # Merge: path1 (ending/starting at u) + path2 (starting/ending at v)
                    p1 = best[idx1]
                    p2 = best[idx2]
                    # Determine orientation
                    if pos1 == 1:  # p1 ends at u
                        left = p1
                    else:  # p1 starts at u
                        left = list(reversed(p1))  # now ends at u

                    if pos2 == 0:  # p2 starts at v
                        right = p2
                    else:  # p2 ends at v
                        right = list(reversed(p2))  # now starts at v

                    merged = left + right[1:]
                    # Validate: all edges in merged must be in edge_set, and simple
                    valid = True
                    seen = set()
                    for i in range(len(merged) - 1):
                        a, b = merged[i], merged[i + 1]
                        ee = tuple(sorted((a, b)))
                        if ee not in edge_set:
                            valid = False
                            break
                        if a in seen and b in seen:
                            pass
                        # Check simplicity
                    for vtx in merged:
                        if vtx in seen:
                            valid = False
                            break
                        seen.add(vtx)

                    if valid:
                        # Perform merge
                        new_paths = []
                        for idx in range(len(best)):
                            if idx != idx1 and idx != idx2:
                                new_paths.append(best[idx])
                        new_paths.append(merged)
                        if len(new_paths) < best_count:
                            best = new_paths
                            best_count = len(new_paths)
                            improved_any = True
                            merged_any = True
                            break
            if merged_any:
                break
        if not improved_any:
            break
    return best


def _solve_one(n, p, seed, deadline):
    edges = _instance(n, p, seed)
    edge_set = set(edges)
    adj = _build_adj(edges)
    if not edges:
        return []

    # Phase 1: Greedy
    paths = _greedy_decomp(edges, adj, deadline)

    # Phase 2: Merge passes (multiple rounds)
    while time.time() < deadline:
        before = len(paths)
        paths = _merge_pass(paths, edges, adj, edge_set, deadline)
        if len(paths) >= before:
            break

    # Final validation
    if not _validate(paths, edge_set):
        # Fallback: each edge as its own path
        paths = [list(e) for e in edges]

    return paths


def construct():
    start = time.time()
    total_deadline = start + 130  # 20s safety margin
    results = []
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))

    for i, spec in enumerate(specs):
        remaining = total_deadline - time.time()
        if remaining < 10:
            # Not enough time; trivial solution
            n, p, seed = spec
            edges = _instance(n, p, seed)
            results.append([list(e) for e in edges])
            continue
        # Allocate time: give more to later (larger) graphs
        sub_deadline = min(total_deadline - 5, time.time() + remaining * 0.8)
        results.append(_solve_one(*spec, sub_deadline))

    return results
# EVOLVE-BLOCK-END
