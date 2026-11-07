import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin for 90s limit

    n = 21
    edges = P['edges']
    edge_set = set()
    adj = defaultdict(set)
    for u, v in edges:
        edge_set.add((min(u, v), max(u, v)))
        adj[u].add(v)
        adj[v].add(u)

    # Precompute edge index
    edge_list = list(edge_set)
    edge_index = {e: i for i, e in enumerate(edge_list)}
    m = len(edge_list)

    # Precompute edge incidence per vertex
    vertex_edges = defaultdict(list)  # vertex -> list of (edge_idx, other_vertex)
    for idx, (u, v) in enumerate(edge_list):
        vertex_edges[u].append((idx, v))
        vertex_edges[v].append((idx, u))

    best_cycles = []
    best_count = 0

    def time_ok():
        return time.time() < deadline

    def find_cycles_greedy_with_backtrack(vertices, max_cycles, deadline):
        """Find edge-disjoint Hamiltonian cycles using DFS with constraint propagation."""
        vertices = frozenset(vertices)
        k = len(vertices)
        if k < 3:
            return []

        used = [False] * m
        cycles = []

        # Precompute adjacency within vertices
        v_list = list(vertices)
        v_set = set(v_list)
        local_adj = {}
        for v in v_list:
            local_adj[v] = []
            for eidx, w in vertex_edges[v]:
                if w in v_set:
                    local_adj[v].append((eidx, w))

        def can_complete(used, path, start, remaining):
            """Check if completion is possible (degree constraints)."""
            if not time_ok():
                return False
            last = path[-1]
            # Check degree constraints for each vertex in remaining + last
            for v in list(remaining) + [last]:
                deg = 0
                for eidx, w in local_adj[v]:
                    if not used[eidx]:
                        deg += 1
                # In the cycle, each vertex needs degree 2
                # Already used edges in path:
                used_deg = 0
                for eidx, w in local_adj[v]:
                    if used[eidx]:
                        # Check if this edge is in our path
                        # Simplified: just count available
                        pass
                # Count edges already assigned to v in the path
                path_edges = set()
                for i in range(len(path) - 1):
                    a, b = path[i], path[i + 1]
                    e = (min(a, b), max(a, b))
                    if e in edge_index:
                        path_edges.add(edge_index[e])
                # Also check if v is start and last connects to start
                assigned = 0
                for eidx, w in local_adj[v]:
                    if eidx in path_edges:
                        assigned += 1
                available = 0
                for eidx, w in local_adj[v]:
                    if not used[eidx] and eidx not in path_edges:
                        available += 1
                if assigned + available < 2:
                    return False
                if assigned > 2:
                    return False
            return True

        def backtrack(path, remaining, start):
            if not time_ok():
                return None
            if len(path) == k:
                last = path[-1]
                e = (min(last, start), max(last, start))
                if e in edge_index:
                    eidx = edge_index[e]
                    if not used[eidx]:
                        # Verify all edges in path are unused
                        for i in range(k):
                            a = path[i]
                            b = path[(i + 1) % k]
                            ee = (min(a, b), max(a, b))
                            ei = edge_index[ee]
                            if used[ei]:
                                return None
                        return path[:]
                return None

            last = path[-1]
            candidates = []
            for eidx, w in local_adj[last]:
                if w in remaining and not used[eidx]:
                    # Heuristic: prefer vertices with fewer remaining options
                    opts = 0
                    for e2, w2 in local_adj[w]:
                        if not used[e2] and w2 != last:
                            opts += 1
                    candidates.append((opts, eidx, w))
            candidates.sort()

            for opts, eidx, w in candidates:
                if not time_ok():
                    return None
                used[eidx] = True
                path.append(w)
                remaining.discard(w)
                result = backtrack(path, remaining, start)
                if result is not None:
                    return result
                path.pop()
                remaining.add(w)
                used[eidx] = False
            return None

        for attempt in range(max_cycles):
            if not time_ok():
                break
            # Try multiple starting vertices
            found = None
            for start in v_list:
                if not time_ok():
                    break
                path = [start]
                remaining = set(v_list) - {start}
                found = backtrack(path, remaining, start)
                if found is not None:
                    break
            if found is None:
                break
            # Mark edges as used
            for i in range(k):
                a = found[i]
                b = found[(i + 1) % k]
                e = (min(a, b), max(a, b))
                used[edge_index[e]] = True
            cycles.append(found)

        return cycles

    def count_induced_edges(vertices):
        count = 0
        v_set = set(vertices)
        for v in vertices:
            for eidx, w in vertex_edges[v]:
                if w in v_set:
                    count += 1
        return count // 2

    def try_vertex_set(vertices):
        nonlocal best_cycles, best_count
        if not time_ok():
            return
        k = len(vertices)
        if k < 3:
            return
        induced = count_induced_edges(vertices)
        upper = induced // k
        if upper <= best_count:
            return
        cycles = find_cycles_greedy_with_backtrack(vertices, upper, deadline)
        if len(cycles) > best_count:
            best_count = len(cycles)
            best_cycles = [list(c) for c in cycles]

    all_vertices = set(range(n))

    # Try full vertex set
    try_vertex_set(all_vertices)

    # Try all single-vertex removals
    for removed in range(n):
        if not time_ok():
            break
        subset = all_vertices - {removed}
        try_vertex_set(subset)

    # Try all pairs of vertex removals
    for i in range(n):
        for j in range(i + 1, n):
            if not time_ok():
                break
            subset = all_vertices - {i, j}
            try_vertex_set(subset)

    # Try random subsets
    random.seed(12345)
    for k in range(17, 20):
        for _ in range(50):
            if not time_ok():
                break
            subset = set(random.sample(range(n), k))
            try_vertex_set(subset)

    # Return best result
    if best_cycles:
        result = []
        for cycle in best_cycles:
            result.append([int(v) for v in cycle])
        return result
    else:
        # Fallback
        cycle = find_cycles_greedy_with_backtrack(all_vertices, 1, deadline)
        if cycle:
            return [[int(v) for v in cycle]]
        return [list(range(n))]

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}
