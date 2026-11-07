"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

_START_TIME = time.time()
_TIME_LIMIT = 85.0  # safety margin before 90s kill

def _time_ok():
    return (time.time() - _START_TIME) < _TIME_LIMIT

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _bfs_distances(adj_masks, n, start):
    dist = [-1] * n
    dist[start] = 0
    q = deque([start])
    while q:
        u = q.popleft()
        mask = adj_masks[u]
        m = mask
        while m:
            lb = m & -m
            v = lb.bit_length() - 1
            m ^= lb
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist

def _check_solution_fast(n, base_edges, added_edges):
    adj_masks = [0] * n
    for u, v in base_edges:
        adj_masks[u] |= (1 << v)
        adj_masks[v] |= (1 << u)
    for u, v in added_edges:
        if (adj_masks[u] >> v) & 1:
            return False
        adj_masks[u] |= (1 << v)
        adj_masks[v] |= (1 << u)
    for i in range(n):
        for j in range(i+1, n):
            if (adj_masks[i] >> j) & 1:
                if (adj_masks[i] & adj_masks[j]) != 0:
                    return False
    for start in range(n):
        dist = _bfs_distances(adj_masks, n, start)
        if -1 in dist:
            return False
        if max(dist) > 4:
            return False
    return True

def _is_triangle_free_add_mask(adj_masks, u, v):
    return (adj_masks[u] & adj_masks[v]) == 0

def _construct_initial(n, base_edges):
    base_set = set()
    for u, v in base_edges:
        base_set.add((min(u, v), max(u, v)))
    adj_masks = [0] * n
    for u, v in base_edges:
        adj_masks[u] |= (1 << v)
        adj_masks[v] |= (1 << u)
    added = []
    # Structured pattern: add edges at specific distances
    for dist in [5, 6, 7, 8, 9, 10, 11, 12]:
        for i in range(n - dist):
            j = i + dist
            if (i, j) in base_set:
                continue
            if _is_triangle_free_add_mask(adj_masks, i, j):
                adj_masks[i] |= (1 << j)
                adj_masks[j] |= (1 << i)
                added.append([i, j])
        # Early termination if diameter is already <= 4
        if _time_ok():
            ok = True
            for start in range(n):
                d = _bfs_distances(adj_masks, n, start)
                if -1 in d or max(d) > 4:
                    ok = False
                    break
            if ok:
                break
    # If still not valid, add remaining triangle-free edges
    if _time_ok():
        ok = True
        for start in range(n):
            d = _bfs_distances(adj_masks, n, start)
            if -1 in d or max(d) > 4:
                ok = False
                break
        if not ok:
            for i in range(n):
                for j in range(i+1, n):
                    if (i, j) in base_set:
                        continue
                    if not ((adj_masks[i] >> j) & 1):
                        if _is_triangle_free_add_mask(adj_masks, i, j):
                            adj_masks[i] |= (1 << j)
                            adj_masks[j] |= (1 << i)
                            added.append([i, j])
    return added

def _prune_edges(n, base, added):
    def is_valid_added(added_list):
        return _check_solution_fast(n, base, added_list)
    for _ in range(3):
        if not _time_ok():
            break
        changed = True
        while changed and _time_ok():
            changed = False
            for i in range(len(added) - 1, -1, -1):
                if not _time_ok():
                    break
                new_added = added[:i] + added[i+1:]
                if is_valid_added(new_added):
                    added = new_added
                    changed = True
    return added

def _local_search(n, base, added):
    base_set = set()
    for u, v in base:
        base_set.add((min(u, v), max(u, v)))
    def is_valid_added(added_list):
        return _check_solution_fast(n, base, added_list)
    improved = True
    while improved and _time_ok():
        improved = False
        # Try removals first
        for i in range(len(added)):
            if not _time_ok():
                break
            new_added = added[:i] + added[i+1:]
            if is_valid_added(new_added):
                added = new_added
                improved = True
                break
        # Try swaps if no removal worked
        if not improved and _time_ok():
            for i in range(len(added)):
                if not _time_ok():
                    break
                for u in range(n):
                    for v in range(u+1, n):
                        if not _time_ok():
                            break
                        if (u, v) in base_set:
                            continue
                        if (u, v) == tuple(added[i]) or (v, u) == tuple(added[i]):
                            continue
                        if any((a, b) == (u, v) or (b, a) == (u, v) for a, b in added):
                            continue
                        new_added = added[:i] + added[i+1:] + [[u, v]]
                        if is_valid_added(new_added) and len(new_added) < len(added):
                            added = new_added
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
    return added

def construct():
    n = P['n']
    base = P['edges']
    added = _construct_initial(n, base)
    added = _prune_edges(n, base, added)
    if _time_ok():
        added = _local_search(n, base, added)
    if _check_solution_fast(n, base, added):
        return [[int(u), int(v)] for u, v in added]
    # Fallback
    if _time_ok():
        adj_masks = [0] * n
        for u, v in base:
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
        added2 = []
        for i in range(n):
            for j in range(i+1, n):
                if (i, j) not in set((min(a,b), max(a,b)) for a,b in base):
                    if _is_triangle_free_add_mask(adj_masks, i, j):
                        adj_masks[i] |= (1 << j)
                        adj_masks[j] |= (1 << i)
                        added2.append([i, j])
        added2 = _prune_edges(n, base, added2)
        if _check_solution_fast(n, base, added2):
            return [[int(u), int(v)] for u, v in added2]
    return []

# EVOLVE-BLOCK-END
