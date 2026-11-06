import time
import sys
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

_N = P['n']
_EDGE_ORDER = P['edge_order']
_NEDGES = len(_EDGE_ORDER)

_EDGE_POS = {}
for _i, (_a, _b) in enumerate(_EDGE_ORDER, 1):
    _EDGE_POS[(_a, _b)] = _i
    _EDGE_POS[(_b, _a)] = _i

_ADJ = [[] for _ in range(_N)]
for _i, (_a, _b) in enumerate(_EDGE_ORDER, 1):
    _ADJ[_a].append((_i, _b))
    _ADJ[_b].append((_i, _a))
for _v in range(_N):
    _ADJ[_v].sort()

sys.setrecursionlimit(10000)

# Precompute edge indices for ILP
_EDGE_INDICES = {}
for _i, (_a, _b) in enumerate(_EDGE_ORDER):
    _EDGE_INDICES[(_a, _b)] = _i
    _EDGE_INDICES[(_b, _a)] = _i


def _build_masks(K):
    masks = [0] * _N
    for i, (u, v) in enumerate(_EDGE_ORDER, 1):
        if i > K:
            break
        masks[u] |= (1 << v)
        masks[v] |= (1 << u)
    return masks


def _is_connected(masks):
    start = -1
    for i in range(_N):
        if masks[i] != 0:
            start = i
            break
    if start == -1:
        return False
    visited = 0
    stack = [start]
    visited |= (1 << start)
    count = 1
    while stack:
        u = stack.pop()
        m = masks[u] & ~visited
        while m:
            lowbit = m & -m
            v = lowbit.bit_length() - 1
            m ^= lowbit
            visited |= (1 << v)
            count += 1
            stack.append(v)
    return count == _N


def _find_cycle_for_k(K, time_limit=None, start_time=None):
    if start_time is None:
        start_time = time.time()
    if time_limit is None:
        time_limit = 60.0

    masks = _build_masks(K)

    degs = [bin(masks[v]).count('1') for v in range(_N)]
    for v in range(_N):
        if degs[v] < 2:
            return None

    if not _is_connected(masks):
        return None

    full_mask = (1 << _N) - 1

    # Precompute sorted neighbor lists (by degree ascending - MRV heuristic)
    neigh_lists = []
    for v in range(_N):
        m = masks[v]
        neighbors = []
        while m:
            lowbit = m & -m
            u = lowbit.bit_length() - 1
            m ^= lowbit
            neighbors.append(u)
        neighbors.sort(key=lambda u: degs[u])
        neigh_lists.append(neighbors)

    if len(neigh_lists[0]) < 2:
        return None

    # Use iterative DFS with proper backtracking
    def _dfs(n1):
        path = [0] * _N
        path[0] = 0
        path[1] = n1
        v_mask = (1 << 0) | (1 << n1)
        # Stack entries: (vertex, depth, next_neighbor_index)
        stack = [(n1, 1, 0)]

        while stack:
            if time.time() - start_time > time_limit:
                return None

            cur, d, nidx = stack[-1]

            if d == _N - 1:
                if masks[cur] & (1 << 0):
                    return list(path[:d + 1])
                # Backtrack
                stack.pop()
                if not stack:
                    return None
                pc, pd, pnidx = stack[-1]
                stack[-1] = (pc, pd, pnidx + 1)
                v_mask &= ~(1 << cur)
                path[d] = 0
                continue

            found_next = False
            cur_neighbors = neigh_lists[cur]

            while nidx < len(cur_neighbors):
                nxt = cur_neighbors[nidx]
                nidx += 1
                stack[-1] = (cur, d, nidx)

                if v_mask & (1 << nxt):
                    continue

                # Check if adding nxt disconnects any unvisited vertex
                unvisited = full_mask ^ (v_mask | (1 << nxt))
                if unvisited:
                    # BFS from nxt through unvisited vertices
                    reach = 1 << nxt
                    bfs_stack = [nxt]
                    while bfs_stack:
                        u = bfs_stack.pop()
                        m = masks[u] & unvisited
                        while m:
                            lowbit = m & -m
                            w = lowbit.bit_length() - 1
                            m ^= lowbit
                            if not (reach & (1 << w)):
                                reach |= (1 << w)
                                bfs_stack.append(w)
                    if reach != (unvisited | (1 << nxt)):
                        continue

                v_mask |= (1 << nxt)
                path[d + 1] = nxt
                stack.append((nxt, d + 1, 0))
                found_next = True
                break

            if not found_next:
                stack.pop()
                if not stack:
                    return None
                pc, pd, pnidx = stack[-1]
                stack[-1] = (pc, pd, pnidx + 1)
                v_mask &= ~(1 << cur)
                path[d] = 0

        return None

    for n1 in neigh_lists[0]:
        if time.time() - start_time > time_limit:
            break
        result = _dfs(n1)
        if result is not None:
            return result

    return None


def _score_cycle(cycle):
    n = len(cycle)
    max_pos = 0
    for i in range(n):
        a = cycle[i]
        b = cycle[(i + 1) % n]
        pos = _EDGE_POS[(a, b)]
        if pos > max_pos:
            max_pos = pos
    return max_pos


def _two_opt_polish(cycle, deadline=None, start_time=None):
    """Apply 2-opt moves to minimize the max edge position in the cycle."""
    if start_time is None:
        start_time = time.time()
    if deadline is None:
        deadline = start_time + 10.0

    n = len(cycle)
    best_cycle = list(cycle)
    best_score = _score_cycle(best_cycle)

    def get_edge_pos(a, b):
        return _EDGE_POS[(a, b)]

    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Precompute edge positions for current cycle
        edge_pos = [get_edge_pos(cycle[k], cycle[(k + 1) % n]) for k in range(n)]
        # Prefix max: pref_max[i] = max(edge_pos[0..i])
        pref_max = [0] * n
        pref_max[0] = edge_pos[0]
        for i in range(1, n):
            pref_max[i] = max(pref_max[i - 1], edge_pos[i])
        # Suffix max: suff_max[i] = max(edge_pos[i..n-1])
        suff_max = [0] * n
        suff_max[n - 1] = edge_pos[n - 1]
        for i in range(n - 2, -1, -1):
            suff_max[i] = max(suff_max[i + 1], edge_pos[i])

        for i in range(n - 1):
            if time.time() >= deadline:
                return best_cycle
            for j in range(i + 2, n):
                if time.time() >= deadline:
                    return best_cycle
                # Edges being removed: edge_pos[i] (cycle[i]->cycle[i+1]) and edge_pos[j] (cycle[j]->cycle[j+1])
                # Edges being added: (cycle[i], cycle[j]) and (cycle[i+1], cycle[(j+1)%n])
                new1 = get_edge_pos(cycle[i], cycle[j])
                new2 = get_edge_pos(cycle[i + 1], cycle[(j + 1) % n])

                # max_other = max of all edge_pos except at indices i and j
                # Split into: [0..i-1], [i+1..j-1], [j+1..n-1]
                max_other = 0
                if i > 0:
                    max_other = max(max_other, pref_max[i - 1])
                if j - 1 > i:
                    max_other = max(max_other, suff_max[i + 1] if i + 1 <= j - 1 else 0)
                    # Actually need max of edge_pos[i+1..j-1]
                    if i + 1 <= j - 1:
                        # Use a different approach: max of range [i+1, j-1]
                        # pref_max[j-1] gives max[0..j-1], but we need to exclude [0..i]
                        # This is tricky with just prefix/suffix. Let me use a segment tree or just accept O(1) approximation
                        # Actually for correctness, let's compute it properly:
                        pass
                if j + 1 < n:
                    max_other = max(max_other, suff_max[j + 1])

                # For the middle segment [i+1, j-1], we need its max.
                # Let's just compute it directly since n=24 is small
                mid_max = 0
                for k in range(i + 1, j):
                    if edge_pos[k] > mid_max:
                        mid_max = edge_pos[k]
                max_other = max(max_other, mid_max)

                new_max = max(max_other, new1, new2)
                if new_max < best_score:
                    # Apply the move
                    segment = cycle[i + 1:j + 1]
                    segment.reverse()
                    cycle[i + 1:j + 1] = segment
                    best_score = new_max
                    best_cycle = list(cycle)
                    improved = True
                    break
            if improved:
                break

    return best_cycle


def _find_cycle_ilp(K, time_limit=5.0, start_time=None):
    if start_time is None:
        start_time = time.time()

    edges = []
    for i, (u, v) in enumerate(_EDGE_ORDER, 1):
        if i > K:
            break
        edges.append((u, v))

    if len(edges) < _N:
        return None

    m = len(edges)
    n = _N

    c = np.zeros(m)

    A_rows = []
    b_lows = []
    b_ups = []

    for v in range(n):
        row = np.zeros(m)
        count = 0
        for i, (u, w) in enumerate(edges):
            if u == v or w == v:
                row[i] = 1
                count += 1
        if count < 2:
            return None
        A_rows.append(row)
        b_lows.append(2)
        b_ups.append(2)

    A = np.array(A_rows)
    b_lows = np.array(b_lows)
    b_ups = np.array(b_ups)

    bounds = Bounds(lb=np.zeros(m), ub=np.ones(m))
    integrality = np.ones(m)

    constraints = [LinearConstraint(A, b_lows, b_ups)]

    max_iterations = 10

    for _ in range(max_iterations):
        if time.time() - start_time > time_limit:
            break

        try:
            res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds)
        except Exception:
            break

        if not res.success:
            return None

        x = res.x
        selected_edges = [edges[i] for i in range(m) if x[i] > 0.5]

        adj = [[] for _ in range(n)]
        for u, v in selected_edges:
            adj[u].append(v)
            adj[v].append(u)

        valid = True
        for v in range(n):
            if len(adj[v]) != 2:
                valid = False
                break

        if not valid:
            return None

        visited = [False] * n
        cycle = []
        if adj[0]:
            curr = 0
            prev = -1
            while True:
                if visited[curr]:
                    break
                visited[curr] = True
                cycle.append(curr)
                nexts = [u for u in adj[curr] if u != prev]
                if not nexts:
                    break
                prev, curr = curr, nexts[0]

        if len(cycle) == n and cycle[0] == 0:
            if 0 in adj[cycle[-1]]:
                return cycle

        visited = [False] * n
        subtour = []
        for start in range(n):
            if not visited[start]:
                curr = start
                prev = -1
                while True:
                    if visited[curr]:
                        break
                    visited[curr] = True
                    subtour.append(curr)
                    nexts = [u for u in adj[curr] if u != prev]
                    if not nexts:
                        break
                    prev, curr = curr, nexts[0]

                if len(subtour) < n:
                    row = np.zeros(m)
                    for i, (u, v) in enumerate(edges):
                        if u in subtour and v in subtour:
                            row[i] = 1
                    constraints.append(LinearConstraint(row.reshape(1, -1), len(subtour) - 1, len(subtour) - 1))
                    break

    return None


def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    baseline_cycle = list(range(_N))
    baseline_score = _score_cycle(baseline_cycle)

    best_cycle = baseline_cycle
    best_score = baseline_score

    # Binary search on K: find smallest K where G_K has a Hamilton cycle
    lo = 24
    hi = best_score

    while lo < hi and time.time() < deadline:
        if time.time() > deadline - 5:
            break
        mid = (lo + hi) // 2
        remaining = deadline - time.time()
        if remaining < 2:
            break

        # Try DFS first (faster for feasible K)
        cycle = _find_cycle_for_k(mid, time_limit=min(remaining - 2, 15.0), start_time=start_time)
        if cycle is None and time.time() < deadline - 3:
            # Fall back to ILP
            cycle = _find_cycle_ilp(mid, time_limit=min(deadline - time.time() - 3, 8.0), start_time=start_time)

        if cycle is not None and len(set(cycle)) == _N and len(cycle) == _N:
            s = _score_cycle(cycle)
            if s <= mid:
                best_cycle = cycle
                best_score = s
                hi = mid
            else:
                if s < best_score:
                    best_cycle = cycle
                    best_score = s
                hi = mid
        else:
            lo = mid + 1

    # Apply 2-opt polish
    if time.time() < deadline - 2:
        remaining = deadline - time.time()
        if remaining > 2:
            polished = _two_opt_polish(best_cycle, deadline=deadline, start_time=start_time)
            if polished is not None:
                s = _score_cycle(polished)
                if s < best_score:
                    best_cycle = polished
                    best_score = s

    # Try to find a cycle with the current best K
    if time.time() < deadline - 2:
        remaining = deadline - time.time()
        if remaining > 2:
            cycle = _find_cycle_for_k(best_score, time_limit=min(remaining - 2, 15.0), start_time=start_time)
            if cycle is None and time.time() < deadline - 2:
                cycle = _find_cycle_ilp(best_score, time_limit=min(deadline - time.time() - 2, 8.0), start_time=start_time)
            if cycle is not None and len(set(cycle)) == _N and len(cycle) == _N:
                s = _score_cycle(cycle)
                if s < best_score:
                    if time.time() < deadline - 1:
                        polished = _two_opt_polish(cycle, deadline=deadline, start_time=start_time)
                        if polished is not None:
                            s2 = _score_cycle(polished)
                            if s2 < best_score:
                                best_cycle = polished
                                best_score = s2
                            elif s < best_score:
                                best_cycle = cycle
                                best_score = s
                    else:
                        best_cycle = cycle
                        best_score = s

    return best_cycle
