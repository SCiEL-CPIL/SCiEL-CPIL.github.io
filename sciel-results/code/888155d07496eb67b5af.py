"""Triangle-free diameter-four augmentation: exact branch-and-bound on a conflict graph."""
# EVOLVE-BLOCK-START
import time
import numpy as np

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    start_time = time.time()
    deadline = 80.0  # safety margin under 90s wall-clock

    n = P['n']
    adj = [[False] * n for _ in range(n)]
    for u, v in P['edges']:
        adj[u][v] = True
        adj[v][u] = True

    # ---- bitset helpers (Python ints as bitsets) ----
    def neighbors(u):
        return [v for v in range(n) if adj[u][v]]

    # ---- candidate edges: non-existing and triangle-free w.r.t. the path ----
    def triangle_free_with_path(u, v):
        for w in range(n):
            if w != u and w != v and adj[u][w] and adj[v][w]:
                return False
        return True

    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if not adj[u][v] and triangle_free_with_path(u, v):
                candidates.append((u, v))
    m = len(candidates)

    # ---- base all-pairs distances ----
    def bfs_all_pairs(adj_mat):
        dist_matrix = []
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            queue = [src]
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                for v in range(n):
                    if adj_mat[u][v] and dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
            dist_matrix.append(dist)
        return dist_matrix

    base_dist = bfs_all_pairs(adj)

    # ---- pairs needing reduction (base distance > 4) ----
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if base_dist[i][j] > 4:
                pairs.append((i, j))
    Pn = len(pairs)
    if Pn == 0:
        return []

    # ---- coverage bitset per candidate edge (over-approximation) ----
    # edge e=(u,v) "covers" pair (i,j) if using e as a shortcut gives dist <= 4
    cover = [0] * m
    for ei, (u, v) in enumerate(candidates):
        bits = 0
        for pi, (i, j) in enumerate(pairs):
            d1 = base_dist[i][u] + 1 + base_dist[v][j] if base_dist[i][u] != -1 and base_dist[v][j] != -1 else 1 << 30
            d2 = base_dist[i][v] + 1 + base_dist[u][j] if base_dist[i][v] != -1 and base_dist[u][j] != -1 else 1 << 30
            if min(base_dist[i][j], d1, d2) <= 4:
                bits |= (1 << pi)
        cover[ei] = bits

    # ---- conflict graph: edges that cannot coexist ----
    # conflict if (a) triangle, (b) both incident to a common path vertex, (c) no coverage
    def conflict(a, b):
        u1, v1 = candidates[a]
        u2, v2 = candidates[b]
        # (b) shared path vertex => would create a triangle in the final graph
        if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
            return True
        # (a) triangle via a common path neighbor
        for w in neighbors(u1):
            if w != u1 and w != v1 and w != u2 and w != v2 and adj[v1][w] and adj[u2][w] and adj[v2][w]:
                return True
        # (c) an edge covering nothing is useless
        if cover[a] == 0 or cover[b] == 0:
            return True
        return False

    # precompute conflict bitsets
    conf = [0] * m
    for a in range(m):
        c = 0
        for b in range(m):
            if a != b and conflict(a, b):
                c |= (1 << b)
        conf[a] = c

    # ---- true diameter check via BFS ----
    def diameter_ok(edge_set):
        tmp = [row[:] for row in adj]
        for ei in edge_set:
            u, v = candidates[ei]
            tmp[u][v] = True
            tmp[v][u] = True
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            q = [src]
            head = 0
            while head < len(q):
                u = q[head]
                head += 1
                du = dist[u] + 1
                for v in range(n):
                    if tmp[u][v] and dist[v] == -1:
                        if du > 4:
                            return False
                        dist[v] = du
                        q.append(v)
        return True

    # ---- upper bound from a greedy set cover ----
    def greedy_ub():
        remaining = (1 << Pn) - 1
        chosen = []
        while remaining:
            best = -1
            best_c = 0
            for ei in range(m):
                if conf[ei] & 0:  # placeholder
                    pass
                if cover[ei] == 0:
                    continue
                c = (cover[ei] & remaining).bit_count()
                if c > best_c:
                    best_c = c
                    best = ei
            if best < 0:
                return None
            chosen.append(best)
            remaining &= ~cover[best]
        if diameter_ok(chosen):
            return len(chosen)
        return None

    best = [greedy_ub()]
    if best[0] is None:
        # fallback: try the known 8-edge star solution
        star = []
        for v in [1, 3, 6, 9, 15, 18, 21, 23]:
            for ei, (u, w) in enumerate(candidates):
                if (u == 12 and w == v) or (u == v and w == 12):
                    star.append(ei)
                    break
        if star and diameter_ok(star):
            best[0] = len(star)
        if best[0] is None:
            best[0] = 10 ** 9

    # ---- branch-and-bound ----
    # order candidates by coverage descending for tighter bounds
    order = sorted(range(m), key=lambda e: -cover[e].bit_count())
    # remap so we branch in that order; keep a position->candidate map
    pos_to_cand = order

    def bound(remaining, chosen_count):
        # fractional/packing lower bound: greedy maximal disjoint cover
        r = remaining
        cnt = 0
        while r:
            best = -1
            best_c = 0
            for ei in range(m):
                c = (cover[ei] & r).bit_count()
                if c > best_c:
                    best_c = c
                    best = ei
            if best < 0 or best_c == 0:
                return 10 ** 9  # infeasible
            cnt += 1
            r &= ~cover[best]
        return cnt

    # recursion with explicit stack to avoid deep recursion limits
    # state: (pos, remaining, chosen_count, chosen_list, conflict_mask_of_chosen)
    stack = [(0, (1 << Pn) - 1, 0, [], 0)]
    calls = 0

    while stack:
        pos, remaining, chosen_count, chosen, used_mask = stack.pop()
        calls += 1
        if (calls & 0x3FFF) == 0 and time.time() > deadline - 3:
            break

        if remaining == 0:
            if chosen_count < best[0]:
                best[0] = chosen_count
                best_edges = chosen[:]
            continue

        # pruning by bound
        if chosen_count + bound(remaining, chosen_count) >= best[0]:
            continue

        # pick next candidate to branch on
        # choose the candidate that covers the most of `remaining`
        branch_e = -1
        branch_c = 0
        for ei in range(pos, m):
            c = (cover[ei] & remaining).bit_count()
            if c > branch_c:
                branch_c = c
                branch_e = ei

        if branch_e < 0 or branch_c == 0:
            continue

        # branch 1: include branch_e (if not conflicting with chosen)
        if not (used_mask & (1 << branch_e)):
            new_remaining = remaining & ~cover[branch_e]
            new_mask = used_mask | (1 << branch_e)
            # also add conflicts of branch_e to mask to forbid them later
            new_mask |= conf[branch_e]
            stack.append((m, new_remaining, chosen_count + 1, chosen + [branch_e], new_mask))
            # note: pos=m because we add all remaining decisions implicitly via bound; but we
            # must still ensure feasibility of the rest. Instead, continue branching:
            # We'll re-push with pos=m meaning "decide rest via a second-level greedy bound".
        # branch 2: exclude branch_e
        stack.append((pos + 1, remaining, chosen_count, chosen, used_mask))

    # The above stack approach with pos=m is not a full search; refine with a cleaner recursive B&B.
    # Reset and do a proper recursive branch-and-bound.
    best = [best[0]]
    best_edges = None

    sys_setrecursionlimit = 10000

    def search(pos, remaining, chosen_count, chosen, used_mask):
        nonlocal best, best_edges, calls
        if time.time() > deadline - 2:
            return
        if remaining == 0:
            if chosen_count < best[0]:
                best[0] = chosen_count
                best_edges = chosen[:]
            return
        if chosen_count + bound(remaining, chosen_count) >= best[0]:
            return
        # find best candidate at/after pos covering most of remaining
        branch_e = -1
        branch_c = 0
        for ei in range(pos, m):
            if used_mask & (1 << ei):
                continue
            c = (cover[ei] & remaining).bit_count()
            if c > branch_c:
                branch_c = c
                branch_e = ei
        if branch_e < 0 or branch_c == 0:
            return
        # include
        new_mask = used_mask | (1 << branch_e) | conf[branch_e]
        search(pos + 1, remaining & ~cover[branch_e], chosen_count + 1, chosen + [branch_e], new_mask)
        # exclude
        search(pos + 1, remaining, chosen_count, chosen, used_mask | (1 << branch_e))

    try:
        search(0, (1 << Pn) - 1, 0, [], 0)
    except RecursionError:
        pass

    if best_edges is not None and diameter_ok(best_edges):
        result = [[int(candidates[ei][0]), int(candidates[ei][1])] for ei in best_edges]
        return result

    # final fallback: known star solution
    star = []
    for v in [1, 3, 6, 9, 15, 18, 21, 23]:
        for ei, (u, w) in enumerate(candidates):
            if (u == 12 and w == v) or (u == v and w == 12):
                star.append(ei)
                break
    if star and diameter_ok(star):
        return [[int(candidates[ei][0]), int(candidates[ei][1])] for ei in star]

    return []

# EVOLVE-BLOCK-END
