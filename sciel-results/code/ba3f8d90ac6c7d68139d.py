"""Large clique subdivisions in a pinned graph: exact backtracking path assignment."""
import time
import sys
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

_N = P['n']
_ADJ = [set() for _ in range(_N)]
for _a, _b in P['edges']:
    _ADJ[_a].add(_b)
    _ADJ[_b].add(_a)
_ADJ_MASK = [0] * _N
for _i in range(_N):
    _m = 0
    for _j in _ADJ[_i]:
        _m |= (1 << _j)
    _ADJ_MASK[_i] = _m

_DEADLINE = time.time() + 80.0


def _time_ok():
    return time.time() < _DEADLINE


def construct():
    sys.setrecursionlimit(10000)
    n = _N
    adj = _ADJ
    adj_mask = _ADJ_MASK

    best = {'branches': [int(P['edges'][0][0]), int(P['edges'][0][1])],
            'paths': [[int(P['edges'][0][0]), int(P['edges'][0][1])]]}
    best_k = 2

    # All candidate branch sets are checked by exact path-assignment backtracking.
    # We enumerate branch sets in increasing size, in a smart order, and stop growing
    # a set when infeasible. We use a global best.

    # Precompute: for a given branch set, the internal vertices are those not in it.
    # For each pair (a,b) in the branch set, we need a simple path from a to b
    # whose interior avoids all branches and is disjoint from other path interiors.

    # We order pairs so that the "hardest" (fewest available paths) are first.
    # For backtracking we assign one path at a time, marking used internal vertices.

    def bfs_paths(a, b, forbidden_mask, max_count):
        """Return up to max_count simple paths from a to b with interior avoiding
        forbidden_mask (and avoiding all branches which are in forbidden_mask)."""
        # We'll do a DFS enumeration but cap it. Use iterative with a stack.
        paths = []
        # interior vertices = all except a,b, minus forbidden
        # do DFS
        stack = [(a, [a])]
        # To keep it bounded, limit path length
        max_len = n  # simple path
        while stack and len(paths) < max_count:
            u, path = stack.pop()
            if u == b:
                if len(path) >= 2:
                    paths.append(path)
                continue
            if len(path) >= max_len:
                continue
            # neighbors not in path and not forbidden (except b)
            seen = set(path)
            mm = adj_mask[u]
            while mm:
                lsb = mm & (-mm)
                v = lsb.bit_length() - 1
                mm ^= lsb
                if v in seen:
                    continue
                if v != b and (forbidden_mask >> v) & 1:
                    continue
                stack.append((v, path + [v]))
        return paths

    def check_and_extract(branches):
        """Return list of paths (one per unordered pair) if feasible, else None."""
        k = len(branches)
        branch_set = set(branches)
        # forbidden mask = all branch vertices (interiors must avoid every branch)
        forbidden = 0
        for v in branches:
            forbidden |= (1 << v)
        # list of unordered pairs
        pairs = [(branches[i], branches[j]) for i in range(k) for j in range(i + 1, k)]
        # Order pairs by difficulty: fewer candidate paths first.
        # Estimate candidate count by BFS count with only branch-forbidden.
        def pair_key(p):
            a, b = p
            cnt = len(bfs_paths(a, b, forbidden, 4))
            return cnt
        pairs.sort(key=pair_key)
        # Backtrack: assign a path to each pair, marking used interior vertices.
        used_mask = 0  # interior vertices already used
        assignment = [None] * len(pairs)

        def bt(idx):
            if not _time_ok():
                return False
            if idx == len(pairs):
                return True
            a, b = pairs[idx]
            # candidate paths must avoid forbidden AND used interior
            eff_forbidden = forbidden | used_mask
            cands = bfs_paths(a, b, eff_forbidden, 6)
            # try each candidate
            for p in cands:
                # interior = p[1:-1]
                interior_mask = 0
                for v in p[1:-1]:
                    if (used_mask >> v) & 1:
                        break
                    interior_mask |= (1 << v)
                else:
                    assignment[idx] = p
                    if bt(idx + 1):
                        return True
                    assignment[idx] = None
            return False

        if bt(0):
            # build result in original unordered-pair order
            res = {}
            for (a, b), p in zip(pairs, assignment):
                res[frozenset((a, b))] = p
            return [res[frozenset((branches[i], branches[j]))]
                    for i in range(k) for j in range(i + 1, k)]
        return None

    # Enumerate branch sets. Strategy: grow greedily by feasibility, then local search
    # (swap) to improve, plus a bounded exhaustive search over small sets.
    order = sorted(range(n), key=lambda v: -len(adj[v]))

    def update_best(branches):
        nonlocal best, best_k
        if len(branches) < 2:
            return
        paths = check_and_extract(branches)
        if paths is not None and len(branches) > best_k:
            best_k = len(branches)
            best = {'branches': [int(v) for v in branches],
                    'paths': [[int(x) for x in p] for p in paths]}

    # Greedy growth
    chosen = []
    for v in order:
        trial = chosen + [v]
        if len(trial) < 2:
            chosen = trial
            continue
        if check_and_extract(trial) is not None:
            chosen = trial
            if not _time_ok():
                break
    if len(chosen) >= 2:
        update_best(chosen)

    # Local search: swap a branch with a non-branch to try to increase size.
    def try_grow(branches):
        # try adding each non-branch vertex
        bset = set(branches)
        for v in order:
            if v in bset:
                continue
            trial = branches + [v]
            if check_and_extract(trial) is not None:
                return trial
        return None

    cur = list(chosen)
    improved = True
    while improved and _time_ok():
        improved = False
        grown = try_grow(cur)
        if grown is not None:
            cur = grown
            improved = True
            update_best(cur)

    # Bounded exhaustive search for small branch sets (size up to best_k+1) to ensure
    # we don't miss a better configuration. Use combinations from `order` prefix.
    max_search_k = best_k + 1
    # limit total combinations tried
    budget = 200000
    tried = 0
    from itertools import combinations
    # Try all combinations of size max_search_k from the top ~12 high-degree vertices
    pool = order[:12]
    for combo in combinations(pool, max_search_k):
        tried += 1
        if tried > budget or not _time_ok():
            break
        if check_and_extract(list(combo)) is not None:
            update_best(list(combo))

    # Also try size best_k+1 from a slightly larger pool if time allows
    if _time_ok() and best_k + 1 <= n:
        pool2 = order[:14]
        tried2 = 0
        for combo in combinations(pool2, best_k + 1):
            tried2 += 1
            if tried2 > budget or not _time_ok():
                break
            if check_and_extract(list(combo)) is not None:
                update_best(list(combo))

    if len(best['branches']) < 2:
        u, v = P['edges'][0]
        best = {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}

    br = [int(x) for x in best['branches']]
    pl = [[int(x) for x in p] for p in best['paths']]
    return {'branches': br, 'paths': pl}
