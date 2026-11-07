"""Mutually orthogonal Latin squares of order 10: deterministic construction
plus a directed, time-budgeted search for additional squares.

The construction starts from a cyclic Latin square and tries to add new
squares that are orthogonal to every square already in the set.  If a square
cannot be found within the remaining time budget, the best verified set
found so far is returned.

The program returns a list of 10-by-10 arrays with Python integer symbols
0..9.  Every row and column contains each symbol exactly once, and every
pair of arrays has distinct ordered symbol pairs at the 100 positions.
"""
import time
import random

N = 10


def _cyclic():
    return [[(i + j) % N for j in range(N)] for i in range(N)]


def _verify(S):
    for sq in S:
        if not isinstance(sq, (list, tuple)):
            return False
        if len(sq) != N:
            return False
        for row in sq:
            if not isinstance(row, (list, tuple)):
                return False
            if len(row) != N:
                return False
            for v in row:
                if type(v) is not int:
                    return False
                if not (0 <= v <= N - 1):
                    return False
        for i in range(N):
            if sorted(sq[i]) != list(range(N)):
                return False
        for j in range(N):
            col = [sq[i][j] for i in range(N)]
            if sorted(col) != list(range(N)):
                return False
    m = len(S)
    for a in range(m):
        for b in range(a + 1, m):
            pairs = set()
            for i in range(N):
                for j in range(N):
                    pairs.add((S[a][i][j], S[b][i][j]))
            if len(pairs) != N * N:
                return False
    return True


def _find_orthogonal(S, node_limit, time_limit, t_start):
    row_used = [set() for _ in range(N)]
    col_used = [set() for _ in range(N)]
    pair_used = [set() for _ in range(len(S))]
    grid = [[-1] * N for _ in range(N)]
    state = {"count": 0, "found": None}
    K = len(S)

    def candidates(i, j):
        ru = row_used[i]
        cu = col_used[j]
        res = []
        for s in range(N):
            if s in ru or s in cu:
                continue
            ok = True
            for k in range(K):
                if (S[k][i][j], s) in pair_used[k]:
                    ok = False
                    break
            if ok:
                res.append(s)
        random.shuffle(res)
        return res

    def solve(idx):
        if state["found"] is not None:
            return
        if state["count"] > node_limit or (time.time() - t_start) > time_limit:
            return
        if idx == N * N:
            state["found"] = [row[:] for row in grid]
            return
        i = idx // N
        j = idx % N
        for s in candidates(i, j):
            grid[i][j] = s
            row_used[i].add(s)
            col_used[j].add(s)
            for k in range(K):
                pair_used[k].add((S[k][i][j], s))
            state["count"] += 1
            solve(idx + 1)
            if state["found"] is not None:
                return
            grid[i][j] = -1
            row_used[i].remove(s)
            col_used[j].remove(s)
            for k in range(K):
                pair_used[k].remove((S[k][i][j], s))

    solve(0)
    return state["found"]


def construct():
    t0 = time.time()
    deadline = t0 + 88.0
    random.seed(2024)

    S = [_cyclic()]

    while time.time() < deadline and len(S) < 9:
        prev_len = len(S)
        remaining = deadline - time.time()
        if remaining <= 0.05:
            break
        progress = False
        extra = 0
        while (time.time() < deadline) and (not progress) and (extra < 4):
            rem = deadline - time.time()
            if rem <= 0.05:
                break
            tl = min(1.4, rem * 0.9)
            nl = 250000 if extra == 0 else 120000
            newq = _find_orthogonal(S, nl, tl, time.time())
            extra += 1
            if newq is not None:
                S.append(newq)
                progress = True
        if not progress:
            break

    if _verify(S):
        return S
    for k in range(len(S), 0, -1):
        sub = S[:k]
        if _verify(sub):
            return sub
    return [_cyclic()]
