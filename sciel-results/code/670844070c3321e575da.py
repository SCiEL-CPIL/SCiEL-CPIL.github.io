import time
import random

N = 10

def _cyclic():
    return [[(i + j) % N for j in range(N)] for i in range(N)]

def _verify(S):
    # Type / structure checks
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
        # Latin: rows and columns are permutations
        for i in range(N):
            if sorted(sq[i]) != list(range(N)):
                return False
        for j in range(N):
            col = [sq[i][j] for i in range(N)]
            if sorted(col) != list(range(N)):
                return False
    # Pairwise orthogonality
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
    """Backtrack a new Latin square orthogonal to every square in S."""
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
    deadline = t0 + 45.0
    random.seed()

    S = [_cyclic()]

    # Try to find second square
    if len(S) < 9:
        prev_len = len(S)
        remaining = deadline - time.time()
        if remaining > 0.05:
            tl = min(10.0, remaining * 0.9) # Give it up to 10 seconds
            nl = 500000
            newq = _find_orthogonal(S, nl, tl, time.time())
            if newq is not None:
                S.append(newq)

    # Try to find third square (expected to fail, but try briefly)
    if len(S) < 9:
        remaining = deadline - time.time()
        if remaining > 0.05:
            tl = min(5.0, remaining * 0.9)
            nl = 200000
            newq = _find_orthogonal(S, nl, tl, time.time())
            if newq is not None:
                S.append(newq)

    # Verify and return
    if _verify(S):
        return S
    
    # Fallback
    for k in range(len(S), 0, -1):
        sub = S[:k]
        if _verify(sub):
            return sub
    return [_cyclic()]
