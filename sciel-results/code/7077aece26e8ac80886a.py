"""Rewritten program: directed backtracking search for mutually orthogonal
Latin squares of order 10, maximizing the number of squares returned."""
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
                if type(v) is not int:  # rejects bool and numpy ints
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

    # Precompute for each cell (i,j) the list of values already forbidden by pairs
    # and use forward checking
    def get_candidates(i, j):
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
        return res

    def forward_check(idx, i, j, s):
        """Check if placing s at (i,j) leaves any unassigned cell with no candidates."""
        for ni in range(N):
            if ni == i:
                continue
            for nj in range(N):
                if nj == j:
                    continue
                if grid[ni][nj] != -1:
                    continue
                # Check if this cell would still have at least one candidate
                # Quick check: row and col constraints
                if s in row_used[ni] or s in col_used[nj]:
                    continue  # s doesn't affect this cell's row/col for value s
                # Actually we need to check all candidates for this cell
                # This is expensive; do a lighter check
                # Check: does this cell have any candidate at all?
                has_cand = False
                for t in range(N):
                    if t in row_used[ni] or t in col_used[nj]:
                        continue
                    ok = True
                    for k in range(K):
                        if (S[k][ni][nj], t) in pair_used[k]:
                            ok = False
                            break
                    if ok:
                        has_cand = True
                        break
                if not has_cand:
                    return False
        return True

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
        cands = get_candidates(i, j)
        random.shuffle(cands)
        for s in cands:
            grid[i][j] = s
            row_used[i].add(s)
            col_used[j].add(s)
            for k in range(K):
                pair_used[k].add((S[k][i][j], s))
            state["count"] += 1
            if forward_check(idx, i, j, s):
                solve(idx + 1)
            else:
                pass  # skip this branch
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
    deadline = t0 + 55.0  # safety margin under the 90s limit
    random.seed()  # fresh random state

    S = [_cyclic()]

    while time.time() < deadline and len(S) < 9:
        prev_len = len(S)
        remaining = deadline - time.time()
        if remaining <= 0.05:
            break
        # A batch of attempts to try to grow the set by one square.
        progress = False
        extra = 0
        while (time.time() < deadline) and (not progress) and (extra < 6):
            rem = deadline - time.time()
            if rem <= 0.05:
                break
            # Give more time to the first attempt, less to subsequent
            if extra == 0:
                tl = min(3.0, rem * 0.85)
                nl = 500000
            else:
                tl = min(1.0, rem * 0.8)
                nl = 200000
            newq = _find_orthogonal(S, nl, tl, time.time())
            extra += 1
            if newq is not None:
                S.append(newq)
                progress = True
        if not progress:
            # Could not extend the current set further.
            break

    # Final safety: return the largest verified set we actually built.
    if _verify(S):
        return S
    # Fallbacks (should not be needed): largest verified prefix, else single.
    for k in range(len(S), 0, -1):
        sub = S[:k]
        if _verify(sub):
            return sub
    return [_cyclic()]
