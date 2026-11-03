import time


def construct(n_rows, k):
    deadline = time.time() + 140.0  # safety margin before the 150 s kill

    # Valid baseline (guarantees we always return something correct).
    def greedy():
        used, rows = set(), []
        for _ in range(n_rows):
            row, x = [0], 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    best = greedy()
    best_scope = max(max(r) for r in best)

    # Counting lower bound on the scope: each row contributes k(k+1)/2 distinct
    # positive differences, all drawn from {1..M}, so M >= n_rows * k(k+1)/2.
    diffs_per_row = k * (k + 1) // 2
    lower = n_rows * diffs_per_row

    # Candidate scopes to probe, ascending, from the counting bound up to just
    # below the greedy scope (any solution found here strictly improves on greedy).
    hi = min(lower + 14, best_scope)
    scopes = list(range(lower, hi + 1))

    total_budget = max(5.0, deadline - time.time() - 6.0)
    per = total_budget / max(1, len(scopes))

    for M in scopes:
        if time.time() > deadline - 4:
            break
        tlim = min(per, deadline - time.time() - 3)
        if tlim < 1.0:
            break
        node_budget = int(max(50000, tlim * 4000))
        res = _search_scope(M, n_rows, k, node_budget, tlim)
        if res is not None:
            scope = max(max(r) for r in res)
            if scope < best_scope:
                best = res
                best_scope = scope
            # First feasible scope (ascending) is the minimal achievable scope.
            return best

    return best


def _search_scope(M, n_rows, k, node_budget, time_limit):
    """Backtracking search for n_rows lex-increasing rows, each an increasing
    (k+1)-tuple from 0 with max entry <= M, whose positive-difference sets are
    pairwise disjoint. Returns the list of rows, or None if not found within
    the node/time budget."""
    used = set()
    rows = []
    prev_holder = [None]
    nodes = [0]
    aborted = [False]
    start = time.time()
    n = k + 1

    def place_row(r):
        if aborted[0]:
            return False
        if r == n_rows:
            return True
        cur = [0] * n
        cur_diffs = set()

        def rec(idx, exceeded):
            if aborted[0]:
                return False
            nodes[0] += 1
            if (nodes[0] & 511) == 0:
                if nodes[0] > node_budget or (time.time() - start) > time_limit:
                    aborted[0] = True
                    return False

            if idx == n:
                row = tuple(cur)
                # Lex symmetry breaking: must be strictly greater than previous row.
                if prev_holder[0] is not None and row <= prev_holder[0]:
                    return False
                # Commit this row.
                used |= cur_diffs
                rows.append(list(cur))
                old_prev = prev_holder[0]
                prev_holder[0] = row
                if place_row(r + 1):
                    return True
                # Undo.
                rows.pop()
                used -= cur_diffs
                prev_holder[0] = old_prev
                return False

            if idx == 0:
                cur[0] = 0  # every row starts at 0
                return rec(1, False)

            prev = prev_holder[0]
            lo = cur[idx - 1] + 1
            if prev is not None and not exceeded and lo < prev[idx]:
                lo = prev[idx]
            # Leave room for the remaining (n-1-idx) strictly larger elements.
            hi_v = M - (n - 1 - idx)
            if hi_v < lo:
                return False

            for v in range(lo, hi_v + 1):
                cur[idx] = v
                new_diffs = {v - c for c in cur[:idx]}
                if (new_diffs & used) or (new_diffs & cur_diffs):
                    continue
                cur_diffs |= new_diffs
                ne = exceeded
                if not ne and prev is not None:
                    if v > prev[idx]:
                        ne = True
                    elif v < prev[idx]:
                        cur_diffs -= new_diffs
                        continue
                if rec(idx + 1, ne):
                    return True
                cur_diffs -= new_diffs
            return False

        return rec(0, False)

    if place_row(0):
        return rows
    return None
