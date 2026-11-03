import time
import random

_DEADLINE = 120.0  # seconds; evaluator kills at 150s, keep safety margin


def _row_diffs(row):
    d = set()
    n = len(row)
    for i in range(n):
        ai = row[i]
        for j in range(i + 1, n):
            d.add(row[j] - ai)
    return d


def _is_valid(rows):
    used = set()
    for row in rows:
        for d in _row_diffs(row):
            if d in used:
                return False
            used.add(d)
    return True


def _lower_bound(n_rows, k):
    # Counting bound: total distinct differences needed = n_rows * k(k+1)/2
    # Each difference in 1..M is usable at most once.
    need = n_rows * k * (k + 1) // 2
    M = 0
    while M * (M + 1) // 2 < need:
        M += 1
    return M


def construct(n_rows, k):
    t0 = time.time()
    rng = random.Random(12345)

    # Trivial cases
    if k <= 0:
        return [[0]] * n_rows

    lb = _lower_bound(n_rows, k)

    best = None
    best_score = None

    def deadline_ok():
        return (time.time() - t0) < (_DEADLINE - 8.0)

    # We'll attempt iterative deepening on M. For each M, run a backtracking search
    # that tries to fill n_rows rows of length k+1 with marks in [0, M].
    # To keep runtime bounded, we cap the number of nodes per M and overall.

    # Order rows: process "larger" rows first is not directly meaningful since all
    # rows have the same length k+1. Instead, we can try to place rows in an order
    # that constrains early. We'll just go in order but use a smart candidate order.

    def backtrack(m_max, node_budget):
        """Try to build n_rows rows with max entry <= m_max.
        Returns a list of rows or None. Uses node_budget as a cap on recursive
        nodes explored (approx)."""
        nonlocal best, best_score

        used = set()  # globally used differences
        rows = []
        # State: we build row by row. Within a row, we place marks in increasing order.
        # For efficiency, we track the current row's marks and the set of differences
        # introduced so far in this row (to check internal conflicts quickly).

        # We'll use a recursive function that takes (row_idx, mark_idx, current_row)
        # and maintains used as a global mutable set with undo.

        # To speed up, precompute for a given current_row and candidate x, the new
        # differences = {x - a for a in current_row}. If any is in used or in
        # current_row's own diff set, reject.

        # We'll store node counter.
        nodes = [0]

        def rec(row_idx, mark_idx, cur_row, cur_diffs):
            if not deadline_ok():
                return False
            nodes[0] += 1
            if nodes[0] > node_budget:
                return False

            if row_idx == n_rows:
                # All rows placed
                nonlocal best, best_score
                candidate = [list(r) for r in rows]
                # Validate just in case
                if _is_valid(candidate):
                    scope = max(max(r) for r in candidate)
                    if best is None or scope < best_score:
                        best = candidate
                        best_score = scope
                return True

            if mark_idx == k + 1:
                # Row complete
                rows.append(list(cur_row))
                if rec(row_idx + 1, 0, [0], set()):
                    return True
                rows.pop()
                return False

            # Place mark at position mark_idx. cur_row[0]=0, cur_row[1..mark_idx-1] placed.
            # Next mark must be > cur_row[-1] (if mark_idx>0) else >0.
            start = cur_row[-1] + 1 if mark_idx > 0 else 1
            # Upper bound: m_max
            # Try candidates in increasing order (greedy-first) but we can also
            # randomize order occasionally to escape local minima.
            candidates = list(range(start, m_max + 1))

            # Heuristic: try candidates that produce "common" differences first?
            # Actually, to find a solution quickly, increasing order is good.
            # But to explore, we can occasionally shuffle the tail.
            if mark_idx >= 2 and rng.random() < 0.3 and len(candidates) > 4:
                # Keep first few in order (to preserve some greediness), shuffle rest
                head = candidates[:2]
                tail = candidates[2:]
                rng.shuffle(tail)
                candidates = head + tail

            for x in candidates:
                if x > m_max:
                    break
                new_diffs = set()
                conflict = False
                for a in cur_row:
                    d = x - a
                    if d in used or d in cur_diffs or d in new_diffs:
                        conflict = True
                        break
                    new_diffs.add(d)
                if conflict:
                    continue

                # Place x
                cur_row.append(x)
                cur_diffs |= new_diffs
                used |= new_diffs

                if rec(row_idx, mark_idx + 1, cur_row, cur_diffs):
                    return True

                # Undo
                cur_row.pop()
                cur_diffs -= new_diffs
                used -= new_diffs

            return False

        # Start
        if rec(0, 0, [0], set()):
            return True
        return False

    # Iterative deepening on M starting from lb
    M = lb
    node_budget = 200000  # nodes per M attempt
    while M <= 200:  # hard cap on M to avoid infinite loop
        if not deadline_ok():
            break
        # Try to find a solution with max entry <= M
        # We'll try a few randomizations per M to improve chances
        attempts = 3
        for _ in range(attempts):
            if not deadline_ok():
                break
            if backtrack(M, node_budget):
                if best is not None:
                    return best
            # If we found a valid best, return it
            if best is not None:
                return best
        # No solution found at this M with our budget; increase M
        M += 1

    # If we exhausted search, return best found (may be None)
    if best is not None:
        return best

    # Fallback: greedy (original seed) to guarantee a valid answer
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
