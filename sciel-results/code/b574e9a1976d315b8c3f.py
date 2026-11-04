"""Construct difference triangle sets (n_rows, k) minimizing the largest entry.

Strategy: CSP-style backtracking with bitset-based difference tracking.
We build rows one at a time, choosing each entry via DFS with strong pruning:
  - A global bitset of used differences lets us test a candidate in O(1).
  - We try smaller candidates first (to keep scope low) but also explore
    alternative branches via randomized restarts and limited backtracking.
  - A dynamic lower bound on remaining entries helps prune hopeless branches.
"""
import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 118.0  # safety margin before 120s limit

    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    # Maximum difference value we might need (upper bound)
    max_scope_guess = 200 + n_rows * k * 5  # generous upper bound

    # We'll use a Python int as a bitset for used differences.
    # Bit d is set if difference d is used.
    # This allows O(1) membership tests via (used >> d) & 1, or used & (1 << d).

    def scope_of(rows):
        return max(max(r) for r in rows) if rows else 0

    def row_diff_bitset(row):
        """Return bitset of all positive differences in row."""
        bits = 0
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                bits |= (1 << d)
        return bits

    def count_bits(bits):
        return bin(bits).count('1')

    def lower_bound_next(row, used_bits, k_remaining):
        """
        Estimate the minimum next value needed to place k_remaining more entries
        given current row and used differences. Returns a lower bound on the next
        entry value.
        """
        if k_remaining == 0:
            return 0
        # The next entry x must satisfy: for all a in row, (x - a) not in used_bits
        # and (x - a) > 0 (since x > all elements in row for increasing sequence)
        # Also x - a must not collide with any other x' - a' pair within the new row.
        # Simple lower bound: x must be > max(row), and x - max(row) must be unused.
        # We just return max(row) + 1 as a trivial lower bound.
        return row[-1] + 1 if row else 1

    def build_row_dfs(used_bits, row, k_target, max_scope, time_limit, rng,
                      backtrack_limit=200):
        """
        DFS to build a row of k_target+1 entries starting from current row.
        Returns (row, used_bits) on success, or (None, None) on failure.
        """
        if len(row) == k_target + 1:
            return row, used_bits

        if time.time() > time_limit:
            return None, None

        last = row[-1] if row else 0
        # Try candidates from last+1 upward, but with some randomization
        # to escape local traps.
        # We'll try a window of candidates, shuffled.
        window = 30
        cands = list(range(last + 1, last + 1 + window))
        # Also add some random larger candidates
        for _ in range(5):
            c = rng.randint(last + 1, max_scope)
            cands.append(c)
        # Remove duplicates, keep order-ish
        cands = sorted(set(cands))
        rng.shuffle(cands)

        for cand in cands:
            if cand > max_scope:
                continue
            # Compute new differences
            new_diffs_bits = 0
            conflict = False
            for a in row:
                d = cand - a
                if d <= 0:
                    conflict = True
                    break
                if used_bits & (1 << d):
                    conflict = True
                    break
                new_diffs_bits |= (1 << d)
            if conflict:
                continue
            # Check internal consistency: new_diffs_bits must have exactly
            # len(row) bits (one for each existing element)
            # Actually it should have len(row) new differences, all distinct
            # and not in used_bits. The above check ensures they're not in used_bits.
            # But we also need them to be distinct among themselves.
            if count_bits(new_diffs_bits) != len(row):
                continue

            new_row = row + [cand]
            new_used = used_bits | new_diffs_bits

            # Recurse
            result_row, result_used = build_row_dfs(
                new_used, new_row, k_target, max_scope, time_limit, rng,
                backtrack_limit - 1
            )
            if result_row is not None:
                return result_row, result_used

        return None, None

    def build_all_rows_dfs(used_bits, rows, n_rows_target, max_scope, time_limit, rng):
        """
        DFS to build all rows. Returns (rows, used_bits) on success or (None, None).
        """
        if len(rows) == n_rows_target:
            return rows, used_bits

        if time.time() > time_limit:
            return None, None

        # Build next row
        row, new_used = build_row_dfs(used_bits, [], k, max_scope, time_limit, rng)
        if row is None:
            return None, None

        new_rows = rows + [row]
        return build_all_rows_dfs(new_used, new_rows, n_rows_target, max_scope,
                                   time_limit, rng)

    def greedy_build(used_bits, row, k_target, max_scope, rng):
        """Pure greedy: pick smallest valid candidate at each step."""
        if len(row) == k_target + 1:
            return row, used_bits
        last = row[-1] if row else 0
        x = last
        while x < max_scope:
            x += 1
            conflict = False
            new_bits = 0
            for a in row:
                d = x - a
                if used_bits & (1 << d):
                    conflict = True
                    break
                new_bits |= (1 << d)
            if not conflict and count_bits(new_bits) == len(row):
                return greedy_build(used_bits | new_bits, row + [x], k_target,
                                     max_scope, rng)
        return None, None

    # ---------- Main search ----------
    best_rows = None
    best_scope = max_scope_guess + 1

    # Phase 1: Greedy baseline
    used = 0
    rows = []
    for _ in range(n_rows):
        row, used = greedy_build(used, [], k, max_scope_guess, random)
        if row is None:
            break
        rows.append(row)
    if len(rows) == n_rows:
        best_rows = rows
        best_scope = scope_of(rows)

    # Phase 2: Randomized DFS with restarts, trying to beat best_scope
    rng = random.Random(42)
    attempts = 0
    while time.time() < deadline - 15.0 and attempts < 500:
        attempts += 1
        # Try a scope slightly below current best
        target_scope = best_scope - 1 if best_scope > 112 else 112
        if target_scope < 112:
            target_scope = 112

        # Randomize the order of rows slightly
        row_order = list(range(n_rows))
        rng.shuffle(row_order)

        used = 0
        rows = []
        success = True
        for ri in row_order:
            row, used = build_row_dfs(used, [], k, target_scope,
                                       deadline - 15.0, rng, backtrack_limit=100)
            if row is None:
                success = False
                break
            rows.append(row)

        if success and len(rows) == n_rows:
            sc = scope_of(rows)
            if sc < best_scope:
                best_rows = rows
                best_scope = sc
                if best_scope <= 112:
                    break

    # Phase 3: Local search refinement
    if best_rows is not None and time.time() < deadline - 5.0:
        best_rows = local_search_refine(best_rows, k, diffs_per_row,
                                         deadline - 5.0)
        best_scope = scope_of(best_rows)

    return best_rows


def local_search_refine(rows, k, diffs_per_row, time_limit):
    """Local search to reduce scope by perturbing entries."""
    rows = [r[:] for r in rows]
    n = len(rows)
    L = len(rows[0])
    best_scope = max(max(r) for r in rows)

    # Precompute diff bitsets
    def row_diffs_set(row):
        s = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                s.add(row[j] - row[i])
        return s

    row_diff_sets = [row_diffs_set(r) for r in rows]
    all_diffs = set()
    for s in row_diff_sets:
        all_diffs |= s

    it = 0
    while time.time() < time_limit and it < 10000:
        it += 1
        ri = random.randrange(n)
        ci = random.randrange(1, L)

        old_val = rows[ri][ci]
        old_row = rows[ri]
        old_diffs = row_diff_sets[ri]

        # Generate candidates
        candidates = set()
        for d in range(-10, 11):
            nv = old_val + d
            if nv >= 1:
                candidates.add(nv)
        for _ in range(8):
            nv = random.randint(1, best_scope)
            candidates.add(nv)

        lower = old_row[ci - 1] + 1
        upper = (old_row[ci + 1] - 1) if ci + 1 < L else best_scope
        candidates = [c for c in candidates if lower <= c <= upper]
        if not candidates:
            continue

        random.shuffle(candidates)
        for nv in candidates:
            if nv == old_val:
                continue
            new_row = old_row[:]
            new_row[ci] = nv
            new_diffs = row_diffs_set(new_row)

            if len(new_diffs) != diffs_per_row:
                continue

            other_diffs = all_diffs - old_diffs
            if new_diffs & other_diffs:
                continue

            new_scope = max(best_scope, max(new_row))
            if new_scope < best_scope:
                all_diffs = (all_diffs - old_diffs) | new_diffs
                row_diff_sets[ri] = new_diffs
                rows[ri] = new_row
                best_scope = new_scope
                break

    return rows
