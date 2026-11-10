import time
import random
import math

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0, minimizing the
    largest entry, subject to all positive differences within rows being distinct
    across all rows.

    Strategy (bin-packing / First-Fit-Decreasing):
      - Model each row as a "bin" whose difference-set must be disjoint from every
        other row's difference-set.
      - Pre-generate a pool of candidate rows at many tight scopes (the "bin sizes").
      - Assign rows greedily in First-Fit-Decreasing order: place the largest
        difference-sets into the tightest scopes first, which avoids the combinatorial
        blowup of general graph independent-set searches.
      - Finish with a per-row tightening pass that shrinks the widest row one scope
        at a time using a bounded backtracking search against the other rows' diffs.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin vs 150s kill

    def now():
        return time.time()

    def diff_set_of_row(row):
        s = set()
        n = len(row)
        for i in range(n):
            ai = row[i]
            for j in range(i + 1, n):
                s.add(row[j] - ai)
        return s

    def validate(rows):
        used = set()
        for r in rows:
            if not r or r[0] != 0:
                return False
            for a, b in zip(r, r[1:]):
                if b <= a:
                    return False
            for d in diff_set_of_row(r):
                if d in used:
                    return False
                used.add(d)
        return True

    def max_val(rows):
        return max(max(r) for r in rows)

    # ---------- Candidate row generators ----------
    def greedy_row(used, temp=0.0, scope_limit=None):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            if scope_limit is not None and x > scope_limit:
                return None
            new_diffs = {x - a for a in row}
            if new_diffs & used:
                continue
            if temp > 0 and random.random() < temp:
                continue
            row.append(x)
            used = used | new_diffs
        return row

    def backtrack_row(used_global, scope_limit, time_limit=None):
        """Find a valid row of size k+1, max <= scope_limit, disjoint from used_global."""
        if time_limit is not None and now() > time_limit:
            return None
        target_len = k + 1

        def dfs(row, used_local):
            if time_limit is not None and now() > time_limit:
                return None
            if len(row) == target_len:
                return row[:]
            last = row[-1]
            rem = target_len - len(row)
            max_x = scope_limit - (rem - 1)
            if last + 1 > max_x:
                return None
            for x in range(last + 1, max_x + 1):
                if time_limit is not None and now() > time_limit:
                    return None
                conflict = False
                for a in row:
                    if (x - a) in used_local:
                        conflict = True
                        break
                if conflict:
                    continue
                new_row = row + [x]
                new_used = set(used_local)
                for a in row:
                    new_used.add(x - a)
                res = dfs(new_row, new_used)
                if res is not None:
                    return res
            return None

        return dfs([0], set(used_global))

    # ---------- First pass: pure greedy baseline ----------
    used = set()
    base_rows = []
    for _ in range(n_rows):
        r = greedy_row(used, temp=0.0)
        if r is None:
            break
        base_rows.append(r)
        used |= diff_set_of_row(r)
    if len(base_rows) < n_rows:
        # extremely unlikely fallback: extend with a big scope
        for _ in range(n_rows - len(base_rows)):
            r = greedy_row(used, temp=0.0, scope_limit=500)
            base_rows.append(r)
            used |= diff_set_of_row(r)
    best_rows = [r[:] for r in base_rows]
    best_max = max_val(best_rows)

    # ---------- Build the candidate pool (the "bins") ----------
    # For a set of tight scopes, generate several candidate rows per scope.
    # Each candidate row has a known max (its scope) and a known diff-set.
    # We will sort candidates by scope (largest diff-set first) and pack them.
    #
    # The scope range: from a lower bound up to best_max.
    # Lower bound: at least k (0..k) but diffs collide; use a practical floor.
    scope_floor = k + 2
    scope_ceiling = best_max  # never worse than baseline

    # Generate candidate rows for each scope in the range.
    # We keep, for each scope, the best (smallest diff-set "spread") candidate
    # that is internally valid. We generate a handful per scope with randomness.
    candidates_by_scope = {}
    rng = random.Random(12345)

    # Determine how many candidates per scope based on available time.
    n_scopes = scope_ceiling - scope_floor + 1
    # Budget: generate at most ~ (time_available / est_cost) total candidates.
    # We'll cap per-scope candidates to keep it fast.
    per_scope_trials = 8

    for scope in range(scope_floor, scope_ceiling + 1):
        if now() > deadline - 40:
            break
        best_cand = None
        best_cand_diffs = None
        for trial in range(per_scope_trials):
            temp = 0.0 if trial == 0 else (0.15 * trial / per_scope_trials)
            r = greedy_row(set(), temp=temp, scope_limit=scope)
            if r is None:
                continue
            if r[-1] > scope:
                continue
            ds = diff_set_of_row(r)
            # Prefer candidates whose max diff is small (tight packing)
            # and whose diff-set is "dense" (fewer gaps) -> use max diff as proxy.
            cand_key = max(ds)
            if best_cand is None or cand_key < max(best_cand_diffs):
                best_cand = r
                best_cand_diffs = ds
        if best_cand is not None:
            # Keep a few alternatives per scope for diversity (helps FFD packing)
            if scope not in candidates_by_scope:
                candidates_by_scope[scope] = []
            candidates_by_scope[scope].append((best_cand, best_cand_diffs))

    # ---------- First-Fit-Decreasing packing ----------
    # Build a flat list of (scope, row, diffset) sorted by scope DESCENDING.
    flat_candidates = []
    for scope, cands in candidates_by_scope.items():
        for (row, ds) in cands:
            flat_candidates.append((scope, row, ds))
    # Sort: largest scope (largest diff-set) first -> First-Fit-Decreasing
    flat_candidates.sort(key=lambda t: (-t[0], -max(t[2])))

    # Greedy assignment: iterate candidates in FFD order, pick a candidate whose
    # diff-set is disjoint from the already-assigned rows' diff-sets, until we
    # have n_rows rows. Among same-scope candidates, prefer the one with the
    # smallest max diff (tightest).
    def try_ffd_pack():
        used = set()
        chosen = []
        chosen_scopes = []
        for scope, row, ds in flat_candidates:
            if len(chosen) == n_rows:
                break
            if ds & used:
                continue
            chosen.append(row)
            chosen_scopes.append(scope)
            used |= ds
        return chosen, chosen_scopes

    ffd_rows, _ = try_ffd_pack()
    if len(ffd_rows) == n_rows and validate(ffd_rows):
        ffd_max = max_val(ffd_rows)
        if ffd_max < best_max:
            best_max = ffd_max
            best_rows = [r[:] for r in ffd_rows]

    # ---------- Tightening pass: shrink the widest row one scope at a time ----------
    improved = True
    while improved and now() < deadline - 12:
        improved = False
        # Work on a copy so we can roll back if a tightening hurts the global max.
        cur_rows = [r[:] for r in best_rows]
        cur_max = best_max
        # Repeatedly pick the row with the largest max and try to shrink it.
        while now() < deadline - 12:
            # Find the index of the row with the current largest max
            idx = max(range(n_rows), key=lambda i: cur_rows[i][-1])
            current_scope = cur_rows[idx][-1]
            target_scope = current_scope - 1
            # Stop shrinking if we'd go below the floor or below the global min possible
            if target_scope < scope_floor:
                break
            # Diffs from all other rows
            other_diffs = set()
            for j in range(n_rows):
                if j != idx:
                    other_diffs |= diff_set_of_row(cur_rows[j])
            new_row = backtrack_row(other_diffs, target_scope, time_limit=deadline - 12)
            if new_row is None:
                # Try a slightly looser target to make progress (shrink by 1 was too tight)
                # Actually if we can't shrink by 1, we likely can't shrink at all; stop.
                break
            # Validate the full set
            candidate_rows = cur_rows[:]
            candidate_rows[idx] = new_row
            if not validate(candidate_rows):
                break
            new_max = max_val(candidate_rows)
            if new_max < cur_max:
                cur_max = new_max
                cur_rows = candidate_rows
                improved = True
            else:
                # Shrinking this row didn't lower the global max (another row is the max).
                # Move on; break to avoid an infinite loop on the same row.
                break
        if improved:
            best_max = cur_max
            best_rows = cur_rows

    return best_rows

# EVOLVE-BLOCK-END
