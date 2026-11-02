import time
import random
import math

_START = None

def construct(n_rows, k):
    global _START
    if _START is None:
        _START = time.time()
    deadline = _START + 132.0  # safety margin before 150s kill

    n = k + 1  # entries per row
    d = k * (k + 1) // 2  # differences per row
    total_diffs = n_rows * d

    def get_scope(rows):
        return max(max(r) for r in rows)

    def full_validate(rows):
        seen = set()
        scope = 0
        for row in rows:
            if len(row) != n:
                return None
            if row[0] != 0:
                return None
            for i in range(n):
                for j in range(i + 1, n):
                    diff = row[j] - row[i]
                    if diff <= 0 or diff in seen:
                        return None
                    seen.add(diff)
            scope = max(scope, max(row))
        return scope

    # ---------- Fast greedy init ----------
    def greedy_init():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < n:
                found = False
                for trial in range(x + 1, x + 60):
                    new = {trial - a for a in row}
                    if not (new & used):
                        row.append(trial)
                        used |= new
                        x = trial
                        found = True
                        break
                if not found:
                    while True:
                        x += 1
                        new = {x - a for a in row}
                        if not (new & used):
                            row.append(x)
                            used |= new
                            break
            rows.append(row)
        return rows

    best_rows = greedy_init()
    best_scope = get_scope(best_rows)

    # ---------- Fast conflict mask ----------
    # Use a bytearray to mark used differences for O(1) conflict checks.
    def make_mask(size):
        return bytearray(size)

    def mask_add(mask, diffs):
        for dd in diffs:
            if dd < len(mask):
                mask[dd] = 1

    def mask_sub(mask, diffs):
        for dd in diffs:
            if dd < len(mask):
                mask[dd] = 0

    def mask_conflict(mask, diffs):
        for dd in diffs:
            if dd < len(mask) and mask[dd]:
                return True
            if dd >= len(mask):
                return True
        return False

    # ---------- Compute diffs and entries from gaps ----------
    def gaps_to_entries(gaps):
        entries = [0]
        s = 0
        for g in gaps:
            s += g
            entries.append(s)
        return entries

    def entries_to_diffs(entries):
        diffs = set()
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(entries[j] - entries[i])
        return diffs

    # ---------- Targeted descent local search (fast) ----------
    def local_search(rows, time_limit):
        start = time.time()
        rows = [r[:] for r in rows]
        best_scope = get_scope(rows)

        # Build global diff set and mask
        all_diffs = set()
        for r in rows:
            all_diffs |= entries_to_diffs(r)
        max_d = max(all_diffs) + 1
        mask = make_mask(max_d + 10)
        mask_add(mask, all_diffs)

        def try_decrease(r_idx, c_idx):
            row = rows[r_idx]
            curr_val = row[c_idx]
            prev_v = row[c_idx - 1]
            # Diffs involving this element
            remove = set()
            for i in range(c_idx):
                remove.add(curr_val - row[i])
            for j in range(c_idx + 1, n):
                remove.add(row[j] - curr_val)
            # Build a temporary mask without these diffs
            # Instead of copying, just check against all_diffs minus remove
            fixed = all_diffs - remove
            for new_val in range(curr_val - 1, prev_v, -1):
                ok = True
                for i in range(c_idx):
                    d = new_val - row[i]
                    if d in fixed:
                        ok = False
                        break
                if not ok:
                    continue
                for j in range(c_idx + 1, n):
                    d = row[j] - new_val
                    if d in fixed:
                        ok = False
                        break
                if not ok:
                    continue
                # internal uniqueness
                newd = set()
                for i in range(c_idx):
                    newd.add(new_val - row[i])
                for j in range(c_idx + 1, n):
                    newd.add(row[j] - new_val)
                if len(newd) != len(new_val and [new_val - row[i] for i in range(c_idx)] + [row[j] - new_val for j in range(c_idx + 1, n)]):
                    continue
                # accept
                row[c_idx] = new_val
                # update all_diffs
                all_diffs -= remove
                all_diffs |= newd
                return True
            return False

        while time.time() < start + time_limit:
            # find global max
            max_val = -1
            mr = mc = -1
            for ri, r in enumerate(rows):
                for ci, v in enumerate(r):
                    if v > max_val:
                        max_val = v
                        mr, mc = ri, ci
            if mc == 0:
                break
            improved = False
            if try_decrease(mr, mc):
                ns = get_scope(rows)
                if ns < best_scope:
                    best_scope = ns
                improved = True
            if improved:
                continue
            # try other elements
            for ri in range(n_rows):
                for ci in range(1, n):
                    if time.time() > start + time_limit:
                        break
                    if try_decrease(ri, ci):
                        ns = get_scope(rows)
                        if ns < best_scope:
                            best_scope = ns
                        improved = True
                        break
                if improved:
                    break
            if improved:
                continue
            break  # stuck

        return rows, best_scope

    # ---------- Deterministic backtracking over gaps ----------
    def backtracking_solve(initial_rows, max_scope, time_budget):
        start_time = time.time()
        # Convert to gaps
        init_gaps = []
        for r in initial_rows:
            gaps = []
            for j in range(1, n):
                gaps.append(r[j] - r[j - 1])
            init_gaps.append(gaps)

        # Order rows by current span descending (most rigid first)
        row_order = sorted(range(n_rows), key=lambda i: sum(init_gaps[i]), reverse=True)

        best_found = None
        best_found_scope = max_scope + 1

        used_diffs = set()
        assigned_gaps = [None] * n_rows

        def compute_diffs(gaps):
            entries = gaps_to_entries(gaps)
            return entries_to_diffs(entries), entries

        def backtrack(idx):
            nonlocal best_found, best_found_scope
            if time.time() - start_time > time_budget:
                return False
            if idx == n_rows:
                # compute scope
                scope = 0
                for ri in range(n_rows):
                    entries = gaps_to_entries(assigned_gaps[ri])
                    scope = max(scope, entries[-1])
                if scope < best_found_scope:
                    best_found_scope = scope
                    best_found = [gaps_to_entries(assigned_gaps[ri]) for ri in range(n_rows)]
                return True

            ri = row_order[idx]
            # Try to find gaps for row ri with sum <= max_scope that don't conflict
            # Use a recursive gap search with forward checking
            gaps = init_gaps[ri][:]  # start from current
            # We'll do a simple search: try to reduce the last gap, or try small perturbations
            # For determinism, try the current gaps first, then try decreasing each gap

            # First try current gaps
            diffs, entries = compute_diffs(gaps)
            if len(diffs) == d and not (diffs & used_diffs):
                assigned_gaps[ri] = gaps[:]
                used_diffs |= diffs
                if backtrack(idx + 1):
                    return True
                used_diffs -= diffs
                assigned_gaps[ri] = None

            # Try perturbing: decrease each gap by 1 and see if valid
            for gi in range(k):
                if gaps[gi] > 1:
                    old = gaps[gi]
                    gaps[gi] = old - 1
                    diffs, entries = compute_diffs(gaps)
                    if len(diffs) == d and not (diffs & used_diffs):
                        # also check internal: entries must be strictly increasing (guaranteed by gaps>=1)
                        assigned_gaps[ri] = gaps[:]
                        used_diffs |= diffs
                        if backtrack(idx + 1):
                            return True
                        used_diffs -= diffs
                        assigned_gaps[ri] = None
                    gaps[gi] = old
                    if time.time() - start_time > time_budget:
                        return False

            # Try increasing the last gap slightly to allow others to decrease
            # (skip for brevity; the decrease search is the main driver)
            return False

        backtrack(0)
        if best_found:
            return best_found, best_found_scope
        return None, None

    # ---------- Run local search ----------
    elapsed = time.time() - _START
    local_time = min(35.0, max(5.0, deadline - elapsed - 30))
    if local_time > 3:
        refined, refined_scope = local_search(best_rows, local_time)
        if full_validate(refined) is not None:
            best_rows = refined
            best_scope = refined_scope

    # ---------- Run backtracking to try to beat current best ----------
    if time.time() < deadline - 15:
        bt_time = min(25.0, deadline - time.time() - 10)
        if bt_time > 3:
            target = best_scope - 1
            if target > d:  # must be at least d
                bt_rows, bt_scope = backtracking_solve(best_rows, target, bt_time)
                if bt_rows and bt_scope < best_scope:
                    if full_validate(bt_rows) is not None:
                        best_rows = bt_rows
                        best_scope = bt_scope

    # ---------- Final polish ----------
    if time.time() < deadline - 2:
        polish_time = min(8.0, deadline - time.time() - 2)
        if polish_time > 2:
            polished, polished_scope = local_search(best_rows, polish_time)
            if full_validate(polished) is not None:
                best_rows = polished
                best_scope = polished_scope

    # Ensure validity; if not, fall back to greedy
    if full_validate(best_rows) is None:
        best_rows = greedy_init()
        best_scope = get_scope(best_rows)

    return best_rows

if __name__ == "__main__":
    rows = construct(4, 7)
    print("Scope:", max(max(r) for r in rows))
    diffs = set()
    valid = True
    for r in rows:
        if r[0] != 0:
            valid = False
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                dd = r[j] - r[i]
                if dd in diffs:
                    valid = False
                diffs.add(dd)
    print("Valid:", valid)
