"""Difference triangle set construction with local search."""
import time
import random

def construct(n_rows, k):
    deadline = time.time() + 120.0
    rng = random.Random(42)

    def greedy_row(used, k):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
        return row

    def all_diffs(rows):
        diffs = []
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    diffs.append(r[j] - r[i])
        return diffs

    def scope(rows):
        return max(max(r) for r in rows)

    def is_valid(rows):
        seen = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    d = r[j] - r[i]
                    if d in seen:
                        return False
                    seen.add(d)
        return True

    # Greedy initialization
    used = set()
    rows = []
    for _ in range(n_rows):
        rows.append(greedy_row(used, k))

    best_scope = scope(rows)
    best_rows = [list(r) for r in rows]

    def try_improve(rows, deadline):
        """Local search: shift individual elements or swap pairs within a row."""
        nonlocal best_scope, best_rows
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for ri in range(len(rows)):
                r = rows[ri]
                # Try shifting each element (except 0) by -1
                for j in range(1, len(r)):
                    if time.time() >= deadline:
                        return
                    old_val = r[j]
                    new_val = old_val - 1
                    if new_val <= r[j - 1]:
                        continue
                    # Check that new differences don't collide
                    new_diffs = {new_val - r[i] for i in range(j)}
                    new_diffs |= {r[i] - new_val for i in range(j + 1, len(r))}
                    old_diffs = {old_val - r[i] for i in range(j)}
                    old_diffs |= {r[i] - old_val for i in range(j + 1, len(r))}
                    # Collect all diffs from other rows
                    other_diffs = set()
                    for oi in range(len(rows)):
                        if oi == ri:
                            continue
                        orow = rows[oi]
                        for a in range(len(orow)):
                            for b in range(a + 1, len(orow)):
                                other_diffs.add(orow[b] - orow[a])
                    # Also need diffs within this row that we're keeping
                    kept_diffs = set()
                    for a in range(len(r)):
                        for b in range(a + 1, len(r)):
                            if a == j or b == j:
                                continue
                            kept_diffs.add(r[b] - r[a])
                    if not (new_diffs & (other_diffs | kept_diffs)):
                        r[j] = new_val
                        new_scope = scope(rows)
                        if new_scope < best_scope:
                            best_scope = new_scope
                            best_rows = [list(x) for x in rows]
                        improved = True

        # Try 2-element swaps within a row
        if time.time() < deadline:
            for ri in range(len(rows)):
                r = rows[ri]
                for j in range(1, len(r)):
                    if time.time() >= deadline:
                        return
                    for jj in range(j + 1, len(r)):
                        if time.time() >= deadline:
                            return
                        # Swap r[j] and r[jj]
                        old_j, old_jj = r[j], r[jj]
                        r[j], r[jj] = old_jj, old_j
                        # Must still be strictly increasing
                        valid_inc = True
                        for idx in range(len(r) - 1):
                            if r[idx] >= r[idx + 1]:
                                valid_inc = False
                                break
                        if not valid_inc:
                            r[j], r[jj] = old_j, old_jj
                            continue
                        # Check all diffs
                        new_diffs = set()
                        for a in range(len(r)):
                            for b in range(a + 1, len(r)):
                                new_diffs.add(r[b] - r[a])
                        other_diffs = set()
                        for oi in range(len(rows)):
                            if oi == ri:
                                continue
                            orow = rows[oi]
                            for a in range(len(orow)):
                                for b in range(a + 1, len(orow)):
                                    other_diffs.add(orow[b] - orow[a])
                        if new_diffs & other_diffs:
                            r[j], r[jj] = old_j, old_jj
                            continue
                        new_scope = scope(rows)
                        if new_scope < best_scope:
                            best_scope = new_scope
                            best_rows = [list(x) for x in rows]
                            improved = True
                        else:
                            r[j], r[jj] = old_j, old_jj

        return improved

    # Run local search
    try_improve(rows, deadline)

    # Randomized restarts with perturbation
    attempt = 0
    while time.time() < deadline - 2.0:
        attempt += 1
        # Perturb best_rows
        pert = [list(r) for r in best_rows]
        n_pert = rng.randint(1, 3)
        for _ in range(n_pert):
            ri = rng.randrange(n_rows)
            j = rng.randint(1, k)
            shift = rng.choice([-2, -1, 1, 2])
            pert[ri][j] += shift
        # Repair: make strictly increasing
        for ri in range(n_rows):
            r = pert[ri]
            r[0] = 0
            for j in range(1, len(r)):
                if r[j] <= r[j - 1]:
                    r[j] = r[j - 1] + 1
        # Check validity; if invalid, skip
        if not is_valid(pert):
            continue
        s = scope(pert)
        if s < best_scope:
            best_scope = s
            best_rows = [list(x) for x in pert]
        # Local search on perturbation
        try_improve(pert, deadline)

    # Final safety: return best found
    if time.time() > deadline - 1.0:
        return best_rows

    # One more greedy rebuild attempt if time allows
    if time.time() < deadline - 5.0:
        used2 = set()
        rows2 = []
        for _ in range(n_rows):
            rows2.append(greedy_row(used2, k))
        s2 = scope(rows2)
        if s2 < best_scope:
            best_scope = s2
            best_rows = [list(x) for x in rows2]
            try_improve(rows2, deadline)

    return best_rows
