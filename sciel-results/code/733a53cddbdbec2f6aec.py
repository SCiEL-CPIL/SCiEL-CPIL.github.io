"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random


def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 118.0  # safety margin before 120s limit

    # Number of distinct positive differences required per row
    diffs_per_row = k * (k + 1) // 2

    # ---------- helpers ----------
    def pure_greedy():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    def scope_of(rows):
        return max(max(r) for r in rows)

    def randomized_greedy():
        """Greedy but with a shuffled candidate window to escape the pure-greedy
        local optimum. Returns a valid set of rows (always succeeds, just
        possibly with a larger scope)."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                # candidate window: from x+1 up to x+window, shuffled
                window = 60
                cands = list(range(x + 1, x + window + 1))
                random.shuffle(cands)
                found = False
                for cand in cands:
                    new = {cand - a for a in row}
                    if not new & used:
                        row.append(cand)
                        used |= new
                        x = cand
                        found = True
                        break
                if not found:
                    # fall back to a larger sequential scan (guaranteed to find one)
                    cand = x + 1
                    while True:
                        new = {cand - a for a in row}
                        if not new & used:
                            row.append(cand)
                            used |= new
                            x = cand
                            break
                        cand += 1
            rows.append(row)
        return rows

    def row_diffs(row):
        out = []
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                out.append(row[j] - row[i])
        return out

    def local_search(rows, max_iters, time_limit):
        """Perturb single entries to new values; accept a move only if all
        differences across all rows remain distinct and the scope does not
        increase. Returns the best rows found."""
        rows = [r[:] for r in rows]
        best = rows
        best_scope = scope_of(rows)
        n = len(rows)
        L = len(rows[0])  # k+1

        # Precompute current difference multiset to detect duplicates fast.
        def build_used(rows):
            used = set()
            for row in rows:
                for d in row_diffs(row):
                    used.add(d)
            return used

        # We accept a move only if it keeps the set of differences distinct.
        # Since the starting rows are valid, we track the used set and, for a
        # candidate move, recompute the affected row's differences and check
        # for any collision with the global used set (minus that row's old diffs).
        it = 0
        while it < max_iters and time.time() < time_limit:
            it += 1
            ri = random.randrange(n)
            # never move index 0 (fixed at 0)
            if L <= 1:
                break
            ci = random.randrange(1, L)

            old_val = rows[ri][ci]

            # old diffs of this row
            old_row = rows[ri]
            old_diffs = row_diffs(old_row)

            # candidate new value: try a small set of nearby values and a few
            # random values within [1, best_scope]
            candidates = set()
            # nearby
            for d in range(-6, 7):
                nv = old_val + d
                if nv >= 1 and nv <= best_scope:
                    candidates.add(nv)
            # random within scope
            for _ in range(4):
                nv = random.randint(1, best_scope)
                candidates.add(nv)

            # enforce strict increasing order
            lower = old_row[ci - 1] + 1
            upper = (old_row[ci + 1] - 1) if ci + 1 < L else best_scope
            candidates = [c for c in candidates if lower <= c <= upper]
            if not candidates:
                continue

            random.shuffle(list(candidates))
            for nv in candidates:
                if nv == old_val:
                    continue
                new_row = old_row[:]
                new_row[ci] = nv
                new_diffs = row_diffs(new_row)

                # Check distinctness: new_diffs must be unique internally and
                # must not collide with differences of other rows.
                # Build the set of diffs from other rows.
                # For efficiency, compute the global used set once per row change
                # is expensive; instead check: (a) no dup within new_diffs,
                # (b) no diff in new_diffs that equals a diff in another row.
                if len(set(new_diffs)) != len(new_diffs):
                    continue
                # collisions with other rows
                collides = False
                for oi in range(n):
                    if oi == ri:
                        continue
                    for d in row_diffs(rows[oi]):
                        if d in new_diffs:
                            collides = True
                            break
                    if collides:
                        break
                if collides:
                    continue

                # valid move; accept if it reduces scope (or keep if equal to
                # allow exploration) -- we only accept strictly better scope to
                # guarantee monotone improvement.
                new_scope = scope_of([new_row if i == ri else rows[i] for i in range(n)])
                if new_scope < best_scope:
                    rows[ri] = new_row
                    best = [r[:] for r in rows]
                    best_scope = new_scope
                break  # one move per iteration

        return best, best_scope

    # ---------- main search ----------
    # Baseline: pure greedy (valid).
    base_rows = pure_greedy()
    best_rows = base_rows
    best_scope = scope_of(base_rows)

    # Phase 1: randomized greedy restarts to get diverse valid starts.
    attempts = 0
    while time.time() < deadline - 30.0 and attempts < 60:
        attempts += 1
        rows = randomized_greedy()
        sc = scope_of(rows)
        if sc < best_scope:
            best_rows = rows
            best_scope = sc
            if best_scope <= 112:  # counting bound
                break

    # Phase 2: local search refinement on the best configuration (and a couple
    # of the best alternatives) to drive scope down within valid configs.
    # Build a small pool of good starting points.
    pool = [best_rows]
    # add a few more randomized starts that are close to best
    for _ in range(3):
        if time.time() >= deadline - 15.0:
            break
        rows = randomized_greedy()
        sc = scope_of(rows)
        if sc <= best_scope + 15:
            pool.append(rows)

    for start in pool:
        if time.time() >= deadline - 8.0:
            break
        improved, sc = local_search(start, max_iters=4000, time_limit=deadline - 8.0)
        if sc < best_scope:
            best_rows = improved
            best_scope = sc
            if best_scope <= 112:
                break

    return best_rows
# EVOLVE-BLOCK-END
