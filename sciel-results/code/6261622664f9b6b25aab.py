"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Build n_rows rows of k+1 increasing integers starting at 0 such that all
    positive differences within rows are distinct across all rows.
    Minimize the largest entry (scope).
    """
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    def get_max_val(rows):
        return max(max(r) for r in rows)

    def is_valid_row(row, used):
        """Check if adding this row's differences doesn't conflict with used."""
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in diffs or d in used:
                    return False
                diffs.add(d)
        return True

    def row_diffs(row):
        """Return the set of all positive differences in a row."""
        s = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                s.add(row[j] - row[i])
        return s

    def greedy_construct(n_rows, k, used_seed=None, rng=None):
        """
        Greedy construction: build rows one by one, each row greedily
        choosing the smallest next mark that doesn't conflict.
        If used_seed is given, those differences are pre-occupied.
        """
        if rng is None:
            rng = random.Random()
        used = set(used_seed) if used_seed else set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows, used

    def randomized_greedy(n_rows, k, used_seed=None, rng=None, bias=0.0):
        """
        Randomized greedy: at each step, consider a window of candidate next marks
        and pick one randomly weighted by how 'tight' it is (smaller preferred).
        bias controls randomness: 0 = pure greedy, higher = more random.
        """
        if rng is None:
            rng = random.Random()
        used = set(used_seed) if used_seed else set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                # Find the next few valid candidates
                candidates = []
                x += 1
                # Look ahead a small window
                window = 1 + int(bias * 5)
                for _ in range(window):
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        candidates.append(x)
                    x += 1
                if not candidates:
                    # Shouldn't happen, but fallback
                    x += 1
                    candidates = [x]
                # Pick: prefer smaller with some randomness
                if bias > 0 and len(candidates) > 1:
                    # Weighted choice: earlier candidates more likely
                    weights = [1.0 / (i + 1) ** (bias * 2) for i in range(len(candidates))]
                    total = sum(weights)
                    r = rng.random() * total
                    cum = 0
                    chosen = candidates[0]
                    for c, w in zip(candidates, weights):
                        cum += w
                        if r <= cum:
                            chosen = c
                            break
                    x = chosen + 1
                else:
                    chosen = candidates[0]
                    x = chosen + 1
                row.append(chosen)
                used |= {chosen - a for a in row[:-1]}
            rows.append(row)
        return rows, used

    def local_improve(rows, k, used, rng, max_iters=200):
        """
        Try to reduce the scope by moving individual marks within a row.
        Returns improved rows and used set, or original if no improvement.
        """
        current_max = get_max_val(rows)
        best_rows = [list(r) for r in rows]
        best_used = set(used)
        best_max = current_max

        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            # Pick a random row and a random non-zero mark to try moving
            ri = rng.randrange(n_rows)
            row = best_rows[ri]
            # Try to reduce the largest mark first, then others
            # Find indices of marks to try (exclude index 0 which is always 0)
            if len(row) <= 1:
                continue
            # Try moving each mark (except first) to a smaller value
            improved = False
            for mi in range(len(row) - 1, 0, -1):  # from largest to smallest
                old_val = row[mi]
                # Compute current diffs for this row
                row_diffs_set = set()
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        row_diffs_set.add(row[j] - row[i])
                # Other rows' diffs
                other_diffs = set()
                for oi, orow in enumerate(best_rows):
                    if oi == ri:
                        continue
                    for i in range(len(orow)):
                        for j in range(i + 1, len(orow)):
                            other_diffs.add(orow[j] - orow[i])

                # Try to move row[mi] to a smaller value
                # New value must be > row[mi-1] and < row[mi+1] (if exists)
                lo = row[mi - 1] + 1
                hi = old_val - 1
                if hi < lo:
                    continue
                # Try values from hi down to lo
                for new_val in range(hi, lo - 1, -1):
                    # Check if new_val creates valid differences
                    new_row = list(row)
                    new_row[mi] = new_val
                    # Check strictly increasing
                    if not all(new_row[i] < new_row[i+1] for i in range(len(new_row)-1)):
                        continue
                    # Compute new diffs for this row
                    new_row_diffs = set()
                    for i in range(len(new_row)):
                        for j in range(i + 1, len(new_row)):
                            d = new_row[j] - new_row[i]
                            new_row_diffs.add(d)
                    # Check no conflict with other rows
                    if new_row_diffs & other_diffs:
                        continue
                    # Check within-row uniqueness
                    if len(new_row_diffs) != len(new_row) * (len(new_row) - 1) // 2:
                        continue
                    # Valid move!
                    best_rows[ri] = new_row
                    # Update used
                    best_used = set()
                    for br in best_rows:
                        for i in range(len(br)):
                            for j in range(i + 1, len(br)):
                                best_used.add(br[j] - br[i])
                    new_max = get_max_val(best_rows)
                    if new_max < best_max:
                        best_max = new_max
                    improved = True
                    break
                if improved:
                    break
            if not improved:
                # Try a random perturbation then local search
                # Pick a random row, change a random mark slightly
                ri = rng.randrange(n_rows)
                row = best_rows[ri]
                if len(row) <= 1:
                    continue
                mi = rng.randrange(1, len(row))
                old_val = row[mi]
                lo = row[mi - 1] + 1
                hi = old_val + 5  # allow slight increase too
                if hi <= lo:
                    continue
                new_val = rng.randint(lo, hi)
                if new_val == old_val:
                    continue
                new_row = list(row)
                new_row[mi] = new_val
                if not all(new_row[i] < new_row[i+1] for i in range(len(new_row)-1)):
                    continue
                # Check validity
                new_row_diffs = set()
                for i in range(len(new_row)):
                    for j in range(i + 1, len(new_row)):
                        d = new_row[j] - new_row[i]
                        new_row_diffs.add(d)
                other_diffs = set()
                for oi, orow in enumerate(best_rows):
                    if oi == ri:
                        continue
                    for i in range(len(orow)):
                        for j in range(i + 1, len(orow)):
                            other_diffs.add(orow[j] - orow[i])
                if len(new_row_diffs) != len(new_row) * (len(new_row) - 1) // 2:
                    continue
                if new_row_diffs & other_diffs:
                    continue
                # Accept (even if scope increases, for exploration)
                best_rows[ri] = new_row
                best_used = set()
                for br in best_rows:
                    for i in range(len(br)):
                        for j in range(i + 1, len(br)):
                            best_used.add(br[j] - br[i])
                new_max = get_max_val(best_rows)
                if new_max < best_max:
                    best_max = new_max

        return best_rows, best_used, best_max

    def row_replacement_sa(rows, k, used, rng, max_iters=100):
        """
        Simulated annealing with row-replacement moves.
        Pick a row, replace it with a newly constructed row that fits
        the remaining differences.
        """
        current_max = get_max_val(rows)
        best_rows = [list(r) for r in rows]
        best_used = set(used)
        best_max = current_max

        # Compute per-row difference sets for fast checking
        def compute_all_diffs(rows):
            all_diffs = []
            for r in rows:
                s = set()
                for i in range(len(r)):
                    for j in range(i + 1, len(r)):
                        s.add(r[j] - r[i])
                all_diffs.append(s)
            return all_diffs

        def check_row_valid(new_row, other_diffs_union):
            """Check new_row's diffs don't conflict with other_diffs_union."""
            nd = set()
            for i in range(len(new_row)):
                for j in range(i + 1, len(new_row)):
                    d = new_row[j] - new_row[i]
                    if d in nd or d in other_diffs_union:
                        return False
                    nd.add(d)
            return True

        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            # Pick a row to replace
            ri = rng.randrange(n_rows)
            # Compute union of all other rows' diffs
            other_diffs = set()
            for oi, orow in enumerate(best_rows):
                if oi == ri:
                    continue
                for i in range(len(orow)):
                    for j in range(i + 1, len(orow)):
                        other_diffs.add(orow[j] - orow[i])

            # Try to construct a new row that avoids other_diffs
            # Use greedy with the other_diffs as pre-occupied
            new_row = [0]
            x = 0
            valid = True
            while len(new_row) < k + 1:
                x += 1
                new_diffs = {x - a for a in new_row}
                if new_diffs & other_diffs:
                    # Try a few more
                    found = False
                    for _ in range(20):
                        x += 1
                        new_diffs = {x - a for a in new_row}
                        if not (new_diffs & other_diffs):
                            found = True
                            break
                    if not found:
                        valid = False
                        break
                new_row.append(x)
            if not valid:
                continue

            # Check the new row's own diffs are all distinct (should be by construction)
            nr_diffs = set()
            for i in range(len(new_row)):
                for j in range(i + 1, len(new_row)):
                    d = new_row[j] - new_row[i]
                    nr_diffs.add(d)
            if len(nr_diffs) != len(new_row) * (len(new_row) - 1) // 2:
                continue

            # Compute new scope
            new_rows = [list(r) for r in best_rows]
            new_rows[ri] = new_row
            new_max = get_max_val(new_rows)

            # Accept if better, or with some probability if worse (annealing)
            if new_max < best_max:
                best_rows = new_rows
                best_max = new_max
            elif new_max <= best_max + 3:
                # Accept with some probability for exploration
                if rng.random() < 0.1:
                    best_rows = new_rows

        return best_rows, best_max

    # Main search loop
    rng = random.Random(42)

    # Start with pure greedy
    initial_rows, initial_used = greedy_construct(n_rows, k, rng=rng)
    best_rows = [list(r) for r in initial_rows]
    best_max = get_max_val(best_rows)
    best_used = set(initial_used)

    # Phase 1: Local improvement
    if time.time() < deadline - 30:
        improved_rows, improved_used, improved_max = local_improve(
            best_rows, k, best_used, rng, max_iters=300
        )
        if improved_max < best_max:
            best_rows = improved_rows
            best_used = improved_used
            best_max = improved_max

    # Phase 2: Row replacement SA
    if time.time() < deadline - 20:
        sa_rows, sa_max = row_replacement_sa(
            best_rows, k, best_used, rng, max_iters=200
        )
        if sa_max < best_max:
            best_rows = sa_rows
            best_max = sa_max
            # Recompute used
            best_used = set()
            for r in best_rows:
                for i in range(len(r)):
                    for j in range(i + 1, len(r)):
                        best_used.add(r[j] - r[i])

    # Phase 3: Multiple randomized greedy restarts with local improvement
    while time.time() < deadline - 10:
        # Try randomized greedy with varying bias
        bias = rng.random() * 0.5
        r_rows, r_used = randomized_greedy(n_rows, k, rng=rng, bias=bias)
        r_max = get_max_val(r_rows)

        if r_max < best_max:
            best_rows = [list(r) for r in r_rows]
            best_used = set(r_used)
            best_max = r_max

        # Local improve the best
        if time.time() < deadline - 10:
            imp_rows, imp_used, imp_max = local_improve(
                best_rows, k, best_used, rng, max_iters=50
            )
            if imp_max < best_max:
                best_rows = imp_rows
                best_used = imp_used
                best_max = imp_max

        # Also try row replacement on best
        if time.time() < deadline - 10:
            sa_rows2, sa_max2 = row_replacement_sa(
                best_rows, k, best_used, rng, max_iters=50
            )
            if sa_max2 < best_max:
                best_rows = sa_rows2
                best_max = sa_max2
                best_used = set()
                for r in best_rows:
                    for i in range(len(r)):
                        for j in range(i + 1, len(r)):
                            best_used.add(r[j] - r[i])

    # Final: ensure validity
    # Recompute all diffs to be safe
    all_diffs = set()
    for r in best_rows:
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                if d in all_diffs:
                    # Shouldn't happen, but if it does, fall back to greedy
                    fallback_rows, _ = greedy_construct(n_rows, k, rng=rng)
                    return fallback_rows
                all_diffs.add(d)

    return best_rows
# EVOLVE-BLOCK-END
