"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    such that all positive differences within rows are distinct across all rows.
    Minimize the largest entry.
    """
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    best_rows = None
    best_max = float('inf')

    # First, try a fast greedy to get a baseline
    greedy_rows = _greedy_construct(n_rows, k)
    if greedy_rows is not None:
        best_rows = greedy_rows
        best_max = max(max(r) for r in greedy_rows)

    # Then try randomized local search / simulated annealing style improvement
    # Strategy: build rows one by one, but with randomized ordering and backtracking
    # Also try to improve an existing solution by swapping/replacing entries

    # Try multiple random restarts with different strategies
    rng = random.Random(42)

    for restart in range(500):
        if time.time() > deadline:
            break

        # Strategy 1: Randomized greedy with shuffled candidate order
        rows = _randomized_greedy(n_rows, k, rng)
        if rows is not None:
            cur_max = max(max(r) for r in rows)
            if cur_max < best_max:
                best_max = cur_max
                best_rows = rows

        # Strategy 2: Try to improve current best via local search
        if best_rows is not None and restart % 5 == 0:
            improved = _local_search_improve(best_rows, n_rows, k, rng, deadline)
            if improved is not None:
                cur_max = max(max(r) for r in improved)
                if cur_max < best_max:
                    best_max = cur_max
                    best_rows = improved

        # Strategy 3: Backtracking search for small cases
        if n_rows <= 3 and k <= 4:
            bt_rows = _backtracking_search(n_rows, k, best_max - 1, rng)
            if bt_rows is not None:
                cur_max = max(max(r) for r in bt_rows)
                if cur_max < best_max:
                    best_max = cur_max
                    best_rows = bt_rows

    if best_rows is None:
        # Fallback: should not happen, but just in case
        return [[0] * (k + 1) for _ in range(n_rows)]

    return best_rows


def _greedy_construct(n_rows, k):
    """Deterministic greedy: pick smallest next value that doesn't conflict."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows


def _randomized_greedy(n_rows, k, rng):
    """
    Randomized greedy: for each position, consider a range of candidate values
    and pick one at random from those that are valid. This can escape local optima.
    """
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            # Look ahead a bit: collect valid candidates in a window
            candidates = []
            window = 15
            for cand in range(x, x + window):
                new_diffs = {cand - a for a in row}
                if not (new_diffs & used):
                    candidates.append(cand)
                if len(candidates) >= 10:
                    break

            if not candidates:
                # Expand window if needed
                for cand in range(x + window, x + window * 3):
                    new_diffs = {cand - a for a in row}
                    if not (new_diffs & used):
                        candidates.append(cand)
                        if len(candidates) >= 10:
                            break

            if not candidates:
                return None  # Failed

            # Pick randomly from candidates (weighted toward smaller)
            # Use exponential weighting to favor smaller values
            weights = [1.0 / (1.0 + i) for i in range(len(candidates))]
            total_w = sum(weights)
            r = rng.random() * total_w
            cum = 0
            chosen = candidates[0]
            for i, c in enumerate(candidates):
                cum += weights[i]
                if r <= cum:
                    chosen = c
                    break

            row.append(chosen)
            used |= {chosen - a for a in row[:-1]}
            x = chosen

        rows.append(row)

    # Verify
    all_diffs = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in all_diffs:
                    return None
                all_diffs.add(d)

    return rows


def _local_search_improve(rows, n_rows, k, rng, deadline):
    """
    Try to improve an existing solution by modifying individual entries.
    """
    if rows is None:
        return None

    cur_rows = [list(r) for r in rows]
    cur_max = max(max(r) for r in cur_rows)

    # Collect all used differences
    all_diffs = set()
    for row in cur_rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                all_diffs.add(row[j] - row[i])

    improved = False
    attempts = 0
    while time.time() < deadline - 5 and attempts < 200:
        attempts += 1

        # Pick a random row and a random position (not the first, which is 0)
        row_idx = rng.randrange(n_rows)
        pos = rng.randrange(1, len(cur_rows[row_idx]))

        old_val = cur_rows[row_idx][pos]
        row = cur_rows[row_idx]

        # Try to find a smaller value that still works
        # The value must be > row[pos-1] and < row[pos+1] if pos < len(row)-1
        lo = row[pos - 1] + 1
        hi = row[pos + 1] - 1 if pos < len(row) - 1 else old_val - 1

        if hi < lo:
            continue

        # Try values from lo to min(hi, old_val - 1)
        for new_val in range(lo, min(hi, old_val - 1) + 1):
            if new_val == old_val:
                continue

            # Check if replacing old_val with new_val keeps all diffs unique
            # Remove old diffs involving this position
            old_diffs = set()
            for i in range(len(row)):
                if i != pos:
                    d = abs(row[pos] - row[i])
                    old_diffs.add(d)

            # Check new diffs
            new_diffs = set()
            for i in range(len(row)):
                if i != pos:
                    d = abs(new_val - row[i])
                    new_diffs.add(d)

            # Check that new_diffs don't conflict with (all_diffs - old_diffs)
            remaining = all_diffs - old_diffs
            if new_diffs & remaining:
                continue

            # Also check that new_val is between neighbors
            if pos > 0 and new_val <= row[pos - 1]:
                continue
            if pos < len(row) - 1 and new_val >= row[pos + 1]:
                continue

            # Accept the change
            all_diffs = remaining | new_diffs
            cur_rows[row_idx][pos] = new_val
            improved = True
            break

        if not improved:
            # Try a random perturbation: swap two adjacent non-zero entries? No, that breaks ordering.
            # Instead, try a random increase to escape, then decrease later
            if rng.random() < 0.3:
                row_idx = rng.randrange(n_rows)
                pos = rng.randrange(1, len(cur_rows[row_idx]))
                row = cur_rows[row_idx]
                old_val = row[pos]
                hi = row[pos + 1] - 1 if pos < len(row) - 1 else old_val + 10
                lo = row[pos - 1] + 1

                candidates = list(range(old_val + 1, min(hi, old_val + 10) + 1))
                if candidates:
                    new_val = rng.choice(candidates)
                    old_diffs = set()
                    for i in range(len(row)):
                        if i != pos:
                            old_diffs.add(abs(row[pos] - row[i]))

                    new_diffs = set()
                    for i in range(len(row)):
                        if i != pos:
                            new_diffs.add(abs(new_val - row[i]))

                    remaining = all_diffs - old_diffs
                    if not (new_diffs & remaining):
                        all_diffs = remaining | new_diffs
                        cur_rows[row_idx][pos] = new_val
                        improved = True

    if improved:
        new_max = max(max(r) for r in cur_rows)
        if new_max < cur_max:
            return cur_rows
    return None


def _backtracking_search(n_rows, k, max_val, rng):
    """
    Backtracking search for small instances.
    """
    if n_rows == 0:
        return []

    used = set()
    rows = []

    def backtrack(row_idx, current_row):
        nonlocal used, rows
        if row_idx == n_rows:
            return True

        if len(current_row) == 0:
            current_row = [0]

        if len(current_row) > k:
            return False

        if current_row[-1] > max_val:
            return False

        if len(current_row) == k + 1:
            rows.append(current_row)
            return backtrack(row_idx + 1, [])

        # Try next values
        last = current_row[-1]
        for x in range(last + 1, max_val + 1):
            new_diffs = {x - a for a in current_row}
            if new_diffs & used:
                continue

            # Pruning: check if we can still fit remaining values
            remaining_slots = k + 1 - (len(current_row) + 1)
            if remaining_slots > 0 and x + remaining_slots > max_val:
                continue

            used |= new_diffs
            if backtrack(row_idx, current_row + [x]):
                return True
            used -= new_diffs

        return False

    if backtrack(0, []):
        return rows
    return None
# EVOLVE-BLOCK-END
