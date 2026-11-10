"""Difference triangle set via greedy init + simulated annealing refinement."""
import time
import random
import math

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct a difference triangle set using greedy initialization followed
    by simulated annealing to minimize the largest entry.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    def time_remaining():
        return deadline - time.time()

    def validate(rows):
        """Check if rows form a valid DTS. Return (valid, max_entry)."""
        all_diffs = set()
        max_entry = 0
        for row in rows:
            if not row or row[0] != 0:
                return False, float('inf')
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in all_diffs:
                        return False, float('inf')
                    all_diffs.add(d)
            max_entry = max(max_entry, row[-1])
        return True, max_entry

    def greedy_initial(n_rows, k, rng, perturb=0):
        """
        Greedy initialization with randomization.
        """
        used_diffs = set()
        rows = []
        for r_idx in range(n_rows):
            row = [0]
            x = 1
            while len(row) < k + 1:
                if time.time() > deadline - 15:
                    return None
                candidates = []
                search_limit = x + 40 + (perturb * 12)
                while x < search_limit:
                    new_diffs = set()
                    valid = True
                    for a in row:
                        d = x - a
                        if d in used_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                    if valid:
                        candidates.append((x, new_diffs))
                    x += 1
                    if time.time() > deadline - 15:
                        return None

                if not candidates:
                    x = search_limit
                    continue

                if len(candidates) > 1 and rng.random() < 0.4:
                    top = min(3, len(candidates))
                    chosen_idx = rng.randint(0, top - 1)
                    chosen_val, chosen_diffs = candidates[chosen_idx]
                else:
                    chosen_val, chosen_diffs = candidates[0]

                row.append(chosen_val)
                used_diffs.update(chosen_diffs)
                x = chosen_val + 1

            rows.append(row)
        return rows

    def compute_used_diffs(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.add(row[j] - row[i])
        return used

    def row_diffs(row, ci):
        """Return set of differences involving position ci in row."""
        old_diffs = set()
        for i in range(ci):
            old_diffs.add(row[ci] - row[i])
        for j in range(ci + 1, len(row)):
            old_diffs.add(row[j] - row[ci])
        return old_diffs

    def try_move_element(rows, ri, ci, new_val, used_diffs):
        """
        Try to move element (ri, ci) to new_val.
        Returns (success, new_used_diffs) or (False, None).
        """
        row = rows[ri]
        if ci > 0 and new_val <= row[ci - 1]:
            return False, None
        if ci < len(row) - 1 and new_val >= row[ci + 1]:
            return False, None

        old_diffs = row_diffs(row, ci)
        temp_used = used_diffs - old_diffs

        new_diffs = set()
        valid = True
        for i in range(ci):
            d = new_val - row[i]
            if d <= 0 or d in temp_used:
                valid = False
                break
            new_diffs.add(d)
        if valid:
            for j in range(ci + 1, len(row)):
                d = row[j] - new_val
                if d <= 0 or d in temp_used:
                    valid = False
                    break
                new_diffs.add(d)

        if not valid:
            return False, None

        new_used = temp_used | new_diffs
        return True, new_used

    def simulated_annealing(rows, n_rows, k, max_iters=30000):
        """
        Simulated annealing to reduce the maximum entry.
        Moves: (1) reduce a random element, (2) swap two rows, (3) perturb an element.
        Accept worsening moves with probability exp(-delta/T).
        """
        rows = [row[:] for row in rows]
        used_diffs = compute_used_diffs(rows)
        best_score = max(row[-1] for row in rows)
        best_rows = [row[:] for row in rows]

        T = 50.0
        T_min = 0.5
        cooling = 0.9995

        for iteration in range(max_iters):
            if time.time() > deadline - 10:
                break

            current_score = max(r[-1] for r in rows)

            # Choose a random move type
            move_type = rng.randint(0, 2)

            if move_type == 0:
                # Try to reduce a random element
                ri = rng.randint(0, n_rows - 1)
                ci = rng.randint(1, len(rows[ri]) - 1)
                current_val = rows[ri][ci]
                lower = rows[ri][ci - 1] + 1
                if ci < len(rows[ri]) - 1:
                    upper = rows[ri][ci + 1] - 1
                else:
                    upper = current_val - 1

                if lower > upper:
                    continue

                new_val = rng.randint(lower, upper)
                if new_val == current_val:
                    continue

                success, new_used = try_move_element(rows, ri, ci, new_val, used_diffs)
                if not success:
                    continue

                new_score = max(r[-1] for r in rows)
                delta = new_score - current_score

                if delta <= 0 or (T > T_min and rng.random() < math.exp(-delta / T)):
                    rows[ri][ci] = new_val
                    used_diffs = new_used
                    if new_score < best_score:
                        best_score = new_score
                        best_rows = [r[:] for r in rows]

            elif move_type == 1:
                # Swap two rows
                r1 = rng.randint(0, n_rows - 1)
                r2 = rng.randint(0, n_rows - 1)
                if r1 == r2:
                    continue

                # Check if swap is valid: all diffs must remain distinct
                # Since rows are independent, swapping doesn't change the set of diffs
                # So any swap is valid! Just do it.
                rows[r1], rows[r2] = rows[r2], rows[r1]
                # used_diffs unchanged
                # Score unchanged, but row order changed
                # This can help because the greedy/local search treats rows asymmetrically

            else:
                # Perturb: increase a random element slightly, then try to reduce it back
                ri = rng.randint(0, n_rows - 1)
                ci = rng.randint(1, len(rows[ri]) - 1)
                current_val = rows[ri][ci]
                # Try a few nearby values
                for offset in [1, -1, 2, -2]:
                    new_val = current_val + offset
                    if ci > 0 and new_val <= rows[ri][ci - 1]:
                        continue
                    if ci < len(rows[ri]) - 1 and new_val >= rows[ri][ci + 1]:
                        continue

                    success, new_used = try_move_element(rows, ri, ci, new_val, used_diffs)
                    if not success:
                        continue

                    new_score = max(r[-1] for r in rows)
                    delta = new_score - current_score

                    if delta <= 0 or (T > T_min and rng.random() < math.exp(-delta / T)):
                        rows[ri][ci] = new_val
                        used_diffs = new_used
                        if new_score < best_score:
                            best_score = new_score
                            best_rows = [r[:] for r in rows]
                        break

            T = max(T_min, T * cooling)

        return best_rows, best_score

    # Main search loop
    best_result = None
    best_score = float('inf')
    rng = random.Random(42)

    # Deterministic greedy first
    det_rows = greedy_initial(n_rows, k, rng, perturb=0)
    if det_rows:
        valid, max_entry = validate(det_rows)
        if valid:
            best_result = det_rows
            best_score = max_entry
            improved, improved_score = simulated_annealing(det_rows, n_rows, k)
            if improved_score < best_score:
                best_score = improved_score
                best_result = improved

    # Randomized restarts with SA
    max_restarts = 80
    for restart in range(max_restarts):
        if time.time() > deadline - 15:
            break

        perturb = rng.randint(0, 6)
        rows = greedy_initial(n_rows, k, rng, perturb=perturb)
        if not rows:
            continue

        valid, max_entry = validate(rows)
        if not valid:
            continue

        if max_entry < best_score:
            best_score = max_entry
            best_result = rows
            improved, improved_score = simulated_annealing(rows, n_rows, k)
            if improved_score < best_score:
                best_score = improved_score
                best_result = improved
        else:
            # Even if not better than best, SA might find a better path
            # Only do SA if we have time
            if time.time() < deadline - 30 and max_entry <= best_score + 5:
                improved, improved_score = simulated_annealing(rows, n_rows, k, max_iters=15000)
                if improved_score < best_score:
                    best_score = improved_score
                    best_result = improved

    if best_result is None:
        # Fallback: simple greedy
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

    return best_result
# EVOLVE-BLOCK-END
