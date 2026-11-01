"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin under 150s kill

    def time_left():
        return deadline - time.time()

    def row_diffs_set(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def is_valid(rows):
        seen = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in seen:
                        return False
                    seen.add(d)
        return True

    def scope(rows):
        return max(max(row) for row in rows)

    # ---------- Greedy baseline (fast, deterministic) ----------
    def greedy():
        used = set()
        rows = []
        for _ in range(n_rows):
            row, x = [0], 0
            row_diffs = set()
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not (new & used) and not (new & row_diffs):
                    row.append(x)
                    row_diffs |= new
                    used |= new
            rows.append(row)
        return rows

    best_rows = greedy()
    best_scope = scope(best_rows)

    # ---------- Conflict-directed repair: replace largest entry with minimal valid ----------
    def repair_row_scope(rows, row_idx, max_attempts=500):
        """Try to reduce the max entry of a specific row by replacing it with a smaller value."""
        rows = [list(r) for r in rows]
        current_row = rows[row_idx]
        current_max = max(current_row)
        best_row = list(current_row)
        best_max = current_max

        # Compute used diffs from other rows
        used_other = set()
        for j in range(n_rows):
            if j != row_idx:
                used_other |= row_diffs_set(rows[j])

        for attempt in range(max_attempts):
            if time_left() < 5:
                break

            # Try to find a smaller replacement for the last (largest) entry
            row_without_last = current_row[:-1]
            row_diffs_wo_last = set()
            for i in range(len(row_without_last)):
                for j in range(i + 1, len(row_without_last)):
                    row_diffs_wo_last.add(row_without_last[j] - row_without_last[i])

            # Find the smallest x > row_without_last[-1] such that all new diffs are valid
            found = False
            for x in range(row_without_last[-1] + 1, current_max):
                new_diffs = {x - a for a in row_without_last}
                if not (new_diffs & used_other) and not (new_diffs & row_diffs_wo_last):
                    candidate_row = row_without_last + [x]
                    # Verify full validity
                    test_rows = [list(r) for r in rows]
                    test_rows[row_idx] = candidate_row
                    if is_valid(test_rows):
                        best_row = candidate_row
                        best_max = x
                        found = True
                        break

            if not found:
                # Try random perturbations: remove a random middle element and rebuild
                if len(current_row) <= 2:
                    break
                pos = random.randint(1, len(current_row) - 2)
                base = current_row[:pos] + current_row[pos+1:]
                row_diffs_base = set()
                for i in range(len(base)):
                    for j in range(i + 1, len(base)):
                        row_diffs_base.add(base[j] - base[i])

                # Rebuild from base
                new_row = list(base)
                new_row_diffs = set(row_diffs_base)
                x = new_row[-1]
                success = True
                while len(new_row) < k + 1:
                    found_x = False
                    for trial_x in range(x + 1, x + 100):
                        new = {trial_x - a for a in new_row}
                        if not (new & used_other) and not (new & new_row_diffs):
                            new_row.append(trial_x)
                            new_row_diffs |= new
                            found_x = True
                            break
                    if not found_x:
                        success = False
                        break
                    x = new_row[-1]

                if success and max(new_row) < best_max:
                    test_rows = [list(r) for r in rows]
                    test_rows[row_idx] = new_row
                    if is_valid(test_rows):
                        best_row = new_row
                        best_max = max(new_row)

        return best_row, best_max

    def local_polish(rows):
        """Reduce scope by locally re-optimizing each row (ordered by scope)."""
        rows = [list(r) for r in rows]
        improved = True
        while improved and time_left() > 15:
            improved = False
            order = sorted(range(n_rows), key=lambda i: -max(rows[i]))
            for i in order:
                if time_left() < 15:
                    break
                new_row, new_max = repair_row_scope(rows, i, max_attempts=300)
                if new_max < max(rows[i]):
                    old_scope = scope(rows)
                    rows[i] = new_row
                    new_scope = scope(rows)
                    if new_scope < old_scope:
                        improved = True
        return rows

    if time_left() > 15:
        polished = local_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(best_rows)

    # ---------- Multi-start with randomized greedy ----------
    def randomized_greedy(seed_val):
        rng = random.Random(seed_val)
        used = set()
        rows = []
        for _ in range(n_rows):
            row, x = [0], 0
            row_diffs = set()
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not (new & used) and not (new & row_diffs):
                    # Occasionally skip to explore alternatives
                    if rng.random() < 0.08 and x < 200:
                        continue
                    row.append(x)
                    row_diffs |= new
                    used |= new
            rows.append(row)
        return rows

    # Try multiple random starts
    best_found = best_rows
    best_found_scope = best_scope

    num_starts = 8
    for s in range(num_starts):
        if time_left() < 25:
            break
        seed_val = 42 + s * 137
        cand = randomized_greedy(seed_val)
        if not is_valid(cand):
            continue
        cand_scope = scope(cand)
        if cand_scope < best_found_scope:
            best_found = cand
            best_found_scope = cand_scope
            # Apply repair to the new candidate
            if time_left() > 20:
                repaired_cand = local_polish(cand)
                if scope(repaired_cand) < best_found_scope:
                    best_found = repaired_cand
                    best_found_scope = scope(repaired_cand)

    if best_found_scope < best_scope:
        best_rows = best_found
        best_scope = best_found_scope

    # ---------- Final local repair pass ----------
    if time_left() > 15:
        final_repaired = local_polish(best_rows)
        if scope(final_repaired) < best_scope:
            best_rows = final_repaired
            best_scope = scope(best_rows)

    # Final validity guarantee.
    if not is_valid(best_rows):
        best_rows = greedy()

    return best_rows
# EVOLVE-BLOCK-END
