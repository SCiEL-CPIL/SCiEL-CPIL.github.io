import time
import random
import math

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Constructs a Difference Triangle Set (DTS) with n_rows rows and k differences (k+1 elements per row).
    Minimizes the largest entry (scope).
    Uses simulated annealing with incremental diff-set maintenance.
    """
    start_time = time.time()
    deadline = 145.0  # 5s margin before 150s limit

    def elapsed():
        return time.time() - start_time

    def greedy_construct(n_rows, k):
        """Greedy construction - always produces a valid solution."""
        used = set()
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
        return rows

    # Start with greedy solution
    best_solution = greedy_construct(n_rows, k)
    best_scope = max(max(row) for row in best_solution)

    # Lower bound
    total_diffs_needed = n_rows * k * (k + 1) // 2
    lb = total_diffs_needed

    if best_scope <= lb:
        return best_solution

    # Simulated annealing on a flat array representation
    # State: flat array of length n_rows*(k+1), each row starts with 0
    # Move: pick a random row and a random position (1..k), change its value

    def flat_to_rows(flat):
        return [list(flat[r*(k+1):(r+1)*(k+1)]) for r in range(n_rows)]

    def rows_to_flat(rows):
        flat = []
        for row in rows:
            flat.extend(row)
        return flat

    def compute_all_diffs(flat):
        """Compute set of all differences from flat array. Returns (diff_set, row_diff_sets)."""
        all_diffs = set()
        row_diff_sets = []
        for r in range(n_rows):
            row = flat[r*(k+1):(r+1)*(k+1)]
            rd = set()
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    d = row[j] - row[i]
                    rd.add(d)
            row_diff_sets.append(rd)
            all_diffs |= rd
        return all_diffs, row_diff_sets

    def get_scope(flat):
        return max(flat[r*(k+1)+k] for r in range(n_rows))

    def is_valid_state(flat):
        """Check if flat array represents a valid DTS (strictly increasing rows, no duplicate diffs)."""
        all_diffs = set()
        for r in range(n_rows):
            row = flat[r*(k+1):(r+1)*(k+1)]
            # Check strictly increasing
            for i in range(1, len(row)):
                if row[i] <= row[i-1]:
                    return False
            # Check diffs
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    d = row[j] - row[i]
                    if d in all_diffs:
                        return False
                    all_diffs.add(d)
        return True

    # Initialize SA state
    current_flat = rows_to_flat(best_solution)
    current_all_diffs, current_row_diffs = compute_all_diffs(current_flat)
    current_scope = get_scope(current_flat)

    best_flat = list(current_flat)
    best_sa_scope = current_scope

    # SA parameters
    T = max(current_scope - lb, 10) * 2.0
    T_min = 0.5
    cooling_rate = 0.9995
    max_iters = 500000

    iter_count = 0
    no_improve = 0
    max_no_improve = 5000

    while elapsed() < deadline - 2.0:
        # Check time more frequently
        if iter_count % 500 == 0 and elapsed() > deadline - 2.0:
            break

        # Pick a random move
        r = random.randint(0, n_rows - 1)
        pos = random.randint(1, k)  # positions 1..k (position 0 is always 0)
        old_val = current_flat[r*(k+1) + pos]

        # Generate new value: biased toward reducing scope
        if random.random() < 0.6:
            # Try to reduce: pick a value smaller than current
            low = current_flat[r*(k+1) + pos - 1] + 1 if pos > 0 else 1
            high = old_val - 1
            if high >= low:
                new_val = random.randint(low, high)
            else:
                new_val = old_val
        else:
            # Try to increase slightly (for escaping local minima)
            low = old_val + 1
            high = min(best_scope + 20, 200)
            if high >= low:
                new_val = random.randint(low, high)
            else:
                new_val = old_val

        if new_val == old_val:
            continue

        # Check strictly increasing constraint
        row_start = r * (k + 1)
        if pos > 0 and new_val <= current_flat[row_start + pos - 1]:
            continue
        if pos < k and new_val >= current_flat[row_start + pos + 1]:
            continue

        # Compute old diffs for this row
        old_row = current_flat[row_start:row_start + k + 1]
        old_row_diffs = set()
        for i in range(k + 1):
            for j in range(i+1, k+1):
                old_row_diffs.add(old_row[j] - old_row[i])

        # Compute new diffs for this row
        new_row = list(old_row)
        new_row[pos] = new_val
        new_row_diffs = set()
        for i in range(k + 1):
            for j in range(i+1, k+1):
                new_row_diffs.add(new_row[j] - new_row[i])

        # Check for conflicts: new diffs that aren't in old_row_diffs must not be in other rows' diffs
        # Diffs removed from this row: old_row_diffs - new_row_diffs
        # Diffs added to this row: new_row_diffs - old_row_diffs
        added_diffs = new_row_diffs - old_row_diffs
        removed_diffs = old_row_diffs - new_row_diffs

        # Check if any added diff conflicts with other rows
        conflict = False
        for d in added_diffs:
            # d is in the global set. It was either in this row's old diffs (removed) or another row's diffs
            # If d was in old_row_diffs, it's being replaced, so we need to check if it's in another row
            # Actually: the global set has d. After the move, d is in new_row_diffs.
            # The issue is: is d in any OTHER row's diff set?
            # d is in current_all_diffs. d is in new_row_diffs (so it'll be in the new global set from this row).
            # But if d is also in another row's diff set, that's a conflict.
            # We need to check: for each other row, is d in that row's diffs?
            # Efficient check: d is in current_all_diffs. If d is NOT in old_row_diffs, then it must be in another row.
            if d not in old_row_diffs:
                conflict = True
                break
            # If d IS in old_row_diffs, it might still be in another row.
            # Check other rows
            for r2 in range(n_rows):
                if r2 == r:
                    continue
                if d in current_row_diffs[r2]:
                    conflict = True
                    break
            if conflict:
                break

        if conflict:
            # Move is invalid (creates duplicate diff). Accept with low probability.
            # Penalty-based: reject unless very hot
            if T > 1.0 and random.random() < math.exp(-1.0 / max(T, 0.01)):
                # Accept invalid move (rare, for exploration)
                pass
            else:
                iter_count += 1
                continue

        # Compute new scope
        new_scope = max(new_val, current_flat[row_start + k] if pos != k else 0)
        # Actually recompute scope properly
        new_scope = 0
        for r2 in range(n_rows):
            v = current_flat[r2*(k+1) + k]
            if r2 == r:
                v = new_val if pos == k else v
            if v > new_scope:
                new_scope = v

        # Delta in scope
        delta_scope = new_scope - current_scope

        # Accept/reject
        if delta_scope <= 0:
            accept = True
        else:
            accept = random.random() < math.exp(-delta_scope / max(T, 0.01))

        if accept:
            # Apply move
            current_flat[row_start + pos] = new_val

            # Update diff sets
            current_all_diffs -= removed_diffs
            current_all_diffs |= added_diffs
            current_row_diffs[r] = new_row_diffs
            current_scope = new_scope

            if new_scope < best_sa_scope:
                best_sa_scope = new_scope
                best_flat = list(current_flat)
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1

        # Cooling
        T *= cooling_rate
        if T < T_min:
            T = T_min

        iter_count += 1

        # Periodic restart if stuck
        if no_improve > max_no_improve and T <= T_min * 1.5:
            # Reheat
            T = max(current_scope - lb, 5)
            no_improve = 0

    # Convert best SA solution back to rows
    best_rows = flat_to_rows(best_flat)

    # Validate
    all_diffs = set()
    valid = True
    for row in best_rows:
        for i in range(len(row)):
            if i > 0 and row[i] <= row[i-1]:
                valid = False
                break
            for j in range(i+1, len(row)):
                d = row[j] - row[i]
                if d in all_diffs:
                    valid = False
                    break
                all_diffs.add(d)
        if not valid:
            break

    if valid:
        sa_scope = max(max(row) for row in best_rows)
        if sa_scope < best_scope:
            best_scope = sa_scope
            best_solution = best_rows
    else:
        # Fallback: use the greedy solution (which is always valid)
        best_solution = greedy_construct(n_rows, k)
        best_scope = max(max(row) for row in best_solution)

    return best_solution
# EVOLVE-BLOCK-END
