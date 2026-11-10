import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 100  # Safety margin for 150s limit

    num_elements = k + 1
    total_diffs = n_rows * k * (k + 1) // 2
    lower_bound = math.ceil((-1 + math.sqrt(1 + 8 * total_diffs)) / 2)

    best_rows = None
    best_scope = float('inf')

    def get_scope(rows):
        return max(max(r) for r in rows)

    def row_to_mask(row):
        """Convert a row to a bitmask of its differences."""
        mask = 0
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                mask |= (1 << (d - 1))
        return mask

    def full_mask(rows):
        """Compute the full difference mask for all rows."""
        mask = 0
        for row in rows:
            mask |= row_to_mask(row)
        return mask

    def is_valid_mask(mask):
        """Check if mask has no duplicate differences (always true by construction if we track properly)."""
        # If we build masks by OR-ing, we can't detect duplicates this way.
        # Instead, we track count: valid iff popcount(mask) == total_diffs
        return bin(mask).count('1') == total_diffs

    def greedy_construct():
        """Greedy construction using bit-mask for speed."""
        used = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < num_elements:
                x += 1
                new_mask = 0
                valid = True
                for a in row:
                    d = x - a
                    bit = 1 << (d - 1)
                    if (used & bit) or (new_mask & bit):
                        valid = False
                        break
                    new_mask |= bit
                if valid:
                    row.append(x)
                    used |= new_mask
            rows.append(row)
        return rows

    def build_row_backtrack(used_mask, row, pos, current_max, rng, max_window, deadline_check):
        """Backtracking to build one row given current used_mask."""
        if deadline_check():
            return None
        if pos == num_elements:
            return row[:]

        start_val = current_max + 1
        # Generate candidates in a window, shuffle for diversity
        candidates = list(range(start_val, start_val + max_window))
        rng.shuffle(candidates)

        for x in candidates:
            if deadline_check():
                return None

            new_mask = 0
            valid = True
            for a in row:
                d = x - a
                bit = 1 << (d - 1)
                if (used_mask & bit) or (new_mask & bit):
                    valid = False
                    break
                new_mask |= bit

            if not valid:
                continue

            row.append(x)
            result = build_row_backtrack(used_mask | new_mask, row, pos + 1, x, rng, max_window, deadline_check)
            row.pop()

            if result is not None:
                return result

        return None

    def elapsed():
        return time.time() - start_time

    def deadline_check():
        return elapsed() > deadline

    # Phase 1: Greedy baseline
    greedy_rows = greedy_construct()
    greedy_scope = get_scope(greedy_rows)
    best_rows = greedy_rows
    best_scope = greedy_scope

    # Phase 2: Randomized backtracking for better initial solutions
    # Limit attempts to ensure we have time for local search
    max_attempts = 50
    for attempt in range(max_attempts):
        if deadline_check() or (time.time() - start_time) > deadline - 40:
            break
        rng = random.Random(attempt * 7919 + 13)

        used = 0
        rows = []
        success = True

        for row_idx in range(n_rows):
            row = [0]
            # Use a wider window for later rows (more constraints)
            max_window = 60 + row_idx * 20
            result = build_row_backtrack(used, row, 1, 0, rng, max_window, deadline_check)
            if result is None:
                success = False
                break
            rows.append(result)
            used |= row_to_mask(result)

        if success and len(rows) == n_rows:
            scope = get_scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [r[:] for r in rows]
                if best_scope <= lower_bound:
                    return best_rows

    # Phase 3: Local search with simulated annealing using bit-mask for speed
    if best_rows is not None and not deadline_check():
        current_rows = [r[:] for r in best_rows]
        current_scope = best_scope
        current_diffs_mask = full_mask(current_rows)

        temperature = max(1.0, (current_scope - lower_bound) / 5.0)
        cooling_rate = 0.9995
        no_improve_count = 0
        max_no_improve = 2000
        max_iterations = 5000 # Hard cap on iterations to prevent infinite loops

        iterations = 0
        while not deadline_check() and no_improve_count < max_no_improve and iterations < max_iterations:
            iterations += 1
            row_idx = random.randint(0, len(current_rows) - 1)
            row = current_rows[row_idx]

            # Move type 1: Shift one element (70% probability)
            if random.random() < 0.7 and len(row) > 1:
                idx = random.randint(1, len(row) - 1)
                delta = random.randint(-30, 30)
                new_val = row[idx] + delta

                if new_val <= row[idx - 1]:
                    continue
                if idx < len(row) - 1 and new_val >= row[idx + 1]:
                    continue

                # Compute old and new differences involving this element
                old_mask = 0
                new_mask = 0
                for j in range(idx):
                    d_old = row[idx] - row[j]
                    d_new = new_val - row[j]
                    old_mask |= (1 << (d_old - 1))
                    new_mask |= (1 << (d_new - 1))
                for j in range(idx + 1, len(row)):
                    d_old = row[j] - row[idx]
                    d_new = row[j] - new_val
                    old_mask |= (1 << (d_old - 1))
                    new_mask |= (1 << (d_new - 1))

                other_mask = current_diffs_mask & ~old_mask
                if new_mask & other_mask:
                    continue

                # Valid move
                # Update current_rows in-place to avoid deep copy cost if accepted, 
                # but we need to revert if rejected. 
                # For simplicity and safety, we copy, but we can optimize scope calc.
                old_val = current_rows[row_idx][idx]
                current_rows[row_idx][idx] = new_val
                new_diffs_mask = other_mask | new_mask
                new_scope = max(max(r) for r in current_rows)

                # Accept or reject based on simulated annealing
                if new_scope <= current_scope or random.random() < math.exp(-(new_scope - current_scope) / temperature):
                    current_scope = new_scope
                    current_diffs_mask = new_diffs_mask
                    no_improve_count = 0
                    if current_scope < best_scope:
                        best_scope = current_scope
                        best_rows = [r[:] for r in current_rows]
                        if best_scope <= lower_bound:
                            return best_rows
                else:
                    current_rows[row_idx][idx] = old_val
                    no_improve_count += 1
            else:
                # Move type 2: Replace entire row using backtracking
                # This is expensive, so only do it occasionally
                if random.random() < 0.2:
                    other_mask = 0
                    for i, r in enumerate(current_rows):
                        if i != row_idx:
                            other_mask |= row_to_mask(r)

                    new_row = [0]
                    rng = random.Random()
                    result = build_row_backtrack(other_mask, new_row, 1, 0, rng, 80, deadline_check)

                    if result is not None:
                        old_row = current_rows[row_idx]
                        current_rows[row_idx] = result
                        new_diffs_mask = other_mask | row_to_mask(result)
                        new_scope = max(max(r) for r in current_rows)

                        if new_scope <= current_scope or random.random() < math.exp(-(new_scope - current_scope) / temperature):
                            current_scope = new_scope
                            current_diffs_mask = new_diffs_mask
                            no_improve_count = 0
                            if current_scope < best_scope:
                                best_scope = current_scope
                                best_rows = [r[:] for r in current_rows]
                                if best_scope <= lower_bound:
                                    return best_rows
                        else:
                            current_rows[row_idx] = old_row
                            no_improve_count += 1
                    else:
                        no_improve_count += 1

            temperature *= cooling_rate

    return best_rows
