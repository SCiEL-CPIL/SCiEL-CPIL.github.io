"""Improved program for difference triangle set (4,7)."""
import time
import random
from collections import Counter

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct across all rows.
    Minimize the largest entry.
    """
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill

    # Greedy construction for initial solution
    def greedy_construct(n_rows, k):
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

    # Compute the largest entry (score)
    def get_score(rows):
        return max(max(row) for row in rows)

    # Check validity
    def is_valid(rows, k):
        all_diffs = []
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            for i in range(k):
                if row[i+1] <= row[i]:
                    return False
            for i in range(k + 1):
                for j in range(i+1, k+1):
                    all_diffs.append(row[j] - row[i])
        # All differences must be distinct
        return len(all_diffs) == len(set(all_diffs))

    # Simulated annealing with local search
    def simulated_annealing(rows, k, time_limit):
        n = len(rows)
        current_rows = [list(r) for r in rows]
        current_score = get_score(current_rows)
        best_rows = [list(r) for r in current_rows]
        best_score = current_score

        used = set()
        for row in current_rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    used.add(row[j] - row[i])

        def rebuild_used(rows):
            used = set()
            for row in rows:
                for i in range(len(row)):
                    for j in range(i+1, len(row)):
                        used.add(row[j] - row[i])
            return used

        def try_move(rows, used, row_idx, k):
            """Try to improve a single row by local search."""
            row = list(rows[row_idx])
            # Remove this row's diffs from used
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    used.discard(row[j] - row[i])

            # Try to find a better row with smaller max
            max_val = row[-1]
            for attempt in range(50):
                # Try to reduce the last element
                new_row = list(row)
                # Try shrinking the last element
                for new_last in range(max_val - 1, 0, -1):
                    if new_last <= new_row[-2]:
                        break
                    new_row[-1] = new_last
                    new_diffs = set()
                    valid = True
                    for i in range(len(new_row)):
                        for j in range(i+1, len(new_row)):
                            d = new_row[j] - new_row[i]
                            if d in used or d in new_diffs:
                                valid = False
                                break
                            new_diffs.add(d)
                        if not valid:
                            break
                    if valid:
                        # Accept this improvement
                        for i in range(len(row)):
                            for j in range(i+1, len(row)):
                                used.discard(row[j] - row[i])
                        rows[row_idx] = new_row
                        for d in new_diffs:
                            used.add(d)
                        return True, new_row[-1]
                # Try random perturbation
                pos = random.randint(1, len(row)-1)
                old_val = row[pos]
                delta = random.choice([-3, -2, -1, 1, 2, 3])
                new_val = old_val + delta
                if new_val > row[pos-1] and (pos == len(row)-1 or new_val < row[pos+1]):
                    row[pos] = new_val
                    # Check validity
                    new_diffs = set()
                    valid = True
                    for i in range(len(row)):
                        for j in range(i+1, len(row)):
                            d = row[j] - row[i]
                            if d in used or d in new_diffs:
                                valid = False
                                break
                            new_diffs.add(d)
                        if not valid:
                            break
                    if valid:
                        for i in range(len(row)):
                            for j in range(i+1, len(row)):
                                used.discard(row[j] - row[i])
                        rows[row_idx] = row
                        for d in new_diffs:
                            used.add(d)
                        return True, row[-1]
                    else:
                        row[pos] = old_val

            return False, row[-1]

        # Main SA loop
        t = 1.0
        t_min = 0.01
        cooling = 0.995
        random.seed(42)

        while time.time() < time_limit:
            # Pick a random row to modify
            row_idx = random.randint(0, n-1)
            success, new_max = try_move(current_rows, used, row_idx, k)

            if success:
                new_score = get_score(current_rows)
                if new_score < best_score:
                    best_score = new_score
                    best_rows = [list(r) for r in current_rows]
                current_score = new_score
            else:
                # Try a more aggressive move: rebuild a row from scratch
                if random.random() < 0.3:
                    # Save current row
                    old_row = list(current_rows[row_idx])
                    # Remove from used
                    for i in range(len(old_row)):
                        for j in range(i+1, len(old_row)):
                            used.discard(old_row[j] - old_row[i])

                    # Try greedy with randomization
                    new_row = [0]
                    x = 0
                    while len(new_row) < k + 1:
                        x += 1
                        if random.random() < 0.1 and x > 5:
                            x += random.randint(1, 5)
                        new_diffs = {x - a for a in new_row}
                        if not new_diffs & used:
                            new_row.append(x)
                            used |= new_diffs

                    new_score = max(get_score(current_rows), new_row[-1])
                    if new_score <= current_score:
                        current_rows[row_idx] = new_row
                        current_score = new_score
                        if new_score < best_score:
                            best_score = new_score
                            best_rows = [list(r) for r in current_rows]

            t *= cooling
            if t < t_min:
                t = t_min

        return best_rows, best_score

    # Get initial solution
    rows = greedy_construct(n_rows, k)
    if not is_valid(rows, k):
        raise ValueError("Initial solution is invalid")

    best_rows = rows
    best_score = get_score(rows)

    # Run simulated annealing
    if time.time() < deadline:
        rows, score = simulated_annealing(rows, k, deadline)
        if score < best_score:
            best_score = score
            best_rows = rows

    # Final polish: try to reduce each row's max further
    if time.time() < deadline - 5.0:
        used = set()
        for row in best_rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    used.add(row[j] - row[i])

        for row_idx in range(n_rows):
            row = list(best_rows[row_idx])
            # Remove from used
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    used.discard(row[j] - row[i])

            max_val = row[-1]
            # Try to reduce max_val
            for new_max in range(max_val - 1, 0, -1):
                if time.time() > deadline - 5.0:
                    break
                # Try to find a valid row ending at new_max
                found = False
                for attempt in range(100):
                    candidate = [0]
                    x = 0
                    while len(candidate) < k + 1:
                        x += 1
                        if x > new_max:
                            break
                        new_diffs = {x - a for a in candidate}
                        if not new_diffs & used:
                            candidate.append(x)
                            used |= new_diffs
                    if len(candidate) == k + 1 and candidate[-1] == new_max:
                        # Found a valid row
                        for i in range(len(row)):
                            for j in range(i+1, len(row)):
                                used.discard(row[j] - row[i])
                        best_rows[row_idx] = candidate
                        for d in new_diffs:
                            used.add(d)
                        best_score = max(best_score, new_max)
                        found = True
                        break
                    else:
                        # Roll back
                        for i in range(len(candidate)):
                            for j in range(i+1, len(candidate)):
                                used.discard(candidate[j] - candidate[i])
                        break
                if found:
                    break

    return best_rows
# EVOLVE-BLOCK-END
