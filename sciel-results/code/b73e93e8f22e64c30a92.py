"""Difference triangle set construction via iterative local search + greedy fallback."""
import time
import random
import sys

def construct(n_rows, k):
    """Construct n_rows rows of k+1 increasing integers starting at 0,
    minimizing the largest entry such that all positive differences are distinct
    across all rows."""
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill

    # Greedy construction to get an initial valid solution
    def greedy_construct():
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

    rows = greedy_construct()
    best_scope = max(max(r) for r in rows)

    def all_diffs(rows):
        diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    diffs.add(row[j] - row[i])
        return diffs

    def is_valid(rows):
        diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    d = row[j] - row[i]
                    if d in diffs:
                        return False
                    diffs.add(d)
        return True

    def scope(rows):
        return max(max(r) for r in rows)

    # Local search: try to reduce scope by moving elements
    def local_search(rows, time_limit):
        nonlocal best_scope
        rows = [list(r) for r in rows]
        current_scope = scope(rows)
        used = all_diffs(rows)
        best_rows = [list(r) for r in rows]
        best_local_scope = current_scope
        no_improve_count = 0

        while time.time() < time_limit:
            if no_improve_count > 200:
                break

            # Pick a random row and a random non-zero element to move
            ri = random.randint(0, n_rows - 1)
            ei = random.randint(1, len(rows[ri]) - 1)  # don't move 0

            old_val = rows[ri][ei]
            # Try to move to a smaller value to reduce scope
            # Or try random perturbation
            if random.random() < 0.7 and old_val > 0:
                # Try to decrease
                candidates = list(range(max(1, old_val - 20), old_val))
                random.shuffle(candidates)
            else:
                # Random perturbation
                delta = random.randint(-15, 15)
                new_val = old_val + delta
                candidates = [new_val] if new_val > 0 else []

            moved = False
            for new_val in candidates:
                if new_val <= 0:
                    continue
                # Check strictly increasing
                if ei > 0 and new_val <= rows[ri][ei-1]:
                    continue
                if ei < len(rows[ri]) - 1 and new_val >= rows[ri][ei+1]:
                    continue

                # Compute old diffs involving this element
                old_diffs = set()
                for j in range(len(rows[ri])):
                    if j != ei:
                        d = abs(rows[ri][ei] - rows[ri][j])
                        old_diffs.add(d)

                # Check new diffs
                new_diffs = set()
                conflict = False
                for j in range(len(rows[ri])):
                    if j != ei:
                        d = abs(new_val - rows[ri][j])
                        if d in used and d not in old_diffs:
                            conflict = True
                            break
                        new_diffs.add(d)

                if conflict:
                    continue

                # Check if new_diffs have internal conflicts (they shouldn't since row is still strictly increasing)
                # Also check that new_diffs don't conflict with each other (they're all different by construction since row is strictly increasing)

                # Apply move
                rows[ri][ei] = new_val
                # Update used set
                for d in old_diffs:
                    if d not in new_diffs:
                        used.discard(d)
                for d in new_diffs:
                    if d not in old_diffs:
                        used.add(d)

                cur_scope = scope(rows)
                if cur_scope < best_local_scope:
                    best_local_scope = cur_scope
                    best_rows = [list(r) for r in rows]
                    if cur_scope < best_scope:
                        best_scope = cur_scope
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                moved = True
                break

            if not moved:
                no_improve_count += 1

        return best_rows, best_local_scope

    # Run local search with time budget
    remaining = deadline - time.time()
    if remaining > 5:
        # Split time among multiple restarts
        num_restarts = max(1, int(remaining // 10))
        per_restart = remaining / num_restarts
        for restart in range(num_restarts):
            if time.time() >= deadline - 2:
                break
            t_limit = time.time() + per_restart * 0.95
            # Sometimes start from best, sometimes from fresh greedy
            if restart > 0 and random.random() < 0.5:
                # Perturb best solution
                start_rows = [list(r) for r in best_rows]
                # Add small random perturbations
                for ri in range(n_rows):
                    for ei in range(1, len(start_rows[ri])):
                        if random.random() < 0.3:
                            delta = random.randint(-5, 5)
                            new_val = start_rows[ri][ei] + delta
                            if new_val > 0:
                                if ei > 0 and new_val <= start_rows[ri][ei-1]:
                                    continue
                                if ei < len(start_rows[ri]) - 1 and new_val >= start_rows[ri][ei+1]:
                                    continue
                                start_rows[ri][ei] = new_val
                if not is_valid(start_rows):
                    start_rows = [list(r) for r in best_rows]
            else:
                start_rows = [list(r) for r in best_rows]

            improved_rows, improved_scope = local_search(start_rows, t_limit)
            if improved_scope < best_scope:
                best_scope = improved_scope
                best_rows = improved_rows

    # Final validity check
    if not is_valid(best_rows):
        # Fallback to greedy
        best_rows = greedy_construct()

    return best_rows
