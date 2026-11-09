import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0

    def get_max_scope(rows):
        return max(max(r) for r in rows)

    def compute_used_diffs(rows):
        used = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    used.add(r[j] - r[i])
        return used

    def get_involved_diffs(row, pos, val):
        diffs = set()
        for i in range(pos):
            diffs.add(val - row[i])
        for j in range(pos + 1, len(row)):
            diffs.add(row[j] - val)
        return diffs

    # Greedy row builder with backtracking for small k
    def build_row_greedy(used_diffs, target_len, max_val=None):
        """Build a row of target_len elements starting at 0, strictly increasing,
        using only differences not in used_diffs."""
        row = [0]
        def backtrack():
            if len(row) == target_len:
                return True
            x = row[-1] + 1
            if max_val is not None and x > max_val:
                return False
            while True:
                new_diffs = {x - a for a in row}
                if not new_diffs & used_diffs:
                    row.append(x)
                    if backtrack():
                        return True
                    row.pop()
                x += 1
                if max_val is not None and x > max_val:
                    return False
        if backtrack():
            return row
        return None

    # Interleaved greedy initialization
    def generate_greedy_interleaved():
        used_diffs = set()
        rows = [[0] for _ in range(n_rows)]
        while any(len(r) < k + 1 for r in rows):
            for r_idx in range(n_rows):
                row = rows[r_idx]
                if len(row) >= k + 1:
                    continue
                x = row[-1] + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used_diffs:
                        row.append(x)
                        used_diffs |= new_diffs
                        break
                    x += 1
        return rows, used_diffs

    # Standard greedy (row-by-row)
    def generate_greedy_sequential():
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used_diffs:
                        row.append(x)
                        used_diffs |= new_diffs
                        break
                    x += 1
            rows.append(row)
        return rows, used_diffs

    # Try both and pick the better initial solution
    rows_a, used_a = generate_greedy_interleaved()
    scope_a = get_max_scope(rows_a)
    rows_b, used_b = generate_greedy_sequential()
    scope_b = get_max_scope(rows_b)

    if scope_a <= scope_b:
        rows = [r[:] for r in rows_a]
        used_diffs = set(used_a)
        current_scope = scope_a
    else:
        rows = [r[:] for r in rows_b]
        used_diffs = set(used_b)
        current_scope = scope_b

    best_rows = [r[:] for r in rows]
    best_scope = current_scope

    # Phase 1: Row-level reconstruction with different orderings
    # For each permutation of row orderings, try rebuilding rows greedily
    # This is a structural change: instead of element-level moves, we rebuild whole rows
    
    def try_row_order(order, used_diffs, current_scope):
        """Try rebuilding rows in the given order, each as compact as possible."""
        new_rows = [None] * n_rows
        new_used = set(used_diffs)
        
        for idx in order:
            # Remove this row's diffs from used set
            if new_rows[idx] is not None:
                for i in range(len(new_rows[idx])):
                    for j in range(i + 1, len(new_rows[idx])):
                        new_used.discard(new_rows[idx][j] - new_rows[idx][i])
            else:
                # Original row
                orig_row = rows[idx]
                for i in range(len(orig_row)):
                    for j in range(i + 1, len(orig_row)):
                        new_used.discard(orig_row[j] - orig_row[i])
            
            # Build new row greedily
            new_row = build_row_greedy(new_used, k + 1)
            if new_row is None:
                return None, None, None
            
            for i in range(len(new_row)):
                for j in range(i + 1, len(new_row)):
                    new_used.add(new_row[j] - new_row[i])
            
            new_rows[idx] = new_row
        
        new_scope = get_max_scope(new_rows)
        return new_rows, new_used, new_scope

    # Try several random orderings
    for attempt in range(20):
        if time.time() > deadline - 15:
            break
        order = list(range(n_rows))
        random.shuffle(order)
        new_rows, new_used, new_scope = try_row_order(order, used_diffs, current_scope)
        if new_rows is not None and new_scope < best_scope:
            best_scope = new_scope
            best_rows = [r[:] for r in new_rows]
            used_diffs = new_used
            current_scope = new_scope
            rows = [r[:] for r in new_rows]

    # Phase 2: Element-level hill climbing (refined)
    def hill_climb_pass(rows, used_diffs, current_scope, max_rounds=5):
        """Perform hill climbing to reduce scope."""
        improved = True
        rounds = 0
        while improved and rounds < max_rounds and time.time() < deadline - 10:
            improved = False
            rounds += 1
            row_order = list(range(n_rows))
            random.shuffle(row_order)

            for r_idx in row_order:
                if time.time() > deadline - 10:
                    break
                row = rows[r_idx]
                for pos in range(1, k + 1):
                    old_val = row[pos]
                    low = row[pos - 1] + 1
                    high = row[pos + 1] - 1 if pos < k else old_val

                    if low >= old_val:
                        continue

                    # Try to reduce value
                    for new_val in range(low, old_val):
                        old_involved = get_involved_diffs(row, pos, old_val)
                        used_diffs.difference_update(old_involved)

                        new_involved = get_involved_diffs(row, pos, new_val)

                        if len(new_involved) != len(old_involved):
                            used_diffs |= old_involved
                            continue

                        if new_involved & used_diffs:
                            used_diffs |= old_involved
                            continue

                        # Valid
                        used_diffs |= new_involved
                        row[pos] = new_val

                        new_scope = get_max_scope(rows)
                        if new_scope < current_scope:
                            current_scope = new_scope
                            improved = True
                            break
                        else:
                            used_diffs.difference_update(new_involved)
                            used_diffs |= old_involved
                            row[pos] = old_val

                    if improved:
                        break
                if improved:
                    break
        return rows, used_diffs, current_scope

    # Apply hill climbing
    rows, used_diffs, current_scope = hill_climb_pass(rows, used_diffs, current_scope)
    if current_scope < best_scope:
        best_scope = current_scope
        best_rows = [r[:] for r in rows]

    # Phase 3: Simulated annealing with row-reconstruction restarts
    T = 5.0
    cooling = 0.9995
    min_T = 0.1

    while time.time() < deadline - 5:
        # Randomly choose between element move and row rebuild
        if random.random() < 0.3:
            # Row rebuild
            r_idx = random.randint(0, n_rows - 1)
            # Remove row's diffs
            for i in range(len(rows[r_idx])):
                for j in range(i + 1, len(rows[r_idx])):
                    used_diffs.discard(rows[r_idx][j] - rows[r_idx][i])
            
            old_row = rows[r_idx][:]
            new_row = build_row_greedy(used_diffs, k + 1)
            if new_row is not None:
                for i in range(len(new_row)):
                    for j in range(i + 1, len(new_row)):
                        used_diffs.add(new_row[j] - new_row[i])
                rows[r_idx] = new_row
                new_scope = get_max_scope(rows)
                if new_scope < current_scope:
                    current_scope = new_scope
                    if current_scope < best_scope:
                        best_scope = current_scope
                        best_rows = [r[:] for r in rows]
                else:
                    delta = new_scope - current_scope
                    if random.random() < math.exp(-delta / T):
                        current_scope = new_scope
                    else:
                        # Revert
                        for i in range(len(new_row)):
                            for j in range(i + 1, len(new_row)):
                                used_diffs.discard(new_row[j] - new_row[i])
                        for i in range(len(old_row)):
                            for j in range(i + 1, len(old_row)):
                                used_diffs.add(old_row[j] - old_row[i])
                        rows[r_idx] = old_row
        else:
            # Element move
            r_idx = random.randint(0, n_rows - 1)
            row = rows[r_idx]
            pos = random.randint(1, k)
            old_val = row[pos]

            low = row[pos - 1] + 1
            high = row[pos + 1] - 1 if pos < k else old_val + 20

            if pos == k:
                if low < old_val:
                    new_val = random.randint(low, old_val - 1)
                else:
                    T *= cooling
                    if T < min_T:
                        T = 5.0
                    continue
            else:
                search_low = max(low, old_val - 10)
                search_high = min(high, old_val + 10)
                if search_low > search_high:
                    T *= cooling
                    if T < min_T:
                        T = 5.0
                    continue
                new_val = random.randint(search_low, search_high)

            if new_val == old_val:
                T *= cooling
                if T < min_T:
                    T = 5.0
                continue

            old_involved = get_involved_diffs(row, pos, old_val)
            used_diffs.difference_update(old_involved)

            new_involved = get_involved_diffs(row, pos, new_val)

            if len(new_involved) != len(old_involved):
                used_diffs |= old_involved
                T *= cooling
                if T < min_T:
                    T = 5.0
                continue

            if new_involved & used_diffs:
                used_diffs |= old_involved
                T *= cooling
                if T < min_T:
                    T = 5.0
                continue

            used_diffs |= new_involved
            row[pos] = new_val

            new_scope = get_max_scope(rows)

            if new_scope < current_scope:
                current_scope = new_scope
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [r[:] for r in rows]
            else:
                delta = new_scope - current_scope
                if random.random() < math.exp(-delta / T):
                    current_scope = new_scope
                else:
                    used_diffs.difference_update(new_involved)
                    used_diffs |= old_involved
                    row[pos] = old_val

            T *= cooling
            if T < min_T:
                T = 5.0
                # Quick hill climb
                rows, used_diffs, current_scope = hill_climb_pass(rows, used_diffs, current_scope, max_rounds=3)
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [r[:] for r in rows]

    # Phase 4: Final polish
    final_used = compute_used_diffs(best_rows)
    for r_idx in range(n_rows):
        row = best_rows[r_idx]
        for pos in range(1, k + 1):
            old_val = row[pos]
            low = row[pos - 1] + 1
            high = row[pos + 1] - 1 if pos < k else old_val
            if low >= old_val:
                continue
            for new_val in range(low, old_val):
                old_involved = get_involved_diffs(row, pos, old_val)
                final_used.difference_update(old_involved)
                new_involved = get_involved_diffs(row, pos, new_val)
                if len(new_involved) != len(old_involved):
                    final_used |= old_involved
                    continue
                if new_involved & final_used:
                    final_used |= old_involved
                    continue
                final_used |= new_involved
                row[pos] = new_val
                new_scope = get_max_scope(best_rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                    break
                else:
                    final_used.difference_update(new_involved)
                    final_used |= old_involved
                    row[pos] = old_val

    return best_rows
