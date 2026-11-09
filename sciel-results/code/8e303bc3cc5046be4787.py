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

    def generate_greedy_random_order():
        used_diffs = set()
        rows = [[0] for _ in range(n_rows)]
        order = list(range(n_rows))
        random.shuffle(order)
        # Grow rows in random order but interleaved
        while any(len(r) < k + 1 for r in rows):
            for r_idx in order:
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

    # Try multiple initializations
    candidates = []
    rows_a, used_a = generate_greedy_interleaved()
    candidates.append((rows_a, used_a))
    rows_b, used_b = generate_greedy_sequential()
    candidates.append((rows_b, used_b))
    rows_c, used_c = generate_greedy_random_order()
    candidates.append((rows_c, used_c))

    best_init = None
    best_init_scope = float('inf')
    for rows_c, used_c in candidates:
        s = get_max_scope(rows_c)
        if s < best_init_scope:
            best_init_scope = s
            best_init = (rows_c, used_c)

    rows = [r[:] for r in best_init[0]]
    used_diffs = set(best_init[1])
    current_scope = best_init_scope

    best_rows = [r[:] for r in rows]
    best_scope = current_scope

    def get_involved_diffs(row, pos, val):
        diffs = set()
        for i in range(pos):
            diffs.add(val - row[i])
        for j in range(pos + 1, len(row)):
            diffs.add(row[j] - val)
        return diffs

    # Gap-space representation: gaps[i] = row[i+1] - row[i] for i in 0..k-1
    # Scope = sum of gaps. Reducing any gap by 1 reduces scope by 1 if that row is the max.

    def gaps_of_row(row):
        return [row[i+1] - row[i] for i in range(len(row) - 1)]

    def row_from_gaps(gaps):
        row = [0]
        for g in gaps:
            row.append(row[-1] + g)
        return row

    # Phase 1: Aggressive hill climbing in gap space
    # Focus on reducing gaps, especially in the row with the largest scope
    def hill_climb_gaps(rows, used_diffs, max_iters=40):
        improved = True
        it = 0
        while improved and it < max_iters and time.time() < deadline - 12:
            improved = False
            it += 1
            # Process rows in order of decreasing max element
            row_order = sorted(range(n_rows), key=lambda i: -max(rows[i]))
            if it > 1:
                # Slight randomization
                if random.random() < 0.3:
                    random.shuffle(row_order)

            for r_idx in row_order:
                if time.time() > deadline - 12:
                    break
                row = rows[r_idx]
                # Try to reduce each position from last to first
                for pos in range(k, 0, -1):
                    old_val = row[pos]
                    low = row[pos - 1] + 1
                    high = row[pos + 1] - 1 if pos < k else old_val

                    if low >= old_val:
                        continue

                    # Try reducing by 1 first (smallest change)
                    for new_val in range(old_val - 1, low - 1, -1):
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
                        improved = True
                        break

                    if improved:
                        break

        return rows, used_diffs

    rows, used_diffs = hill_climb_gaps(rows, used_diffs)
    current_scope = get_max_scope(rows)
    if current_scope < best_scope:
        best_scope = current_scope
        best_rows = [r[:] for r in rows]

    # Phase 2: Simulated Annealing with row rebuilds and gap-space moves
    T = 4.0
    cooling = 0.9995
    min_T = 0.1

    def rebuild_row(r_idx):
        row = rows[r_idx]
        old_involved = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                old_involved.add(row[j] - row[i])

        trial_used = used_diffs - old_involved

        new_row = [0]
        for pos in range(1, k + 1):
            x = new_row[-1] + 1
            found = False
            max_search = row[-1] + 30
            while x <= max_search:
                new_diffs = {x - a for a in new_row}
                if not new_diffs & trial_used:
                    new_row.append(x)
                    trial_used |= new_diffs
                    found = True
                    break
                x += 1

            if not found:
                return None, None, current_scope, False

        other_maxes = [max(r) for i, r in enumerate(rows) if i != r_idx]
        temp_scope = max(other_maxes + [max(new_row)])

        return new_row, trial_used, temp_scope, True

    sa_steps = 0
    while time.time() < deadline - 5:
        sa_steps += 1
        move_type = random.random()

        if move_type < 0.25:
            # Row Rebuild Move
            r_idx = random.randint(0, n_rows - 1)
            old_row = rows[r_idx]
            old_scope = current_scope

            new_row, new_used, new_scope_val, success = rebuild_row(r_idx)

            if success:
                if new_scope_val < current_scope:
                    rows[r_idx] = new_row
                    used_diffs = new_used
                    current_scope = new_scope_val
                    if current_scope < best_scope:
                        best_scope = current_scope
                        best_rows = [r[:] for r in rows]
                else:
                    delta = new_scope_val - current_scope
                    if delta > 0 and random.random() < math.exp(-delta / T):
                        rows[r_idx] = new_row
                        used_diffs = new_used
                        current_scope = new_scope_val

        elif move_type < 0.6:
            # Gap-space move: pick a row and try to reduce a gap
            r_idx = random.randint(0, n_rows - 1)
            row = rows[r_idx]
            # Bias toward the last position (affects scope directly)
            if random.random() < 0.5:
                pos = k  # last element
            else:
                pos = random.randint(1, k)

            old_val = row[pos]
            low = row[pos - 1] + 1
            high = row[pos + 1] - 1 if pos < k else old_val + 20

            if pos == k:
                # Try to reduce
                if low < old_val:
                    new_val = random.randint(low, old_val - 1)
                else:
                    continue
            else:
                search_low = max(low, old_val - 10)
                search_high = min(high, old_val + 10)
                if search_low > search_high:
                    continue
                new_val = random.randint(search_low, search_high)

            if new_val == old_val:
                continue

            old_involved = get_involved_diffs(row, pos, old_val)
            used_diffs.difference_update(old_involved)
            new_involved = get_involved_diffs(row, pos, new_val)

            if len(new_involved) != len(old_involved):
                used_diffs |= old_involved
                continue

            if new_involved & used_diffs:
                used_diffs |= old_involved
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
                if delta > 0 and random.random() < math.exp(-delta / T):
                    current_scope = new_scope
                else:
                    used_diffs.difference_update(new_involved)
                    used_diffs |= old_involved
                    row[pos] = old_val

        else:
            # Two-position swap within a row (structural move)
            r_idx = random.randint(0, n_rows - 1)
            row = rows[r_idx]
            if k >= 2:
                p1 = random.randint(1, k - 1)
                p2 = random.randint(p1 + 1, k)
                v1, v2 = row[p1], row[p2]
                # Try swapping values if valid ordering is maintained
                if v1 < v2:  # already in order, try swapping
                    row[p1], row[p2] = v2, v1
                    # Check validity
                    old_involved = set()
                    for i in range(len(row)):
                        for j in range(i + 1, len(row)):
                            old_involved.add(row[j] - row[i])
                    # This is complex, so skip swap and just do a local move
                    row[p1], row[p2] = v1, v2
                    # Fall through to a simple local move
                    pos = random.randint(1, k)
                    old_val = row[pos]
                    low = row[pos - 1] + 1
                    high = row[pos + 1] - 1 if pos < k else old_val + 15
                    if low < high:
                        new_val = random.randint(low, high)
                        if new_val != old_val:
                            old_inv = get_involved_diffs(row, pos, old_val)
                            used_diffs.difference_update(old_inv)
                            new_inv = get_involved_diffs(row, pos, new_val)
                            if len(new_inv) == len(old_inv) and not (new_inv & used_diffs):
                                used_diffs |= new_inv
                                row[pos] = new_val
                                new_scope = get_max_scope(rows)
                                if new_scope < current_scope:
                                    current_scope = new_scope
                                    if current_scope < best_scope:
                                        best_scope = current_scope
                                        best_rows = [r[:] for r in rows]
                                else:
                                    delta = new_scope - current_scope
                                    if delta > 0 and random.random() < math.exp(-delta / T):
                                        current_scope = new_scope
                                    else:
                                        used_diffs.difference_update(new_inv)
                                        used_diffs |= old_inv
                                        row[pos] = old_val

        T *= cooling
        if T < min_T:
            # Restart with quick hill climb
            T = 4.0
            improved_hc = True
            while improved_hc and time.time() < deadline - 5:
                improved_hc = False
                for r_idx2 in range(n_rows):
                    row2 = rows[r_idx2]
                    for pos2 in range(k, 0, -1):
                        old_val2 = row2[pos2]
                        low2 = row2[pos2 - 1] + 1
                        high2 = row2[pos2 + 1] - 1 if pos2 < k else old_val2
                        if low2 >= old_val2:
                            continue
                        for new_val2 in range(old_val2 - 1, low2 - 1, -1):
                            old_inv2 = get_involved_diffs(row2, pos2, old_val2)
                            used_diffs.difference_update(old_inv2)
                            new_inv2 = get_involved_diffs(row2, pos2, new_val2)
                            if len(new_inv2) != len(old_inv2):
                                used_diffs |= old_inv2
                                continue
                            if new_inv2 & used_diffs:
                                used_diffs |= old_inv2
                                continue
                            used_diffs |= new_inv2
                            row2[pos2] = new_val2
                            new_scope2 = get_max_scope(rows)
                            if new_scope2 < current_scope:
                                current_scope = new_scope2
                                if current_scope < best_scope:
                                    best_scope = current_scope
                                    best_rows = [r[:] for r in rows]
                                improved_hc = True
                                break
                            else:
                                used_diffs.difference_update(new_inv2)
                                used_diffs |= old_inv2
                                row2[pos2] = old_val2
                        if improved_hc:
                            break
                    if improved_hc:
                        break

    # Phase 3: Final polish
    final_used = compute_used_diffs(best_rows)
    for r_idx in range(n_rows):
        row = best_rows[r_idx]
        for pos in range(k, 0, -1):
            old_val = row[pos]
            low = row[pos - 1] + 1
            high = row[pos + 1] - 1 if pos < k else old_val
            if low >= old_val:
                continue
            for new_val in range(old_val - 1, low - 1, -1):
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
