def construct(n_rows, k):
    import sys
    import time

    sys.setrecursionlimit(100000)
    start_time = time.time()
    time_limit = 115.0

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

    rows = greedy_construct(n_rows, k)
    best_max = max(max(r) for r in rows)
    best_rows = [list(r) for r in rows]

    def solve_with_max(M, n_rows, k):
        """Find n_rows mutually non-conflicting rows with all entries <= M."""
        if M < k:
            return None

        found = [False]
        result_rows = []
        node_count = [0]
        node_limit = 4000000

        def build_rows(row_idx, current_row, current_mask, all_used_mask, prev_row):
            """Build row_idx-th row with lexicographic symmetry breaking."""
            if found[0] or node_count[0] > node_limit:
                return
            if time.time() - start_time > time_limit - 1:
                return
            node_count[0] += 1

            if len(current_row) == k + 1:
                # Symmetry: rows must be in lexicographic non-decreasing order
                if prev_row is not None and current_row < prev_row:
                    return

                result_rows.append(current_row[:])

                if row_idx + 1 == n_rows:
                    found[0] = True
                    return

                build_rows(row_idx + 1, [0], current_mask, all_used_mask | current_mask, current_row)
                if not found[0]:
                    result_rows.pop()
                return

            last = current_row[-1]
            for x in range(last + 1, M + 1):
                valid = True
                new_mask = 0
                for a in current_row:
                    d = x - a
                    bit = 1 << d
                    if (all_used_mask | current_mask) & bit:
                        valid = False
                        break
                    new_mask |= bit
                if not valid:
                    continue

                build_rows(row_idx, current_row + [x], current_mask | new_mask, all_used_mask, prev_row)
                if found[0]:
                    return

        build_rows(0, [0], 0, 0, None)

        if not found[0]:
            return None
        return [list(r) for r in result_rows]

    for target_max in range(best_max - 1, max(best_max - 25, k), -1):
        if time.time() - start_time > time_limit:
            break
        result = solve_with_max(target_max, n_rows, k)
        if result is not None:
            best_rows = result
            best_max = target_max
            break

    return best_rows
