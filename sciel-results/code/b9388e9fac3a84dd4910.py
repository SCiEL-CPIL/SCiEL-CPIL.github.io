import time

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0

    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    lower_bound = max(total_diffs, k)

    def greedy_construct():
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

    greedy_rows = greedy_construct()
    greedy_scope = max(max(r) for r in greedy_rows)
    best_solution = greedy_rows
    best_scope = greedy_scope

    def backtrack(rows, used_diffs, M):
        if len(rows) == n_rows:
            return rows

        if time.time() > deadline:
            return None

        last_row = rows[-1] if rows else None

        def build_row(current_row, min_val, is_greater):
            if len(current_row) == k + 1:
                if last_row is not None and not is_greater:
                    return None
                new_rows = rows + [current_row]
                new_used = used_diffs.union({b - a for i, a in enumerate(current_row) for b in current_row[i+1:]})
                return backtrack(new_rows, new_used, M)

            if time.time() > deadline:
                return None

            if last_row is not None and not is_greater:
                min_val = max(min_val, last_row[len(current_row)])

            for x in range(min_val, M + 1):
                valid = True
                for a in current_row:
                    if (x - a) in used_diffs:
                        valid = False
                        break
                if not valid:
                    continue

                new_is_greater = is_greater
                if last_row is not None and not is_greater:
                    if x > last_row[len(current_row)]:
                        new_is_greater = True

                new_row = current_row + [x]
                res = build_row(new_row, x + 1, new_is_greater)
                if res is not None:
                    return res
            return None

        start_val = 1
        start_is_greater = True
        if last_row is not None:
            start_val = last_row[1]
            start_is_greater = False

        return build_row([0], start_val, start_is_greater)

    for M in range(lower_bound, best_scope):
        if time.time() > deadline:
            break
        solution = backtrack([], set(), M)
        if solution is not None:
            best_solution = solution
            best_scope = M
            break

    return best_solution
