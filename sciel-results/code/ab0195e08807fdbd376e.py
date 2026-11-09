<<<<<<< SEARCH
    # --- Phase 6: Final polish – try to reduce scope by 1 via targeted search ---
    if best_rows is not None and best_scope > 105:
        remaining = _DEADLINE - time.time() - 1.0
        if remaining > 2.0:
            target = best_scope - 1
            # Try backtracking with tight target
            bt_rows = _backtrack_search(n_rows, k, target, rnd, time_budget=remaining)
            if bt_rows is not None and _valid(bt_rows, n_rows, k):
                sc = _scope(bt_rows)
                if sc < best_scope:
                    best_scope = sc
                    best_rows = bt_rows

    return best_rows
=======
    # --- Phase 6: Final polish – try to reduce scope via targeted backtracking ---
    if best_rows is not None and best_scope > 105:
        remaining = _DEADLINE - time.time() - 1.0
        if remaining > 2.0:
            for target in range(best_scope - 1, max(104, best_scope - 5), -1):
                if _dead():
                    break
                bt_rows = _backtrack_search(n_rows, k, target, rnd, time_budget=min(remaining, 5.0))
                if bt_rows is not None and _valid(bt_rows, n_rows, k):
                    sc = _scope(bt_rows)
                    if sc < best_scope:
                        best_scope = sc
                        best_rows = bt_rows
                        break
                remaining = _DEADLINE - time.time() - 1.0
                if remaining < 1.0:
                    break

    # --- Phase 7: One more local search pass if time remains ---
    if best_rows is not None:
        remaining = _DEADLINE - time.time() - 0.5
        if remaining > 1.0:
            best = [r[:] for r in best_rows]
            best_sc = _scope(best)
            max_val = best_sc + 5
            used_arr = [0] * (max_val + 1)
            for r in best:
                for i in range(len(r)):
                    for j in range(i + 1, len(r)):
                        d = r[j] - r[i]
                        if d <= max_val:
                            used_arr[d] += 1

            deadline_local = time.time() + remaining
            improved = True
            while improved and time.time() < deadline_local:
                improved = False
                for ri in range(n_rows):
                    row = best[ri]
                    for i in range(len(row)):
                        for j in range(i + 1, len(row)):
                            d = row[j] - row[i]
                            if d <= max_val:
                                used_arr[d] -= 1
                    for pi in range(1, len(row)):
                        old_val = row[pi]
                        found = False
                        for new_val in range(old_val - 1, row[pi-1], -1):
                            if pi < len(row) - 1 and new_val >= row[pi + 1]:
                                continue
                            bad = False
                            for j in range(pi):
                                d = new_val - row[j]
                                if d > max_val or used_arr[d] > 0:
                                    bad = True
                                    break
                            if bad:
                                continue
                            for j in range(pi + 1, len(row)):
                                d = row[j] - new_val
                                if d > max_val or used_arr[d] > 0:
                                    bad = True
                                    break
                            if bad:
                                continue
                            row[pi] = new_val
                            for j in range(pi):
                                d = new_val - row[j]
                                used_arr[d] += 1
                            for j in range(pi + 1, len(row)):
                                d = row[j] - new_val
                                used_arr[d] += 1
                            found = True
                            break
                        if found:
                            improved = True
                    if not found:
                        for i in range(len(row)):
                            for j in range(i + 1, len(row)):
                                d = row[j] - row[i]
                                if d <= max_val:
                                    used_arr[d] += 1
                    new_sc = _scope(best)
                    if new_sc < best_sc:
                        best_sc = new_sc
                    if time.time() > deadline_local:
                        break

            if best_sc < best_scope:
                best_scope = best_sc
                best_rows = best

    return best_rows
>>>>>>> REPLACE
