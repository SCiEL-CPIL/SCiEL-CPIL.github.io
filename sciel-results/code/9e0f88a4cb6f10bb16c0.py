<<<<<<< SEARCH
    def simulate_annealing(init_rows, time_limit):
        nonlocal best_rows, best_max, best_cost

        rows = [r[:] for r in init_rows]
        current_max = scope_of(rows)
        counts = build_diff_counts(rows)
        current_collisions = get_collisions(counts)
        current_cost = current_collisions * 100000 + current_max

        best_local_cost = current_cost
        best_local_rows = [r[:] for r in rows]
        best_local_max = current_max

        T = 100.0
        cooling = 0.99995
        start_time = time.time()
        steps = 0
        rng = random.Random(int(start_time * 1000))

        while time.time() - start_time < time_limit:
            r_idx = rng.randint(0, n_rows - 1)
            i_idx = rng.randint(1, row_len - 1)

            lower = rows[r_idx][i_idx - 1] + 1
            upper_limit = best_max + 20
            if i_idx < row_len - 1:
                upper = rows[r_idx][i_idx + 1] - 1
                upper = min(upper, upper_limit)
            else:
                upper = upper_limit

            if lower > upper:
                continue

            new_val = rng.randint(lower, upper)
            if new_val == rows[r_idx][i_idx]:
                continue

            old_val = rows[r_idx][i_idx]

            if new_val <= rows[r_idx][i_idx - 1]:
                continue
            if i_idx < row_len - 1 and new_val >= rows[r_idx][i_idx + 1]:
                continue

            delta_coll = 0
            for j in range(row_len):
                if j == i_idx:
                    continue
                old_d = abs(old_val - rows[r_idx][j])
                new_d = abs(new_val - rows[r_idx][j])
                if old_d != new_d:
                    oc = counts.get(old_d, 0)
                    if oc > 0:
                        if oc == 1:
                            del counts[old_d]
                        else:
                            counts[old_d] = oc - 1
                            delta_coll -= 1
                    nc = counts.get(new_d, 0)
                    if nc == 0:
                        counts[new_d] = 1
                    else:
                        counts[new_d] = nc + 1
                        delta_coll += 1

            rows[r_idx][i_idx] = new_val
            new_max = scope_of(rows)
            new_collisions = current_collisions + delta_coll
            new_cost = new_collisions * 100000 + new_max

            if new_cost < current_cost:
                current_cost = new_cost
                current_collisions = new_collisions
                current_max = new_max
                if new_cost < best_local_cost:
                    best_local_cost = new_cost
                    best_local_max = new_max
                    best_local_rows = [r[:] for r in rows]
            else:
                delta = new_cost - current_cost
                if delta == 0:
                    if rng.random() < 0.3:
                        current_cost = new_cost
                        current_collisions = new_collisions
                        current_max = new_max
                    else:
                        rows[r_idx][i_idx] = old_val
                        for j in range(row_len):
                            if j == i_idx:
                                continue
                            old_d = abs(old_val - rows[r_idx][j])
                            new_d = abs(new_val - rows[r_idx][j])
                            if old_d != new_d:
                                nc = counts.get(new_d, 0)
                                if nc == 1:
                                    del counts[new_d]
                                else:
                                    counts[new_d] = nc - 1
                                oc = counts.get(old_d, 0)
                                if oc == 0:
                                    counts[old_d] = 1
                                else:
                                    counts[old_d] = oc + 1
                else:
                    if delta < 10000:
                        p = math.exp(-delta / T)
                    else:
                        p = 0.0

                    if rng.random() < p:
                        current_cost = new_cost
                        current_collisions = new_collisions
                        current_max = new_max
                    else:
                        rows[r_idx][i_idx] = old_val
                        for j in range(row_len):
                            if j == i_idx:
                                continue
                            old_d = abs(old_val - rows[r_idx][j])
                            new_d = abs(new_val - rows[r_idx][j])
                            if old_d != new_d:
                                nc = counts.get(new_d, 0)
                                if nc == 1:
                                    del counts[new_d]
                                else:
                                    counts[new_d] = nc - 1
                                oc = counts.get(old_d, 0)
                                if oc == 0:
                                    counts[old_d] = 1
                                else:
                                    counts[old_d] = oc + 1

            T *= cooling
            if T < 0.1:
                T = 0.1
            steps += 1

            if steps % 2000 == 0:
                actual_coll = get_collisions(counts)
                if actual_coll != current_collisions:
                    counts = build_diff_counts(rows)
                    current_collisions = get_collisions(counts)
                    current_max = scope_of(rows)
                    current_cost = current_collisions * 100000 + current_max

        if best_local_cost < best_cost:
            best_cost = best_local_cost
            best_max = best_local_max
            best_rows = best_local_rows

        return best_local_rows, best_local_max, best_local_cost
=======
    def simulate_annealing(init_rows, time_limit):
        nonlocal best_rows, best_max, best_cost

        rows = [r[:] for r in init_rows]
        current_max = scope_of(rows)
        counts = build_diff_counts(rows)
        current_collisions = get_collisions(counts)
        current_cost = current_collisions * 100000 + current_max

        best_local_cost = current_cost
        best_local_rows = [r[:] for r in rows]
        best_local_max = current_max

        T = 100.0
        cooling = 0.99995
        start_time = time.time()
        steps = 0
        rng = random.Random(int(start_time * 1000))

        # Track max value per row for quick scope updates
        row_maxes = [r[-1] for r in rows]

        while time.time() - start_time < time_limit:
            r_idx = rng.randint(0, n_rows - 1)
            i_idx = rng.randint(1, row_len - 1)

            lower = rows[r_idx][i_idx - 1] + 1
            upper_limit = best_max + 30
            if i_idx < row_len - 1:
                upper = rows[r_idx][i_idx + 1] - 1
                upper = min(upper, upper_limit)
            else:
                upper = upper_limit

            if lower > upper:
                continue

            new_val = rng.randint(lower, upper)
            if new_val == rows[r_idx][i_idx]:
                continue

            old_val = rows[r_idx][i_idx]

            if new_val <= rows[r_idx][i_idx - 1]:
                continue
            if i_idx < row_len - 1 and new_val >= rows[r_idx][i_idx + 1]:
                continue

            # Compute delta in collisions using set-based approach
            old_diffs_set = set()
            new_diffs_set = set()
            for j in range(row_len):
                if j == i_idx:
                    continue
                old_d = abs(old_val - rows[r_idx][j])
                new_d = abs(new_val - rows[r_idx][j])
                old_diffs_set.add(old_d)
                new_diffs_set.add(new_d)

            removed_diffs = old_diffs_set - new_diffs_set
            added_diffs = new_diffs_set - old_diffs_set

            delta_coll = 0
            # Process removals first
            for d in removed_diffs:
                oc = counts.get(d, 0)
                if oc == 1:
                    del counts[d]
                elif oc > 1:
                    counts[d] = oc - 1
                    delta_coll -= 1
            # Process additions
            for d in added_diffs:
                nc = counts.get(d, 0)
                if nc == 0:
                    counts[d] = 1
                else:
                    counts[d] = nc + 1
                    delta_coll += 1

            rows[r_idx][i_idx] = new_val
            if i_idx == row_len - 1:
                row_maxes[r_idx] = new_val
            new_max = max(row_maxes)
            new_collisions = current_collisions + delta_coll
            new_cost = new_collisions * 100000 + new_max

            if new_cost < current_cost:
                current_cost = new_cost
                current_collisions = new_collisions
                current_max = new_max
                if new_cost < best_local_cost:
                    best_local_cost = new_cost
                    best_local_max = new_max
                    best_local_rows = [r[:] for r in rows]
            else:
                delta = new_cost - current_cost
                if delta == 0:
                    if rng.random() < 0.3:
                        current_cost = new_cost
                        current_collisions = new_collisions
                        current_max = new_max
                    else:
                        # Revert
                        rows[r_idx][i_idx] = old_val
                        if i_idx == row_len - 1:
                            row_maxes[r_idx] = old_val
                        for d in added_diffs:
                            nc = counts.get(d, 0)
                            if nc == 1:
                                del counts[d]
                            else:
                                counts[d] = nc - 1
                        for d in removed_diffs:
                            oc = counts.get(d, 0)
                            if oc == 0:
                                counts[d] = 1
                            else:
                                counts[d] = oc + 1
                else:
                    if delta < 10000:
                        p = math.exp(-delta / T)
                    else:
                        p = 0.0

                    if rng.random() < p:
                        current_cost = new_cost
                        current_collisions = new_collisions
                        current_max = new_max
                    else:
                        # Revert
                        rows[r_idx][i_idx] = old_val
                        if i_idx == row_len - 1:
                            row_maxes[r_idx] = old_val
                        for d in added_diffs:
                            nc = counts.get(d, 0)
                            if nc == 1:
                                del counts[d]
                            else:
                                counts[d] = nc - 1
                        for d in removed_diffs:
                            oc = counts.get(d, 0)
                            if oc == 0:
                                counts[d] = 1
                            else:
                                counts[d] = oc + 1

            T *= cooling
            if T < 0.1:
                T = 0.1
            steps += 1

            if steps % 2000 == 0:
                actual_coll = get_collisions(counts)
                if actual_coll != current_collisions:
                    counts = build_diff_counts(rows)
                    current_collisions = get_collisions(counts)
                    current_max = scope_of(rows)
                    current_cost = current_collisions * 100000 + current_max

        if best_local_cost < best_cost:
            best_cost = best_local_cost
            best_max = best_local_max
            best_rows = best_local_rows

        return best_local_rows, best_local_max, best_local_cost
>>>>>>> REPLACE

