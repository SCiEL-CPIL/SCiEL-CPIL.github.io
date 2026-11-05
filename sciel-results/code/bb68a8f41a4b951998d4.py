import random
import time

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 115.0  # Leave 5s margin for 120s limit

    best_rows = None
    best_score = float('inf')

    # Precompute expected number of differences per row
    expected_diffs = (k + 1) * k // 2

    def row_diffs(row):
        return {row[j] - row[i] for i in range(len(row)) for j in range(i + 1, len(row))}

    def is_valid_row(row):
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(1, len(row)):
            if row[i] <= row[i - 1]:
                return False
        diffs = row_diffs(row)
        return len(diffs) == expected_diffs

    def get_valid_next(row, used, cap):
        valids = []
        last = row[-1]
        for x in range(last + 1, cap + 1):
            new_diffs = {x - a for a in row}
            if not (new_diffs & used) and len(new_diffs) == len(row):
                valids.append(x)
        return valids

    def randomized_greedy(max_scope_hint=None):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                cap = max_scope_hint if max_scope_hint else 200
                if best_score < float('inf'):
                    cap = min(cap, best_score)

                valids = get_valid_next(row, used, cap)
                if not valids:
                    return None

                # Weighted random choice, favoring smaller x
                if len(valids) > 1:
                    consider = valids[:30]
                    weights = [1.0 / (v + 1) for v in consider]
                    total_w = sum(weights)
                    r = random.uniform(0, total_w)
                    acc = 0
                    chosen = consider[0]
                    for v, w in zip(consider, weights):
                        acc += w
                        if r <= acc:
                            chosen = v
                            break
                else:
                    chosen = valids[0]

                row.append(chosen)
                used.update({chosen - a for a in row[:-1]})
            rows.append(row)
        return rows

    def local_search(rows):
        current_max = max(max(r) for r in rows)
        
        # Precompute diff sets for all rows
        row_diff_sets = [row_diffs(r) for r in rows]
        
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            # Shuffle row order to avoid bias
            row_order = list(range(n_rows))
            random.shuffle(row_order)

            # Try to improve each row
            for i in row_order:
                if time.time() - start_time >= time_limit:
                    break
                row = rows[i]
                # Iterate from last element to first (excluding 0)
                for pos in range(len(row) - 1, 0, -1):
                    if time.time() - start_time >= time_limit:
                        break
                    current_val = row[pos]
                    # Try smaller values
                    min_val = row[pos - 1] + 1
                    max_val = current_val - 1  # We want to reduce

                    if max_val < min_val:
                        continue

                    # Compute other_diffs once per position (not per value)
                    other_diffs = set()
                    for r_idx in range(n_rows):
                        if r_idx != i:
                            other_diffs.update(row_diff_sets[r_idx])

                    # Try values from smallest to largest
                    found = False
                    for v in range(min_val, max_val + 1):
                        # Quick check: strictly increasing
                        if v <= row[pos - 1]:
                            continue
                        if pos < len(row) - 1 and v >= row[pos + 1]:
                            continue

                        # Create candidate row
                        candidate_row = list(row)
                        candidate_row[pos] = v

                        # Get diffs of candidate row
                        cand_diffs = row_diffs(candidate_row)

                        # Check for internal duplicates
                        if len(cand_diffs) != expected_diffs:
                            continue

                        # Check for global duplicates with other rows
                        if cand_diffs & other_diffs:
                            continue

                        # Accept the change
                        rows[i] = candidate_row
                        row_diff_sets[i] = cand_diffs
                        new_max = max(max(r) for r in rows)
                        if new_max < current_max:
                            current_max = new_max
                            improved = True
                        found = True
                        break  # Break inner for v loop
                    if found:
                        break  # Break inner for pos loop
                if improved:
                    break  # Break outer for i loop
        return rows, current_max

    def simulated_annealing(rows, max_iters=5000):
        """Try to reduce scope via simulated annealing moves."""
        current_rows = [list(r) for r in rows]
        current_max = max(max(r) for r in current_rows)
        row_diff_sets = [row_diffs(r) for r in current_rows]
        
        best_rows_sa = [list(r) for r in current_rows]
        best_max_sa = current_max
        
        T = 10.0  # Initial temperature
        cooling = 0.9995
        
        for it in range(max_iters):
            if time.time() - start_time >= time_limit - 2:
                break
            if T < 0.01:
                break
                
            # Randomly pick a row and a position
            i = random.randint(0, n_rows - 1)
            pos = random.randint(1, len(current_rows[i]) - 1)
            
            # Compute other diffs
            other_diffs = set()
            for r_idx in range(n_rows):
                if r_idx != i:
                    other_diffs.update(row_diff_sets[r_idx])
            
            # Try a random perturbation
            current_val = current_rows[i][pos]
            min_val = current_rows[i][pos - 1] + 1
            max_val = min(current_val + 5, current_max + 5)
            
            if max_val <= min_val:
                continue
                
            v = random.randint(min_val, max_val)
            if v == current_val:
                continue
                
            # Quick validity checks
            if v <= current_rows[i][pos - 1]:
                continue
            if pos < len(current_rows[i]) - 1 and v >= current_rows[i][pos + 1]:
                continue
            
            candidate_row = list(current_rows[i])
            candidate_row[pos] = v
            cand_diffs = row_diffs(candidate_row)
            
            if len(cand_diffs) != expected_diffs:
                continue
            if cand_diffs & other_diffs:
                continue
            
            # Valid move - compute new max
            new_max = max(max(r) for r in current_rows)
            # Replace row and compute actual new max
            old_row = current_rows[i]
            current_rows[i] = candidate_row
            row_diff_sets[i] = cand_diffs
            actual_new_max = max(max(r) for r in current_rows)
            
            delta = actual_new_max - current_max
            
            if delta <= 0 or random.random() < pow(2.718281828, -delta / T):
                current_max = actual_new_max
                if actual_new_max < best_max_sa:
                    best_max_sa = actual_new_max
                    best_rows_sa = [list(r) for r in current_rows]
            else:
                # Reject - revert
                current_rows[i] = old_row
                row_diff_sets[i] = row_diffs(old_row)
            
            T *= cooling
        
        return best_rows_sa, best_max_sa

    def validate_solution(rows):
        all_diffs = set()
        for r in rows:
            if not is_valid_row(r):
                return False
            d_set = row_diffs(r)
            if all_diffs & d_set:
                return False
            all_diffs.update(d_set)
        return True

    # Main loop
    attempts = 0
    while time.time() - start_time < time_limit:
        # Try randomized greedy
        rows = randomized_greedy()
        if rows is not None:
            score = max(max(r) for r in rows)
            if score < best_score:
                best_score = score
                best_rows = [list(r) for r in rows]
                # Try local search
                if time.time() - start_time < time_limit - 10:
                    improved_rows, improved_score = local_search(best_rows)
                    # Validate improved_rows before accepting
                    if validate_solution(improved_rows):
                        if improved_score < best_score:
                            best_score = improved_score
                            best_rows = improved_rows
                    # Try simulated annealing on the locally improved solution
                    if time.time() - start_time < time_limit - 15:
                        sa_rows, sa_score = simulated_annealing(best_rows)
                        if validate_solution(sa_rows):
                            if sa_score < best_score:
                                best_score = sa_score
                                best_rows = sa_rows
                if best_score <= 105:
                    break

        attempts += 1
        if attempts % 50 == 0:
            # Occasionally try a more aggressive local search by shuffling row order
            if best_rows is not None and time.time() - start_time < time_limit - 20:
                shuffled = list(best_rows)
                random.shuffle(shuffled)
                improved_rows, improved_score = local_search(shuffled)
                if validate_solution(improved_rows):
                    if improved_score < best_score:
                        best_score = improved_score
                        best_rows = improved_rows
                        # Follow up with SA
                        if time.time() - start_time < time_limit - 25:
                            sa_rows, sa_score = simulated_annealing(best_rows)
                            if validate_solution(sa_rows):
                                if sa_score < best_score:
                                    best_score = sa_score
                                    best_rows = sa_rows

    if best_rows is None:
        # Fallback to deterministic greedy
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used) and len(new_diffs) == len(row):
                        break
                    x += 1
                row.append(x)
                used.update({x - a for a in row[:-1]})
            rows.append(row)
        best_rows = rows

    return best_rows
