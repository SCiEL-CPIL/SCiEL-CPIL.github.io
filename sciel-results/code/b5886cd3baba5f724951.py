import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0

    diffs_per_row = (k + 1) * k // 2
    total_diffs = n_rows * diffs_per_row

    def get_scope(rows):
        return max(max(r) for r in rows)

    def is_valid(rows):
        used = set()
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            for i in range(1, len(row)):
                if row[i] <= row[i - 1]:
                    return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    def greedy_construct(scope_limit=None):
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                found = False
                last_val = row[-1]
                max_val = scope_limit if scope_limit is not None else 10000
                for x in range(last_val + 1, max_val + 1):
                    valid = True
                    for y in row:
                        d = x - y
                        if d in used_diffs:
                            valid = False
                            break
                    if valid:
                        row.append(x)
                        for y in row[:-1]:
                            used_diffs.add(x - y)
                        found = True
                        break
                if not found:
                    return None
            rows.append(row)
        return rows

    def diff_coloring_construct():
        """
        Graph-coloring-inspired construction: assign difference values to slots
        using a greedy approach that prioritizes small difference values.
        Each row has k(k+1)/2 difference slots. We build rows by assigning
        differences in a way that minimizes the maximum entry.
        """
        used_diffs = set()
        rows = []
        
        for row_idx in range(n_rows):
            # For each row, we need to choose k+1 positions starting at 0
            # such that all pairwise differences are unique and not in used_diffs
            # Strategy: greedily assign the next position to minimize the final value
            
            row = [0]
            # We'll build the row by choosing each next element
            while len(row) < k + 1:
                last_val = row[-1]
                # Try candidate values starting from last_val + 1
                best_candidate = None
                best_score = float('inf')
                
                # Limit search to reasonable range
                max_search = 500
                for x in range(last_val + 1, last_val + max_search + 1):
                    # Check all new differences
                    new_diffs = []
                    valid = True
                    for y in row:
                        d = x - y
                        if d in used_diffs:
                            valid = False
                            break
                        new_diffs.append(d)
                    
                    if not valid:
                        continue
                    
                    # Score: prefer smaller x, but also consider future feasibility
                    # Use a heuristic: score = x + penalty for large gaps
                    score = x
                    # Add a small penalty if the gap is very large
                    if x - last_val > 20:
                        score += (x - last_val - 20) * 0.1
                    
                    if score < best_score:
                        best_score = score
                        best_candidate = x
                
                if best_candidate is None:
                    # Fallback: try a wider range
                    for x in range(last_val + 1, last_val + 2000 + 1):
                        new_diffs = []
                        valid = True
                        for y in row:
                            d = x - y
                            if d in used_diffs:
                                valid = False
                                break
                        if valid:
                            best_candidate = x
                            break
                
                if best_candidate is None:
                    return None
                
                row.append(best_candidate)
                for y in row[:-1]:
                    used_diffs.add(best_candidate - y)
            
            rows.append(row)
        
        return rows

    def simulated_annealing_bitmask(initial_rows, time_limit):
        """
        Simulated annealing using a boolean array for fast difference collision checks.
        Improved with shrink bias: prefer smaller values to reduce scope.
        """
        rows = [r[:] for r in initial_rows]
        current_scope = get_scope(rows)
        best_rows = [r[:] for r in rows]
        best_scope = current_scope

        # Determine max possible difference value for array sizing
        max_diff = best_scope + 100
        # Use a boolean list as a bitset for fast membership tests
        used = [False] * (max_diff + 1)

        # Fill used array
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d <= max_diff:
                        used[d] = True
                    else:
                        max_diff = d
                        used = [False] * (max_diff + 1)
                        for r2 in rows:
                            for i2 in range(len(r2)):
                                for j2 in range(i2 + 1, len(r2)):
                                    d2 = r2[j2] - r2[i2]
                                    used[d2] = True
                        break

        t_end = time.time() + time_limit
        initial_temp = max(1.0, current_scope / 4.0)
        cooling_rate = 0.9998
        steps = 0

        while time.time() < t_end:
            row_idx = random.randint(0, n_rows - 1)
            # Bias toward positions in the row that has the max value
            if rows[row_idx][-1] == max(max(r) for r in rows):
                pos = random.randint(1, k)
            else:
                pos = random.randint(1, k)

            old_val = rows[row_idx][pos]
            min_val = rows[row_idx][pos - 1] + 1
            max_val = rows[row_idx][pos + 1] - 1 if pos < k else best_scope + 20

            if min_val > max_val:
                initial_temp *= cooling_rate
                continue

            attempts = random.randint(1, 5)
            accepted = False
            for _ in range(attempts):
                # Shrink bias: 60% chance to try a smaller value
                if random.random() < 0.6:
                    # Try shrinking: pick from [min_val, old_val - 1] if possible
                    if old_val - 1 >= min_val:
                        new_val = random.randint(min_val, old_val - 1)
                    else:
                        new_val = random.randint(min_val, max_val)
                else:
                    new_val = random.randint(min_val, max_val)
                
                if new_val == old_val:
                    continue

                # Collect old diffs involving position pos
                old_diffs = []
                for i in range(k + 1):
                    if i != pos:
                        if i < pos:
                            d = old_val - rows[row_idx][i]
                        else:
                            d = rows[row_idx][i] - old_val
                        old_diffs.append(d)

                # Remove old diffs from used
                for d in old_diffs:
                    if d <= max_diff:
                        used[d] = False

                # Try adding new diffs
                new_diffs = []
                valid = True
                for i in range(k + 1):
                    if i != pos:
                        if i < pos:
                            d = new_val - rows[row_idx][i]
                        else:
                            d = rows[row_idx][i] - new_val
                        if d < 1 or d > max_diff:
                            valid = False
                            break
                        if used[d]:
                            valid = False
                            break
                        new_diffs.append(d)

                if not valid:
                    # Restore old diffs
                    for d in old_diffs:
                        if d <= max_diff:
                            used[d] = True
                    continue

                # Compute new scope
                new_row_max = max(rows[row_idx][:pos] + [new_val] + rows[row_idx][pos+1:])
                other_max = 0
                for i in range(n_rows):
                    if i != row_idx:
                        if rows[i][-1] > other_max:
                            other_max = rows[i][-1]
                new_scope = max(other_max, new_row_max)

                # Accept or reject with improved acceptance criterion
                if new_scope < current_scope:
                    accept = True
                elif new_scope == current_scope:
                    # Accept sideways moves with some probability to explore
                    accept = random.random() < 0.3
                else:
                    # Worse scope: accept with SA probability
                    delta = new_scope - current_scope
                    accept = random.random() < pow(2.718281828, -delta / initial_temp)
                
                if accept:
                    rows[row_idx][pos] = new_val
                    for d in new_diffs:
                        used[d] = True
                    current_scope = new_scope

                    if current_scope < best_scope:
                        best_scope = current_scope
                        best_rows = [r[:] for r in rows]
                    accepted = True
                    break
                else:
                    # Restore old diffs
                    for d in old_diffs:
                        if d <= max_diff:
                            used[d] = True

            initial_temp *= cooling_rate
            steps += 1

        return best_rows, best_scope

    def local_search_polish(rows, time_limit):
        rows = [r[:] for r in rows]
        current_scope = get_scope(rows)
        best_rows = [r[:] for r in rows]
        best_scope = current_scope
        t_end = time.time() + time_limit

        max_diff = best_scope + 100
        used = [False] * (max_diff + 1)
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d <= max_diff:
                        used[d] = True

        def try_shrink_pos(row_idx, pos):
            """Try to shrink element at position pos in row row_idx. Return True if improved."""
            nonlocal current_scope, best_scope, best_rows, used, max_diff
            row = rows[row_idx]
            old_val = row[pos]
            if pos == 0:
                return False
            min_allowed = row[pos - 1] + 1
            max_allowed = row[pos + 1] - 1 if pos < k else max_diff
            
            # Try shrinking by 1 first
            new_val = old_val - 1
            if new_val < min_allowed:
                return False
            
            # Collect old diffs involving position pos
            old_diffs = []
            for i in range(k + 1):
                if i != pos:
                    if i < pos:
                        d = old_val - row[i]
                    else:
                        d = row[i] - old_val
                    old_diffs.append(d)
            
            # Remove old diffs
            for d in old_diffs:
                if d <= max_diff:
                    used[d] = False
            
            # Check new diffs
            new_diffs = []
            valid = True
            for i in range(k + 1):
                if i != pos:
                    if i < pos:
                        d = new_val - row[i]
                    else:
                        d = row[i] - new_val
                    if d < 1 or d > max_diff or used[d]:
                        valid = False
                        break
                    new_diffs.append(d)
            
            if not valid:
                # Restore
                for d in old_diffs:
                    if d <= max_diff:
                        used[d] = True
                return False
            
            # Accept
            row[pos] = new_val
            for d in new_diffs:
                used[d] = True
            
            # Update scope
            new_scope = max(max(r) for r in rows)
            if new_scope < current_scope:
                current_scope = new_scope
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [r[:] for r in rows]
            return True
        
        # Iteratively try to shrink any position in any row
        improved = True
        while improved and time.time() < t_end:
            improved = False
            # Prioritize positions in rows that contribute to the max scope
            max_val = max(max(r) for r in rows)
            # Try all rows, positions from last to first (more likely to shrink scope)
            for row_idx in range(n_rows):
                if time.time() >= t_end:
                    break
                row = rows[row_idx]
                if row[-1] < max_val and row[-1] < max_val - 5:
                    continue  # Skip rows far from max
                for pos in range(k, 0, -1):
                    if time.time() >= t_end:
                        break
                    if try_shrink_pos(row_idx, pos):
                        improved = True
                        # Re-check max
                        max_val = max(max(r) for r in rows)
        
        return best_rows, best_scope

    # Step 1: Try multiple construction strategies
    rows = None
    best_initial_scope = float('inf')
    
    # Strategy 1: Standard greedy with tight scope limits first
    for scope_limit in [112, 115, 120, 125, 130, 134, 140, 150, 160, 200, 300, 500, 1000]:
        rows_candidate = greedy_construct(scope_limit)
        if rows_candidate is not None:
            scope = get_scope(rows_candidate)
            if scope < best_initial_scope:
                best_initial_scope = scope
                rows = rows_candidate
            if scope <= 134:
                break
    
    # Strategy 2: Diff-coloring-inspired construction
    rows_coloring = diff_coloring_construct()
    if rows_coloring is not None:
        scope_coloring = get_scope(rows_coloring)
        if scope_coloring < best_initial_scope:
            best_initial_scope = scope_coloring
            rows = rows_coloring
    
    if rows is None:
        rows = greedy_construct()
        if rows is None:
            used_diffs = set()
            rows = []
            for _ in range(n_rows):
                row = [0]
                x = 0
                while len(row) < k + 1:
                    x += 1
                    valid = True
                    for y in row:
                        if x - y in used_diffs:
                            valid = False
                            break
                    if valid:
                        row.append(x)
                        for y in row[:-1]:
                            used_diffs.add(x - y)
                rows.append(row)
            return rows

    current_scope = get_scope(rows)

    # Step 2: Simulated annealing with bitmask optimization
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        sa_time = remaining_time - 20  # Leave 20s for polish
        best_rows, best_scope = simulated_annealing_bitmask(rows, sa_time)
        if is_valid(best_rows):
            rows = best_rows
            current_scope = best_scope

    # Step 3: Final high-precision polish
    remaining_time = deadline - time.time()
    if remaining_time > 3:
        polish_time = remaining_time - 3
        polished_rows, polished_scope = local_search_polish(rows, polish_time)
        if is_valid(polished_rows):
            rows = polished_rows
            current_scope = polished_scope

    return rows
