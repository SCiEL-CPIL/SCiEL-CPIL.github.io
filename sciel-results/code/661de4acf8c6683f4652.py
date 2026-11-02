import time
import random
import copy

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140
    
    def get_row_diffs(row):
        diffs = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        return diffs
    
    def get_all_diffs(rows):
        all_diffs = set()
        for row in rows:
            all_diffs |= get_row_diffs(row)
        return all_diffs
    
    def is_valid_row(row):
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row) - 1):
            if row[i] >= row[i + 1]:
                return False
        diffs = get_row_diffs(row)
        return len(diffs) == k * (k + 1) // 2
    
    def get_scope(rows):
        return max(max(row) for row in rows)
    
    # Phase 1: Greedy initial solution
    def greedy():
        used, rows = set(), []
        for _ in range(n_rows):
            row, x = [0], 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows
    
    best_rows = greedy()
    best_scope = get_scope(best_rows)
    
    # Phase 2: Local search
    def try_improve_row(row, all_other_diffs):
        """Try to find a better version of row with smaller max."""
        # Try to reduce the last element
        for new_last in range(row[-1] - 1, 0, -1):
            if new_last <= row[-2]:
                break
            test_row = row[:-1] + [new_last]
            if not is_valid_row(test_row):
                continue
            test_diffs = get_row_diffs(test_row)
            if not (test_diffs & all_other_diffs):
                return test_row
        return None
    
    def local_search(rows):
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            for i in range(len(rows)):
                all_other_diffs = get_all_diffs(rows[:i] + rows[i+1:])
                new_row = try_improve_row(rows[i], all_other_diffs)
                if new_row is not None:
                    rows[i] = new_row
                    improved = True
        return rows
    
    best_rows = local_search(best_rows)
    best_scope = get_scope(best_rows)
    
    # Phase 3: Simulated annealing for further improvement
    def random_perturbation(rows):
        rows = copy.deepcopy(rows)
        row_idx = random.randint(0, len(rows) - 1)
        row = rows[row_idx]
        
        # Strategy 1: Try to reduce last element
        if random.random() < 0.4:
            for new_last in range(row[-1] - 1, 0, -1):
                if new_last <= row[-2]:
                    break
                test_row = row[:-1] + [new_last]
                if not is_valid_row(test_row):
                    continue
                all_other_diffs = get_all_diffs(rows[:row_idx] + rows[row_idx+1:])
                test_diffs = get_row_diffs(test_row)
                if not (test_diffs & all_other_diffs):
                    rows[row_idx] = test_row
                    return rows
            return rows
        
        # Strategy 2: Shift a random element
        elif len(row) > 2:
            pos = random.randint(1, len(row) - 2)
            delta = random.choice([-3, -2, -1, 1, 2, 3])
            new_val = row[pos] + delta
            if new_val > row[pos - 1] and new_val < row[pos + 1]:
                test_row = row[:pos] + [new_val] + row[pos + 1:]
                if is_valid_row(test_row):
                    all_other_diffs = get_all_diffs(rows[:row_idx] + rows[row_idx+1:])
                    test_diffs = get_row_diffs(test_row)
                    if not (test_diffs & all_other_diffs):
                        rows[row_idx] = test_row
                        return rows
            return rows
        
        # Strategy 3: Rebuild a row from scratch
        else:
            all_other_diffs = get_all_diffs(rows[:row_idx] + rows[row_idx+1:])
            # Try to build a new row with smaller max
            max_target = row[-1] - 1
            for target in range(max_target, 0, -1):
                # Try random combinations
                for _ in range(100):
                    combo = sorted(random.sample(range(1, target + 1), k))
                    test_row = [0] + combo
                    if is_valid_row(test_row):
                        test_diffs = get_row_diffs(test_row)
                        if not (test_diffs & all_other_diffs):
                            rows[row_idx] = test_row
                            return rows
            return rows
    
    current_rows = copy.deepcopy(best_rows)
    current_scope = best_scope
    temp = 5.0
    cooling = 0.9995
    
    for step in range(100000):
        if time.time() > deadline:
            break
        
        new_rows = random_perturbation(current_rows)
        new_scope = get_scope(new_rows)
        
        delta = new_scope - current_scope
        if delta <= 0 or random.random() < pow(2.718281828, -delta / max(temp, 0.1)):
            current_rows = new_rows
            current_scope = new_scope
            if current_scope < best_scope:
                best_rows = copy.deepcopy(current_rows)
                best_scope = current_scope
        
        temp *= cooling
    
    return best_rows
