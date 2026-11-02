import time
import random
from itertools import combinations

def construct(n_rows, k):
    """Construct n_rows rows of k+1 increasing integers starting at 0,
    minimizing the largest entry, such that all within-row differences
    are distinct across all rows."""
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # Total number of differences needed
    diffs_per_row = k * (k + 1) // 2  # C(k+1, 2)
    total_diffs_needed = n_rows * diffs_per_row
    
    def get_row_diffs(row):
        """Get all positive differences within a row."""
        diffs = []
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.append(row[j] - row[i])
        return diffs
    
    def is_valid_row(row, used_diffs):
        """Check if a row's differences don't conflict with used_diffs."""
        rd = get_row_diffs(row)
        if len(rd) != len(set(rd)):
            return False  # internal duplicates
        return not any(d in used_diffs for d in rd)
    
    def greedy_row(used_diffs, max_val):
        """Greedily construct a row given used differences."""
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            if x > max_val:
                return None
            new_diffs = {x - a for a in row}
            if not new_diffs & used_diffs and len(new_diffs) == len(row):
                row.append(x)
        return row
    
    def backtrack_row(used_diffs, max_val, row=None):
        """Backtracking to find a valid row within max_val."""
        if row is None:
            row = [0]
        if len(row) == k + 1:
            return row
        last = row[-1]
        for x in range(last + 1, max_val + 1):
            new_diffs = {x - a for a in row}
            if new_diffs & used_diffs:
                continue
            if len(new_diffs) != len(row):
                continue
            result = backtrack_row(used_diffs, max_val, row + [x])
            if result is not None:
                return result
        return None
    
    def build_greedy():
        """Build all rows greedily."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row = None
            # Try increasing max values
            for max_val in range(total_diffs_needed, total_diffs_needed + 50):
                row = backtrack_row(used, max_val)
                if row is not None:
                    break
            if row is None:
                return None
            rows.append(row)
            used |= set(get_row_diffs(row))
        return rows
    
    def get_scope(rows):
        """Get the maximum entry across all rows."""
        return max(max(r) for r in rows)
    
    # Initial greedy construction
    best_rows = build_greedy()
    if best_rows is None:
        # Fallback: pure greedy without bound
        used = set()
        best_rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            best_rows.append(row)
    
    best_scope = get_scope(best_rows)
    
    # Local search: try to improve by replacing rows
    random.seed(42)
    
    def try_replace_row(rows, row_idx, max_val, used_without_row):
        """Try to find a better row for position row_idx."""
        # Backtrack with the constraint that we can't use diffs from other rows
        # and the new row must fit within max_val
        row = backtrack_row(used_without_row, max_val)
        if row is None:
            return None
        return row
    
    def remove_row_diffs(rows, row_idx):
        """Get used diffs excluding row_idx."""
        used = set()
        for i, r in enumerate(rows):
            if i == row_idx:
                continue
            used |= set(get_row_diffs(r))
        return used
    
    # Simulated annealing / randomized local search
    temp = 1.0
    cooling = 0.95
    
    while time.time() < deadline - 10.0:
        # Pick a random row to replace
        row_idx = random.randint(0, n_rows - 1)
        current_row = best_rows[row_idx]
        current_max = max(current_row)
        
        used_wo = remove_row_diffs(best_rows, row_idx)
        
        # Try to find a row with smaller max
        target_max = current_max - 1
        if target_max < diffs_per_row:
            target_max = diffs_per_row
        
        new_row = backtrack_row(used_wo, target_max)
        
        if new_row is not None:
            new_max = max(new_row)
            new_scope = max(get_scope(best_rows[:row_idx] + [new_row] + best_rows[row_idx+1:]))
            
            # Accept if better or with some probability (annealing)
            if new_scope < best_scope:
                best_rows[row_idx] = new_row
                best_scope = new_scope
                temp = max(0.1, temp * cooling)
            else:
                # Try with slightly larger max for diversity
                pass
        
        # Also try: rebuild a random row with randomized greedy
        if time.time() < deadline - 10.0:
            used_wo2 = remove_row_diffs(best_rows, row_idx)
            # Randomized greedy: shuffle candidate order
            row = [0]
            x = 0
            max_attempts = 0
            while len(row) < k + 1 and max_attempts < 5000:
                x += 1
                max_attempts += 1
                if x > best_scope:
                    break
                new_diffs = {x - a for a in row}
                if not new_diffs & used_wo2 and len(new_diffs) == len(row):
                    # Randomly accept with some probability to add diversity
                    if random.random() < 0.7:
                        row.append(x)
            if len(row) == k + 1:
                candidate_rows = best_rows[:row_idx] + [row] + best_rows[row_idx+1:]
                cand_scope = get_scope(candidate_rows)
                if cand_scope < best_scope:
                    best_rows = candidate_rows
                    best_scope = cand_scope
                    temp = max(0.1, temp * cooling)
        
        temp = max(0.1, temp * cooling)
    
    return best_rows
