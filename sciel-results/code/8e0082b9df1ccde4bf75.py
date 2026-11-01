import sys
import random
import math

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0.
    All positive differences within rows must be distinct across all rows.
    Minimizes the largest entry (scope).
    
    Uses local search (simulated annealing) to minimize scope.
    """
    random.seed(42)  # For reproducibility
    
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    def get_diffs(row):
        """Get all positive differences in a row."""
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs
    
    def is_valid(rows):
        """Check if all differences are globally unique."""
        all_diffs = set()
        for row in rows:
            diffs = get_diffs(row)
            if all_diffs & diffs:
                return False
            all_diffs |= diffs
        return True
    
    def scope(rows):
        """Return the maximum entry across all rows."""
        return max(max(row) for row in rows)
    
    # Initial solution: greedy construction
    def greedy_initial():
        rows = [[0] for _ in range(n_rows)]
        used_diffs = set()
        
        for r in range(n_rows):
            for c in range(1, k + 1):
                # Find smallest x > rows[r][c-1] such that all diffs are new
                x = rows[r][c-1] + 1 if c > 1 else 1
                while True:
                    valid = True
                    new_diffs = []
                    for i in range(c):
                        d = x - rows[r][i]
                        if d in used_diffs:
                            valid = False
                            break
                        new_diffs.append(d)
                    if valid:
                        used_diffs.update(new_diffs)
                        rows[r].append(x)
                        break
                    x += 1
                    if x > 500:
                        return None
        return rows
    
    initial = greedy_initial()
    if initial is None:
        # Fallback to backtracking
        return construct_backtrack(n_rows, k)
    
    current = [row[:] for row in initial]
    current_scope = scope(current)
    best = [row[:] for row in initial]
    best_scope = current_scope
    
    # Simulated annealing
    T = 10.0
    T_min = 0.1
    cooling = 0.995
    max_iters = 50000
    
    def perturb(rows):
        """Make a random perturbation to a row."""
        rows = [row[:] for row in rows]
        r = random.randrange(n_rows)
        # Modify one element in row r
        c = random.randrange(1, k + 1)
        old_val = rows[r][c]
        # New value: random in valid range
        min_val = rows[r][c-1] + 1 if c > 1 else 1
        max_val = rows[r][c+1] - 1 if c < k else old_val + 10
        if max_val < min_val:
            max_val = min_val + 10
        new_val = random.randint(min_val, max_val)
        rows[r][c] = new_val
        return rows
    
    def is_row_valid(row, all_rows, except_r):
        """Check if row has strictly increasing values and its diffs don't conflict with others."""
        # Check increasing
        for i in range(len(row) - 1):
            if row[i] >= row[i+1]:
                return False
        # Check diffs don't conflict
        my_diffs = get_diffs(row)
        for i, other_row in enumerate(all_rows):
            if i == except_r:
                continue
            other_diffs = get_diffs(other_row)
            if my_diffs & other_diffs:
                return False
        return True
    
    iter_count = 0
    while T > T_min and iter_count < max_iters:
        iter_count += 1
        candidate = perturb(current)
        
        # Check if candidate is valid
        if not is_row_valid(candidate, candidate, -1):
            # Check all rows
            valid = True
            all_diffs = set()
            for i, row in enumerate(candidate):
                # Check increasing
                for j in range(len(row) - 1):
                    if row[j] >= row[j+1]:
                        valid = False
                        break
                if not valid:
                    break
                diffs = get_diffs(row)
                if all_diffs & diffs:
                    valid = False
                    break
                all_diffs |= diffs
            if not valid:
                T *= cooling
                continue
        
        cand_scope = scope(candidate)
        delta = cand_scope - current_scope
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            current = candidate
            current_scope = cand_scope
            if cand_scope < best_scope:
                best = [row[:] for row in candidate]
                best_scope = cand_scope
        
        T *= cooling
    
    return best

def construct_backtrack(n_rows, k):
    """Fallback: simple backtracking."""
    rows = [[0] for _ in range(n_rows)]
    used_diffs = set()
    sys.setrecursionlimit(10000)
    
    def dfs(r, c):
        if r == n_rows:
            return True
        if c == 1:
            start_x = 1 if r == 0 else rows[r-1][1]
        else:
            start_x = rows[r][c-1] + 1
        
        while start_x <= 500:
            valid = True
            new_diffs = []
            for i in range(c):
                d = start_x - rows[r][i]
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.append(d)
            if valid:
                used_diffs.update(new_diffs)
                rows[r].append(start_x)
                if c == k:
                    if dfs(r + 1, 1):
                        return True
                else:
                    if dfs(r, c + 1):
                        return True
                rows[r].pop()
                used_diffs.difference_update(new_diffs)
            start_x += 1
        return False
    
    dfs(0, 1)
    return rows
