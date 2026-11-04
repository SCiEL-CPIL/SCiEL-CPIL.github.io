import random
import time
import sys

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    such that all positive differences within rows are distinct across all rows.
    Minimize the largest entry.
    """
    deadline = time.time() + 145.0  # safety margin before 150s kill
    
    # Each row has C(k+1, 2) differences
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # Phase 1: Greedy construction to get a feasible starting solution
    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = set()
                ok = True
                for a in row:
                    d = x - a
                    if d in used:
                        ok = False
                        break
                    new_diffs.add(d)
                if ok and not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows
    
    best_rows = greedy_construct()
    best_max = max(max(r) for r in best_rows)
    
    # Precompute difference sets for quick checking
    def get_all_diffs(rows):
        """Return list of sets, one per row, of all positive differences."""
        result = []
        for row in rows:
            s = set()
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    s.add(row[j] - row[i])
            result.append(s)
        return result
    
    def get_diffs_for_row(row):
        s = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                s.add(row[j] - row[i])
        return s
    
    def is_valid(rows):
        """Check if all differences across all rows are distinct."""
        all_diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    d = row[j] - row[i]
                    if d in all_diffs:
                        return False
                    all_diffs.add(d)
        return True
    
    def max_entry(rows):
        return max(max(r) for r in rows)
    
    # Phase 2: Local search / simulated annealing to reduce max entry
    # Strategy: try to decrease entries in the row with the largest entry
    # by randomly perturbing and checking validity
    
    random.seed(42)
    
    def try_local_search(rows, time_limit):
        """Try to improve the solution by reducing max entry."""
        rows = [list(r) for r in rows]
        best = max_entry(rows)
        best_solution = [list(r) for r in rows]
        
        # Precompute diff sets
        diff_sets = get_all_diffs(rows)
        
        def check_row_valid(row, other_diffs):
            """Check if row's differences don't collide with other_diffs or internally."""
            row_diffs = get_diffs_for_row(row)
            if len(row_diffs) != diffs_per_row:
                return False  # internal duplicates
            if row_diffs & other_diffs:
                return False
            return True
        
        # Simulated annealing style: try random modifications
        temperature = best * 0.5
        cooling = 0.999
        
        while time.time() < time_limit:
            # Pick a row to modify - prefer the one with largest entry
            row_idx = random.randint(0, n_rows - 1)
            # Bias toward rows with larger max entries
            if random.random() < 0.7:
                max_row_idx = max(range(n_rows), key=lambda i: max(rows[i]))
                row_idx = max_row_idx
            
            row = rows[row_idx]
            # Try to decrease a random entry (not index 0)
            pos = random.randint(1, len(row) - 1)
            
            # Try decreasing by a random small amount
            decrease = random.randint(1, min(5, row[pos] - (row[pos-1] + 1)))
            if decrease <= 0:
                continue
            
            old_val = row[pos]
            new_val = old_val - decrease
            if new_val <= row[pos-1]:
                continue
            
            row[pos] = new_val
            
            # Check validity: compute this row's diffs and check against others
            other_diffs = set()
            for i in range(n_rows):
                if i != row_idx:
                    other_diffs |= diff_sets[i]
            
            row_diffs = get_diffs_for_row(row)
            valid = (len(row_diffs) == diffs_per_row) and not (row_diffs & other_diffs)
            
            if valid:
                # Update diff_sets
                diff_sets[row_idx] = row_diffs
                new_max = max_entry(rows)
                if new_max < best:
                    best = new_max
                    best_solution = [list(r) for r in rows]
                    temperature = best * 0.3
                # Accept
            else:
                # Reject, revert
                row[pos] = old_val
                diff_sets[row_idx] = get_diffs_for_row(row)
            
            temperature *= cooling
            if temperature < 0.01:
                temperature = 0.01
        
        return best_solution, best
    
    # Run local search with time budget
    search_limit = deadline - 5.0
    best_rows, best_max = try_local_search(best_rows, search_limit)
    
    # Verify the best solution is valid
    if not is_valid(best_rows):
        # Fallback to greedy
        best_rows = greedy_construct()
    
    return best_rows
