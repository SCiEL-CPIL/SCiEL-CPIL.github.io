"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all positive differences within rows are distinct across all rows.
    Minimize the largest entry (scope).
    
    Uses a randomized local search (ILS) with difference-set tracking.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s kill
    
    # Total number of differences needed: n_rows * C(k+1, 2)
    num_diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * num_diffs_per_row
    
    # Lower bound on scope: need at least total_diffs distinct positive differences
    # The largest difference in a row with scope m is at most m.
    # We need total_diffs distinct values in [1, m], so m >= total_diffs.
    # But also each row has C(k+1,2) diffs, and the largest diff in row i is <= scope.
    
    # Simple counting bound: m >= total_diffs (since all diffs are in [1, m] and distinct)
    # Actually more precisely, the diffs are positive integers <= m, and there are total_diffs of them,
    # so we need m >= total_diffs.
    
    # Better bound: for a single row with scope m, the max number of distinct positive diffs
    # from a set of k+1 points in [0,m] is C(k+1,2), and the largest is <= m.
    # Across n_rows rows, all diffs must be distinct, so we need at least total_diffs distinct values in [1, m].
    # Thus m >= total_diffs.
    
    lb = total_diffs
    # A tighter bound: the sum of the largest diffs... but let's start with lb.
    
    best_rows = None
    best_scope = float('inf')
    
    def get_diffs(row):
        """Get all positive differences in a row."""
        return set(row[j] - row[i] for i in range(len(row)) for j in range(i+1, len(row)))
    
    def is_valid(all_rows):
        """Check if all differences across all rows are distinct."""
        seen = set()
        for row in all_rows:
            for d in get_diffs(row):
                if d in seen:
                    return False
                seen.add(d)
        return True
    
    def scope_of(rows):
        return max(max(row) for row in rows)
    
    def greedy_construct(n_rows, k, max_scope):
        """
        Greedy construction: for each row, pick the smallest next mark such that
        all new differences don't conflict with already-used differences.
        Returns None if we can't complete within max_scope.
        """
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                if x > max_scope:
                    return None
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows
    
    def random_row(scope, k):
        """Generate a random row of k+1 increasing integers starting at 0, ending <= scope."""
        # Choose k-1 intermediate points from [1, scope-1]
        if k == 1:
            return [0, random.randint(1, scope)]
        mids = sorted(random.sample(range(1, scope), k-1))
        return [0] + mids + [scope]
    
    def local_search(rows, n_rows, k, max_iterations, deadline):
        """
        Local search: try to reduce the scope by modifying rows.
        Returns improved rows or None.
        """
        current_rows = [list(r) for r in rows]
        current_scope = scope_of(current_rows)
        
        for iteration in range(max_iterations):
            if time.time() > deadline:
                break
            
            # Pick a random row to modify
            row_idx = random.randint(0, n_rows - 1)
            old_row = list(current_rows[row_idx])
            
            # Try to reduce the last element (scope of this row)
            # Or try a random perturbation
            if random.random() < 0.5:
                # Try to reduce the last mark
                new_row = list(old_row)
                if len(new_row) > 1:
                    new_last = random.randint(new_row[-2] + 1, max(1, new_row[-1] - 1))
                    if new_last > new_row[-2]:
                        new_row[-1] = new_last
                    else:
                        continue
                # Validate
                all_diffs = set()
                valid = True
                for i, r in enumerate(current_rows):
                    if i == row_idx:
                        r = new_row
                    for d in get_diffs(r):
                        if d in all_diffs:
                            valid = False
                            break
                        all_diffs.add(d)
                    if not valid:
                        break
                if valid:
                    current_rows[row_idx] = new_row
                    new_scope = scope_of(current_rows)
                    if new_scope < current_scope:
                        current_scope = new_scope
            else:
                # Random perturbation: change a random element
                new_row = list(old_row)
                pos = random.randint(1, len(new_row) - 1)
                low = new_row[pos-1] + 1
                high = new_row[pos+1] - 1 if pos < len(new_row) - 1 else current_scope
                if low <= high:
                    new_val = random.randint(low, high)
                    new_row[pos] = new_val
                    # Validate
                    all_diffs = set()
                    valid = True
                    for i, r in enumerate(current_rows):
                        if i == row_idx:
                            r = new_row
                        for d in get_diffs(r):
                            if d in all_diffs:
                                valid = False
                                break
                            all_diffs.add(d)
                        if not valid:
                            break
                    if valid:
                        current_rows[row_idx] = new_row
                        new_scope = scope_of(current_rows)
                        if new_scope < current_scope:
                            current_scope = new_scope
        
        return current_rows
    
    # Phase 1: Greedy construction
    # Start with a reasonable upper bound
    # The greedy will find some scope, let's just run it without a cap first
    greedy_result = greedy_construct(n_rows, k, 10000)
    if greedy_result is not None:
        best_rows = greedy_result
        best_scope = scope_of(best_rows)
    
    if best_rows is None:
        # Fallback: simple greedy without scope limit
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        best_rows = rows
        best_scope = scope_of(best_rows)
    
    # Phase 2: Iterated Local Search
    if time.time() < deadline:
        current_rows = [list(r) for r in best_rows]
        current_scope = best_scope
        
        max_ils_iterations = 10000
        for ils_iter in range(max_ils_iterations):
            if time.time() > deadline:
                break
            
            # Local search
            improved = local_search(current_rows, n_rows, k, 50, deadline)
            if improved is not None:
                improved_scope = scope_of(improved)
                if improved_scope < current_scope:
                    current_rows = improved
                    current_scope = improved_scope
                    if current_scope < best_scope:
                        best_rows = [list(r) for r in current_rows]
                        best_scope = current_scope
                        # If we hit the lower bound, we're done
                        if best_scope <= lb:
                            break
            else:
                current_rows = [list(r) for r in best_rows]
                current_scope = best_scope
            
            # Perturbation for ILS: random restart of one row
            if random.random() < 0.3:
                row_idx = random.randint(0, n_rows - 1)
                old_row = list(current_rows[row_idx])
                # Perturb: change 1-2 elements
                for _ in range(random.randint(1, 2)):
                    pos = random.randint(1, len(old_row) - 1)
                    low = old_row[pos-1] + 1
                    high = old_row[pos+1] - 1 if pos < len(old_row) - 1 else best_scope
                    if low <= high:
                        old_row[pos] = random.randint(low, high)
                current_rows[row_idx] = old_row
        
        if scope_of(current_rows) < best_scope:
            best_rows = current_rows
            best_scope = scope_of(best_rows)
    
    return best_rows
# EVOLVE-BLOCK-END
