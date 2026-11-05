import time
import random
import math

def construct(n_rows, k):
    """
    Constructs a set of n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct across all rows.
    Uses difference-based CSP with column-wise scheduling and SA refinement.
    """
    start_time = time.time()
    time_limit = 140.0  # Safety margin before 150s kill
    
    def elapsed():
        return time.time() - start_time
    
    def time_ok():
        return elapsed() < time_limit
    
    # Each row has k+1 elements, so C(k+1,2) = k*(k+1)/2 differences per row
    diffs_per_row = k * (k + 1) // 2
    total_diffs_needed = n_rows * diffs_per_row
    
    # Counting bound: scope must be at least such that C(scope+1,2) >= total_diffs_needed
    # C(s+1,2) = (s+1)*s/2 >= total_diffs_needed
    # s^2 + s - 2*total_diffs_needed >= 0
    # s >= (-1 + sqrt(1 + 8*total_diffs_needed)) / 2
    min_scope_by_count = math.ceil((-1 + math.sqrt(1 + 8 * total_diffs_needed)) / 2)
    
    def generate_row_from_diffs(diff_set, scope):
        """
        Given a set of diffs that should appear in one row, try to reconstruct
        the row (k+1 elements starting at 0) that produces exactly these diffs.
        Returns the row if found, else None.
        """
        if len(diff_set) != diffs_per_row:
            return None
        
        # The row must have elements r_0=0 < r_1 < ... < r_k
        # The diffs are {r_j - r_i : i < j}
        # r_k is the max element = scope candidate
        # The largest diff in the set must be r_k - r_0 = r_k
        max_diff = max(diff_set)
        
        # Try to reconstruct: r_0 = 0, and we need to find r_1, ..., r_k
        # such that all pairwise differences match diff_set exactly
        
        # Use backtracking on the elements
        row = [0]
        used_diffs = set()
        
        def bt(idx):
            if idx == k + 1:
                return True
            last = row[-1]
            # Try values from last+1 to max_diff
            # Must ensure new diffs don't conflict and eventually all diffs are used
            for x in range(last + 1, max_diff + 1):
                new_diffs = set()
                conflict = False
                for a in row:
                    d = x - a
                    if d in used_diffs:
                        conflict = True
                        break
                    new_diffs.add(d)
                
                if conflict:
                    continue
                
                # Check that all new_diffs are in diff_set
                if not new_diffs.issubset(diff_set):
                    continue
                
                row.append(x)
                used_diffs.update(new_diffs)
                
                if bt(idx + 1):
                    return True
                
                row.pop()
                for d in new_diffs:
                    used_diffs.discard(d)
            
            return False
        
        if bt(0):
            return row
        return None
    
    def greedy_initial(scope, seed=0):
        """
        Generate an initial solution using greedy with column-wise awareness.
        Returns list of rows or None.
        """
        rng = random.Random(seed)
        used = set()
        rows = []
        
        for r in range(n_rows):
            row = [0]
            # Try to build row greedily, preferring smaller values but with some randomization
            x = 1
            attempts = 0
            while len(row) < k + 1 and x <= scope:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
                    x = 1  # Reset to try small values again
                    attempts = 0
                else:
                    x += 1
                    attempts += 1
                    if attempts > scope * 2:
                        break
            
            if len(row) < k + 1:
                return None
            rows.append(row[:])
        
        return rows
    
    def compute_scope(rows):
        """Compute the largest entry across all rows."""
        return max(max(row) for row in rows)
    
    def is_valid(rows):
        """Verify all positive differences are distinct across all rows."""
        seen = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in seen:
                        return False
                    seen.add(d)
        return True
    
    def sa_refine(rows, scope_limit, seed=0):
        """
        Simulated annealing to reduce scope.
        Tries to swap elements or rows to reduce the maximum entry.
        Returns refined rows.
        """
        rng = random.Random(seed)
        current = [row[:] for row in rows]
        current_scope = compute_scope(current)
        best = [row[:] for row in current]
        best_scope = current_scope
        
        if not is_valid(current):
            return best
        
        # SA parameters
        t_start = 50.0
        t_end = 0.1
        iterations = 50000
        cooling = (t_end / t_start) ** (1.0 / iterations)
        
        t = t_start
        for it in range(iterations):
            if not time_ok():
                break
            if t < t_end:
                break
            
            # Pick a random row and try to reduce one of its elements
            r = rng.randrange(n_rows)
            row = current[r]
            
            # Try to move a random element (not the first, not the last necessarily) to a smaller value
            if len(row) <= 1:
                continue
            pos = rng.randrange(1, len(row))
            old_val = row[pos]
            
            # Try smaller values
            new_val = rng.randrange(1, old_val) if old_val > 1 else None
            if new_val is None:
                continue
            
            # Check if new_val maintains strict increase
            if pos > 0 and new_val <= row[pos - 1]:
                continue
            if pos < len(row) - 1 and new_val >= row[pos + 1]:
                continue
            
            # Check difference conflicts
            new_diffs = set()
            conflict = False
            for i, a in enumerate(row):
                if i == pos:
                    continue
                d = abs(new_val - a)
                if d == 0:
                    conflict = True
                    break
                new_diffs.add(d)
            
            if conflict:
                continue
            
            # Check against all other differences in all rows
            # Remove old diffs for this row, check new ones
            old_diffs = set()
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    old_diffs.add(row[j] - row[i])
            
            # All new_diffs must not be in (all_diffs - old_diffs)
            all_diffs = set()
            for row2 in current:
                for i in range(len(row2)):
                    for j in range(i + 1, len(row2)):
                        all_diffs.add(row2[j] - row2[i])
            
            available = all_diffs - old_diffs
            if new_diffs & available:
                continue
            
            # Apply move
            row[pos] = new_val
            new_scope = compute_scope(current)
            
            if new_scope < best_scope:
                best = [row2[:] for row2 in current]
                best_scope = new_scope
            
            # Accept/reject
            delta = new_scope - current_scope
            if delta <= 0 or rng.random() < math.exp(-delta / t):
                current_scope = new_scope
            else:
                row[pos] = old_val
            
            t *= cooling
        
        return best
    
    def column_wise_construction(scope, seed=0):
        """
        Column-wise list scheduling: build rows by assigning differences to columns.
        This respects global difference constraints earlier.
        Returns list of rows or None.
        """
        rng = random.Random(seed)
        
        # We'll build each row by choosing elements one at a time
        # but track which differences are used globally
        used_diffs = set()
        rows = []
        
        for r in range(n_rows):
            row = [0]
            # Build the row element by element
            for col in range(1, k + 1):
                # Find a valid value for this column
                # Must be > row[-1] and <= scope
                # And all new differences must not be in used_diffs
                found = False
                # Try values in a somewhat random order, biased toward smaller
                candidates = list(range(row[-1] + 1, scope + 1))
                # Sort by a heuristic: prefer values that create "rare" differences
                # and keep the row compact
                def score_candidate(x):
                    new_diffs = {x - a for a in row}
                    # Count how many of these diffs are already "crowded" (small)
                    small_count = sum(1 for d in new_diffs if d <= 10)
                    return (small_count, x)
                
                candidates.sort(key=score_candidate)
                # Add some randomization
                if rng.random() < 0.3 and len(candidates) > 5:
                    # Pick a random candidate from the top half
                    idx = rng.randrange(len(candidates) // 2)
                    candidates.insert(0, candidates.pop(idx))
                
                for x in candidates:
                    new_diffs = {x - a for a in row}
                    if new_diffs & used_diffs:
                        continue
                    # Check that x - a for all a in row are all in [1, scope]
                    # (they are, since x > a and x <= scope)
                    row.append(x)
                    used_diffs |= new_diffs
                    found = True
                    break
                
                if not found:
                    return None
            
            rows.append(row[:])
        
        return rows
    
    # Main algorithm:
    # 1. Try to find the minimum scope using binary search with column-wise construction
    # 2. Refine with SA to potentially reduce scope further
    
    best_result = None
    best_scope = float('inf')
    
    # First, try greedy to get an initial solution
    initial_scope = min_scope_by_count + 20  # Start a bit above counting bound
    for seed in range(10):
        if not time_ok():
            break
        result = greedy_initial(initial_scope, seed)
        if result is not None and is_valid(result):
            s = compute_scope(result)
            if s < best_scope:
                best_scope = s
                best_result = result
            break
    
    # Try column-wise construction at various scopes
    for scope in range(min_scope_by_count, min_scope_by_count + 30):
        if not time_ok():
            break
        for seed in range(5):
            if not time_ok():
                break
            result = column_wise_construction(scope, seed)
            if result is not None and is_valid(result):
                s = compute_scope(result)
                if s < best_scope:
                    best_scope = s
                    best_result = result
                break
    
    # If we have a result, try to refine with SA
    if best_result is not None and time_ok():
        for seed in range(3):
            if not time_ok():
                break
            refined = sa_refine(best_result, best_scope, seed)
            if refined is not None and is_valid(refined):
                s = compute_scope(refined)
                if s < best_scope:
                    best_scope = s
                    best_result = refined
    
    # Fallback: simple greedy if nothing found
    if best_result is None:
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 1
            while len(row) < k + 1:
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
                    x = 1
                else:
                    x += 1
                if x > 500:  # Safety
                    break
            if len(row) < k + 1:
                # Force complete
                while len(row) < k + 1:
                    row.append(row[-1] + 1)
            rows.append(row)
        best_result = rows
    
    return best_result
