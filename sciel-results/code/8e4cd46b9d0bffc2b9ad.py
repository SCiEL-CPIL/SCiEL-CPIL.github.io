import time
import random
import math
import sys

sys.setrecursionlimit(10000)

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    # Total number of differences needed
    total_diffs_needed = n_rows * k * (k + 1) // 2
    
    # Lower bound on scope: we need at least total_diffs_needed distinct differences.
    # Since differences are integers from 1 to scope, scope >= total_diffs_needed.
    # Actually, the differences are a_j - a_i. The maximum difference is the scope.
    # So the number of distinct differences is at most scope.
    # Thus scope >= total_diffs_needed.
    lb = total_diffs_needed

    def _row_diffs(row):
        diffs = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        return diffs

    def _validate(rows):
        all_diffs = set()
        for row in rows:
            if len(row) != k + 1 or row[0] != 0:
                return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in all_diffs:
                        return False
                    all_diffs.add(d)
        return True

    # --- Greedy Construction ---
    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    # --- Randomized Greedy Row Generation ---
    def randomized_greedy_row(forbidden_diffs, max_limit=None, max_candidates=5):
        """
        Generates a valid row given a set of forbidden differences (from other rows).
        Returns None if no row can be found within max_limit (if specified).
        """
        row = [0]
        used_in_row = set()
        last_val = 0
        
        for _ in range(k):
            candidates = []
            x = last_val + 1
            
            # We need to find x such that:
            # 1. x > last_val
            # 2. {x - a for a in row} are all distinct (implied if x is unique and a's are distinct? No, x-a must be distinct for fixed x. Since a's are distinct, x-a are distinct. So this is always true for a single x).
            # 3. {x - a for a in row} does not intersect with forbidden_diffs
            # 4. {x - a for a in row} does not intersect with used_in_row
            
            # Actually, condition 3 and 4 can be combined: new_diffs & (forbidden_diffs | used_in_row) == empty.
            
            # To speed up, we scan x upwards.
            # We stop if we find enough candidates or if we exceed a limit.
            
            # Limit for scanning x:
            # If max_limit is set, we must have x <= max_limit.
            # Also, we don't want to scan forever.
            
            scan_limit = 500
            if max_limit is not None:
                scan_limit = min(scan_limit, max_limit - last_val)
            
            while len(candidates) < max_candidates and x < last_val + scan_limit:
                new_diffs = {x - a for a in row}
                if not (new_diffs & forbidden_diffs) and not (new_diffs & used_in_row):
                    candidates.append(x)
                x += 1
                
            if not candidates:
                return None
            
            # Choose randomly from candidates
            # Bias towards smaller values to keep scope low?
            # Uniform is fine for diversity, but weighted might help minimize max.
            # Let's use uniform for now to allow exploration.
            chosen_x = random.choice(candidates)
            
            row.append(chosen_x)
            new_diffs = {chosen_x - a for a in row[:-1]}
            used_in_row |= new_diffs
            last_val = chosen_x
            
        return row

    # --- Initial Solution ---
    best_rows = greedy_construct()
    best_scope = max(max(r) for r in best_rows)
    
    if not _validate(best_rows):
        return best_rows

    # --- Simulated Annealing ---
    # State: rows, current_scope
    # Move: Replace one row with a new randomized greedy row.
    
    T = 10.0
    cooling_rate = 0.999
    min_T = 0.01
    
    current_rows = [r[:] for r in best_rows]
    current_scope = best_scope
    
    # Pre-compute row diffs for current rows
    current_row_diffs = [_row_diffs(r) for r in current_rows]
    
    steps = 0
    while time.time() < deadline:
        # Check for timeout frequently
        if time.time() > deadline - 5.0:
            break
            
        i = random.randint(0, n_rows - 1)
        
        # Calculate forbidden diffs for row i
        forbidden = set()
        for j in range(n_rows):
            if j != i:
                forbidden |= current_row_diffs[j]
        
        # Try to generate a new row
        # We want to keep the scope low.
        # Strategy:
        # 1. Try to find a row with max < current_scope.
        # 2. If not, try to find a row with max == current_scope.
        # 3. If not, allow a slightly larger row to escape local minima.
        
        new_row = None
        
        # Try tight constraint first
        if current_scope > lb:
            new_row = randomized_greedy_row(forbidden, max_limit=current_scope - 1, max_candidates=10)
        
        if new_row is None:
            new_row = randomized_greedy_row(forbidden, max_limit=current_scope, max_candidates=10)
            
        if new_row is None:
            # Allow expansion to escape
            new_row = randomized_greedy_row(forbidden, max_limit=None, max_candidates=10)
            
        if new_row is None:
            continue
            
        new_max_row = new_row[-1]
        
        # Calculate new global scope
        # We replace row i.
        # new_scope = max( max(r[-1] for r in current_rows if r is not current_rows[i]), new_max_row )
        # Note: current_rows[i] is the old row.
        
        other_max = 0
        for j in range(n_rows):
            if j != i:
                if current_rows[j][-1] > other_max:
                    other_max = current_rows[j][-1]
        
        new_scope = max(other_max, new_max_row)
        
        delta = new_scope - current_scope
        
        accept = False
        if delta < 0:
            accept = True
        else:
            if T > min_T:
                # Avoid overflow in exp
                if delta / T < 50:
                    prob = math.exp(-delta / T)
                    if random.random() < prob:
                        accept = True
        
        if accept:
            # Accept move
            old_row = current_rows[i]
            old_diffs = current_row_diffs[i]
            
            current_rows[i] = new_row
            current_row_diffs[i] = _row_diffs(new_row)
            current_scope = new_scope
            
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [r[:] for r in current_rows]
        else:
            # Reject move (state remains unchanged)
            pass
            
        T *= cooling_rate
        if T < min_T:
            T = min_T
            
        steps += 1
        
        # Optional: Periodic restart or perturbation if stuck?
        # For now, simple SA.
        
    return best_rows

