import time
import random
import sys
import math
import copy
from collections import defaultdict

def _is_valid(rows, k=None):
    seen = set()
    for row in rows:
        if k is not None and len(row) != k + 1:
            return False
        if len(row) < 2:
            continue
        if row[0] != 0:
            return False
        for i in range(len(row)):
            if i > 0 and row[i] <= row[i - 1]:
                return False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in seen:
                    return False
                seen.add(d)
    return True

def _get_scope(rows):
    if not rows:
        return 0
    return max(max(row) for row in rows)

def _compute_row_diffs(row):
    diffs = set()
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            diffs.add(row[j] - row[i])
    return diffs

def _greedy_init(n_rows, k, rng=None):
    if rng is None:
        rng = random.Random()
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows, used

def _csp_search(n_rows, k, max_scope, rng, time_limit=5.0):
    """
    Backtracking CSP search for a solution with all entries <= max_scope.
    Each row has k+1 elements: a[0]=0 < a[1] < ... < a[k].
    We build rows one at a time, and within each row build elements left to right.
    """
    start_time = time.time()
    used = set()
    rows = []
    
    def backtrack(row_idx, pos_in_row, current_row):
        if time.time() - start_time > time_limit:
            return False, "timeout"
        
        if row_idx == n_rows:
            return True, "success"
        
        if pos_in_row == k + 1:
            rows.append(current_row)
            return backtrack(row_idx + 1, 0, [0])
        
        if pos_in_row == 0:
            # a[0] must be 0
            return backtrack(row_idx, 1, [0])
        
        # Choose next element a[pos_in_row]
        lo = current_row[-1] + 1
        hi = max_scope
        
        # Generate candidates in random-ish order, biased towards smaller values
        candidates = list(range(lo, hi + 1))
        rng.shuffle(candidates)
        
        # Sort by: prefer smaller, prefer values that create fewer "tight" diffs
        # Actually, let's just try them in shuffled order
        for val in candidates:
            # Check differences
            new_diffs = set()
            valid = True
            for i in range(pos_in_row):
                d = val - current_row[i]
                if d in used:
                    valid = False
                    break
                new_diffs.add(d)
            
            if not valid:
                continue
            
            # Also check new diffs don't conflict with each other (they can't since all positive and distinct in a row)
            # But we need to make sure they don't conflict with each other within the row
            # Actually, within a single row, all differences are distinct if the row is strictly increasing
            # Wait, no. a_j - a_i and a_m - a_n could be equal? No, if a_j - a_i = a_m - a_n with i<j and n<m,
            # it's possible. But we're building the row, so all diffs involving the new element are val - current_row[i]
            # These are all distinct since current_row[i] are distinct.
            
            used.update(new_diffs)
            current_row.append(val)
            
            ok, reason = backtrack(row_idx, pos_in_row + 1, current_row)
            if ok:
                return True, reason
            
            current_row.pop()
            used.difference_update(new_diffs)
        
        return False, "dead_end"
    
    ok, reason = backtrack(0, 0, [])
    if ok:
        return rows
    return None

def _sa_optimize(rows, n_rows, k, deadline, rng=None):
    """
    Simulated Annealing on the actual row elements.
    Neighborhood: 
      1. Pick a row, pick an element (not the first), try to move it to a new position.
      2. Special: try to reduce the max element.
    Cost: scope (max entry).
    """
    if rng is None:
        rng = random.Random()
    
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_scope = _get_scope(rows)
    
    # Build used set
    used = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                used.add(row[j] - row[i])
    
    T = 10.0
    cooling = 0.9995
    min_T = 0.01
    
    no_improve = 0
    max_no_improve = 5000
    
    while time.time() < deadline - 1 and T > min_T:
        # Pick a move
        r_idx = rng.randint(0, n_rows - 1)
        row = rows[r_idx]
        
        # Decide which element to move
        if rng.random() < 0.4:
            # Try to reduce the last element
            pos = len(row) - 1
        else:
            pos = rng.randint(1, len(row) - 1)
        
        old_val = row[pos]
        
        # Remove differences involving this element
        removed = []
        for i in range(len(row)):
            if i == pos:
                continue
            if i < pos:
                d = old_val - row[i]
                removed.append(d)
                used.discard(d)
            else:
                d = row[i] - old_val
                removed.append(d)
                used.discard(d)
        
        # Generate new value
        if rng.random() < 0.6:
            # Local move
            delta = rng.choice([-3, -2, -1, 1, 2, 3])
            new_val = old_val + delta
        else:
            # Larger jump
            if rng.random() < 0.5:
                new_val = rng.randint(1, old_val - 1) if old_val > 1 else 1
            else:
                new_val = rng.randint(1, best_scope + 5)
        
        # Check validity
        valid = True
        if new_val <= 0:
            valid = False
        else:
            # Check ordering
            if pos > 0 and new_val <= row[pos - 1]:
                valid = False
            if pos < len(row) - 1 and new_val >= row[pos + 1]:
                valid = False
            else:
                # Check differences
                for i in range(len(row)):
                    if i == pos:
                        continue
                    if i < pos:
                        d = new_val - row[i]
                    else:
                        d = row[i] - new_val
                    if d <= 0:
                        valid = False
                        break
                    if d in used:
                        valid = False
                        break
                    if d in removed:
                        # This is fine, we removed it
                        pass
        
        if valid:
            # Apply move
            row[pos] = new_val
            for i in range(len(row)):
                if i == pos:
                    continue
                if i < pos:
                    d = new_val - row[i]
                    used.add(d)
                else:
                    d = row[i] - new_val
                    used.add(d)
            
            new_scope = _get_scope(rows)
            
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [list(r) for r in rows]
                no_improve = 0
            else:
                delta_cost = new_scope - best_scope
                if rng.random() < math.exp(-delta_cost / max(T, min_T)):
                    no_improve = 0
                else:
                    # Revert
                    row[pos] = old_val
                    used.difference_update(removed)
                    for i in range(len(row)):
                        if i == pos:
                            continue
                        if i < pos:
                            d = old_val - row[i]
                            used.add(d)
                        else:
                            d = row[i] - old_val
                            used.add(d)
                    no_improve += 1
        else:
            # Revert (we didn't apply anything)
            for d in removed:
                used.add(d)
            no_improve += 1
        
        if no_improve > max_no_improve:
            # Restart with greedy
            new_rows, _ = _greedy_init(n_rows, k, rng=rng)
            if _is_valid(new_rows, k):
                ns = _get_scope(new_rows)
                if ns < best_scope:
                    best_scope = ns
                    best_rows = [list(r) for r in new_rows]
                    rows = [list(r) for r in new_rows]
                    used = set()
                    for row in rows:
                        for i in range(len(row)):
                            for j in range(i + 1, len(row)):
                                used.add(row[j] - row[i])
                    T = 10.0
                    no_improve = 0
            no_improve = 0
            max_no_improve = 3000
        
        T *= cooling
    
    return best_rows

def _local_polish(rows, n_rows, k, deadline, rng=None):
    """
    Aggressive local search to reduce scope.
    For each row, try to find the smallest possible values for elements.
    """
    if rng is None:
        rng = random.Random()
    
    rows = [list(r) for r in rows]
    best_scope = _get_scope(rows)
    
    used = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                used.add(row[j] - row[i])
    
    improved = True
    while improved and time.time() < deadline - 1:
        improved = False
        
        # Sort rows by their max element (descending) to tackle the biggest first
        row_indices = sorted(range(n_rows), key=lambda i: -max(rows[i]))
        
        for r_idx in row_indices:
            if time.time() > deadline - 1:
                break
            
            row = rows[r_idx]
            
            # For each position from right to left, try to reduce the value
            for pos in range(len(row) - 1, 0, -1):
                old_val = row[pos]
                
                # Try values from old_val - 1 down to row[pos-1] + 1
                for new_val in range(old_val - 1, row[pos - 1], -1):
                    # Check if new_val can be placed
                    removed = []
                    for i in range(len(row)):
                        if i == pos:
                            continue
                        if i < pos:
                            d = old_val - row[i]
                            removed.append(d)
                            used.discard(d)
                        else:
                            d = row[i] - old_val
                            removed.append(d)
                            used.discard(d)
                    
                    valid = True
                    for i in range(len(row)):
                        if i == pos:
                            continue
                        if i < pos:
                            d = new_val - row[i]
                        else:
                            d = row[i] - new_val
                        if d in used:
                            valid = False
                            break
                    
                    if valid:
                        # Apply
                        row[pos] = new_val
                        for i in range(len(row)):
                            if i == pos:
                                continue
                            if i < pos:
                                d = new_val - row[i]
                                used.add(d)
                            else:
                                d = row[i] - new_val
                                used.add(d)
                        improved = True
                        break
                    else:
                        # Revert
                        for d in removed:
                            used.add(d)
    
    return rows

def construct(n_rows, k):
    deadline = time.time() + 148.0  # Safety margin
    rng = random.Random(42)
    
    best_rows = None
    best_scope = float('inf')
    
    # Phase 1: Greedy initialization with multiple seeds
    for seed in [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]:
        r = random.Random(seed)
        rows, _ = _greedy_init(n_rows, k, rng=r)
        if _is_valid(rows, k):
            scope = _get_scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r2) for r2 in rows]
    
    if best_rows is None:
        rows, _ = _greedy_init(n_rows, k, rng=rng)
        return rows
    
    # Phase 2: CSP search for smaller scopes
    # Try to find solutions with scope < best_scope
    current_scope = best_scope
    # Lower bound: counting argument
    # Total diffs needed: n_rows * k*(k+1)/2
    # Max diff <= scope
    # So scope >= ceil(n_rows * k*(k+1)/2 / scope) ... this is circular
    # Simple lower bound: n_rows * k*(k+1)/2 distinct diffs, each <= scope
    # So scope must be at least n_rows * k*(k+1)/2
    # But that's not right. The diffs are from 1 to scope, so there are scope possible diffs.
    # We need n_rows * k*(k+1)/2 distinct diffs, so scope >= n_rows * k*(k+1)/2
    # For n_rows=7, k=5: 7 * 15 = 105. So scope >= 105.
    
    lower_bound = n_rows * k * (k + 1) // 2
    
    # Try CSP from lower_bound up to current_scope - 1
    for scope in range(lower_bound, current_scope):
        if time.time() > deadline - 30:
            break
        
        # Give each scope attempt a time limit
        time_per_scope = min(5.0, deadline - 30 - time.time())
        if time_per_scope < 1.0:
            break
        
        sol = _csp_search(n_rows, k, scope, rng, time_limit=time_per_scope)
        if sol is not None and _is_valid(sol, k):
            best_scope = scope
            best_rows = [list(r) for r in sol]
            break  # Found the smallest possible (or close to it)
    
    # Phase 3: SA optimization
    if time.time() < deadline - 10:
        sa_rows = _sa_optimize(best_rows, n_rows, k, deadline - 10, rng)
        if _is_valid(sa_rows, k):
            sa_scope = _get_scope(sa_rows)
            if sa_scope < best_scope:
                best_scope = sa_scope
                best_rows = [list(r) for r in sa_rows]
    
    # Phase 4: Local polish
    if time.time() < deadline - 2:
        polish_rows = _local_polish(best_rows, n_rows, k, deadline - 2, rng)
        if _is_valid(polish_rows, k):
            polish_scope = _get_scope(polish_rows)
            if polish_scope < best_scope:
                best_scope = polish_scope
                best_rows = [list(r) for r in polish_rows]
    
    # Final validation
    if not _is_valid(best_rows, k):
        # Fallback
        rows, _ = _greedy_init(n_rows, k, rng=rng)
        return rows
    
    return best_rows
