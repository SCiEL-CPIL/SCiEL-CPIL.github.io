import time
import random
import math
from collections import defaultdict

# EVOLVE-BLOCK-START

def _greedy_init(n_rows, k):
    """
    Greedy initialization: each row takes the smallest next mark with no difference collision.
    """
    rows = []
    used = set()
    for r in range(n_rows):
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

def _row_diffs(row):
    """Compute the set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _all_used(rows):
    """Compute the union of all differences across all rows."""
    all_used = set()
    for row in rows:
        all_used |= _row_diffs(row)
    return all_used

def _global_max(rows):
    """Compute the maximum entry across all rows."""
    return max(max(row) for row in rows)

def _validate_move_fast(cand_row, old_diffs, all_used):
    """
    Check if changing a row to cand_row is valid (no internal collisions, no cross-row collisions).
    Returns (new_diffs, is_valid).
    """
    n = len(cand_row)
    new_diffs = set()
    for i in range(n):
        for j in range(i + 1, n):
            d = cand_row[j] - cand_row[i]
            if d in new_diffs:
                return None, False
            new_diffs.add(d)
    
    # Check cross-row conflicts
    for d in new_diffs:
        if d in all_used and d not in old_diffs:
            return None, False
    
    return new_diffs, True

def _backtrack_construct(n_rows, k, max_scope, timeout=5.0):
    """
    Backtracking construction that tries to find a valid set of rows with scope <= max_scope.
    This is a different structural approach: rather than local search from a greedy start,
    we build rows one by one with constraint propagation.
    Returns list of rows if found, else None.
    """
    start_time = time.time()
    rows = []
    used = set()
    
    def backtrack(r, current_row, current_diffs, current_max):
        if time.time() - start_time > timeout:
            return False
        if r == n_rows:
            rows.extend([list(cr) for cr in current_row])
            return True
        
        # Try to construct row r
        # We need k+1 elements starting at 0
        # Use backtracking within the row
        row = [0]
        row_diffs = set()
        
        def build_row(pos, last_val):
            if time.time() - start_time > timeout:
                return False
            if pos == k + 1:
                # Check if this row's diffs conflict with used
                if row_diffs & used:
                    return False
                rows.append(list(row))
                return True
            
            # Try next values
            min_next = last_val + 1
            max_next = max_scope
            # Pruning: need enough room for remaining elements
            remaining = k + 1 - pos
            if max_next < min_next + remaining - 1:
                return False
            
            for x in range(min_next, max_next + 1):
                # Check internal diff conflicts
                new_diffs = set()
                ok = True
                for a in row:
                    d = x - a
                    if d in row_diffs or d in new_diffs:
                        ok = False
                        break
                    new_diffs.add(d)
                if not ok:
                    continue
                
                # Check against global used (for early pruning)
                # Only check if this is the last element or periodically
                if pos == k:
                    if new_diffs & used:
                        continue
                
                row.append(x)
                old_diffs = row_diffs.copy()
                row_diffs |= new_diffs
                
                if build_row(pos + 1, x):
                    return True
                
                row.pop()
                row_diffs = old_diffs
            
            return False
        
        if build_row(1, 0):
            used |= row_diffs
            return True
        
        # If we can't build row r with current used, we need to backtrack
        # For simplicity in this heuristic, we just return False
        # A more complete implementation would backtrack previous rows
        return False
    
    if backtrack(0, [], set(), 0):
        return rows
    return None

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # 1. Multiple diverse initializations
    best_rows = None
    best_val = float('inf')
    
    # Try backtracking construction first with a tight bound
    # The counting bound is 105, so try from 105 upwards
    for scope in range(105, 130):
        bt_rows = _backtrack_construct(n_rows, k, scope, timeout=3.0)
        if bt_rows is not None:
            val = _global_max(bt_rows)
            if val < best_val:
                best_val = val
                best_rows = [r[:] for r in bt_rows]
            break
    
    # If backtracking didn't find anything good, use greedy
    if best_rows is None or best_val > 120:
        for init_trial in range(5):
            rows = _greedy_init(n_rows, k)
            val = _global_max(rows)
            if val < best_val:
                best_val = val
                best_rows = [r[:] for r in rows]
    
    # 2. Iterated Local Search with Simulated Annealing
    current_rows = [r[:] for r in best_rows]
    current_val = best_val
    row_maxes = [max(r) for r in current_rows]
    all_used = _all_used(current_rows)
    row_diffs = [_row_diffs(r) for r in current_rows]
    
    # SA parameters
    T = 500.0
    T_min = 0.5
    cool = 0.99998
    
    no_improve = 0
    max_no_improve = 30000
    restarts = 0
    max_restarts = 20
    
    while time.time() < deadline - 10.0 and no_improve < max_no_improve:
        # Pick a row and position to modify
        ri = random.randrange(n_rows)
        pi = random.randrange(1, k + 1)
        
        old_val = current_rows[ri][pi]
        
        # Generate candidates
        candidates = []
        
        # Decreasing candidates (primary optimization direction)
        min_allowed = current_rows[ri][pi - 1] + 1
        max_decrease = min(80, old_val - min_allowed)
        for d in range(1, max_decrease + 1):
            candidates.append(old_val - d)
        
        # Increasing candidates (exploration)
        max_allowed = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 50
        max_increase = min(30, max_allowed - old_val)
        for d in range(1, max_increase + 1):
            candidates.append(old_val + d)
        
        # Random jumps
        if random.random() < 0.3:
            low = current_rows[ri][pi - 1] + 1
            high = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 40
            if low <= high:
                candidates.append(random.randint(low, high))
        
        if not candidates:
            no_improve += 1
            T = max(T_min, T * cool)
            continue
        
        random.shuffle(candidates)
        
        moved = False
        for new_val in candidates:
            if new_val <= current_rows[ri][pi - 1]:
                continue
            if pi < k and new_val >= current_rows[ri][pi + 1]:
                continue
            
            cand_row = current_rows[ri][:]
            cand_row[pi] = new_val
            
            new_diffs, valid = _validate_move_fast(cand_row, row_diffs[ri], all_used)
            if not valid:
                continue
            
            # Compute new global max
            cand_row_max = max(cand_row)
            old_row_max = row_maxes[ri]
            row_maxes[ri] = cand_row_max
            new_global_max = max(row_maxes)
            row_maxes[ri] = old_row_max
            
            delta = new_global_max - current_val
            
            # Acceptance criterion
            accept = False
            if delta < 0:
                accept = True
            elif delta == 0:
                if random.random() < 0.2:
                    accept = True
            else:
                if T > T_min:
                    prob = math.exp(-delta / max(T, 0.1))
                    if random.random() < prob:
                        accept = True
            
            if accept:
                all_used -= row_diffs[ri]
                all_used |= new_diffs
                current_rows[ri] = cand_row
                row_diffs[ri] = new_diffs
                row_maxes[ri] = cand_row_max
                current_val = max(row_maxes)
                
                if current_val < best_val:
                    best_val = current_val
                    best_rows = [r[:] for r in current_rows]
                    no_improve = 0
                else:
                    no_improve += 1
                
                moved = True
                break
        
        if not moved:
            no_improve += 1
        
        T = max(T_min, T * cool)
        
        # Restart if stuck
        if no_improve > max_no_improve:
            restarts += 1
            if restarts > max_restarts:
                break
            # Restart from best with perturbation
            current_rows = [r[:] for r in best_rows]
            current_val = best_val
            all_used = _all_used(current_rows)
            row_diffs = [_row_diffs(r) for r in current_rows]
            row_maxes = [max(r) for r in current_rows]
            T = 300.0
            no_improve = 0
    
    # 3. Final intensive polish with decreasing-only moves
    polish_deadline = min(deadline, start_time + 145.0)
    
    current_rows = [r[:] for r in best_rows]
    current_val = best_val
    all_used = _all_used(current_rows)
    row_diffs = [_row_diffs(r) for r in current_rows]
    row_maxes = [max(r) for r in current_rows]
    
    T = 2.0
    T_min = 0.1
    cool = 0.99999
    no_improve = 0
    max_no_improve_polish = 50000
    
    while time.time() < polish_deadline and no_improve < max_no_improve_polish:
        ri = random.randrange(n_rows)
        pi = random.randrange(1, k + 1)
        
        old_val = current_rows[ri][pi]
        
        # Only decreasing moves in polish
        min_allowed = current_rows[ri][pi - 1] + 1
        max_decrease = min(40, old_val - min_allowed)
        
        if max_decrease <= 0:
            no_improve += 1
            continue
        
        candidates = []
        for d in range(1, max_decrease + 1):
            candidates.append(old_val - d)
        
        random.shuffle(candidates)
        
        moved = False
        for new_val in candidates:
            if new_val <= current_rows[ri][pi - 1]:
                continue
            if pi < k and new_val >= current_rows[ri][pi + 1]:
                continue
            
            cand_row = current_rows[ri][:]
            cand_row[pi] = new_val
            
            new_diffs, valid = _validate_move_fast(cand_row, row_diffs[ri], all_used)
            if not valid:
                continue
            
            cand_row_max = max(cand_row)
            old_row_max = row_maxes[ri]
            row_maxes[ri] = cand_row_max
            new_global_max = max(row_maxes)
            row_maxes[ri] = old_row_max
            
            delta = new_global_max - current_val
            
            if delta < 0:
                accept = True
            else:
                if T > T_min:
                    prob = math.exp(-delta / max(T, 0.1))
                    if random.random() < prob:
                        accept = True
                    else:
                        continue
                else:
                    continue
            
            if accept:
                all_used -= row_diffs[ri]
                all_used |= new_diffs
                current_rows[ri] = cand_row
                row_diffs[ri] = new_diffs
                row_maxes[ri] = cand_row_max
                current_val = max(row_maxes)
                
                if current_val < best_val:
                    best_val = current_val
                    best_rows = [r[:] for r in current_rows]
                    no_improve = 0
                else:
                    no_improve += 1
                
                moved = True
                break
        
        if not moved:
            no_improve += 1
        
        T = max(T_min, T * cool)
    
    # Final validity check
    if not _is_valid(best_rows):
        # Fallback: construct a trivial valid set
        rows = []
        used = set()
        for r in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if len(new_diffs) != len(row):
                    continue
                if new_diffs & used:
                    continue
                row.append(x)
                used |= new_diffs
            rows.append(row)
        return rows
    
    return best_rows

def _is_valid(rows):
    """Check if the set of rows is valid (all diffs distinct globally)."""
    used = set()
    for row in rows:
        diffs = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d in diffs:
                    return False
                diffs.add(d)
        if diffs & used:
            return False
        used |= diffs
    return True

# EVOLVE-BLOCK-END
