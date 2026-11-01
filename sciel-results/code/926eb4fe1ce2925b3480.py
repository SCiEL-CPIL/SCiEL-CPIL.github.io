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

def _validate_move(cand_row, old_diffs, all_used):
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

def _backtrack_construct(n_rows, k, max_entry, time_limit=5.0):
    """
    Backtracking search to find a valid construction where all entries are <= max_entry.
    Uses forward checking and symmetry breaking.
    Returns (rows, found).
    """
    start_time = time.time()
    deadline = start_time + time_limit
    rows = [None] * n_rows
    used_diffs = set()
    
    # Pre-compute some constraints if possible? 
    # For now, standard DFS with pruning.
    
    def build_row(pos, current_row, current_diffs, row_idx):
        """
        Recursively build a single row.
        pos: current position in the row (0..k)
        current_row: list of values chosen so far
        current_diffs: set of diffs generated within this row so far
        row_idx: which row we are building (for potential symmetry breaking, though rows are distinct in order)
        """
        if time.time() > deadline:
            return None
        
        if pos == k + 1:
            return current_row[:]
        
        if pos == 0:
            return build_row(1, [0], set(), row_idx)
        
        prev_val = current_row[-1]
        remaining_slots = k + 1 - pos
        
        # Heuristic: The next value must be > prev_val and <= max_entry.
        # Also, the smallest possible value for the last slot is current_max + (remaining_slots - 1).
        # So current_val + (remaining_slots - 1) <= max_entry => current_val <= max_entry - (remaining_slots - 1)
        max_start = max_entry - (remaining_slots - 1)
        
        # We can also prune based on differences.
        # The difference between the last element and the first element (0) is the max entry itself.
        # But we need to ensure all intermediate differences are unique and not in used_diffs.
        
        start_val = prev_val + 1
        end_val = min(max_entry, max_start)
        
        # Optimization: If we are filling the last few positions, we can be more specific.
        # For pos == k (last element), we just need to find a value v such that:
        # 1. v > prev_val
        # 2. v <= max_entry
        # 3. {v - a for a in current_row} are all new and not in used_diffs.
        
        if pos == k:
            for v in range(start_val, end_val + 1):
                new_diffs = {v - a for a in current_row}
                if len(new_diffs) != len(current_row):
                    continue
                if new_diffs & current_diffs:
                    continue
                if new_diffs & used_diffs:
                    continue
                return current_row + [v]
            return None
        
        # General case: try values
        # To improve speed, we can shuffle the range if we allow randomness, but deterministic is better for reproducibility of "found" status.
        # However, for backtracking, order matters. Smallest first is usually good for finding *a* solution, but might not be best for *tight* bounds if it gets stuck.
        # Let's try smallest first.
        
        for v in range(start_val, end_val + 1):
            new_diffs = {v - a for a in current_row}
            # Check internal consistency
            if len(new_diffs) != len(current_row):
                continue
            if new_diffs & current_diffs:
                continue
            # Check global consistency
            if new_diffs & used_diffs:
                continue
            
            # Forward Check: Can we still fill the rest?
            # The minimum value for the next position is v + 1.
            # The minimum value for the last position will be v + (remaining_slots - 1).
            # We already ensured v + (remaining_slots - 1) <= max_entry via end_val.
            # We should also check if the differences required by the *minimum* future configuration
            # would conflict. This is hard to compute exactly, so we skip deep forward checking on diffs for speed,
            # relying on the fact that k is small (5) so the branching factor is manageable.
            
            result = build_row(pos + 1, current_row + [v], current_diffs | new_diffs, row_idx)
            if result is not None:
                return result
        
        return None

    def backtrack(row_idx):
        if time.time() > deadline:
            return False
        
        if row_idx == n_rows:
            return True
        
        # Try to build row row_idx
        # Symmetry breaking: The first row can start with 0.
        # Subsequent rows also start with 0.
        # We don't have much symmetry to break between rows since they are distinct sets of differences,
        # but we can enforce that the max entry of row i is non-decreasing? No, that's not required.
        
        new_row = build_row(0, [], set(), row_idx)
        if new_row is not None:
            row_diffs = _row_diffs(new_row)
            rows[row_idx] = new_row
            used_diffs.update(row_diffs)
            
            if backtrack(row_idx + 1):
                return True
            
            # Backtrack
            used_diffs.difference_update(row_diffs)
            rows[row_idx] = None
        
        return False

    found = backtrack(0)
    if found:
        return rows, True
    return None, False

def _sa_polish(rows, n_rows, k, time_limit):
    """
    Simulated Annealing polish to minimize the global max.
    """
    start_time = time.time()
    deadline = start_time + time_limit
    
    current_rows = [r[:] for r in rows]
    current_val = _global_max(current_rows)
    best_rows = [r[:] for r in current_rows]
    best_val = current_val
    
    all_used = _all_used(current_rows)
    row_diffs = [_row_diffs(r) for r in current_rows]
    row_maxes = [max(r) for r in current_rows]
    
    T = 10.0
    T_min = 0.1
    cool = 0.9999
    
    no_improve = 0
    max_no_improve = 10000
    
    while time.time() < deadline and no_improve < max_no_improve:
        ri = random.randrange(n_rows)
        pi = random.randrange(1, k + 1)
        
        old_val = current_rows[ri][pi]
        
        # Generate candidates: mostly decreasing, some increasing
        candidates = []
        min_allowed = current_rows[ri][pi - 1] + 1
        max_decrease = min(20, old_val - min_allowed)
        for d in range(1, max_decrease + 1):
            candidates.append(old_val - d)
            
        max_allowed = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 10
        max_increase = min(10, max_allowed - old_val)
        for d in range(1, max_increase + 1):
            candidates.append(old_val + d)
            
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
            
            new_diffs, valid = _validate_move(cand_row, row_diffs[ri], all_used)
            if not valid:
                continue
            
            cand_row_max = max(cand_row)
            old_row_max = row_maxes[ri]
            row_maxes[ri] = cand_row_max
            new_global_max = max(row_maxes)
            row_maxes[ri] = old_row_max
            
            delta = new_global_max - current_val
            
            accept = False
            if delta < 0:
                accept = True
            elif delta == 0:
                if random.random() < 0.1:
                    accept = True
            else:
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
        
    return best_rows

def construct(n_rows, k):
    start_time = time.time()
    total_deadline = start_time + 145.0  # Safety margin
    
    # 1. Initial Greedy Construction
    best_rows = _greedy_init(n_rows, k)
    best_val = _global_max(best_rows)
    
    # 2. Backtracking Phase: Try to find a solution with smaller scope
    # We know the counting bound is 105. Greedy might give ~115-120.
    # We try to prove existence of solutions with scope M, starting from best_val - 1 down to 105.
    # We limit the time spent here to leave room for SA.
    
    bt_time_budget = 30.0
    bt_start = time.time()
    
    # Try a few specific values near the best known or just below greedy
    # If greedy is 115, try 114, 113...
    # If we find one, we update best_val.
    
    # To be efficient, we can binary search the scope if the gap is large,
    # but since the gap is small (usually < 10), linear decrement is fine.
    
    current_search_max = best_val - 1
    
    # Ensure we don't search below the theoretical lower bound if known, 
    # but 105 is the counting bound for (7,5). For general n,k, we don't have a tight bound.
    # We'll stop searching if we hit 105 or if time runs out.
    
    while current_search_max >= 105 and (time.time() - bt_start) < bt_time_budget:
        # Allocate time for this specific backtracking attempt
        time_per_attempt = 5.0
        if time.time() - bt_start + time_per_attempt > bt_time_budget:
            time_per_attempt = bt_time_budget - (time.time() - bt_start)
            if time_per_attempt < 1.0:
                break
                
        rows, found = _backtrack_construct(n_rows, k, current_search_max, time_limit=time_per_attempt)
        
        if found:
            val = _global_max(rows)
            if val < best_val:
                best_val = val
                best_rows = [r[:] for r in rows]
            # If found, we can try to go even lower
            # But if we found it quickly, we can try the next lower bound.
            # If it took long, we might stop.
            # Let's just decrement and try again.
            current_search_max -= 1
        else:
            # If not found, it might be infeasible or we ran out of time.
            # If we ran out of time, we should probably stop trying lower bounds 
            # because higher bounds are more likely to be solvable? 
            # Actually, if M is infeasible, M-1 is also infeasible. 
            # So if backtracking *proves* infeasibility, we can stop. 
            # But our backtracker is time-limited. If it times out, it's inconclusive.
            # If it exhausts the search space (returns False quickly), then it's infeasible.
            # Distinguishing "timeout" from "infeasible" is hard without instrumentation.
            # For safety, if we fail, we assume it might be infeasible or just hard.
            # Given the small k, if M is close to the bound, it's likely infeasible.
            # Let's assume if we fail, we stop the backtracking phase to save time for SA.
            break
    
    # 3. SA Polish Phase
    # Use the remaining time to polish the best solution found.
    remaining_time = total_deadline - time.time()
    if remaining_time > 5.0:
        polished_rows = _sa_polish(best_rows, n_rows, k, remaining_time - 2.0)
        if polished_rows:
            val = _global_max(polished_rows)
            if val < best_val:
                best_val = val
                best_rows = polished_rows
    
    return best_rows

# EVOLVE-BLOCK-END
