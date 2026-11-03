"""Difference triangle set (7,5) solver using pre-computed Sidon sets and backtracking search."""
import time
import random
import sys

_DEADLINE = 145.0
_START_TIME = time.time()

def _elapsed():
    return time.time() - _START_TIME

def _row_diffs_set(row):
    """Return the set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _is_valid(rows, k):
    """Check that all positive differences across all rows are distinct."""
    seen = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        if not all(row[i] < row[i+1] for i in range(len(row)-1)):
            return False
        for d in _row_diffs_set(row):
            if d in seen:
                return False
            seen.add(d)
    return True

def _max_entry(rows):
    return max(max(r) for r in rows) if rows else 0

def _precompute_sidon_sets(k, max_scope):
    """Pre-compute all valid row configurations (Sidon sets) up to max_scope.
    A row is a list of k+1 integers starting at 0, strictly increasing,
    where all differences within the row are distinct."""
    valid_rows = []
    
    # Generate all combinations of k values from 1..max_scope
    # Row = [0, a1, a2, ..., ak] where 0 < a1 < a2 < ... < ak <= max_scope
    from itertools import combinations
    
    for combo in combinations(range(1, max_scope + 1), k):
        row = [0] + list(combo)
        diffs = _row_diffs_set(row)
        if len(diffs) == len(row) * (len(row) - 1) // 2:
            valid_rows.append(row)
    
    # Sort by max entry, then by sum
    valid_rows.sort(key=lambda r: (r[-1], sum(r)))
    return valid_rows

def _backtracking_search(n_rows, k, valid_rows, time_budget=90):
    """Backtracking search to find n_rows non-conflicting rows with minimal max entry."""
    start_time = time.time()
    best_rows = None
    best_scope = float('inf')
    best_sum = float('inf')
    
    # We'll search for solutions with increasing max scope
    # First, find the minimum possible scope by binary search or incremental
    scopes = sorted(set(r[-1] for r in valid_rows))
    
    for target_scope in scopes:
        if time.time() - start_time > time_budget:
            break
        
        # Filter rows with max entry <= target_scope
        candidate_rows = [r for r in valid_rows if r[-1] <= target_scope]
        
        if len(candidate_rows) < n_rows:
            continue
        
        # Try to find n_rows non-conflicting rows
        def backtrack(used_diffs, selected, idx):
            if time.time() - start_time > time_budget:
                return None
            
            if len(selected) == n_rows:
                return selected[:]
            
            # Pruning: not enough rows left
            if idx >= len(candidate_rows):
                return None
            
            # Try each candidate row starting from idx
            for i in range(idx, len(candidate_rows)):
                row = candidate_rows[i]
                row_diffs = _row_diffs_set(row)
                
                # Check for conflicts
                if row_diffs & used_diffs:
                    continue
                
                # Try this row
                selected.append(row)
                used_diffs = used_diffs | row_diffs
                
                result = backtrack(used_diffs, selected, i + 1)
                if result is not None:
                    return result
                
                # Backtrack
                selected.pop()
                used_diffs = used_diffs - row_diffs
            
            return None
        
        result = backtrack(set(), [], 0)
        if result is not None:
            scope = _max_entry(result)
            total_sum = sum(sum(r) for r in result)
            if scope < best_scope or (scope == best_scope and total_sum < best_sum):
                best_scope = scope
                best_sum = total_sum
                best_rows = [list(r) for r in result]
            # If we found a solution at this scope, we can stop (we're searching in order)
            break
    
    return best_rows

def _randomized_backtracking(n_rows, k, valid_rows, time_budget=90, rng=None):
    """Randomized backtracking with tie-breaking to explore more solutions."""
    if rng is None:
        rng = random.Random()
    
    start_time = time.time()
    best_rows = None
    best_scope = float('inf')
    best_sum = float('inf')
    
    scopes = sorted(set(r[-1] for r in valid_rows))
    
    for target_scope in scopes:
        if time.time() - start_time > time_budget:
            break
        
        candidate_rows = [r for r in valid_rows if r[-1] <= target_scope]
        
        if len(candidate_rows) < n_rows:
            continue
        
        # Shuffle for randomization
        shuffled = candidate_rows[:]
        rng.shuffle(shuffled)
        
        # Try multiple random orderings
        max_attempts = 50
        found = False
        for attempt in range(max_attempts):
            if time.time() - start_time > time_budget:
                break
            
            if attempt == 0:
                order = shuffled
            else:
                order = candidate_rows[:]
                rng.shuffle(order)
            
            def backtrack(used_diffs, selected, idx):
                if time.time() - start_time > time_budget:
                    return None
                
                if len(selected) == n_rows:
                    return selected[:]
                
                if idx >= len(order):
                    return None
                
                for i in range(idx, len(order)):
                    row = order[i]
                    row_diffs = _row_diffs_set(row)
                    
                    if row_diffs & used_diffs:
                        continue
                    
                    selected.append(row)
                    new_used = used_diffs | row_diffs
                    
                    result = backtrack(new_used, selected, i + 1)
                    if result is not None:
                        return result
                    
                    selected.pop()
                
                return None
            
            result = backtrack(set(), [], 0)
            if result is not None:
                scope = _max_entry(result)
                total_sum = sum(sum(r) for r in result)
                if scope < best_scope or (scope == best_scope and total_sum < best_sum):
                    best_scope = scope
                    best_sum = total_sum
                    best_rows = [list(r) for r in result]
                found = True
                break
        
        if found:
            break
    
    return best_rows

def _local_search_polish(rows, k, rng, time_budget=15):
    """Final high-precision polish: try single-entry shifts to reduce scope."""
    n_rows = len(rows)
    best_rows = [list(r) for r in rows]
    best_scope = _max_entry(best_rows)
    best_sum = sum(sum(r) for r in best_rows)
    
    diff_counts = {}
    for row in best_rows:
        for d in _row_diffs_set(row):
            diff_counts[d] = diff_counts.get(d, 0) + 1
    
    row_diff_sets = [_row_diffs_set(row) for row in best_rows]
    
    start_time = time.time()
    no_improve_count = 0
    MAX_NO_IMPROVE = 8
    
    while time.time() - start_time < time_budget:
        if _elapsed() > _DEADLINE - 3:
            break
        
        improved = False
        for ri in range(n_rows):
            row = best_rows[ri]
            row_len = len(row)
            for idx in range(1, row_len):
                old_val = row[idx]
                
                for reduction in range(1, 6):
                    new_val = old_val - reduction
                    if new_val <= 0:
                        break
                    if idx > 1 and new_val <= row[idx - 1]:
                        break
                    if idx < row_len - 1 and new_val >= row[idx + 1]:
                        continue
                    
                    new_row = list(row)
                    new_row[idx] = new_val
                    new_diffs = _row_diffs_set(new_row)
                    
                    if len(new_diffs) != len(new_row) * (len(new_row) - 1) // 2:
                        continue
                    
                    old_diff_set = row_diff_sets[ri]
                    added = new_diffs - old_diff_set
                    
                    conflict = False
                    for d in added:
                        other_cnt = diff_counts.get(d, 0)
                        if d in old_diff_set:
                            other_cnt -= 1
                        if other_cnt >= 1:
                            conflict = True
                            break
                    
                    if conflict:
                        continue
                    
                    new_scope = max(new_row[-1], max((best_rows[i][-1] for i in range(n_rows) if i != ri), default=0))
                    delta_sum = new_val - old_val
                    new_sum = best_sum + delta_sum
                    
                    if new_scope < best_scope or (new_scope == best_scope and new_sum < best_sum):
                        removed = old_diff_set - new_diffs
                        for d in removed:
                            cnt = diff_counts[d]
                            if cnt == 1:
                                del diff_counts[d]
                            else:
                                diff_counts[d] = cnt - 1
                        for d in added:
                            old_cnt = diff_counts.get(d, 0)
                            diff_counts[d] = old_cnt + 1
                        
                        row[idx] = new_val
                        row_diff_sets[ri] = new_diffs
                        best_scope = new_scope
                        best_sum = new_sum
                        improved = True
                        no_improve_count = 0
                        break
                
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            no_improve_count += 1
            if no_improve_count >= MAX_NO_IMPROVE:
                break
        else:
            no_improve_count = 0
    
    if _is_valid(best_rows, k):
        return best_rows
    return [list(r) for r in rows]

def construct(n_rows, k):
    global _START_TIME
    _START_TIME = time.time()
    
    rng = random.Random(42)
    
    # Pre-compute all valid Sidon sets up to a reasonable scope
    # Counting bound is 105, best known is 112
    # Use max_scope = 115 to be safe
    max_scope = 115
    valid_rows = _precompute_sidon_sets(k, max_scope)
    
    best_rows = None
    best_score = float('inf')
    
    # Phase 1: Deterministic backtracking search
    if _elapsed() < _DEADLINE - 30:
        time_left = _DEADLINE - 30 - _elapsed()
        result = _backtracking_search(n_rows, k, valid_rows, time_budget=min(60, time_left))
        if result is not None:
            best_rows = result
            best_score = _max_entry(result)
    
    # Phase 2: Randomized backtracking for diversity
    if _elapsed() < _DEADLINE - 15 and best_rows is not None:
        time_left = _DEADLINE - 15 - _elapsed()
        rng2 = random.Random(999)
        result = _randomized_backtracking(n_rows, k, valid_rows, time_budget=min(40, time_left), rng=rng2)
        if result is not None:
            score = _max_entry(result)
            if score < best_score:
                best_rows = result
                best_score = score
    
    # Phase 3: Local search polish
    if _elapsed() < _DEADLINE - 3 and best_rows is not None:
        polish_time = min(15, _DEADLINE - 3 - _elapsed())
        if polish_time > 0:
            polished_rows = _local_search_polish(best_rows, k, rng, time_budget=polish_time)
            if polished_rows and _is_valid(polished_rows, k):
                new_score = _max_entry(polished_rows)
                if new_score < best_score:
                    best_rows = polished_rows
                    best_score = new_score
    
    # Fallback: if no solution found, use greedy
    if best_rows is None:
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
    
    # Final validation
    if not _is_valid(best_rows, k):
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
    
    return best_rows
