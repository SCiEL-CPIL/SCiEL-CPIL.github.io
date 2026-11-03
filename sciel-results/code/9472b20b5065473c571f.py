"""Difference triangle set (7,5) solver with structured search and aggressive local search."""
import time
import random
import math

_DEADLINE = 145.0
_START_TIME = time.time()

def _elapsed():
    return time.time() - _START_TIME

def _row_diffs(row):
    """Return the list of all positive differences in a row."""
    diffs = []
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.append(row[j] - row[i])
    return diffs

def _row_diffs_set(row):
    """Return the set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _max_entry(rows):
    return max(max(r) for r in rows) if rows else 0

def _sum_entries(rows):
    return sum(sum(r) for r in rows)

def _is_valid(rows, k):
    """Check that all positive differences across all rows are distinct."""
    seen = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in seen:
                    return False
                seen.add(d)
    return True

def _build_row_given_used(used, k, rng, max_val=None, jump_prob=0.15, jump_max=5):
    """Build one row greedily, choosing the smallest available next mark."""
    row = [0]
    x = 0
    while len(row) < k + 1:
        x += 1
        if max_val is not None and x > max_val:
            break
        if rng.random() < jump_prob:
            x += rng.randint(1, jump_max)
            if max_val is not None and x > max_val:
                break
        new_diffs = {x - a for a in row}
        if not new_diffs & used:
            row.append(x)
            used |= new_diffs
    return row

def _initial_construction(n_rows, k, rng, jump_prob=0.15, jump_max=5):
    """Build all rows greedily with randomization."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _build_row_given_used(used, k, rng, jump_prob=jump_prob, jump_max=jump_max)
        if len(row) < k + 1:
            return None
        rows.append(row)
    return rows

def _compute_all_diffs(rows):
    """Compute a dictionary mapping each difference to its count."""
    diff_counts = {}
    for row in rows:
        for d in _row_diffs(row):
            diff_counts[d] = diff_counts.get(d, 0) + 1
    return diff_counts

def _count_conflicts(diff_counts):
    """Count the number of conflict pairs."""
    c = 0
    for cnt in diff_counts.values():
        if cnt > 1:
            c += cnt - 1
    return c

def _sa_main(rows, k, rng, time_budget=60):
    """Simulated Annealing to minimize the scope (max entry)."""
    n_rows = len(rows)
    best_rows = [list(r) for r in rows]
    current_rows = [list(r) for r in best_rows]
    
    diff_counts = _compute_all_diffs(current_rows)
    row_diffs_list = [_row_diffs(row) for row in current_rows]
    
    def get_scope():
        return _max_entry(current_rows)
    
    current_scope = get_scope()
    current_sum = _sum_entries(current_rows)
    current_conflicts = _count_conflicts(diff_counts)
    
    PENALTY = 500
    SUM_WEIGHT = 0.05
    current_cost = current_scope + PENALTY * current_conflicts + SUM_WEIGHT * current_sum
    
    best_scope = current_scope
    best_conflicts = current_conflicts
    best_cost = current_cost
    
    start_time = time.time()
    temp = 20.0
    cooling = 0.9993
    
    while time.time() - start_time < time_budget:
        if _elapsed() > _DEADLINE - 5:
            break
            
        move_type = rng.random()
        
        if move_type < 0.35:
            # Rebuild a random row
            ri = rng.randrange(n_rows)
            old_row = current_rows[ri]
            old_diffs = row_diffs_list[ri]
            
            for d in old_diffs:
                old_cnt = diff_counts[d]
                if old_cnt == 2:
                    current_conflicts -= 1
                elif old_cnt > 2:
                    current_conflicts -= 1
                if old_cnt == 1:
                    del diff_counts[d]
                else:
                    diff_counts[d] = old_cnt - 1
            
            used = set(diff_counts.keys())
            max_limit = current_scope + rng.randint(0, 10)
            new_row = _build_row_given_used(used, k, rng, max_limit, jump_prob=0.15, jump_max=5)
            
            if len(new_row) < k + 1:
                for d in old_diffs:
                    old_cnt = diff_counts.get(d, 0)
                    if old_cnt == 1:
                        current_conflicts += 1
                    elif old_cnt > 1:
                        current_conflicts += 1
                    diff_counts[d] = old_cnt + 1
                continue
            
            new_diffs = _row_diffs(new_row)
            for d in new_diffs:
                old_cnt = diff_counts.get(d, 0)
                if old_cnt == 1:
                    current_conflicts += 1
                elif old_cnt > 1:
                    current_conflicts += 1
                diff_counts[d] = old_cnt + 1
            
            current_rows[ri] = new_row
            row_diffs_list[ri] = new_diffs
            
        elif move_type < 0.80:
            # Shift a single entry
            ri = rng.randrange(n_rows)
            row = current_rows[ri]
            row_len = len(row)
            if row_len < 2:
                continue
            
            idx = rng.randint(1, row_len - 1)
            old_val = row[idx]
            
            r = rng.random()
            if r < 0.6:
                new_val = old_val - rng.randint(1, 6)
            elif r < 0.85:
                new_val = old_val + rng.randint(-4, 4)
            else:
                new_val = old_val + rng.randint(-10, 10)
            
            if new_val <= 0 or new_val == old_val:
                continue
            if idx > 1 and new_val <= row[idx - 1]:
                continue
            if idx < row_len - 1 and new_val >= row[idx + 1]:
                continue
            
            old_diffs = row_diffs_list[ri]
            old_diff_set = set(old_diffs)
            
            new_row = list(row)
            new_row[idx] = new_val
            new_diffs = _row_diffs(new_row)
            new_diff_set = set(new_diffs)
            
            if len(new_diffs) != len(new_diff_set):
                continue
            
            removed = old_diff_set - new_diff_set
            added = new_diff_set - old_diff_set
            
            delta_conflict = 0
            for d in removed:
                cnt = diff_counts.get(d, 0)
                if cnt == 2:
                    delta_conflict -= 1
                elif cnt > 2:
                    delta_conflict -= 1
            for d in added:
                cnt = diff_counts.get(d, 0)
                if cnt == 1:
                    delta_conflict += 1
                elif cnt > 1:
                    delta_conflict += 1
            
            other_max = 0
            for i in range(n_rows):
                if i != ri:
                    other_max = max(other_max, current_rows[i][-1])
            new_scope = max(new_row[-1], other_max)
            
            delta_scope = new_scope - current_scope
            delta_sum = sum(new_row) - sum(row)
            delta_cost = delta_scope + PENALTY * delta_conflict + SUM_WEIGHT * delta_sum
            
            if delta_cost < 0 or (temp > 0 and rng.random() < math.exp(-delta_cost / temp)):
                # Accept
                for d in removed:
                    cnt = diff_counts[d]
                    if cnt == 2:
                        current_conflicts -= 1
                    elif cnt > 2:
                        current_conflicts -= 1
                    if cnt == 1:
                        del diff_counts[d]
                    else:
                        diff_counts[d] = cnt - 1
                for d in added:
                    old_cnt = diff_counts.get(d, 0)
                    if old_cnt == 1:
                        current_conflicts += 1
                    elif old_cnt > 1:
                        current_conflicts += 1
                    diff_counts[d] = old_cnt + 1
                
                current_rows[ri] = new_row
                row_diffs_list[ri] = new_diffs
                current_scope = new_scope
                current_sum += delta_sum
                current_conflicts += delta_conflict
                current_cost += delta_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_scope = new_scope
                    best_conflicts = current_conflicts
                    best_rows = [list(r) for r in current_rows]
            else:
                # Reject - no state change needed since we didn't modify current_rows yet
                pass
        
        else:
            # Swap two entries in different rows
            ri = rng.randrange(n_rows)
            rj = rng.randrange(n_rows)
            if ri == rj:
                continue
            
            row_i = current_rows[ri]
            row_j = current_rows[rj]
            if len(row_i) < 2 or len(row_j) < 2:
                continue
            
            idx_i = rng.randint(1, len(row_i) - 1)
            idx_j = rng.randint(1, len(row_j) - 1)
            
            val_i = row_i[idx_i]
            val_j = row_j[idx_j]
            
            # Check validity after swap
            new_row_i = list(row_i)
            new_row_j = list(row_j)
            new_row_i[idx_i] = val_j
            new_row_j[idx_j] = val_i
            
            if not all(new_row_i[p] < new_row_i[p+1] for p in range(len(new_row_i)-1)):
                continue
            if not all(new_row_j[p] < new_row_j[p+1] for p in range(len(new_row_j)-1)):
                continue
            
            old_diffs_i = row_diffs_list[ri]
            old_diffs_j = row_diffs_list[rj]
            old_diff_set_i = set(old_diffs_i)
            old_diff_set_j = set(old_diffs_j)
            
            new_diffs_i = _row_diffs(new_row_i)
            new_diffs_j = _row_diffs(new_row_j)
            new_diff_set_i = set(new_diffs_i)
            new_diff_set_j = set(new_diffs_j)
            
            if len(new_diffs_i) != len(new_diff_set_i) or len(new_diffs_j) != len(new_diff_set_j):
                continue
            
            removed_i = old_diff_set_i - new_diff_set_i
            added_i = new_diff_set_i - old_diff_set_i
            removed_j = old_diff_set_j - new_diff_set_j
            added_j = new_diff_set_j - old_diff_set_j
            
            delta_conflict = 0
            for d in removed_i:
                cnt = diff_counts.get(d, 0)
                if cnt == 2:
                    delta_conflict -= 1
                elif cnt > 2:
                    delta_conflict -= 1
            for d in added_i:
                cnt = diff_counts.get(d, 0)
                if cnt == 1:
                    delta_conflict += 1
                elif cnt > 1:
                    delta_conflict += 1
            for d in removed_j:
                cnt = diff_counts.get(d, 0)
                if cnt == 2:
                    delta_conflict -= 1
                elif cnt > 2:
                    delta_conflict -= 1
            for d in added_j:
                cnt = diff_counts.get(d, 0)
                if cnt == 1:
                    delta_conflict += 1
                elif cnt > 1:
                    delta_conflict += 1
            
            new_scope = max(new_row_i[-1], new_row_j[-1], 
                           max(current_rows[m][-1] for m in range(n_rows) if m != ri and m != rj))
            delta_scope = new_scope - current_scope
            delta_sum = (sum(new_row_i) - sum(row_i)) + (sum(new_row_j) - sum(row_j))
            delta_cost = delta_scope + PENALTY * delta_conflict + SUM_WEIGHT * delta_sum
            
            if delta_cost < 0 or (temp > 0 and rng.random() < math.exp(-delta_cost / temp)):
                # Accept
                for d in removed_i:
                    cnt = diff_counts[d]
                    if cnt == 2:
                        current_conflicts -= 1
                    elif cnt > 2:
                        current_conflicts -= 1
                    if cnt == 1:
                        del diff_counts[d]
                    else:
                        diff_counts[d] = cnt - 1
                for d in added_i:
                    old_cnt = diff_counts.get(d, 0)
                    if old_cnt == 1:
                        current_conflicts += 1
                    elif old_cnt > 1:
                        current_conflicts += 1
                    diff_counts[d] = old_cnt + 1
                for d in removed_j:
                    cnt = diff_counts[d]
                    if cnt == 2:
                        current_conflicts -= 1
                    elif cnt > 2:
                        current_conflicts -= 1
                    if cnt == 1:
                        del diff_counts[d]
                    else:
                        diff_counts[d] = cnt - 1
                for d in added_j:
                    old_cnt = diff_counts.get(d, 0)
                    if old_cnt == 1:
                        current_conflicts += 1
                    elif old_cnt > 1:
                        current_conflicts += 1
                    diff_counts[d] = old_cnt + 1
                
                current_rows[ri] = new_row_i
                current_rows[rj] = new_row_j
                row_diffs_list[ri] = new_diffs_i
                row_diffs_list[rj] = new_diffs_j
                current_scope = new_scope
                current_sum += delta_sum
                current_conflicts += delta_conflict
                current_cost += delta_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_scope = new_scope
                    best_conflicts = current_conflicts
                    best_rows = [list(r) for r in current_rows]
        
        temp *= cooling
        if temp < 0.01:
            temp = 0.01

    if _is_valid(best_rows, k):
        return best_rows
    else:
        if _is_valid(current_rows, k):
            return current_rows
        return best_rows

def _aggressive_polish(rows, k, rng, time_budget=5):
    """Final high-precision polish: try single-entry shifts to reduce scope."""
    n_rows = len(rows)
    best_rows = [list(r) for r in rows]
    best_scope = _max_entry(best_rows)
    best_cost = best_scope
    
    start_time = time.time()
    
    all_diffs = set()
    for row in best_rows:
        for d in _row_diffs(row):
            all_diffs.add(d)
    
    while time.time() - start_time < time_budget:
        if _elapsed() > _DEADLINE - 3:
            break
        
        improved = False
        for ri in range(n_rows):
            row = best_rows[ri]
            for idx in range(1, len(row)):
                old_val = row[idx]
                new_val = old_val - 1
                if new_val <= 0:
                    continue
                if idx > 1 and new_val <= row[idx - 1]:
                    continue
                if idx < len(row) - 1 and new_val >= row[idx + 1]:
                    continue
                
                old_diffs = _row_diffs(row)
                row[idx] = new_val
                new_diffs = _row_diffs(row)
                
                temp_diffs = set(all_diffs)
                for d in old_diffs:
                    temp_diffs.discard(d)
                conflict = False
                for d in new_diffs:
                    if d in temp_diffs:
                        conflict = True
                        break
                    temp_diffs.add(d)
                
                if not conflict:
                    all_diffs = temp_diffs
                    new_scope = _max_entry(best_rows)
                    new_cost = new_scope
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_scope = new_scope
                        improved = True
                else:
                    row[idx] = old_val
        
        if not improved:
            break
    
    if _is_valid(best_rows, k):
        return best_rows
    return [list(r) for r in rows]

def construct(n_rows, k):
    global _START_TIME
    _START_TIME = time.time()
    
    rng = random.Random(42)
    
    best_rows = None
    best_score = float('inf')
    
    # Use fewer restarts with better diversity parameters
    n_restarts = 80
    for i in range(n_restarts):
        if _elapsed() > _DEADLINE - 25:
            break
        rng2 = random.Random(12345 + i * 7919)
        # Vary jump parameters for diversity
        jump_prob = 0.10 + 0.10 * (i % 3) / 2.0
        jump_max = 3 + (i % 4)
        rows = _initial_construction(n_rows, k, rng2, jump_prob=jump_prob, jump_max=jump_max)
        if rows is None:
            continue
        if not _is_valid(rows, k):
            continue
        score = _max_entry(rows)
        if score < best_score:
            best_score = score
            best_rows = [list(r) for r in rows]
    
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
        best_score = _max_entry(best_rows)
    
    # Run SA with more time budget since we have fewer restarts
    if _elapsed() < _DEADLINE - 15:
        time_left = _DEADLINE - 15 - _elapsed()
        sa_time = min(100, time_left)
        improved_rows = _sa_main(best_rows, k, rng, time_budget=sa_time)
        
        if improved_rows and _is_valid(improved_rows, k):
            new_score = _max_entry(improved_rows)
            if new_score < best_score:
                best_rows = improved_rows
                best_score = new_score
    
    # Final polish
    if _elapsed() < _DEADLINE - 5:
        polish_time = min(8, _DEADLINE - 5 - _elapsed())
        if polish_time > 0:
            polished_rows = _aggressive_polish(best_rows, k, rng, time_budget=polish_time)
            if polished_rows and _is_valid(polished_rows, k):
                new_score = _max_entry(polished_rows)
                if new_score < best_score:
                    best_rows = polished_rows
                    best_score = new_score
    
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
