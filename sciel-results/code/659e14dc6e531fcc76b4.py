"""Difference triangle set (7,5) – structural rewrite: multi-restart backtracking + aggressive local search."""
import time
import random
import math
import sys

_START = time.time()
_DEADLINE = _START + 145.0


def _elapsed():
    return time.time() - _START


def _dead():
    return time.time() >= _DEADLINE


def _diffs_of_row(row):
    n = len(row)
    d = set()
    for i in range(n):
        ai = row[i]
        for j in range(i + 1, n):
            d.add(row[j] - ai)
    return d


def _scope(rows):
    return max(max(r) for r in rows)


def _valid(rows, n_rows, k):
    if len(rows) != n_rows:
        return False
    used = set()
    for r in rows:
        if len(r) != k + 1 or r[0] != 0:
            return False
        for i in range(k + 1):
            for j in range(i + 1, k + 1):
                if r[j] <= r[i]:
                    return False
        for d in _diffs_of_row(r):
            if d in used:
                return False
            used.add(d)
    return True


def _backtracking_solve(n_rows, k, max_val, seed, time_limit=8.0):
    """
    Backtracking search for a complete valid solution within max_val.
    Tries to fill all n_rows rows of k+1 elements with values in [0, max_val].
    Uses constraint propagation: for each row, tracks which differences are used globally.
    """
    rnd = random.Random(seed)
    start_time = time.time()
    
    # We'll build rows one at a time, using backtracking within each row
    rows = []
    used_diffs = set()
    
    def backtrack_row(row_idx, pos, current_row, max_val):
        """Recursively build row row_idx, currently at position pos."""
        if time.time() - start_time > time_limit:
            return None
        
        if pos == k + 1:
            return current_row[:]
        
        lower = current_row[-1] + 1 if pos > 0 else 1
        upper = max_val
        
        # Try values in randomized order for diversity
        candidates = list(range(lower, upper + 1))
        rnd.shuffle(candidates)
        
        for val in candidates:
            if time.time() - start_time > time_limit:
                return None
            
            # Check if val can be placed at this position
            new_diffs = set()
            valid = True
            for i in range(len(current_row)):
                d = val - current_row[i]
                if d <= 0:
                    valid = False
                    break
                if d in used_diffs:
                    valid = False
                    break
                if d in new_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            
            if not valid:
                continue
            
            # Place val
            old_used = set(used_diffs)
            used_diffs.update(new_diffs)
            current_row.append(val)
            
            result = backtrack_row(row_idx, pos + 1, current_row, max_val)
            if result is not None:
                return result
            
            # Undo
            used_diffs.clear()
            used_diffs.update(old_used)
            current_row.pop()
        
        return None
    
    for r in range(n_rows):
        if time.time() - start_time > time_limit:
            return None
        row_result = backtrack_row(r, 0, [], max_val)
        if row_result is None:
            return None
        rows.append(row_result)
    
    return rows


def _greedy_row_with_limits(used_diffs, k, max_val, rnd):
    """Greedy row construction with randomization and limits."""
    row = [0]
    local_diffs = set()
    
    while len(row) < k + 1:
        found = False
        # Try in randomized order from row[-1]+1 up to max_val
        start = row[-1] + 1
        candidates = list(range(start, max_val + 1))
        rnd.shuffle(candidates)
        
        for val in candidates:
            new_diffs = set()
            valid = True
            for a in row:
                d = val - a
                if d <= 0:
                    valid = False
                    break
                if d in used_diffs or d in local_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            
            if valid:
                row.append(val)
                local_diffs.update(new_diffs)
                found = True
                break
        
        if not found:
            return None
    
    return row


def _local_search_aggressive(rows, n_rows, k, time_limit=30.0):
    """
    Aggressive local search: repeatedly try to reduce every element to minimum valid value.
    Uses a global difference count array for O(1) conflict checks.
    """
    start_time = time.time()
    current_rows = [list(r) for r in rows]
    best_rows = [list(r) for r in current_rows]
    best_scope = _scope(current_rows)
    
    # Build difference count array
    max_val = max(max(r) for r in current_rows) + 20
    diff_counts = [0] * (max_val + 1)
    
    for row in current_rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d < len(diff_counts):
                    diff_counts[d] += 1
    
    def try_reduce_element(r_idx, pos):
        """Try to reduce element at position pos in row r_idx to minimum valid value."""
        nonlocal best_rows, best_scope, current_rows
        
        row = current_rows[r_idx]
        old_val = row[pos]
        lower_bound = row[pos - 1] + 1 if pos > 0 else 1
        
        if lower_bound >= old_val:
            return False
        
        # Remove old differences from counts
        old_diffs = []
        for i in range(len(row)):
            if i != pos:
                d = abs(old_val - row[i])
                if d < len(diff_counts):
                    diff_counts[d] -= 1
                old_diffs.append(d)
        
        # Try values from lower_bound upward
        for new_val in range(lower_bound, old_val):
            valid = True
            for i in range(len(row)):
                if i == pos:
                    continue
                d = abs(new_val - row[i])
                if d < len(diff_counts) and diff_counts[d] > 0:
                    valid = False
                    break
            
            if valid:
                row[pos] = new_val
                for i in range(len(row)):
                    if i != pos:
                        d = abs(new_val - row[i])
                        if d < len(diff_counts):
                            diff_counts[d] += 1
                return True
        
        # Restore old value
        row[pos] = old_val
        for d in old_diffs:
            if d < len(diff_counts):
                diff_counts[d] += 1
        return False
    
    # Multiple passes
    improved = True
    passes = 0
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        passes += 1
        
        # Process rows in random order for diversity
        row_order = list(range(n_rows))
        random.shuffle(row_order)
        
        for r_idx in row_order:
            if time.time() - start_time > time_limit:
                break
            row = current_rows[r_idx]
            
            # Try reducing from right to left (more impactful for scope)
            for pos in range(len(row) - 1, 0, -1):
                if time.time() - start_time > time_limit:
                    break
                if try_reduce_element(r_idx, pos):
                    improved = True
        
        current_scope = _scope(current_rows)
        if current_scope < best_scope:
            best_scope = current_scope
            best_rows = [list(r) for r in current_rows]
    
    return best_rows, best_scope


def _simulated_annealing_structural(rows, n_rows, k, time_budget, seed=42):
    """
    Simulated annealing with structural moves: row replacement, element shifts, and pair swaps.
    """
    rnd = random.Random(seed)
    best = [r[:] for r in rows]
    best_scope = _scope(best)
    current = [r[:] for r in best]
    current_scope = best_scope
    
    max_val = max(max(r) for r in current) + 15
    diff_counts = [0] * (max_val + 1)
    
    for row in current:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d < len(diff_counts):
                    diff_counts[d] += 1
    
    def remove_row(row):
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d < len(diff_counts):
                    diff_counts[d] -= 1
    
    def add_row(row):
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d >= len(diff_counts):
                    diff_counts.extend([0] * (d - len(diff_counts) + 1))
                diff_counts[d] += 1
    
    def remove_pos(row, pos):
        val = row[pos]
        for i in range(len(row)):
            if i != pos:
                d = abs(val - row[i])
                if d < len(diff_counts):
                    diff_counts[d] -= 1
    
    def add_pos(row, pos):
        val = row[pos]
        for i in range(len(row)):
            if i != pos:
                d = abs(val - row[i])
                if d >= len(diff_counts):
                    diff_counts.extend([0] * (d - len(diff_counts) + 1))
                diff_counts[d] += 1
    
    def check_valid_pos(row, pos, val):
        for i in range(len(row)):
            if i == pos:
                continue
            d = abs(val - row[i])
            if d < len(diff_counts) and diff_counts[d] > 0:
                return False
        return True
    
    def compute_scope(exclude_row=None):
        mx = 0
        for i, r in enumerate(current):
            if exclude_row is not None and i == exclude_row:
                continue
            if max(r) > mx:
                mx = max(r)
        return mx
    
    deadline = time.time() + time_budget
    T = 20.0
    cooling = 0.9975
    
    while time.time() < deadline and T > 0.1:
        if _dead():
            break
        
        move_type = rnd.random()
        
        if move_type < 0.45:
            # Row replacement
            ri = rnd.randint(0, n_rows - 1)
            old_row = current[ri]
            remove_row(old_row)
            
            max_mark = min(current_scope + 2, best_scope + 2)
            new_row = _greedy_row_with_limits(set(d for d, c in enumerate(diff_counts) if c > 0), k, max_mark, rnd)
            
            if new_row is None:
                add_row(old_row)
                T *= cooling
                continue
            
            # Validate new_row against current diff_counts
            valid = True
            n = len(new_row)
            for i in range(n):
                for j in range(i + 1, n):
                    d = new_row[j] - new_row[i]
                    if d < len(diff_counts) and diff_counts[d] > 0:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                add_row(old_row)
                T *= cooling
                continue
            
            add_row(new_row)
            other_max = compute_scope(ri)
            new_scope = max(other_max, max(new_row))
            delta = new_scope - current_scope
            
            if delta <= 0 or rnd.random() < math.exp(-delta / T):
                current[ri] = new_row
                current_scope = new_scope
                if new_scope < best_scope:
                    best_scope = new_scope
                    best = [r[:] for r in current]
            else:
                remove_row(new_row)
                add_row(old_row)
        
        elif move_type < 0.80:
            # Element shift down
            ri = rnd.randint(0, n_rows - 1)
            row = current[ri]
            pi = rnd.randint(1, len(row) - 1)
            old_val = row[pi]
            
            shift = rnd.choice([1, 1, 1, 2, 2, 3])
            new_val = old_val - shift
            
            if new_val <= row[pi - 1]:
                T *= cooling
                continue
            if pi < len(row) - 1 and new_val >= row[pi + 1]:
                T *= cooling
                continue
            
            remove_pos(row, pi)
            if check_valid_pos(row, pi, new_val):
                row[pi] = new_val
                add_pos(row, pi)
                
                other_max = compute_scope(ri)
                new_scope = max(other_max, max(row))
                delta = new_scope - current_scope
                
                if delta <= 0 or rnd.random() < math.exp(-delta / T):
                    current_scope = new_scope
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best = [r[:] for r in current]
                else:
                    remove_pos(row, pi)
                    row[pi] = old_val
                    add_pos(row, pi)
            else:
                row[pi] = old_val
                add_pos(row, pi)
        
        else:
            # Swap adjacent elements
            ri = rnd.randint(0, n_rows - 1)
            row = current[ri]
            if len(row) < 3:
                T *= cooling
                continue
            
            pi = rnd.randint(1, len(row) - 2)
            pj = pi + 1
            
            remove_pos(row, pi)
            remove_pos(row, pj)
            
            row[pi], row[pj] = row[pj], row[pi]
            
            valid = True
            for i in range(1, len(row)):
                if row[i] <= row[i - 1]:
                    valid = False
                    break
            
            if valid:
                # Check all diffs in this row
                n = len(row)
                for i in range(n):
                    for j in range(i + 1, n):
                        d = row[j] - row[i]
                        if d < len(diff_counts) and diff_counts[d] > 0:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                add_pos(row, pi)
                add_pos(row, pj)
                
                other_max = compute_scope(ri)
                new_scope = max(other_max, max(row))
                delta = new_scope - current_scope
                
                if delta <= 0 or rnd.random() < math.exp(-delta / T):
                    current_scope = new_scope
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best = [r[:] for r in current]
                else:
                    remove_pos(row, pi)
                    remove_pos(row, pj)
                    row[pi], row[pj] = row[pj], row[pi]
                    add_pos(row, pi)
                    add_pos(row, pj)
            else:
                row[pi], row[pj] = row[pj], row[pi]
                add_pos(row, pi)
                add_pos(row, pj)
        
        T *= cooling
    
    return best, best_scope


def _final_polish(rows, n_rows, k, time_limit=10.0):
    """Final polish: repeated minimum-element reduction."""
    start_time = time.time()
    current_rows = [list(r) for r in rows]
    
    max_val = max(max(r) for r in current_rows) + 10
    diff_counts = [0] * (max_val + 1)
    
    for row in current_rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d < len(diff_counts):
                    diff_counts[d] += 1
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for r_idx in range(n_rows):
            if time.time() - start_time > time_limit:
                break
            row = current_rows[r_idx]
            for pos in range(len(row) - 1, 0, -1):
                if time.time() - start_time > time_limit:
                    break
                
                old_val = row[pos]
                lower_bound = row[pos - 1] + 1 if pos > 0 else 1
                if lower_bound >= old_val:
                    continue
                
                old_diffs = []
                for i in range(len(row)):
                    if i != pos:
                        d = abs(old_val - row[i])
                        if d < len(diff_counts):
                            diff_counts[d] -= 1
                        old_diffs.append(d)
                
                found = False
                for new_val in range(lower_bound, old_val):
                    valid = True
                    for i in range(len(row)):
                        if i == pos:
                            continue
                        d = abs(new_val - row[i])
                        if d < len(diff_counts) and diff_counts[d] > 0:
                            valid = False
                            break
                    if valid:
                        row[pos] = new_val
                        for i in range(len(row)):
                            if i != pos:
                                d = abs(new_val - row[i])
                                if d < len(diff_counts):
                                    diff_counts[d] += 1
                        improved = True
                        found = True
                        break
                
                if not found:
                    for d in old_diffs:
                        if d < len(diff_counts):
                            diff_counts[d] += 1
    
    return current_rows


def construct(n_rows, k):
    """Construct difference triangle set with n_rows rows of k+1 increasing integers."""
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_rows = None
    best_scope = float('inf')
    
    # Phase 1: Multi-restart with backtracking + greedy
    # Try different max_val values and seeds
    seeds = list(range(20))
    random.shuffle(seeds)
    
    for seed in seeds:
        if time.time() > deadline - 80.0:
            break
        
        # Try backtracking with decreasing max_val
        for max_val in [112, 108, 105, 100, 95, 90]:
            if time.time() > deadline - 80.0:
                break
            
            rows = _backtracking_solve(n_rows, k, max_val, seed, time_limit=5.0)
            if rows is not None:
                sc = _scope(rows)
                if sc < best_scope:
                    best_scope = sc
                    best_rows = [list(r) for r in rows]
                break
        
        # Also try greedy with randomization
        if time.time() > deadline - 80.0:
            break
        
        rnd = random.Random(seed)
        used_diffs = set()
        rows = []
        success = True
        
        for r in range(n_rows):
            row = _greedy_row_with_limits(used_diffs, k, 120, rnd)
            if row is None:
                success = False
                break
            rows.append(row)
            for d in _diffs_of_row(row):
                used_diffs.add(d)
        
        if success:
            sc = _scope(rows)
            if sc < best_scope:
                best_scope = sc
                best_rows = [list(r) for r in rows]
    
    if best_rows is None:
        # Fallback: simple sequential greedy
        rows = []
        used = set()
        for _ in range(n_rows):
            row, x = [0], 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows
    
    # Phase 2: Optimization
    time_remaining = deadline - time.time()
    
    # 2a: Aggressive local search
    if time_remaining > 30.0:
        t_ls = min(35.0, time_remaining * 0.35)
        optimized, sc = _local_search_aggressive(best_rows, n_rows, k, time_limit=t_ls)
        if sc < best_scope:
            best_scope = sc
            best_rows = optimized
    
    # 2b: Simulated annealing
    time_remaining = deadline - time.time()
    if time_remaining > 25.0:
        t_sa = min(55.0, time_remaining * 0.55)
        sa_result, sa_scope = _simulated_annealing_structural(best_rows, n_rows, k, time_budget=t_sa, seed=99)
        if sa_scope < best_scope:
            best_scope = sa_scope
            best_rows = sa_result
    
    # 2c: Another local search round
    time_remaining = deadline - time.time()
    if time_remaining > 20.0:
        t_ls2 = min(20.0, time_remaining * 0.4)
        optimized2, sc2 = _local_search_aggressive(best_rows, n_rows, k, time_limit=t_ls2)
        if sc2 < best_scope:
            best_scope = sc2
            best_rows = optimized2
    
    # 2d: Final polish
    time_remaining = deadline - time.time()
    if time_remaining > 8.0:
        t_polish = min(10.0, time_remaining * 0.8)
        polished = _final_polish(best_rows, n_rows, k, time_limit=t_polish)
        sc3 = _scope(polished)
        if sc3 < best_scope:
            best_scope = sc3
            best_rows = polished
    
    return best_rows
