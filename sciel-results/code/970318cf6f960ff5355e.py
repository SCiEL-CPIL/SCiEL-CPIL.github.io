import time
import random
import math
import numpy as np
from collections import Counter

_DEADLINE = 0.0
_BEST = None

def _scope(rows):
    return max(a for r in rows for a in r[1:])

def _valid(rows, k):
    seen = set()
    for r in rows:
        if r[0] != 0 or len(r) != k + 1:
            return False
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                if d <= 0 or d in seen:
                    return False
                seen.add(d)
    return True

def _fast_greedy(n_rows, k, scope, rng):
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        for pos in range(1, k + 1):
            found = False
            for v in range(row[-1] + 1, scope + 1):
                ok = True
                for a in row:
                    if v - a in used:
                        ok = False
                        break
                if ok:
                    row.append(v)
                    for a in row[:-1]:
                        used.add(v - a)
                    found = True
                    break
            if not found:
                return None
        rows.append(row)
    return rows

def _build_diff_counts(matrix, k):
    n_rows = matrix.shape[0]
    total_diffs = k * (k + 1) // 2
    diff_counts = np.zeros(total_diffs + 1, dtype=np.int32)
    for i in range(n_rows):
        row = matrix[i]
        for j in range(k + 1):
            for l in range(j):
                d = row[j] - row[l]
                if 0 <= d <= total_diffs:
                    diff_counts[d] += 1
    return diff_counts

def _remove_row_diffs(diff_counts, row, k):
    for j in range(k + 1):
        for l in range(j):
            d = row[j] - row[l]
            if 0 <= d <= len(diff_counts) - 1:
                diff_counts[d] -= 1

def _add_row_diffs(diff_counts, row, k):
    for j in range(k + 1):
        for l in range(j):
            d = row[j] - row[l]
            if 0 <= d <= len(diff_counts) - 1:
                diff_counts[d] += 1

def _row_rebuild(used_diffs, k, scope, rng, strategy='tight'):
    row = [0]
    local_diffs = set()
    for pos in range(1, k + 1):
        min_val = row[-1] + 1
        found = False
        if strategy == 'tight':
            candidates = range(min_val, scope + 1)
        else:
            candidates = list(range(min_val, scope + 1))
            rng.shuffle(candidates)
        
        for v in candidates:
            ok = True
            new_diffs = []
            for a in row:
                d = v - a
                if d in used_diffs or d in local_diffs:
                    ok = False
                    break
                new_diffs.append(d)
            if ok:
                row.append(v)
                for d in new_diffs:
                    local_diffs.add(d)
                found = True
                break
        if not found:
            return None
    return row

def _sa_vectorized(matrix, k, time_limit, rng):
    n_rows = matrix.shape[0]
    total_diffs = k * (k + 1) // 2
    diff_counts = _build_diff_counts(matrix, k)
    initial_conflicts = np.sum(diff_counts > 1)
    current_conflicts = initial_conflicts
    best_conflicts = initial_conflicts
    best_matrix = matrix.copy()
    
    start_time = time.time()
    t = 1.0
    cooling = 0.9995
    min_t = 0.01
    
    while time.time() < time_limit and t > min_t:
        if rng.random() < 0.3:
            i = rng.randint(0, n_rows - 1)
            j = rng.randint(0, n_rows - 1)
            if i == j:
                continue
            old_i = matrix[i].copy()
            old_j = matrix[j].copy()
            matrix[i], matrix[j] = matrix[j], matrix[i]
            
            _remove_row_diffs(diff_counts, old_i, k)
            _remove_row_diffs(diff_counts, old_j, k)
            _add_row_diffs(diff_counts, matrix[i], k)
            _add_row_diffs(diff_counts, matrix[j], k)
            
            new_conflicts = np.sum(diff_counts > 1)
            delta = new_conflicts - current_conflicts
            
            if delta < 0 or rng.random() < math.exp(-delta / t):
                current_conflicts = new_conflicts
                if current_conflicts < best_conflicts:
                    best_conflicts = current_conflicts
                    best_matrix = matrix.copy()
            else:
                matrix[i], matrix[j] = matrix[j], matrix[i]
                _remove_row_diffs(diff_counts, matrix[i], k)
                _remove_row_diffs(diff_counts, matrix[j], k)
                _add_row_diffs(diff_counts, old_i, k)
                _add_row_diffs(diff_counts, old_j, k)
        else:
            i = rng.randint(0, n_rows - 1)
            row = matrix[i]
            old_row = row.copy()
            _remove_row_diffs(diff_counts, old_row, k)
            
            # Re-greedy with some randomness
            new_row = [0]
            local_diffs = set()
            for pos in range(1, k + 1):
                min_val = new_row[-1] + 1
                found = False
                # Random window
                window_size = 20 + rng.randint(0, 20)
                end = min(old_row[-1] + 50, min_val + window_size)
                candidates = list(range(min_val, end + 1))
                rng.shuffle(candidates)
                
                for v in candidates:
                    ok = True
                    for a in new_row:
                        d = v - a
                        if d in local_diffs:
                            ok = False
                            break
                    if ok:
                        new_row.append(v)
                        for a in new_row[:-1]:
                            local_diffs.add(v - a)
                        found = True
                        break
                if not found:
                    # Fallback to tight
                    for v in range(min_val, old_row[-1] + 50):
                        ok = True
                        for a in new_row:
                            d = v - a
                            if d in local_diffs:
                                ok = False
                                break
                        if ok:
                            new_row.append(v)
                            for a in new_row[:-1]:
                                local_diffs.add(v - a)
                            found = True
                            break
            if not found or len(new_row) < k + 1:
                matrix[i] = old_row
                _add_row_diffs(diff_counts, old_row, k)
                continue
            
            matrix[i] = new_row
            _add_row_diffs(diff_counts, new_row, k)
            
            new_conflicts = np.sum(diff_counts > 1)
            delta = new_conflicts - current_conflicts
            
            if delta < 0 or rng.random() < math.exp(-delta / t):
                current_conflicts = new_conflicts
                if current_conflicts < best_conflicts:
                    best_conflicts = current_conflicts
                    best_matrix = matrix.copy()
            else:
                matrix[i] = old_row
                _remove_row_diffs(diff_counts, new_row, k)
                _add_row_diffs(diff_counts, old_row, k)
                
        t *= cooling
    return best_matrix, best_conflicts

def construct(n_rows, k):
    global _BEST, _DEADLINE
    _BEST = None
    _DEADLINE = time.time() + 140.0
    
    rng = random.Random(42)
    total_diffs = n_rows * k * (k + 1) // 2
    lb = int(math.ceil(math.sqrt(2 * total_diffs)))
    
    # 1. Initial greedy
    initial_scope = lb * 2 + 20
    rows = _fast_greedy(n_rows, k, initial_scope, rng)
    if rows is None:
        rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                ok = True
                for a in row:
                    if x - a in used:
                        ok = False
                        break
                if ok:
                    row.append(x)
                    for a in row[:-1]:
                        used.add(x - a)
            rows.append(row)
            
    matrix = np.array(rows, dtype=np.int64)
    current_scope = _scope(rows)
    _BEST = (current_scope, [list(r) for r in rows])
    
    # 2. SA Phase
    sa_deadline = time.time() + 60
    best_matrix, best_conflicts = _sa_vectorized(matrix, k, sa_deadline, rng)
    
    if best_conflicts == 0:
        new_scope = _scope(best_matrix)
        if new_scope < current_scope:
            current_scope = new_scope
            _BEST = (current_scope, [list(r) for r in best_matrix])
    
    # 3. Hill Climbing Phase
    hc_deadline = time.time() + 130
    improved = True
    while improved and time.time() < hc_deadline:
        improved = False
        used_diffs = set()
        for r in best_matrix:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    used_diffs.add(r[j] - r[i])
        
        for i in range(n_rows):
            if time.time() > hc_deadline:
                break
            old_row = best_matrix[i].copy()
            old_scope = _scope(best_matrix)
            
            # Rebuild row with tighter scope
            # Try to rebuild with scope = current_scope - 1
            target_scope = current_scope - 1
            new_row = _row_rebuild(used_diffs - set(d for d in (old_row[j] - old_row[l] for l in range(k+1) for j in range(l+1, k+1))), k, target_scope, rng, 'tight')
            if new_row is None:
                new_row = _row_rebuild(used_diffs - set(d for d in (old_row[j] - old_row[l] for l in range(k+1) for j in range(l+1, k+1))), k, current_scope, rng, 'random')
            
            if new_row is not None:
                new_scope = max(max(new_row), max(max(r) for r in best_matrix if r is not old_row))
                if new_scope < old_scope:
                    # Update used_diffs
                    for l in range(k+1):
                        for j in range(l+1, k+1):
                            used_diffs.remove(old_row[j] - old_row[l])
                    for l in range(k+1):
                        for j in range(l+1, k+1):
                            used_diffs.add(new_row[j] - new_row[l])
                    
                    best_matrix[i] = new_row
                    current_scope = new_scope
                    _BEST = (current_scope, [list(r) for r in best_matrix])
                    improved = True
                    
    if _BEST is not None:
        return _BEST[1]
    return [list(r) for r in best_matrix]
