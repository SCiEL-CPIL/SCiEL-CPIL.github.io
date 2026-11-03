import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140
    
    # Get greedy baseline
    best = _greedy(n_rows, k)
    best_scope = max(max(row) for row in best)
    
    num_diffs_per_row = (k + 1) * k // 2
    lower_bound = n_rows * num_diffs_per_row
    
    # Try backtracking for M from lower_bound to best_scope - 1
    for M in range(lower_bound, min(best_scope, lower_bound + 10)):
        if time.time() > deadline - 30:
            break
        time_limit = min(deadline - 30, time.time() + 15)
        result = _backtrack_search(n_rows, k, M, time_limit)
        if result is not None:
            best = result
            best_scope = M
            break
    
    # Local search to try to reduce scope further
    for _ in range(500):
        if time.time() > deadline - 5:
            break
        improved = False
        for i in range(n_rows):
            new_row = _improve_row(i, best, n_rows, k, deadline)
            if new_row is not None:
                if max(new_row) < max(best[i]):
                    best[i] = new_row
                    best_scope = max(max(row) for row in best)
                    improved = True
                    break
        if not improved:
            # Randomized restart
            new_best = _randomized_greedy(n_rows, k, deadline)
            if new_best is not None:
                new_scope = max(max(row) for row in new_best)
                if new_scope < best_scope:
                    best = new_best
                    best_scope = new_scope
    
    return best


def _greedy(n_rows, k):
    used, rows = set(), []
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


def _randomized_greedy(n_rows, k, deadline):
    used, rows = set(), []
    for _ in range(n_rows):
        if time.time() > deadline:
            return None
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            if x > 200:
                return None
            new = {x - a for a in row}
            if not new & used:
                if random.random() < 0.02:
                    continue
                row.append(x)
                used |= new
    return rows


def _backtrack_search(n_rows, k, M, time_limit):
    used_diffs = set()
    rows = []
    
    def build_row(pos, current_row):
        if time.time() > time_limit:
            return None
        if pos == k + 1:
            return current_row[:]
        if pos == 0:
            return build_row(1, [0])
        
        prev = current_row[-1]
        for val in range(prev + 1, M + 1):
            if time.time() > time_limit:
                return None
            
            ok = True
            new_diffs = []
            for elem in current_row:
                d = val - elem
                if d in used_diffs:
                    ok = False
                    break
                new_diffs.append(d)
            
            if not ok:
                continue
            
            remaining = k + 1 - pos - 1
            if M - val < remaining:
                continue
            
            current_row.append(val)
            for d in new_diffs:
                used_diffs.add(d)
            
            result = build_row(pos + 1, current_row)
            if result is not None:
                return result
            
            current_row.pop()
            for d in new_diffs:
                used_diffs.discard(d)
        
        return None
    
    for i in range(n_rows):
        row = build_row(0, [])
        if row is None:
            return None
        rows.append(row)
    
    return rows


def _improve_row(idx, rows, n_rows, k, deadline):
    used_diffs = set()
    for i, row in enumerate(rows):
        if i == idx:
            continue
        for j in range(len(row)):
            for l in range(j+1, len(row)):
                used_diffs.add(row[l] - row[j])
    
    old_max = max(rows[idx])
    
    for target in range(old_max - 1, 0, -1):
        if time.time() > deadline:
            return None
        result = _build_row_limited(k, target, used_diffs, deadline)
        if result is not None:
            return result
    
    return None


def _build_row_limited(k, max_val, used_diffs, deadline):
    row = [0]
    
    def bt(pos):
        if time.time() > deadline:
            return False
        if pos == k + 1:
            return True
        if pos == 0:
            return bt(1)
        
        prev = row[-1]
        # Limit candidates for speed
        for val in range(prev + 1, min(prev + 15, max_val + 1)):
            if time.time() > deadline:
                return False
            
            ok = True
            new_diffs = []
            for elem in row:
                d = val - elem
                if d in used_diffs:
                    ok = False
                    break
                new_diffs.append(d)
            
            if not ok:
                continue
            
            remaining = k + 1 - pos - 1
            if max_val - val < remaining:
                continue
            
            row.append(val)
            for d in new_diffs:
                used_diffs.add(d)
            
            if bt(pos + 1):
                return True
            
            row.pop()
            for d in new_diffs:
                used_diffs.discard(d)
        
        return False
    
    if bt(1):
        return row[:]
    return None
