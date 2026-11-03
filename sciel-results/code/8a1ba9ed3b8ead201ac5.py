import time
import random
import math

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

def _greedy_construct(n_rows, k, rng, bias=1.0):
    """Fast greedy construction."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        last = 0
        for j in range(1, k + 1):
            x = last + 1
            found = False
            while x < 500:
                valid = True
                new_diffs = []
                for i in range(len(row)):
                    d = x - row[i]
                    if d in used:
                        valid = False
                        break
                    new_diffs.append(d)
                if valid:
                    row.append(x)
                    last = x
                    used.update(new_diffs)
                    found = True
                    break
                x += 1
            if not found:
                x = last + 100
                row.append(x)
                last = x
                for i in range(len(row)-1):
                    used.add(x - row[i])
        rows.append(row)
    return rows

def _local_search(rows, k, deadline, rng):
    """
    Optimized local search with incremental diff tracking.
    """
    state = [list(r) for r in rows]
    best = [list(r) for r in rows]
    best_scope = _scope(best)
    
    # Build global diff counts
    diff_counts = {}
    for r in state:
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                diff_counts[d] = diff_counts.get(d, 0) + 1
    
    T = 50.0
    cooling = 0.999
    min_T = 0.5
    n_rows = len(state)
    
    while time.time() < deadline:
        ri = rng.randint(0, n_rows - 1)
        pos = rng.randint(1, k)
        row = state[ri]
        old_val = row[pos]
        
        if rng.random() < 0.8:
            lo = row[pos-1] + 1
            hi = old_val - 1
            if hi < lo:
                T = max(T * cooling, min_T)
                continue
            new_val = rng.randint(lo, hi)
        else:
            delta = rng.randint(1, 5)
            new_val = old_val + delta
            
        if pos < k:
            if new_val >= row[pos+1]:
                T = max(T * cooling, min_T)
                continue
        else:
            if new_val > 200:
                T = max(T * cooling, min_T)
                continue
                
        if new_val == old_val:
            T = max(T * cooling, min_T)
            continue
            
        diffs_to_remove = set()
        for t in range(k + 1):
            if t == pos: continue
            d = abs(old_val - row[t])
            diffs_to_remove.add(d)
            
        diffs_to_add = set()
        for t in range(k + 1):
            if t == pos: continue
            d = abs(new_val - row[t])
            diffs_to_add.add(d)
            
        valid = True
        for d in diffs_to_add:
            if d in diffs_to_remove:
                continue
            if d in diff_counts:
                valid = False
                break
                
        if not valid:
            T = max(T * cooling, min_T)
            continue
            
        row[pos] = new_val
        
        for d in diffs_to_remove:
            diff_counts[d] -= 1
            if diff_counts[d] == 0:
                del diff_counts[d]
                
        for d in diffs_to_add:
            diff_counts[d] = diff_counts.get(d, 0) + 1
            
        new_scope = _scope(state)
        
        if new_scope < best_scope:
            best_scope = new_scope
            best = [list(r) for r in state]
            
        T = max(T * cooling, min_T)
        
    return best_scope, best

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0
    
    rng = random.Random(42)
    
    best_rows = None
    best_scope = float('inf')
    
    init_deadline = start_time + 10.0
    candidates = []
    
    for trial in range(50):
        if time.time() > init_deadline:
            break
        bias = rng.uniform(0.5, 2.0)
        rows = _greedy_construct(n_rows, k, rng, bias=bias)
        if _valid(rows, k):
            s = _scope(rows)
            candidates.append((s, rows))
            if s < best_scope:
                best_scope = s
                best_rows = [list(r) for r in rows]
                
    if not candidates:
        rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            for j in range(1, k+1):
                x = len(used) + j + 1
                while any((x - row[i]) in used for i in range(len(row))):
                    x += 1
                for i in range(len(row)):
                    used.add(x - row[i])
                row.append(x)
            rows.append(row)
        return rows
        
    candidates.sort(key=lambda x: x[0])
    
    remaining = deadline - time.time()
    top_n = min(5, len(candidates))
    time_per = remaining / max(1, top_n)
    
    for idx in range(top_n):
        if time.time() > deadline - 5:
            break
        s, rows = candidates[idx]
        ls_deadline = time.time() + time_per * 0.9
        ls_scope, ls_rows = _local_search(rows, k, ls_deadline, rng)
        
        if ls_rows is not None and _valid(ls_rows, k):
            actual = _scope(ls_rows)
            if actual < best_scope:
                best_scope = actual
                best_rows = ls_rows
    
    return best_rows
