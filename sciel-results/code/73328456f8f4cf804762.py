import time
import random
import sys

def _check_valid(rows, k):
    """Return True iff all positive differences within rows are distinct."""
    used = set()
    for row in rows:
        L = len(row)
        for i in range(L):
            for j in range(i + 1, L):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True

def _max_entry(rows):
    return max(max(r) for r in rows)

def _greedy_construct(n_rows, k, rng=None):
    """Greedy ruler construction with optional randomization."""
    if rng is None:
        rng = random.Random(0)
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            if rng.random() < 0.02:
                x += 1
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows

def _generate_candidates(k, max_scope):
    """
    Generate all valid rows (Sidon sets of size k+1 starting with 0)
    with max element <= max_scope.
    """
    candidates = []
    L = k + 1
    
    def backtrack_opt(current_row, used_diffs):
        if len(current_row) == L:
            candidates.append(list(current_row))
            return
        
        last_val = current_row[-1]
        
        for next_val in range(last_val + 1, max_scope + 1):
            # Check new differences
            conflict = False
            new_diffs = []
            for prev_val in current_row:
                d = next_val - prev_val
                if d in used_diffs:
                    conflict = True
                    break
                new_diffs.append(d)
            
            if conflict:
                continue
            
            current_row.append(next_val)
            new_used = used_diffs.union(new_diffs)
            backtrack_opt(current_row, new_used)
            current_row.pop()

    backtrack_opt([0], set())
    return candidates

def construct(n_rows, k):
    start_time = time.time()
    time_limit = start_time + 140
    
    best_rows = None
    best_max = float('inf')
    
    # Try increasing scopes
    for scope in [90, 100, 110, 120, 130, 140, 150]:
        if time.time() > time_limit - 10:
            break
            
        candidates = _generate_candidates(k, scope)
        if not candidates:
            continue
            
        # Sort candidates by max element
        candidates.sort(key=lambda r: r[-1])
        
        # Pre-compute difference sets
        diff_sets = []
        for c in candidates:
            ds = set()
            for i in range(len(c)):
                for j in range(i+1, len(c)):
                    ds.add(c[j] - c[i])
            diff_sets.append(ds)
        
        N = len(candidates)
        if N < n_rows:
            continue
        
        selected_indices = []
        
        def dfs(last_idx, selected_count):
            if time.time() > time_limit - 5:
                return False
            if selected_count == n_rows:
                return True
            
            # Pruning
            if N - last_idx - 1 < n_rows - selected_count:
                return False
            
            for j in range(last_idx + 1, N):
                compatible = True
                for s_idx in selected_indices:
                    if diff_sets[s_idx].intersection(diff_sets[j]):
                        compatible = False
                        break
                
                if compatible:
                    selected_indices.append(j)
                    if dfs(j, selected_count + 1):
                        return True
                    selected_indices.pop()
            
            return False
        
        if dfs(-1, 0):
            sol_indices = selected_indices
            best_rows = [candidates[i] for i in sol_indices]
            best_max = max(max(r) for r in best_rows)
            break
        
    if best_rows is None:
        best_rows = _greedy_construct(n_rows, k, rng=random.Random(0))
        
    return best_rows
