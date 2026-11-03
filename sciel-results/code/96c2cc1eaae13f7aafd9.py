"""Randomized greedy set-packing for difference triangle set."""
import random
import time
from collections import defaultdict

def _generate_sidon_rows(k, limit):
    """Generate all rows of length k+1 starting at 0, strictly increasing,
    with max entry <= limit, whose internal differences are all distinct (Sidon)."""
    rows = []
    
    def backtrack(current, diffs):
        if len(current) == k + 1:
            rows.append(list(current))
            return
        start = current[-1] + 1
        remaining = k + 1 - len(current)
        # Prune: need 'remaining' more elements, last must be <= limit
        # Minimum possible last: current[-1] + remaining
        if current[-1] + remaining > limit:
            return
        for x in range(start, limit + 1):
            # Check new differences
            ok = True
            new_diffs = []
            for a in current:
                d = x - a
                if d in diffs:
                    ok = False
                    break
                new_diffs.append(d)
            if not ok:
                continue
            # Also check new_diffs among themselves (but they're all distinct since a's are distinct and x is fixed)
            current.append(x)
            diffs_new = diffs | set(new_diffs)
            backtrack(current, diffs_new)
            current.pop()
    
    backtrack([0], set())
    return rows

def construct(n_rows, k):
    limit = 130
    start_time = time.time()
    time_budget = 110
    
    # Generate all valid Sidon rows
    all_rows = _generate_sidon_rows(k, limit)
    
    # Compute difference sets
    row_info = []
    for row in all_rows:
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        row_info.append((row, diffs, row[-1]))
    
    # Sort by max entry
    row_info.sort(key=lambda x: (x[2], x[0]))
    
    # Group by scope
    scope_groups = defaultdict(list)
    for row, diffs, max_e in row_info:
        scope_groups[max_e].append((row, diffs))
    scopes = sorted(scope_groups.keys())
    
    best_solution = None
    best_score = float('inf')
    
    # Try multiple randomized trials
    num_trials = 100
    
    for trial in range(num_trials):
        if time.time() - start_time > time_budget:
            break
        
        rng = random.Random(trial * 7919 + 3)
        
        # Backtracking search: pick n_rows rows with disjoint diffs, minimizing max scope
        selected = []
        used_diffs = set()
        
        def search():
            if len(selected) == n_rows:
                return True
            if time.time() - start_time > time_budget:
                return False
            
            # Try each scope level
            for scope in scopes:
                if time.time() - start_time > time_budget:
                    return False
                candidates = scope_groups[scope]
                # Shuffle for randomization
                shuffled = candidates[:]
                rng.shuffle(shuffled)
                
                for (row, diffs) in shuffled:
                    if used_diffs & diffs:
                        continue
                    # Select this row
                    selected.append(row)
                    used_diffs.update(diffs)
                    
                    if search():
                        return True
                    
                    # Backtrack
                    used_diffs -= diffs
                    selected.pop()
                
                # If we've tried all candidates at this scope and none led to a solution,
                # move to next scope. But note: we must pick exactly one row per "level" of selection,
                # not one per scope. The above loop tries all rows at this scope; if none work,
                # we continue to the next scope. This is correct: we're trying to pick the next row
                # from any scope, and we try scopes in order.
            
            return False
        
        if search():
            max_entry = max(max(r) for r in selected)
            if max_entry < best_score:
                best_score = max_entry
                best_solution = selected[:]
                if best_score <= 105:
                    break
    
    if best_solution is None:
        # Fallback: simple greedy
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
    
    return best_solution
