"""Difference triangle set (4,7): construct 4 rows of 8 increasing integers starting at 0
with all pairwise differences distinct across all rows, minimizing the largest entry."""
import time
import random
import sys

def construct(n_rows, k):
    _start = time.time()
    _deadline = 140.0  # safety margin before 150s kill
    
    def _time_left():
        return _deadline - (time.time() - _start)
    
    def _compute_diffs(row):
        """Return list of all positive differences in a row."""
        n = len(row)
        diffs = []
        for i in range(n):
            for j in range(i + 1, n):
                diffs.append(row[j] - row[i])
        return diffs
    
    def _is_valid(rows):
        """Check all differences across all rows are distinct."""
        seen = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in seen:
                        return False
                    seen.add(d)
        return True
    
    def _max_entry(rows):
        return max(max(row) for row in rows)
    
    def _greedy_build(n_rows, k, seed=0, strategy='sequential'):
        """Build rows greedily. Returns list of rows."""
        rng = random.Random(seed)
        used = set()
        rows = []
        
        if strategy == 'sequential':
            for r in range(n_rows):
                row = [0]
                x = 0
                while len(row) < k + 1:
                    x += 1
                    new_diffs = set(x - a for a in row)
                    if not (new_diffs & used):
                        row.append(x)
                        used |= new_diffs
                rows.append(row)
        elif strategy == 'round_robin':
            # Build all rows in parallel, one element at a time
            rows = [[0] for _ in range(n_rows)]
            x = 0
            while any(len(row) < k + 1 for row in rows):
                x += 1
                for r in range(n_rows):
                    if len(rows[r]) >= k + 1:
                        continue
                    new_diffs = set(x - a for a in rows[r])
                    if not (new_diffs & used):
                        rows[r].append(x)
                        used |= new_diffs
                        break  # Only add to one row per x value to keep them distinct? 
                        # Actually we can add to multiple rows if no conflict
            # This doesn't work well because x is shared. Let's use per-row counters.
            rows = [[0] for _ in range(n_rows)]
            counters = [0] * n_rows
            while any(len(row) < k + 1 for row in rows):
                progressed = False
                for r in range(n_rows):
                    if len(rows[r]) >= k + 1:
                        continue
                    x = counters[r]
                    new_diffs = set(x - a for a in rows[r])
                    if not (new_diffs & used):
                        rows[r].append(x)
                        used |= new_diffs
                        counters[r] += 1
                        progressed = True
                if not progressed:
                    # Increment all counters
                    for r in range(n_rows):
                        if len(rows[r]) < k + 1:
                            counters[r] += 1
            return rows
        elif strategy == 'random_skip':
            for r in range(n_rows):
                row = [0]
                x = 0
                while len(row) < k + 1:
                    x += rng.randint(1, 3)
                    new_diffs = set(x - a for a in row)
                    if not (new_diffs & used):
                        row.append(x)
                        used |= new_diffs
                rows.append(row)
        return rows
    
    def _backtrack_row(used_diffs, k, max_val, seed=0):
        """Try to find a row of k+1 elements starting at 0, max <= max_val,
        with no conflicts with used_diffs."""
        rng = random.Random(seed)
        
        def backtrack(row, used, remaining_max):
            if len(row) == k + 1:
                return row[:]
            
            last = row[-1]
            # Try values from last+1 to remaining_max
            # Randomize order sometimes
            candidates = list(range(last + 1, remaining_max + 1))
            if rng.random() < 0.5:
                rng.shuffle(candidates)
            
            for x in candidates:
                new_diffs = set(x - a for a in row)
                if new_diffs & used:
                    continue
                row.append(x)
                result = backtrack(row, used | new_diffs, remaining_max)
                if result is not None:
                    return result
                row.pop()
            return None
        
        return backtrack([0], set(used_diffs), max_val)
    
    def _targeted_descent(rows, n_rows, k, time_limit):
        """Iteratively try to reduce the maximum entry by rebuilding the row
        containing the maximum element."""
        rng = random.Random()
        best_rows = [list(r) for r in rows]
        best_max = _max_entry(best_rows)
        start_time = time.time()
        
        while time.time() - start_time < time_limit and _time_left() > 5:
            # Find row with largest max entry
            row_maxes = [max(row) for row in best_rows]
            r = row_maxes.index(max(row_maxes))
            current_max = row_maxes[r]
            
            if current_max <= k * (k + 1) // 2:  # Theoretical minimum for one row
                break
            
            # Try to rebuild row r with max <= current_max - 1
            other_rows = [best_rows[i] for i in range(n_rows) if i != r]
            used = set()
            for row in other_rows:
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        used.add(row[j] - row[i])
            
            target_max = current_max - 1
            # Try a few random seeds for backtracking
            found = False
            for attempt in range(50):
                if time.time() - start_time > time_limit or _time_left() < 5:
                    break
                new_row = _backtrack_row(used, k, target_max, seed=attempt)
                if new_row is not None:
                    new_rows = [best_rows[i] for i in range(n_rows)]
                    new_rows[r] = new_row
                    if _is_valid(new_rows):
                        best_rows = new_rows
                        best_max = _max_entry(best_rows)
                        found = True
                        break
            
            if not found:
                # Try a more aggressive approach: random perturbation of all rows
                # then local search
                if _time_left() < 10:
                    break
                for _ in range(20):
                    if _time_left() < 5:
                        break
                    # Perturb a random element
                    r2 = rng.randint(0, n_rows - 1)
                    pos = rng.randint(1, k)
                    old_val = best_rows[r2][pos]
                    lo = best_rows[r2][pos - 1] + 1
                    hi = old_val - 1 if pos < k else old_val
                    if lo > hi:
                        continue
                    new_val = rng.randint(lo, hi)
                    if pos < k and new_val >= best_rows[r2][pos + 1]:
                        continue
                    if pos > 1 and new_val <= best_rows[r2][pos - 1]:
                        continue
                    best_rows[r2][pos] = new_val
                    if _is_valid(best_rows):
                        cur_max = _max_entry(best_rows)
                        if cur_max < best_max:
                            best_max = cur_max
                        # Accept if valid, even if not improved (simulated annealing style)
                    else:
                        best_rows[r2][pos] = old_val
        
        return best_rows, best_max
    
    def _multi_start(n_rows, k, time_limit):
        """Try multiple initial constructions and pick the best."""
        rng = random.Random()
        best_rows = None
        best_max = float('inf')
        start_time = time.time()
        
        strategies = ['sequential', 'random_skip']
        seeds = list(range(20))
        
        for strategy in strategies:
            for seed in seeds:
                if time.time() - start_time > time_limit or _time_left() < 10:
                    break
                rows = _greedy_build(n_rows, k, seed=seed, strategy=strategy)
                if rows is None:
                    continue
                if not _is_valid(rows):
                    continue
                cur_max = _max_entry(rows)
                if cur_max < best_max:
                    best_max = cur_max
                    best_rows = [list(r) for r in rows]
        
        return best_rows, best_max
    
    # Phase 1: Multi-start greedy
    initial_rows, initial_max = _multi_start(n_rows, k, 10)
    
    if initial_rows is None:
        # Fallback: simple sequential greedy
        initial_rows = _greedy_build(n_rows, k, seed=42, strategy='sequential')
        initial_max = _max_entry(initial_rows)
    
    # Phase 2: Targeted descent to reduce max entry
    remaining = _time_left() - 5
    if remaining > 1:
        best_rows, best_max = _targeted_descent(initial_rows, n_rows, k, remaining)
    else:
        best_rows = initial_rows
        best_max = initial_max
    
    return best_rows

# Quick self-test
if __name__ == "__main__":
    rows = construct(4, 7)
    print("Rows:", rows)
    print("Max entry:", max(max(r) for r in rows))
    # Verify validity
    diffs = set()
    valid = True
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in diffs:
                    valid = False
                    print(f"Duplicate difference {d}")
                diffs.add(d)
    print("Valid:", valid)
    print("Num differences:", len(diffs))
