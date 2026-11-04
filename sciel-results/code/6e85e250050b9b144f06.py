"""Difference triangle set (n_rows, k) — randomized greedy with backtracking and deadline guard."""
import time
import random
import sys

# EVOLVE-BLOCK-START

def _scope_of(rows):
    """Return the largest entry across all rows."""
    if not rows:
        return 0
    return max(max(r) for r in rows)

def _validate(rows, n_rows, k):
    """Check that rows form a valid difference set."""
    if len(rows) != n_rows:
        return False
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row)):
            if i > 0 and row[i] <= row[i-1]:
                return False
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True

def _greedy_row(used, k, rng, max_attempts=50):
    """Build a single row greedily with some randomization."""
    row = [0]
    x = 1
    attempts = 0
    while len(row) < k + 1:
        attempts += 1
        if attempts > max_attempts * 100:
            return None
        # Try a small window of candidates
        best_x = None
        best_conflicts = None
        for trial in range(min(20, max_attempts)):
            candidate = x + rng.randint(0, 5)
            new_diffs = {candidate - a for a in row}
            conflicts = len(new_diffs & used)
            if conflicts == 0:
                best_x = candidate
                break
            if best_conflicts is None or conflicts < best_conflicts:
                best_conflicts = conflicts
                best_x = candidate
        if best_x is None:
            return None
        # If we found a zero-conflict candidate, use it
        if best_conflicts == 0 or best_x is not None:
            # Recompute to be safe
            new_diffs = {best_x - a for a in row}
            if not new_diffs & used:
                row.append(best_x)
                used |= new_diffs
                x = best_x + 1
            else:
                # Fallback: just increment
                x += 1
                if x > 5000:
                    return None
                continue
        else:
            return None
    return row

def _greedy_greedy(n_rows, k, rng):
    """Pure greedy: build rows one at a time, smallest valid marks."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 1
        while len(row) < k + 1:
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
            x += 1
            if x > 10000:
                return None
        rows.append(row)
    return rows

def _randomized_greedy(n_rows, k, rng, jitter_range=(0, 3)):
    """Randomized greedy: at each step, pick from a small random window."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 1
        while len(row) < k + 1:
            found = False
            for _jitter in range(jitter_range[1] + 1):
                candidate = x + rng.randint(0, jitter_range[1])
                new = {candidate - a for a in row}
                if not new & used:
                    row.append(candidate)
                    used |= new
                    x = candidate + 1
                    found = True
                    break
            if not found:
                x += 1
                if x > 50000:
                    return None
        rows.append(row)
    return rows

def _backtrack_row(used, k, rng, max_depth=15, time_limit=2.0):
    """Try to build a row with backtracking when greedy gets stuck."""
    start_time = time.time()
    row = [0]
    used_local = set(used)
    
    def dfs(idx, current_x):
        if time.time() - start_time > time_limit:
            return None
        if idx == k + 1:
            return row[:]
        
        # Try candidates starting from current_x
        for x in range(current_x, current_x + 50):
            new_diffs = {x - a for a in row}
            if new_diffs & used_local:
                continue
            row.append(x)
            used_local |= new_diffs
            result = dfs(idx + 1, x + 1)
            if result is not None:
                return result
            row.pop()
            used_local -= new_diffs
        
        return None
    
    result = dfs(1, 1)
    if result is not None:
        # Update the global used set
        for i in range(len(result)):
            for j in range(i+1, len(result)):
                used.add(result[j] - result[i])
    return result

def construct(n_rows, k):
    """
    Construct a difference triangle set with n_rows rows, each of k+1 integers
    starting at 0, minimizing the largest entry.
    """
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin out of 150s
    
    best_rows = None
    best_scope = float('inf')
    
    # First try pure greedy as a baseline
    rng = random.Random(42)
    rows = _greedy_greedy(n_rows, k, rng)
    if rows is not None and _validate(rows, n_rows, k):
        scope = _scope_of(rows)
        best_rows = rows
        best_scope = scope
    
    # Try randomized greedy with various seeds and jitter levels
    for seed in range(200):
        if time.time() > deadline:
            break
        
        rng = random.Random(seed * 7919 + 13)
        
        # Vary the jitter
        if seed % 3 == 0:
            jitter = (0, 2)
        elif seed % 3 == 1:
            jitter = (0, 5)
        else:
            jitter = (0, 10)
        
        rows = _randomized_greedy(n_rows, k, rng, jitter_range=jitter)
        if rows is not None and _validate(rows, n_rows, k):
            scope = _scope_of(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = rows
                # Early exit if we hit the theoretical lower bound
                # Lower bound: need n_rows * k*(k+1)/2 distinct differences, max diff = m
                # So m*(m+1)/2 >= n_rows * k*(k+1)/2
                lb = int((2 * n_rows * k * (k+1) / 2) ** 0.5)
                if best_scope <= lb + 5:
                    break
    
    # Try a few more with different strategies if time permits
    for seed in range(200, 400):
        if time.time() > deadline:
            break
        
        rng = random.Random(seed * 104729 + 7)
        jitter = (0, rng.randint(1, 15))
        
        rows = _randomized_greedy(n_rows, k, rng, jitter_range=jitter)
        if rows is not None and _validate(rows, n_rows, k):
            scope = _scope_of(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = rows
    
    # Final safety: if we never found anything, fall back to a simple construction
    if best_rows is None:
        # Very conservative: space out marks widely
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 1
            while len(row) < k + 1:
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
                x += 1
            rows.append(row)
        best_rows = rows
    
    return best_rows

# EVOLVE-BLOCK-END
