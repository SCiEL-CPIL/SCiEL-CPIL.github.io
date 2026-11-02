"""Difference triangle set (4,7) — optimized backtracking with local search."""
import time
import random
import sys

# EVOLVE-BLOCK-START


def _all_diffs(row):
    """Return set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs


def _max_entry(rows):
    return max(max(r) for r in rows)


def _is_valid(rows):
    """Check if all differences across all rows are distinct."""
    seen = set()
    for row in rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d in seen:
                    return False
                seen.add(d)
    return True


def _greedy_build(n_rows, k, rng, bias=1.0):
    """Greedy build with optional randomness."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                if bias >= 1.0 or rng.random() < bias:
                    row.append(x)
                    used |= new
                else:
                    if rng.random() > 0.3:
                        row.append(x)
                        used |= new
        rows.append(row)
    return rows


def _try_build_row(used_diffs, k, rng, greedy_bias=0.9, max_x=200):
    """
    Try to build a single row of length k+1 starting at 0,
    avoiding differences in used_diffs.
    Returns (row, new_diffs) or (None, None) if failed.
    """
    row = [0]
    x = 0
    while len(row) < k + 1:
        x += 1
        if x > max_x:
            return None, None
        new = {x - a for a in row}
        if new & used_diffs:
            continue
        if rng.random() < greedy_bias:
            row.append(x)
        # else skip this value (adds diversity)
    
    if len(row) != k + 1:
        return None, None
    
    # Compute all diffs for this row
    diffs = set()
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            diffs.add(row[j] - row[i])
    
    return row, diffs


def _construct_backtrack(n_rows, k, deadline, rng):
    """
    Backtracking: build rows one at a time, with multiple attempts per row.
    Uses MRV-like heuristic: try to keep rows compact.
    """
    used_diffs = set()
    rows = []
    
    def solve(idx):
        if time.time() > deadline:
            return None
        if idx == n_rows:
            return rows[:]
        
        # Try multiple variants for this row
        best_row = None
        best_diffs = None
        best_max = float('inf')
        
        n_attempts = 30
        for attempt in range(n_attempts):
            if time.time() > deadline:
                return None
            
            # Vary the bias for diversity
            if attempt == 0:
                bias = 1.0  # Pure greedy
            else:
                bias = 0.7 + 0.3 * rng.random()
            
            row, diffs = _try_build_row(used_diffs, k, rng, greedy_bias=bias)
            if row is not None:
                mx = row[-1]
                if mx < best_max:
                    best_max = mx
                    best_row = row
                    best_diffs = diffs
                    if mx < 40:  # Good enough for early rows
                        break
        
        if best_row is None:
            return None
        
        rows.append(best_row)
        used_diffs |= best_diffs
        
        result = solve(idx + 1)
        if result is not None:
            return result
        
        # Backtrack
        used_diffs -= best_diffs
        rows.pop()
        return None
    
    return solve(0)


def _local_search(rows, deadline, rng):
    """
    Local search: try to reduce the maximum entry by rebuilding individual rows.
    Uses simulated annealing-like acceptance.
    """
    best = [r[:] for r in rows]
    best_score = _max_entry(best)
    current_score = best_score
    n_rows = len(best)
    k = len(best[0]) - 1
    
    # Precompute diffs for each row
    row_diffs = [_all_diffs(r) for r in best]
    
    def rebuild_row(row_idx, used_others, rng, bias=0.85):
        """Rebuild a single row avoiding used_others differences."""
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            if x > 200:
                return None
            new = {x - a for a in row}
            if new & used_others:
                continue
            if rng.random() < bias:
                row.append(x)
        
        if len(row) != k + 1:
            return None
        return row
    
    while time.time() < deadline:
        # Pick a random row to modify
        row_idx = rng.randrange(n_rows)
        
        # Compute used differences from all other rows
        used_others = set()
        for i in range(n_rows):
            if i != row_idx:
                used_others |= row_diffs[i]
        
        # Try to rebuild this row
        new_row = rebuild_row(row_idx, used_others, rng, bias=rng.choice([1.0, 0.9, 0.85, 0.8]))
        
        if new_row is not None:
            # Check full validity
            trial_rows = [r[:] for r in best]
            trial_rows[row_idx] = new_row
            if _is_valid(trial_rows):
                sc = _max_entry(trial_rows)
                if sc < current_score:
                    current_score = sc
                    best = trial_rows
                    row_diffs = [_all_diffs(r) for r in best]
                    if sc < best_score:
                        best_score = sc
                # Simulated annealing: accept worse solutions with some probability
                elif rng.random() < 0.05 * (1.0 + (best_score - sc) / 10.0):
                    best = trial_rows
                    row_diffs = [_all_diffs(r) for r in best]
                    current_score = sc
        
        # Occasionally try a pure greedy rebuild
        if rng.random() < 0.3:
            new_row = rebuild_row(row_idx, used_others, rng, bias=1.0)
            if new_row is not None:
                trial_rows = [r[:] for r in best]
                trial_rows[row_idx] = new_row
                if _is_valid(trial_rows):
                    sc = _max_entry(trial_rows)
                    if sc < current_score:
                        current_score = sc
                        best = trial_rows
                        row_diffs = [_all_diffs(r) for r in best]
                        if sc < best_score:
                            best_score = sc
    
    return best


def construct(n_rows, k):
    deadline = time.time() + 115.0  # Safety margin before 120s limit
    rng = random.Random(42)
    
    # Phase 1: Greedy baseline with multiple seeds
    best = _greedy_build(n_rows, k, random.Random(0), bias=1.0)
    best_score = _max_entry(best)
    
    # Try multiple greedy variants
    for seed in range(1, 30):
        if time.time() > deadline:
            break
        trial = _greedy_build(n_rows, k, random.Random(seed), bias=rng.choice([1.0, 0.95, 0.9, 0.85]))
        sc = _max_entry(trial)
        if sc < best_score:
            best_score = sc
            best = trial
    
    # Phase 2: Backtracking construction
    if time.time() < deadline - 20.0:
        bt_result = _construct_backtrack(n_rows, k, deadline - 15.0, rng)
        if bt_result is not None:
            sc = _max_entry(bt_result)
            if sc < best_score:
                best_score = sc
                best = bt_result
    
    # Phase 3: Local search to improve
    if time.time() < deadline - 10.0:
        best = _local_search(best, deadline - 5.0, rng)
        best_score = _max_entry(best)
    
    # Phase 4: If time remains, try more restarts with local search
    while time.time() < deadline - 15.0:
        trial = _greedy_build(n_rows, k, rng, bias=rng.choice([1.0, 0.9, 0.85, 0.8]))
        sc = _max_entry(trial)
        if sc < best_score:
            best_score = sc
            best = trial
            if time.time() < deadline - 10.0:
                best = _local_search(best, min(deadline - 5.0, time.time() + 25.0), rng)
                best_score = _max_entry(best)
    
    return best

# EVOLVE-BLOCK-END
