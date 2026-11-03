"""Difference triangle set (7,5) solver with randomized greedy + local search."""
import time
import random
import sys

# Deadline guard: return best result before 145s (5s safety margin under 150s kill)
_DEADLINE = 145.0

def _elapsed():
    return time.time() - _START_TIME

_START_TIME = time.time()

def _row_diffs(row):
    """Return the set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _max_entry(rows):
    return max(max(r) for r in rows) if rows else 0

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

def _greedy_row(used, k, rng):
    """Build one row greedily, choosing the smallest available next mark.
    With randomization: sometimes skip a few candidates to explore variety."""
    row = [0]
    x = 0
    while len(row) < k + 1:
        x += 1
        # Occasionally jump ahead to create diversity
        if rng.random() < 0.15:
            x += rng.randint(1, 4)
        new = {x - a for a in row}
        if not new & used:
            row.append(x)
            used |= new
    return row

def _greedy_full(n_rows, k, rng):
    """Build all rows greedily with randomization."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_row(used, k, rng)
        rows.append(row)
    return rows

def _local_search(rows, k, rng, max_iters=2000):
    """Try to improve by swapping/replacing entries in rows.
    We try to reduce the maximum entry."""
    best = [list(r) for r in rows]
    best_score = _max_entry(best)
    
    for it in range(max_iters):
        if _elapsed() > _DEADLINE - 5:
            break
        if not _is_valid(best, k):
            break
        
        # Pick a random row and try to modify it
        ri = rng.randrange(len(best))
        row = list(best[ri])
        
        # Try to shift some entries down
        improved = False
        for attempt in range(50):
            new_row = list(row)
            # Pick a random position (not 0) and try to reduce it
            pos = rng.choice(range(1, len(new_row)))
            # Try reducing by a small random amount
            reduction = rng.randint(1, 3)
            new_val = new_row[pos] - reduction
            if new_val <= 0 or (pos > 1 and new_val <= new_row[pos - 1]):
                continue
            
            new_row[pos] = new_val
            # Check if still strictly increasing
            if any(new_row[i] >= new_row[i+1] for i in range(len(new_row)-1)):
                continue
            
            # Check differences
            new_diffs = _row_diffs(new_row)
            # Check against other rows
            conflict = False
            for oi, orow in enumerate(best):
                if oi == ri:
                    continue
                if new_diffs & _row_diffs(orow):
                    conflict = True
                    break
            if not conflict:
                best[ri] = new_row
                new_score = _max_entry(best)
                if new_score < best_score:
                    best_score = new_score
                improved = True
                break
        
        if not improved:
            # Try a different strategy: rebuild one row from scratch with a tighter bound
            pass
    
    return best, best_score

def _iterative_improve(rows, k, rng, time_budget=10):
    """Repeatedly apply local search with random restarts."""
    best = [list(r) for r in rows]
    best_score = _max_entry(best)
    start = time.time()
    
    while time.time() - start < time_budget:
        if _elapsed() > _DEADLINE - 5:
            break
        
        # Random perturbation: rebuild one random row
        candidate = [list(r) for r in best]
        ri = rng.randrange(len(candidate))
        # Rebuild row ri with a randomized greedy, but constrained to be <= best_score
        used = set()
        for oi, orow in enumerate(candidate):
            if oi != ri:
                used |= _row_diffs(orow)
        
        # Try to build row ri with max entry <= best_score
        row = [0]
        x = 0
        max_x = best_score
        success = False
        while len(row) < k + 1 and x < max_x:
            x += 1
            if rng.random() < 0.1:
                x += rng.randint(0, 2)
            if x > max_x:
                break
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        if len(row) == k + 1:
            candidate[ri] = row
            if _is_valid(candidate, k):
                score = _max_entry(candidate)
                if score < best_score:
                    best = candidate
                    best_score = score
        
        # Also try local search on current best
        ls_result, ls_score = _local_search(best, k, rng, max_iters=500)
        if ls_score < best_score:
            best = ls_result
            best_score = ls_score
    
    return best, best_score

def construct(n_rows, k):
    global _START_TIME
    _START_TIME = time.time()
    
    rng = random.Random(42)
    
    # Phase 1: Multiple randomized greedy restarts
    best_rows = None
    best_score = float('inf')
    
    n_restarts = 50
    for i in range(n_restarts):
        if _elapsed() > _DEADLINE - 20:
            break
        rng2 = random.Random(12345 + i * 7919)
        rows = _greedy_full(n_rows, k, rng2)
        if not _is_valid(rows, k):
            continue
        score = _max_entry(rows)
        if score < best_score:
            best_score = score
            best_rows = [list(r) for r in rows]
    
    if best_rows is None:
        # Fallback: pure greedy
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
    
    # Phase 2: Local search / iterative improvement
    if _elapsed() < _DEADLINE - 15:
        time_left = min(60, _DEADLINE - 15 - _elapsed())
        best_rows, best_score = _iterative_improve(best_rows, k, rng, time_budget=time_left)
    
    # Final validation
    if not _is_valid(best_rows, k):
        # Shouldn't happen, but fallback
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
