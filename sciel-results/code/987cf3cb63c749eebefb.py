"""Rewritten solver for Difference triangle set (7,5).
Strategy: greedy construction with lookahead + local-search polish (simulated annealing)
to minimize the maximum entry (scope)."""
import time
import random
from itertools import combinations

def construct(n_rows, k):
    deadline = time.time() + 120.0
    # Phase 1: Greedy construction with a few random restarts, keep best.
    best_rows = None
    best_scope = float('inf')
    
    # Try several greedy variants (different random seeds for tie-breaking)
    for seed in range(50):
        if time.time() > deadline - 60:
            break
        rng = random.Random(seed)
        rows, scope = _greedy_with_lookahead(n_rows, k, rng)
        if scope < best_scope:
            best_scope = scope
            best_rows = [r[:] for r in rows]
    
    if best_rows is None:
        # Fallback: pure greedy
        best_rows, best_scope = _greedy_plain(n_rows, k)
    
    # Phase 2: Local search / simulated annealing to reduce scope
    if time.time() < deadline - 10:
        best_rows, best_scope = _local_search_polish(best_rows, best_scope, deadline)
    
    return best_rows


def _greedy_plain(n_rows, k):
    """Original greedy: each row picks smallest next mark with unused differences."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
        rows.append(row)
    scope = max(max(r) for r in rows)
    return rows, scope


def _greedy_with_lookahead(n_rows, k, rng):
    """Greedy with occasional random perturbation to escape bad local choices."""
    used = set()
    rows = []
    for row_idx in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                # Small chance to skip this candidate and try a larger one
                if rng.random() < 0.05:
                    skip = rng.randint(1, 3)
                    for _ in range(skip):
                        x += 1
                        new_diffs = {x - a for a in row}
                        if new_diffs & used:
                            break
                    if new_diffs & used:
                        # Revert to the valid one
                        x -= skip
                        new_diffs = {x - a for a in row}
                        if new_diffs & used:
                            # Shouldn't happen, but safety
                            continue
                row.append(x)
                used |= new_diffs
        rows.append(row)
    scope = max(max(r) for r in rows)
    return rows, scope


def _local_search_polish(rows, scope, deadline):
    """Simulated annealing to reduce the maximum entry."""
    if not rows:
        return rows, scope
    
    current = [r[:] for r in rows]
    current_scope = scope
    best = [r[:] for r in rows]
    best_scope = scope
    
    # Temperature schedule
    T_start = 50.0
    T_end = 0.1
    n_steps = 200000
    step = 0
    
    # Precompute difference sets for fast validation
    def get_used_diffs(rws):
        used = set()
        for r in rws:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    used.add(r[j] - r[i])
        return used
    
    def is_valid(rws):
        used = set()
        for r in rws:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    d = r[j] - r[i]
                    if d in used:
                        return False
                    used.add(d)
        return True
    
    def get_scope(rws):
        return max(max(r) for r in rws)
    
    while step < n_steps:
        if time.time() > deadline - 5:
            break
        
        # Annealing schedule
        progress = step / n_steps
        T = T_start * (T_end / T_start) ** progress
        
        # Choose a random row and a random position to modify
        row_idx = random.randint(0, len(current) - 1)
        row = current[row_idx]
        # Pick a position (not the first, which is 0)
        pos = random.randint(1, len(row) - 1)
        old_val = row[pos]
        
        # Propose a new value: try to reduce scope
        if old_val == current_scope and current_scope > 0:
            # Try to reduce this entry
            new_val = old_val - random.randint(1, 3)
        else:
            # Random perturbation
            delta = random.choice([-2, -1, 1, 2])
            new_val = old_val + delta
        
        if new_val <= 0 or new_val >= row[pos + 1] if pos + 1 < len(row) else False:
            step += 1
            continue
        if pos > 0 and new_val <= row[pos - 1]:
            step += 1
            continue
        if pos + 1 < len(row) and new_val >= row[pos + 1]:
            step += 1
            continue
        
        # Apply change
        row[pos] = new_val
        
        # Check validity
        if is_valid(current):
            new_scope = get_scope(current)
            delta_scope = new_scope - current_scope
            if delta_scope <= 0 or random.random() < _boltzmann(delta_scope, T):
                current_scope = new_scope
                if new_scope < best_scope:
                    best_scope = new_scope
                    best = [r[:] for r in current]
            else:
                row[pos] = old_val
        else:
            row[pos] = old_val
        
        step += 1
    
    return best, best_scope


def _boltzmann(delta, T):
    if delta <= 0:
        return 1.0
    if T <= 0:
        return 0.0
    import math
    return math.exp(-delta / T)
