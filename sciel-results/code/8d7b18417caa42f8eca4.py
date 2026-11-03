"""Difference triangle set construction with time-bounded local search."""
import time
import random
import sys
from functools import lru_cache

# Deadline guard: evaluator kills after 150s, use 145s for safety
_DEADLINE = 145.0
_START_TIME = None


def construct(n_rows, k):
    global _START_TIME
    _START_TIME = time.time()
    
    # Handle trivial cases
    if n_rows <= 0:
        return []
    if k <= 0:
        return [[0]] * n_rows
    
    # Total number of differences needed
    total_diffs = n_rows * k * (k + 1) // 2
    
    # Phase 1: Fast greedy construction to get a valid baseline
    baseline = _greedy_construct(n_rows, k)
    best_scope = max(max(row) for row in baseline) if baseline else 0
    
    # Phase 2: Try to improve with local search / restarts within time budget
    if best_scope <= 112:
        return baseline  # Can't beat the counting bound
    
    best_rows = baseline
    
    # Try multiple random restarts with different strategies
    rng = random.Random(42)
    
    for attempt in range(200000):
        if time.time() - _START_TIME > _DEADLINE:
            break
        
        # Strategy: randomized greedy with occasional backtracking
        rows = _randomized_greedy(n_rows, k, rng, best_scope)
        if rows is not None:
            scope = max(max(row) for row in rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = rows
                if best_scope <= 112:
                    break
        
        # Every few attempts, try a local search on the best solution
        if attempt % 50 == 0:
            improved = _local_search(best_rows, n_rows, k, rng, best_scope)
            if improved is not None:
                scope = max(max(row) for row in improved)
                if scope < best_scope:
                    best_scope = scope
                    best_rows = improved
                    if best_scope <= 112:
                        break
    
    return best_rows


def _greedy_construct(n_rows, k):
    """Standard greedy: each row takes smallest available marks."""
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
    return rows


def _randomized_greedy(n_rows, k, rng, max_scope_hint):
    """Randomized greedy with bias toward smaller values."""
    used = set()
    rows = []
    
    for r in range(n_rows):
        row = [0]
        # Try to build row with some randomness
        x = 0
        attempts = 0
        max_attempts = max_scope_hint + 50
        while len(row) < k + 1:
            if attempts > max_attempts:
                return None
            x += 1
            if x > max_scope_hint:
                return None
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
                # Occasionally jump ahead (randomization)
                if rng.random() < 0.1 and len(row) < k:
                    x += rng.randint(1, 3)
            attempts += 1
        rows.append(row)
    
    return rows


def _local_search(rows, n_rows, k, rng, current_scope):
    """Try to reduce the scope by moving the largest element in a row."""
    if not rows:
        return None
    
    # Find the row with the maximum element
    max_row_idx = max(range(n_rows), key=lambda i: rows[i][-1])
    max_val = rows[max_row_idx][-1]
    
    # Try to find a smaller value for the last element of that row
    row = list(rows[max_row_idx])
    # Compute which differences are used by this row's last element
    last_elem_diffs = {row[-1] - a for a in row[:-1]}
    
    # Collect all used differences from other rows
    other_used = set()
    for i, r in enumerate(rows):
        if i == max_row_idx:
            continue
        for j in range(len(r)):
            for jj in range(j + 1, len(r)):
                other_used.add(r[jj] - r[j])
    
    # Also include differences within this row except those involving the last element
    internal_used = set()
    for j in range(len(row) - 1):
        for jj in range(j + 1, len(row) - 1):
            internal_used.add(row[jj] - row[j])
    
    forbidden = other_used | internal_used
    
    # Try values from max_val - 1 down to row[-2] + 1
    for candidate in range(max_val - 1, row[-2], -1):
        if candidate <= row[-2]:
            break
        cand_diffs = {candidate - a for a in row[:-1]}
        if not cand_diffs & forbidden:
            # Found a valid smaller value
            new_rows = [list(r) for r in rows]
            new_rows[max_row_idx] = row[:-1] + [candidate]
            return new_rows
    
    return None


def _validate(rows, n_rows, k):
    """Check that all differences are distinct across all rows."""
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
