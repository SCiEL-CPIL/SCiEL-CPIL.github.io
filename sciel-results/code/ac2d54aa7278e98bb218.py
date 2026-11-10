"""Difference triangle set (4,7): minimize the largest entry."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _row_differences(row):
    """Return the set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs


def _greedy_row(used, k, rng):
    """Build one row greedily: start at 0, pick smallest next mark with no conflict.
    Returns the row (list of k+1 ints) or None if stuck."""
    row = [0]
    x = 1
    while len(row) < k + 1:
        # Try x, x+1, ... until we find a mark whose differences to row are all unused
        found = False
        while True:
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used.update(new_diffs)
                found = True
                break
            x += 1
        if not found:
            return None
        x += 1
    return row


def _greedy_all_rows(n_rows, k, rng):
    """Build all rows greedily. Returns (rows, used_diffs)."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_row(used, k, rng)
        if row is None:
            return None, None
        rows.append(row)
    return rows, used


def _score(rows):
    """Return the largest entry across all rows."""
    return max(max(r) for r in rows)


def _swap_improve(rows, k, rng, max_iterations=5000):
    """Local search: try swapping the last element of one row with a smaller valid value.
    Returns improved rows or None if no improvement found."""
    best = rows
    best_score = _score(best)
    
    for iteration in range(max_iterations):
        improved = False
        for r_idx in range(len(rows)):
            row = rows[r_idx]
            last = row[-1]
            # Compute used diffs excluding this row's diffs
            row_diffs = _row_differences(row)
            other_diffs = set()
            for o_idx, o_row in enumerate(rows):
                if o_idx != r_idx:
                    other_diffs |= _row_differences(o_row)
            
            # Try replacing last element with a smaller value
            # The new value must be > row[-2] and its diffs to row[:-1] must not conflict
            prefix = row[:-1]
            for new_val in range(row[-2] + 1, last):
                new_diffs = {new_val - a for a in prefix}
                if new_diffs & other_diffs:
                    continue
                # Check that new_diffs don't conflict with the rest of this row's own diffs
                # (they can't, since new_val > all prefix elements)
                # Build new row
                new_row = prefix + [new_val]
                new_row_diffs = _row_differences(new_row)
                # Verify no conflict with other rows
                if new_row_diffs & other_diffs:
                    continue
                # This is valid; check if it improves the score
                new_score = max(max(r) for r in rows[:r_idx] + [new_row] + rows[r_idx+1:])
                if new_score < best_score:
                    rows[r_idx] = new_row
                    best_score = new_score
                    best = [list(r) for r in rows]
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return best


def _backtrack_row(prefix, used, k, rng, max_val=None):
    """Backtrack to complete a row given a prefix. Returns completed row or None."""
    if len(prefix) == k + 1:
        return list(prefix)
    
    last = prefix[-1]
    start = last + 1
    end = max_val + 1 if max_val is not None else start + 200
    
    # Shuffle candidate positions for diversity
    candidates = list(range(start, end))
    rng.shuffle(candidates)
    
    for x in candidates:
        new_diffs = {x - a for a in prefix}
        if new_diffs & used:
            continue
        # Recurse
        result = _backtrack_row(prefix + [x], used | new_diffs, k, rng, max_val)
        if result is not None:
            return result
    return None


def construct(n_rows, k):
    """Construct n_rows rows of k+1 increasing integers starting at 0.
    All positive differences within rows must be distinct across all rows.
    Minimize the largest entry."""
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    
    best_rows = None
    best_score = float('inf')
    
    # Phase 1: Greedy initialization with randomization
    for attempt in range(200):
        if time.time() > deadline - 30:
            break
        rng = random.Random(attempt * 7919 + 13)
        rows, used = _greedy_all_rows(n_rows, k, rng)
        if rows is None:
            continue
        score = _score(rows)
        if score < best_score:
            best_score = score
            best_rows = [list(r) for r in rows]
    
    if best_rows is None:
        # Fallback: pure greedy without randomization
        used = set()
        best_rows = []
        for _ in range(n_rows):
            row = [0]
            x = 1
            while len(row) < k + 1:
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
                x += 1
            best_rows.append(row)
        best_score = _score(best_rows)
    
    # Phase 2: Local search improvement
    if time.time() < deadline - 10:
        improved = _swap_improve([list(r) for r in best_rows], k, random.Random(42), 
                                  max_iterations=2000)
        if improved is not None:
            score = _score(improved)
            if score < best_score:
                best_score = score
                best_rows = improved
    
    # Phase 3: Try randomized restarts with backtracking on the last row
    if time.time() < deadline - 5:
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            rng = random.Random(attempt * 104729 + 97)
            # Build first n_rows-1 rows greedily
            used = set()
            rows = []
            success = True
            for i in range(n_rows - 1):
                row = _greedy_row(used, k, rng)
                if row is None:
                    success = False
                    break
                rows.append(row)
            if not success:
                continue
            
            # Try backtracking for the last row with a tighter bound
            # Current best score gives us a target
            target = best_score
            last_row = _backtrack_row([0], used, k, rng, max_val=target - 1)
            if last_row is not None:
                rows.append(last_row)
                score = _score(rows)
                if score < best_score:
                    best_score = score
                    best_rows = [list(r) for r in rows]
                    # Continue trying to improve
                    if best_score <= 112:  # counting bound
                        break
    
    # Phase 4: Final local search pass
    if time.time() < deadline - 2 and best_rows is not None:
        improved = _swap_improve([list(r) for r in best_rows], k, random.Random(12345),
                                  max_iterations=5000)
        if improved is not None:
            score = _score(improved)
            if score < best_score:
                best_score = score
                best_rows = improved
    
    return best_rows

# EVOLVE-BLOCK-END
