"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START

def _validate(rows):
    """Check that all positive differences across all rows are distinct."""
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


def _scope(rows):
    """Return the largest entry (scope) of the set."""
    return max(max(row) for row in rows)


def _greedy_row(used, k, start_range=200, rng=None):
    """
    Build one row of k+1 increasing integers starting at 0,
    choosing the smallest next mark whose differences are all unused.
    If greedy fails to complete within start_range, try a few random offsets.
    """
    row = [0]
    while len(row) < k + 1:
        found = False
        for x in range(row[-1] + 1, row[-1] + 1 + start_range):
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                row.append(x)
                used |= new_diffs
                found = True
                break
        if not found:
            # Try a larger search window
            for x in range(row[-1] + 1, row[-1] + 1 + 2000):
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
                    found = True
                    break
        if not found:
            return None
    return row


def _greedy_construct(n_rows, k, rng):
    """Standard greedy: build rows one by one with smallest available marks."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_row(used, k, start_range=200, rng=rng)
        if row is None:
            return None
        rows.append(row)
    return rows


def _randomized_greedy(n_rows, k, rng):
    """
    Randomized greedy: at each step, instead of always picking the smallest
    available x, pick from a small random window of candidates.
    """
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        success = True
        while len(row) < k + 1:
            # Collect candidates in a window
            window_size = 30
            candidates = []
            for x in range(row[-1] + 1, row[-1] + 1 + window_size):
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    candidates.append(x)
            if not candidates:
                # Expand window
                for x in range(row[-1] + 1, row[-1] + 1 + 500):
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        candidates.append(x)
            if not candidates:
                success = False
                break
            # Pick randomly from top few or the smallest
            if len(candidates) > 5:
                # Bias toward smaller but allow randomness
                pick_idx = rng.randint(0, min(4, len(candidates) - 1))
            else:
                pick_idx = rng.randint(0, len(candidates) - 1)
            x = candidates[pick_idx]
            new_diffs = {x - a for a in row}
            row.append(x)
            used |= new_diffs
        if not success:
            return None
        rows.append(row)
    return rows


def _local_search(rows, k, rng, max_iter=5000):
    """
    Try to improve the scope by swapping or shifting elements.
    We attempt to reduce the maximum value by replacing the last element
    of some row with a smaller unused one, or by reordering.
    """
    best = rows[:]
    best_scope = _scope(best)
    
    for _ in range(max_iter):
        # Pick a random row and try to reduce its last element
        r_idx = rng.randint(0, len(rows) - 1)
        row = rows[r_idx]
        if len(row) < 2:
            continue
        last = row[-1]
        
        # Collect all used diffs from all rows except this row's contribution
        # This is expensive; instead, try candidate values and check
        improved = False
        for candidate in range(last - 1, row[-2] if len(row) > 2 else 0, -1):
            if candidate <= row[-2]:
                break
            # Check if candidate works: all diffs with other elements in this row
            # must not conflict with diffs used by other rows
            new_row = row[:-1] + [candidate]
            # Compute all diffs in new_row
            new_diffs = set()
            for i in range(len(new_row)):
                for j in range(i + 1, len(new_row)):
                    d = new_row[j] - new_row[i]
                    new_diffs.add(d)
            
            # Check against all other rows' diffs
            ok = True
            for other_idx, other_row in enumerate(rows):
                if other_idx == r_idx:
                    continue
                for i in range(len(other_row)):
                    for j in range(i + 1, len(other_row)):
                        d = other_row[j] - other_row[i]
                        if d in new_diffs:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            
            if ok:
                rows[r_idx] = new_row
                new_scope = _scope(rows)
                if new_scope < best_scope:
                    best = [r[:] for r in rows]
                    best_scope = new_scope
                    improved = True
                break
        
        if not improved:
            # Also try reducing an interior element
            pos = rng.randint(1, len(row) - 2) if len(row) > 3 else -1
            if pos > 0:
                for candidate in range(row[pos] - 1, row[pos - 1], -1):
                    if candidate <= row[pos - 1]:
                        break
                    new_row = row[:]
                    new_row[pos] = candidate
                    new_diffs = set()
                    for i in range(len(new_row)):
                        for j in range(i + 1, len(new_row)):
                            d = new_row[j] - new_row[i]
                            new_diffs.add(d)
                    
                    ok = True
                    for other_idx, other_row in enumerate(rows):
                        if other_idx == r_idx:
                            continue
                        for i in range(len(other_row)):
                            for j in range(i + 1, len(other_row)):
                                d = other_row[j] - other_row[i]
                                if d in new_diffs:
                                    ok = False
                                    break
                            if not ok:
                                break
                        if not ok:
                            break
                    
                    if ok:
                        rows[r_idx] = new_row
                        new_scope = _scope(rows)
                        if new_scope < best_scope:
                            best = [r[:] for r in rows]
                            best_scope = new_scope
                        break
    
    return best


def construct(n_rows, k):
    """
    Construct a difference triangle set with n_rows rows, each of k+1 integers
    starting at 0, minimizing the largest entry.
    """
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    rng = random.Random(42)
    
    best_rows = None
    best_scope = float('inf')
    
    # Phase 1: Deterministic greedy
    rows = _greedy_construct(n_rows, k, rng)
    if rows is not None:
        best_rows = rows
        best_scope = _scope(rows)
    
    # Phase 2: Multiple randomized greedy restarts
    n_restarts = 50
    for i in range(n_restarts):
        if time.time() > deadline:
            break
        rows = _randomized_greedy(n_rows, k, rng)
        if rows is not None:
            scope = _scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = rows
    
    # Phase 3: Local search on the best solution found
    if best_rows is not None and time.time() < deadline - 10:
        # Run local search with time budget
        remaining = deadline - 10 - time.time()
        # Estimate iterations per second: roughly 1000-5000 depending on k
        max_iter = max(100, int(remaining * 500))
        improved = _local_search([r[:] for r in best_rows], k, rng, max_iter=max_iter)
        if _scope(improved) < best_scope:
            best_scope = _scope(improved)
            best_rows = improved
    
    # Phase 4: More randomized restarts if time allows
    while time.time() < deadline - 5:
        rows = _randomized_greedy(n_rows, k, rng)
        if rows is not None:
            scope = _scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = rows
                # Quick local search on new best
                if time.time() < deadline - 5:
                    improved = _local_search([r[:] for r in best_rows], k, rng, max_iter=2000)
                    if _scope(improved) < best_scope:
                        best_scope = _scope(improved)
                        best_rows = improved
    
    if best_rows is None:
        # Fallback: simple greedy
        used = set()
        best_rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    new = {x - a for a in row}
                    if not (new & used):
                        break
                    x += 1
                row.append(x)
                used |= {x - a for a in row}
            best_rows.append(row)
    
    return best_rows

# EVOLVE-BLOCK-END
