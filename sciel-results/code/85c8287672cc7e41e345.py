"""Find longest strictly increasing sequence in [1,256] with distinct
consecutive-block sums using descending-order DFS with restarts."""
import time
import random

def _block_sums(values):
    """Compute all consecutive-block sums for a sequence."""
    sums = set()
    n = len(values)
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += values[j]
            sums.add(s)
    return sums

def _can_append(values, x, sums):
    """Check if appending x keeps all block sums distinct.
    Returns new_sums set if valid, else None."""
    new_sums = set()
    total = x
    new_sums.add(total)
    for a in reversed(values):
        total += a
        new_sums.add(total)
    if new_sums & sums:
        return None
    return new_sums

def _greedy_descending(max_val=256, start=1):
    """Greedy: pick largest valid x first (descending order)."""
    values = []
    sums = set()
    for x in range(max_val, start - 1, -1):
        new_sums = _can_append(values, x, sums)
        if new_sums is not None:
            values.append(x)
            sums.update(new_sums)
    # Return in ascending order (strictly increasing sequence)
    values.sort()
    return values

def _dfs_extend(values, sums, min_val, max_val, deadline, best_len, depth=0):
    """DFS with descending candidate order to extend the sequence."""
    best = list(values)
    
    if time.time() > deadline:
        return best
    
    # Try candidates in descending order (larger values first - fewer collisions)
    for x in range(max_val, min_val - 1, -1):
        if time.time() > deadline:
            break
        
        new_sums = _can_append(values, x, sums)
        if new_sums is None:
            continue
        
        values.append(x)
        sums.update(new_sums)
        
        if len(values) > len(best):
            best = list(values)
        
        # Recurse deeper
        result = _dfs_extend(values, sums, x + 1, max_val, deadline, len(best), depth + 1)
        if len(result) > len(best):
            best = result
        
        # Undo
        values.pop()
        for s in new_sums:
            sums.discard(s)
    
    return best

def _truncated_dfs(prefix, deadline, best_len, max_val=256):
    """DFS from a truncated prefix with descending order."""
    if not prefix:
        values = []
        sums = set()
        min_val = 1
    else:
        values = list(prefix)
        sums = _block_sums(values)
        min_val = values[-1] + 1
    
    return _dfs_extend(values, sums, min_val, max_val, deadline, best_len)

def _randomized_greedy(max_val=256, start=1, rng=None):
    """Randomized greedy: shuffle candidates and pick valid ones."""
    if rng is None:
        rng = random.Random()
    values = []
    sums = set()
    candidates = list(range(max_val, start - 1, -1))
    rng.shuffle(candidates)
    for x in candidates:
        if x <= (values[-1] if values else 0):
            continue
        new_sums = _can_append(values, x, sums)
        if new_sums is not None:
            values.append(x)
            sums.update(new_sums)
    values.sort()
    return values

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin before 90s kill
    
    best = []
    best_len = 0
    
    # Phase 1: Descending greedy baseline
    greedy_result = _greedy_descending()
    best = greedy_result
    best_len = len(best)
    
    # Phase 2: DFS extending the greedy result
    if len(greedy_result) < 256 and time.time() < deadline - 5:
        values = list(greedy_result)
        sums = _block_sums(values)
        min_val = values[-1] + 1 if values else 1
        result = _dfs_extend(values, sums, min_val, 256, deadline, best_len)
        if len(result) > best_len:
            best = result
            best_len = len(result)
    
    # Phase 3: Try DFS from empty with descending order (fresh start)
    if time.time() < deadline - 10:
        result = _dfs_extend([], set(), 1, 256, deadline, best_len)
        if len(result) > best_len:
            best = result
            best_len = len(result)
    
    # Phase 4: Truncated DFS - try different truncation points
    if time.time() < deadline - 10:
        # Use the greedy result as base, try truncating and re-extending
        base = list(greedy_result)
        for trunc in range(min(len(base), 15), max(len(base) - 3, 0), -1):
            if time.time() > deadline - 5:
                break
            prefix = base[:trunc]
            result = _truncated_dfs(prefix, deadline, best_len)
            if len(result) > best_len:
                best = result
                best_len = len(result)
    
    # Phase 5: Randomized restarts
    if time.time() < deadline - 5:
        rng = random.Random(42)
        for trial in range(20):
            if time.time() > deadline - 3:
                break
            # Randomized greedy
            rr = _randomized_greedy(rng=rng)
            if len(rr) > best_len:
                best = rr
                best_len = len(rr)
            # Try to extend the randomized result with DFS
            if len(rr) < 256 and time.time() < deadline - 5:
                values = list(rr)
                sums = _block_sums(values)
                min_val = values[-1] + 1 if values else 1
                result = _dfs_extend(values, sums, min_val, 256, deadline, best_len)
                if len(result) > best_len:
                    best = result
                    best_len = len(result)
    
    # Phase 6: Final polish - try extending current best
    if time.time() < deadline - 2 and len(best) < 256:
        values = list(best)
        sums = _block_sums(values)
        min_val = values[-1] + 1 if values else 1
        result = _dfs_extend(values, sums, min_val, 256, deadline, best_len)
        if len(result) > best_len:
            best = result
            best_len = len(result)
    
    return best
