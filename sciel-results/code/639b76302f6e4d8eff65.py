"""Increasing sequences with unique consecutive-block sums.
Uses randomized greedy + iterative local search (swap/replace) to escape
local optima, plus a fast greedy baseline."""
import time
import random

def _all_block_sums(seq):
    """Compute all nonempty consecutive-block sums for a sequence."""
    n = len(seq)
    sums = set()
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += seq[j]
            sums.add(s)
    return sums

def _can_append_fast(values, x, sums):
    """Check if x can be appended without creating duplicate block sums.
    Returns the set of new sums if valid, else None."""
    new_sums = set()
    total = x
    new_sums.add(total)
    for a in reversed(values):
        total += a
        new_sums.add(total)
    if new_sums & sums:
        return None
    return new_sums

def _greedy_greedy(values, sums, start_val, max_val, deadline):
    """Greedy: pick smallest valid x >= start_val."""
    for x in range(start_val, max_val + 1):
        if time.time() > deadline:
            return False
        new_sums = _can_append_fast(values, x, sums)
        if new_sums is not None:
            values.append(x)
            sums.update(new_sums)
            return True
    return False

def _randomized_greedy(start_val, max_val, deadline, seed=0):
    """Greedy with random tie-breaking among valid candidates.
    Returns (values, sums, best_len)."""
    rng = random.Random(seed)
    values = []
    sums = set()
    
    while True:
        if time.time() > deadline:
            break
        # Collect all valid candidates
        valid = []
        for x in range(max(values[-1] + 1 if values else 1, start_val), max_val + 1):
            new_sums = _can_append_fast(values, x, sums)
            if new_sums is not None:
                valid.append((x, new_sums))
        if not valid:
            break
        # Pick randomly from valid candidates (bias toward smaller values)
        # Use weighted selection: smaller values more likely
        weights = [1.0 / (1.0 + (x - start_val) * 0.05) for x, _ in valid]
        total_w = sum(weights)
        r = rng.random() * total_w
        cum = 0
        chosen_idx = 0
        for i, w in enumerate(weights):
            cum += w
            if r <= cum:
                chosen_idx = i
                break
        x, new_sums = valid[chosen_idx]
        values.append(x)
        sums.update(new_sums)
    
    return values, sums, len(values)

def _local_search(values, sums, deadline, rng, max_iters=500):
    """Iterative local search: try replacing or swapping elements to enable
    further extension. Returns improved (values, sums, length)."""
    best = list(values)
    best_sums = set(sums)
    best_len = len(best)
    iters = 0
    
    while time.time() < deadline and iters < max_iters:
        iters += 1
        current = list(values)
        cur_sums = set(sums)
        cur_len = len(current)
        
        # Try to extend from current state
        extended = False
        for x in range(current[-1] + 1 if current else 1, 257):
            if time.time() > deadline:
                break
            new_sums = _can_append_fast(current, x, cur_sums)
            if new_sums is not None:
                current.append(x)
                cur_sums.update(new_sums)
                extended = True
                break
        
        if extended and len(current) > best_len:
            best = list(current)
            best_sums = set(cur_sums)
            best_len = len(best)
            values = current
            sums = cur_sums
            continue
        
        # Try a random perturbation: replace one element
        if len(current) > 1:
            pos = rng.randint(0, len(current) - 1)
            old_val = current[pos]
            # Try replacing with a nearby value
            for delta in [1, -1, 2, -2, 3, -3]:
                new_val = old_val + delta
                if new_val < 1 or new_val > 256:
                    continue
                # Check strict increasing
                if pos > 0 and new_val <= current[pos - 1]:
                    continue
                if pos < len(current) - 1 and new_val >= current[pos + 1]:
                    continue
                # Try the replacement
                test = list(current)
                test[pos] = new_val
                test_sums = _all_block_sums(test)
                if len(test_sums) == len(test) * (len(test) + 1) // 2:
                    # Valid! Try to extend
                    ext = list(test)
                    ext_sums = set(test_sums)
                    for x in range(ext[-1] + 1, 257):
                        ns = _can_append_fast(ext, x, ext_sums)
                        if ns is not None:
                            ext.append(x)
                            ext_sums.update(ns)
                            break
                    if len(ext) > best_len:
                        best = list(ext)
                        best_sums = set(ext_sums)
                        best_len = len(best)
                        values = ext
                        sums = ext_sums
                        break
        else:
            # Single element, just try to extend
            for x in range(current[-1] + 1 if current else 2, 257):
                ns = _can_append_fast(current, x, cur_sums)
                if ns is not None:
                    current.append(x)
                    cur_sums.update(ns)
                    if len(current) > best_len:
                        best = list(current)
                        best_sums = set(cur_sums)
                        best_len = len(best)
                        values = current
                        sums = cur_sums
                    break
    
    return best, best_sums, best_len

def _multi_start_search(deadline, n_starts=10):
    """Run multiple randomized greedy starts and local search from each."""
    best = []
    best_len = 0
    rng = random.Random(42)
    
    for i in range(n_starts):
        if time.time() > deadline:
            break
        # Vary the starting strategy
        start_val = rng.randint(1, 5)
        values, sums, length = _randomized_greedy(start_val, 256, deadline, seed=i)
        
        if length > best_len:
            best = values
            best_len = length
        
        # Local search from this state
        if time.time() < deadline - 2:
            improved_vals, improved_sums, improved_len = _local_search(
                values, sums, deadline, rng, max_iters=100
            )
            if improved_len > best_len:
                best = improved_vals
                best_len = improved_len
    
    return best, best_len

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin before 90s kill
    
    # Phase 1: Fast greedy baseline (deterministic)
    values = []
    sums = set()
    for x in range(1, 257):
        if time.time() > deadline:
            break
        new_sums = _can_append_fast(values, x, sums)
        if new_sums is not None:
            values.append(x)
            sums.update(new_sums)
    
    best = list(values)
    best_len = len(best)
    
    # Phase 2: Multi-start randomized search + local search
    if time.time() < deadline - 5:
        ms_best, ms_len = _multi_start_search(deadline - 3, n_starts=15)
        if ms_len > best_len:
            best = ms_best
            best_len = ms_len
    
    # Phase 3: Final local search refinement on the best
    if time.time() < deadline - 2 and best_len < 256:
        rng = random.Random(123)
        refined_vals, refined_sums, refined_len = _local_search(
            best, _all_block_sums(best), deadline, rng, max_iters=200
        )
        if refined_len > best_len:
            best = refined_vals
            best_len = refined_len
    
    return best
