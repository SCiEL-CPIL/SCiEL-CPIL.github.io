"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _greedy():
    values, sums = [], set()
    for x in range(1, 257):
        tails, total = [x], x
        for a in reversed(values):
            total += a
            tails.append(total)
        if not sums.intersection(tails):
            values.append(x)
            sums.update(tails)
    return values

def _check_extend(values, sums, x):
    """Check if x can be appended. Returns (tails, new_sums) or None."""
    tails, total = [x], x
    for a in reversed(values):
        total += a
        tails.append(total)
    for t in tails:
        if t in sums:
            return None
    return tails

def _backtrack(values, sums, start_val, target_len, deadline):
    """Try to extend values to reach target_len. Returns extended list or None."""
    if len(values) >= target_len:
        return values[:]
    if time.time() > deadline:
        return None
    for x in range(start_val, 257):
        result = _check_extend(values, sums, x)
        if result is not None:
            tails = result
            new_sums = sums | set(tails)
            ext = _backtrack(values + [x], new_sums, x + 1, target_len, deadline)
            if ext is not None:
                return ext
    return None

def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    
    # Get greedy baseline
    base = _greedy()
    base_len = len(base)
    
    # Try to extend beyond greedy using backtracking
    # First, try to add one more element
    best = base
    
    # Compute sums for base
    sums_set = set()
    for i in range(len(base)):
        s = 0
        for j in range(i, len(base)):
            s += base[j]
            sums_set.add(s)
    
    # Try to extend by backtracking from the end
    # Strategy: try different endings
    # First try: just try to add one more
    if time.time() < deadline - 5:
        for x in range(base[-1] + 1, 257):
            result = _check_extend(base, sums_set, x)
            if result is not None:
                best = base + [x]
                break
    
    # Now try to find even longer sequences via more aggressive search
    # Use a different strategy: restart with backtracking from scratch
    # but guided by the greedy solution
    
    # Try: can we find a sequence of length base_len + 1 or more?
    # Use iterative deepening with backtracking
    if time.time() < deadline - 10:
        # Try to build a longer sequence from scratch with backtracking
        # Use the greedy as a hint for ordering
        for target in range(base_len + 1, min(base_len + 5, 257)):
            if time.time() > deadline - 5:
                break
            found = _backtrack([], set(), 1, target, deadline)
            if found is not None:
                best = found
                break
    
    return best
# EVOLVE-BLOCK-END
