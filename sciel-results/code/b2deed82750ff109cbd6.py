"""Combined program: integer LCM-based meet-in-the-middle with multi-strategy search."""
import time
import math
import random

# Precompute LCM of 1..24
def _lcm(a, b):
    return a * b // math.gcd(a, b)

_LCM = 1
for i in range(1, 25):
    _LCM = _lcm(_LCM, i)

# v[i] = LCM / i
_V = [0] * 25
for i in range(1, 25):
    _V[i] = _LCM // i

_HALF_SIZE = 12
_LEFT = list(range(1, _HALF_SIZE + 1))   # 1..12
_RIGHT = list(range(_HALF_SIZE + 1, 25)) # 13..24

# Precompute per-index values for fast subset sum
_LEFT_VALS = [_V[_LEFT[i]] for i in range(12)]
_RIGHT_VALS = [_V[_RIGHT[i]] for i in range(12)]

# Lazy subset-sum computation with memoization
_LeftSumsByMask = [None] * 4096
_RightSumsByMask = [None] * 4096

def _get_sums(mask, vals, cache):
    """Get all subset sums for a mask using the given value array."""
    if cache[mask] is not None:
        return cache[mask]
    sums = {0}
    sub = mask
    while sub:
        s = 0
        # Fast bit iteration
        m = sub
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            s += vals[idx]
            m ^= lsb
        sums.add(s)
        sub = (sub - 1) & mask
    cache[mask] = sums
    return sums

def _get_left_sums(mask):
    return _get_sums(mask, _LEFT_VALS, _LeftSumsByMask)

def _get_right_sums(mask):
    return _get_sums(mask, _RIGHT_VALS, _RightSumsByMask)

def _check_safe_fast(a_idx, S_mask):
    """Check if element a_idx (0-based) in set S_mask is safe."""
    val = a_idx + 1
    target = _V[val]
    avail_mask = S_mask & ~(1 << a_idx)
    
    if avail_mask == 0:
        return True
    
    left_avail = avail_mask & 0xFFF
    right_avail = (avail_mask >> 12) & 0xFFF
    
    L_sums = _get_left_sums(left_avail)
    R_sums = _get_right_sums(right_avail)
    
    # Check if target can be formed by sL + sR (non-trivial subset)
    if len(L_sums) < len(R_sums):
        for sL in L_sums:
            needR = target - sL
            if needR in R_sums:
                if sL != 0 or needR != 0:
                    return False
    else:
        for sR in R_sums:
            needL = target - sR
            if needL in L_sums:
                if sR != 0 or needL != 0:
                    return False
                    
    return True

def _is_valid_set(S_mask):
    """Check if the entire set is valid (no reciprocal subsum violations)."""
    m = S_mask
    while m:
        lsb = m & -m
        i = lsb.bit_length() - 1
        if not _check_safe_fast(i, S_mask):
            return False
        m ^= lsb
    return True

def _greedy_build(order, deadline):
    """Greedy construction following a given element order (list of 0-based indices)."""
    S = 0
    for idx in order:
        if time.time() > deadline:
            break
        trial = S | (1 << idx)
        ok = True
        # Check all elements in trial for safety
        m = trial
        while m:
            lsb = m & -m
            i = lsb.bit_length() - 1
            if not _check_safe_fast(i, trial):
                ok = False
                break
            m ^= lsb
        if ok:
            S = trial
    return S

def _local_search(S_mask, deadline, max_iters=200):
    """Try to improve the set by adding or swapping elements."""
    best = S_mask
    best_size = S_mask.bit_count()
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        improved = False
        
        # Phase 1: Try adding each element not in the set (cheapest improvement)
        for i in range(24):
            if time.time() > deadline:
                break
            if not (best >> i & 1):
                trial = best | (1 << i)
                if _is_valid_set(trial):
                    best = trial
                    best_size += 1
                    improved = True
                    break
        
        if improved:
            continue
        
        # Phase 2: Try 1-for-1 swaps
        for i in range(24):
            if time.time() > deadline:
                break
            if best >> i & 1:
                base = best & ~(1 << i)
                for j in range(24):
                    if time.time() > deadline:
                        break
                    if not (base >> j & 1):
                        trial = base | (1 << j)
                        if _is_valid_set(trial):
                            best = trial
                            improved = True
                            break
                if improved:
                    break
        
        if not improved:
            break
    
    return best

def _remove_violations(S_mask, deadline):
    """Remove elements causing violations until the set is valid."""
    S = S_mask
    changed = True
    while changed and time.time() < deadline:
        changed = False
        m = S
        while m:
            if time.time() > deadline:
                return S
            lsb = m & -m
            i = lsb.bit_length() - 1
            if not _check_safe_fast(i, S):
                S &= ~(1 << i)
                changed = True
                break
            m ^= lsb
    return S

def construct():
    """Return a nonempty subset of [1,24] maximizing cardinality."""
    start = time.time()
    deadline = start + 82.0  # Safety margin before 90s kill
    
    best_set = 0
    best_size = 0
    
    def update_best(S):
        nonlocal best_set, best_size
        sz = S.bit_count()
        if sz > best_size:
            best_set = S
            best_size = sz
    
    # Strategy 1: Descending order (large elements first)
    order_desc = list(range(23, -1, -1))
    S = _greedy_build(order_desc, deadline)
    update_best(S)
    
    # Strategy 2: Ascending order
    order_asc = list(range(24))
    S = _greedy_build(order_asc, deadline)
    update_best(S)
    
    # Strategy 3: Alternating large/small
    order_alt = []
    large = list(range(23, 11, -1))  # 24 down to 13
    small = list(range(11, -1, -1))   # 12 down to 1
    for i in range(12):
        if i < len(large):
            order_alt.append(large[i])
        if i < len(small):
            order_alt.append(small[i])
    S = _greedy_build(order_alt, deadline)
    update_best(S)
    
    # Strategy 4: Random restarts with local search
    random.seed(42)
    num_restarts = 500
    for attempt in range(num_restarts):
        if time.time() > deadline:
            break
        
        elems = list(range(24))
        random.shuffle(elems)
        
        S = _greedy_build(elems, deadline)
        if S.bit_count() >= best_size:
            # Try local search to improve
            S2 = _local_search(S, deadline, max_iters=30)
            update_best(S2)
    
    # Strategy 5: Full set repair
    if time.time() < deadline - 2:
        S_full = (1 << 24) - 1
        S_repaired = _remove_violations(S_full, deadline)
        update_best(S_repaired)
        if time.time() < deadline - 2:
            S_repaired = _local_search(S_repaired, deadline, max_iters=50)
            update_best(S_repaired)

    # Final: try more structured orderings if time permits
    if time.time() < deadline - 3:
        orderings = [
            [23, 12, 22, 11, 21, 10, 20, 9, 19, 8, 18, 7, 17, 6, 16, 5, 15, 4, 14, 3, 13, 2, 0, 1],
            [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
        ]
        for order in orderings:
            if time.time() > deadline:
                break
            S = _greedy_build(order, deadline)
            update_best(S)
            if S.bit_count() > best_size and time.time() < deadline - 2:
                S2 = _local_search(S, deadline, max_iters=50)
                update_best(S2)
    
    if best_set != 0:
        result = [i + 1 for i in range(24) if best_set >> i & 1]
    else:
        result = [1]
        
    return result

if __name__ == "__main__":
    t0 = time.time()
    r = construct()
    t1 = time.time()
    print(f"Result: {r}, size={len(r)}, time={t1-t0:.2f}s")
