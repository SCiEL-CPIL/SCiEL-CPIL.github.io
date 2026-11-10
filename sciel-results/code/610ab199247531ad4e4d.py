"""Structural rewrite: Cantor base + fast bitset subset-sum check + local search."""
# EVOLVE-BLOCK-START
import time
import random
from functools import lru_cache

_DEADLINE = 85.0  # safety margin under 90s kill
_start_t = time.time()

def _elapsed():
    return time.time() - _start_t

def _deadline_exceeded():
    return _elapsed() > _DEADLINE


def _cantor_numbers(limit):
    """Numbers with base-3 digits in {0,1} only, up to limit.
    These are 3-AP-free and structurally close to mean-free."""
    out = []
    # generate all 3^n with digits 0/1
    n = 0
    while 3**n <= limit:
        n += 1
    for mask in range(1, 2**n):
        v = 0
        m = mask
        p = 1
        while m:
            if m & 1:
                v += p
            m >>= 1
            p *= 3
        if v <= limit:
            out.append(v)
    out.sort()
    return out


def _is_mean_free(selected, candidate):
    """Check that adding candidate to selected keeps mean-free property.
    For each a in selected+{candidate}, no subset of others (size>=2) has mean = a.
    Equivalent: for each a, no subset S of others with |S|>=2 and sum(S) == |S|*a.
    Use bitset subset-sum per a.
    """
    full = selected + [candidate]
    n = len(full)
    for i, a in enumerate(full):
        others = [full[j] for j in range(n) if j != i]
        m = len(others)
        # We need: for some k in [2, m], sum of k distinct others == k*a
        # Bitset: dp[s] = bitmask of subset sizes achievable with sum s
        # But sums can be up to m * max(others). Use dict of {sum: bitmask of sizes}.
        # Better: for each target k, check if subset of size k sums to k*a.
        # Use DP: dp[k][s] = can we pick k elements summing to s.
        # m <= ~60, max sum ~ 60*512=30720. dp as list of sets or bitsets.
        # Use bitset over sums for each k.
        max_sum = sum(others)
        # dp[k] = bitset (int) where bit s set means sum s achievable with exactly k elements
        dp = [0] * (m + 1)
        dp[0] = 1  # sum 0 with 0 elements
        for x in others:
            for k in range(m, 0, -1):
                dp[k] |= (dp[k-1] << x)
        for k in range(2, m + 1):
            target = k * a
            if target <= max_sum:
                if (dp[k] >> target) & 1:
                    return False
    return True


def _validate_set(selected):
    """Full validation of a set (no candidate added)."""
    n = len(selected)
    for i, a in enumerate(selected):
        others = [selected[j] for j in range(n) if j != i]
        m = len(others)
        max_sum = sum(others)
        dp = [0] * (m + 1)
        dp[0] = 1
        for x in others:
            for k in range(m, 0, -1):
                dp[k] |= (dp[k-1] << x)
        for k in range(2, m + 1):
            target = k * a
            if target <= max_sum and ((dp[k] >> target) & 1):
                return False
    return True


def _greedy_from_base(base, limit, max_size):
    """Greedy: try to add each base element in order, plus try all candidates."""
    selected = []
    # Try base elements first in order
    for x in base:
        if x > limit:
            continue
        if len(selected) >= max_size:
            break
        if _is_mean_free(selected, x):
            selected.append(x)
    # Then try all remaining candidates 1..limit
    for x in range(1, limit + 1):
        if x in selected:
            continue
        if len(selected) >= max_size:
            break
        if _is_mean_free(selected, x):
            selected.append(x)
    return selected


def _local_search(selected, limit, max_size, rng):
    """Try to grow the set by swapping or adding."""
    selected = list(selected)
    current_size = len(selected)
    if current_size >= max_size:
        return selected
    # Try adding any element not in selected
    in_set = set(selected)
    for x in range(1, limit + 1):
        if x in in_set:
            continue
        if _is_mean_free(selected, x):
            selected.append(x)
            in_set.add(x)
            if len(selected) >= max_size:
                break
    # Try swaps: remove one, add two
    for _ in range(200):
        if _deadline_exceeded():
            break
        if len(selected) >= max_size:
            break
        i = rng.randrange(len(selected))
        a = selected[i]
        # try to add x without removing
        found = False
        for x in rng.sample(range(1, limit + 1), min(50, limit)):
            if x in set(selected):
                continue
            trial = selected + [x]
            if _is_mean_free(selected, x):
                selected = trial
                found = True
                break
        if found:
            continue
        # try swap
        for x in rng.sample(range(1, limit + 1), min(30, limit)):
            if x == a or x in set(selected):
                continue
            trial = [y for j, y in enumerate(selected) if j != i] + [x]
            if _is_mean_free(selected, x) or True:
                # need full check
                pass
            # simpler: check if set with a removed and x added is valid
            new_set = [y for j, y in enumerate(selected) if j != i]
            if _is_mean_free(new_set, x):
                # also need to ensure new_set itself is valid (it was a subset of valid set, so yes)
                selected = new_set + [x]
                found = True
                break
    return selected


def construct():
    global _start_t
    _start_t = time.time()
    rng = random.Random(42)
    limit = 512
    max_size = 48

    # Build Cantor base
    base = _cantor_numbers(limit)
    
    # Greedy from base
    selected = _greedy_from_base(base, limit, max_size)
    
    # If we hit max_size, validate and return
    if len(selected) >= max_size:
        if _validate_set(selected):
            return selected[:max_size]
        # trim to valid
        # find max valid prefix
        for k in range(len(selected), 0, -1):
            if _validate_set(selected[:k]):
                return selected[:k]
        return selected[:max_size]
    
    # Local search to grow
    if not _deadline_exceeded():
        selected = _local_search(selected, limit, max_size, rng)
    
    # Try multiple random restarts if time permits
    best = selected
    attempts = 0
    while not _deadline_exceeded() and attempts < 50:
        attempts += 1
        # random greedy
        rng2 = random.Random(attempts)
        order = list(range(1, limit + 1))
        rng2.shuffle(order)
        sel2 = []
        for x in order:
            if len(sel2) >= max_size:
                break
            if _is_mean_free(sel2, x):
                sel2.append(x)
        if len(sel2) > len(best):
            best = sel2
        if len(best) >= max_size:
            break
    
    # Final validation
    if len(best) > max_size:
        best = best[:max_size]
    # ensure validity
    if not _validate_set(best):
        # trim
        for k in range(len(best), 0, -1):
            if _validate_set(best[:k]):
                best = best[:k]
                break
    return best
# EVOLVE-BLOCK-END
