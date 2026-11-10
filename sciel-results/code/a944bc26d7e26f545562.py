"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

N = 12
MAX_M = 1000000
DEADLINE = 85.0  # safety margin before 90s kill

def _hall_check(m, L):
    """Check Hall's condition for (m, m+L] with indices 1..N.
    
    For every subset S of {1,...,N}, the number of values in (m, m+L]
    divisible by at least one element of S must be >= |S|.
    """
    if L < 1:
        return False
    
    # For each value v in (m, m+L], compute bitmask of indices i that divide v
    # Then for Hall's condition, for each subset S, count values with mask & S != 0
    
    # Build divisor masks
    div_masks = []
    for v in range(m + 1, m + L + 1):
        mask = 0
        for i in range(1, N + 1):
            if v % i == 0:
                mask |= (1 << i)
        div_masks.append(mask)
    
    # For Hall's condition: for each non-empty subset S (bitmask),
    # count = #{v : div_mask(v) & S != 0} must be >= |S|
    #
    # Optimization: Instead of checking all 4096 subsets, note that if S1 ⊂ S2
    # and |N(S1)| >= |S1|, it doesn't guarantee |N(S2)| >= |S2|.
    # But if |N(S)| < |S| for some S, then for any S' ⊃ S, |N(S')| >= |N(S)|
    # but |S'| > |S|, so it could still fail or not.
    #
    # The binding constraints are typically for small subsets. Let's check
    # subsets by size, and early-exit if a smaller subset fails.
    
    # Group subsets by size
    # Actually, let's just check all but with early termination
    # For speed, check smaller subsets first
    
    # Precompute: for each possible divisor mask, count how many values have it
    # Then for subset S, count = sum of counts for masks that intersect S
    
    # Better approach: use the fact that there are at most L distinct masks
    # and L <= 144
    
    # For each subset S, count values where mask & S != 0
    # = L - count of values where mask & S == 0
    # = L - count of values whose mask is a subset of ~S
    
    # Let's precompute a frequency table of masks
    from collections import Counter
    mask_freq = Counter(div_masks)
    
    # For each subset S (bitmask), we need count of values with (mask & S) == 0
    # i.e., mask is a subset of (full_mask ^ S)
    # full_mask = (1 << (N+1)) - 1 - 1  # bits 1..N set
    full_mask = (1 << (N + 1)) - 2  # bits 1 through N set
    
    # For subset S, forbidden masks are those that are subsets of (full_mask ^ S)
    # i.e., mask & S == 0, meaning all bits in mask are in (full_mask ^ S)
    
    # Precompute for all 4096 subsets S: count of values with mask & S == 0
    # This can be done via subset DP (zeta transform)
    
    # freq[mask] = number of values with that exact mask
    # For each S, count_no_intersect[S] = sum_{mask: mask & S == 0} freq[mask]
    # = sum_{mask ⊆ (full_mask ^ S)} freq[mask]
    
    # Let T = full_mask ^ S, then we want sum_{mask ⊆ T} freq[mask]
    # This is the "subset sum" which can be computed via zeta transform
    
    # Build freq array of size 2^(N+1)
    size = 1 << (N + 1)
    freq = [0] * size
    for mask, cnt in mask_freq.items():
        freq[mask] = cnt
    
    # Zeta transform: F[T] = sum_{S ⊆ T} freq[S]
    F = freq[:]
    for i in range(1, N + 1):
        for T in range(size):
            if T & (1 << i):
                F[T] += F[T ^ (1 << i)]
    
    # Now for subset S, count of values with mask & S == 0 is F[full_mask ^ S]
    # And count with mask & S != 0 is L - F[full_mask ^ S]
    
    # Check Hall: for all non-empty S, L - F[full_mask ^ S] >= |S|
    # i.e., F[full_mask ^ S] <= L - |S|
    
    for S in range(1, size):
        if S == 0:
            continue
        # Only consider S that are subsets of full_mask (bits 1..N)
        if S & ~full_mask:
            continue
        T = full_mask ^ S
        no_intersect = F[T]
        intersect = L - no_intersect
        if intersect < S.bit_count():
            return False
    
    return True

def _compute_f(m):
    """Compute f(N, m): least positive L such that (m, m+L] has valid matching."""
    lo = 1
    hi = 144
    while lo < hi:
        mid = (lo + hi) // 2
        if _hall_check(m, mid):
            hi = mid
        else:
            lo = mid + 1
    return lo

def _search_optimal_m():
    """Search for m that maximizes f(12, m)."""
    start_time = time.time()
    best_m = 12
    best_f = _compute_f(12)
    
    # The function f(12, m) is periodic with period lcm(1..12) = 27720.
    # So we search m in [0, 27719].
    # With optimized Hall check, each _compute_f should be fast.
    
    # Phase 1: Exhaustive search over one period
    for m in range(0, 27720):
        if time.time() - start_time > DEADLINE - 5:
            break
        f_val = _compute_f(m)
        if f_val > best_f:
            best_f = f_val
            best_m = m
    
    # Phase 2: Local search around best (in case period assumption is slightly off)
    if time.time() - start_time < DEADLINE - 5:
        for delta in range(-50, 51):
            m = best_m + delta
            if 0 <= m <= MAX_M and time.time() - start_time > DEADLINE - 5:
                break
            f_val = _compute_f(m)
            if f_val > best_f:
                best_f = f_val
                best_m = m
    
    # Phase 3: Check a few other periods to verify
    if time.time() - start_time < DEADLINE - 5:
        for offset in [27720, 55440, 1000000 - 27720]:
            for delta in range(-10, 11):
                m = best_m + offset + delta
                if 0 <= m <= MAX_M and time.time() - start_time > DEADLINE - 5:
                    break
                f_val = _compute_f(m)
                if f_val > best_f:
                    best_f = f_val
                    best_m = m
    
    return best_m

# Perform search at import time
_OPTIMAL_M = _search_optimal_m()

def construct():
    return int(_OPTIMAL_M)
# EVOLVE-BLOCK-END
