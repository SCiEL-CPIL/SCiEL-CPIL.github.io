"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from functools import lru_cache

def _hall_check(m, L):
    """Check Hall's condition: for every subset S of {1..12}, 
    count of numbers in (m, m+L] divisible by at least one member of S >= |S|.
    Uses inclusion-exclusion over subsets of {1..12}.
    """
    # For each subset S of {1..12}, we need:
    # |{x in (m, m+L] : exists i in S, i|x}| >= |S|
    # 
    # A number x is divisible by "at least one member of S" iff x is NOT coprime to lcm(S)
    # in the sense that... actually let's use: x is divisible by at least one i in S
    # iff x is NOT in the set of numbers coprime to all elements of S simultaneously... 
    # No, that's wrong. Let me think differently.
    #
    # Actually: x is divisible by at least one member of S means x shares a common factor 
    # with at least one element of S. This is NOT simply related to lcm.
    #
    # Better approach: For a given subset S, count numbers in (m, m+L] that are divisible 
    # by at least one element of S. We can use inclusion-exclusion.
    #
    # But with 12 elements, subsets can have up to 12 elements, and inclusion-exclusion 
    # over 12 elements requires 2^12 = 4096 terms per subset, and we have 2^12 = 4096 subsets.
    # That's 4096 * 4096 = 16M operations per (m,L) check - too slow.
    #
    # Alternative: Precompute for each number x in (m, m+L], which indices divide it.
    # Then for Hall's condition, we need: for each subset S, the count of x that have 
    # at least one divisor in S >= |S|.
    #
    # Even better: Since L <= 144 and the interval is small, we can enumerate all numbers 
    # in (m, m+L] and for each, compute which i in 1..12 divide it. Then Hall's condition 
    # becomes: for each subset S, count of numbers whose divisor-set intersects S >= |S|.
    #
    # With at most 144 numbers, and 4096 subsets, this is 144 * 4096 = ~590K ops per check.
    # Still might be slow for exhaustive search.
    #
    # FASTEST approach: Use the fact that we only need to check Hall for the "critical" 
    # subsets. Actually, Hall's condition is equivalent to: the maximum matching has size 12.
    # But we can use a faster check: 
    # 
    # Key insight: For Hall's theorem with left side {1..12}, the condition is:
    # For ALL subsets S of {1..12}: |N(S) ∩ (m, m+L]| >= |S|
    # where N(S) = {x : exists i in S, i|x}
    #
    # Since the interval is small (L <= 144), let's just enumerate all numbers in the interval
    # and for each, record which i divide it. Then for each subset S, count how many numbers
    # have at least one divisor in S.
    
    # Collect all numbers in (m, m+L]
    # For each number, record bitmask of which i in 1..12 divide it
    num_divisors = []  # list of bitmasks
    for x in range(m + 1, m + L + 1):
        mask = 0
        for i in range(1, 13):
            if x % i == 0:
                mask |= (1 << (i - 1))
        num_divisors.append(mask)
    
    # For Hall's condition: for each non-empty subset S (represented as bitmask),
    # count numbers whose divisor mask intersects S, must be >= popcount(S)
    # 
    # To speed up: precompute for each possible divisor mask (0 to 2^12-1), 
    # how many numbers have exactly that mask. Then for subset S, the count of numbers
    # whose mask intersects S = total - count of numbers whose mask is disjoint from S.
    # mask disjoint from S means mask & S == 0, i.e., mask is a subset of (~S) & (2^12-1)
    
    mask_counts = [0] * (1 << 12)
    for mask in num_divisors:
        mask_counts[mask] += 1
    
    # Precompute: for each subset S, count of numbers with mask disjoint from S
    # = sum of mask_counts[m] for all m where m & S == 0
    # = sum of mask_counts[m] for all m that are subsets of complement of S
    # 
    # We can use SOS DP (subset sum) to compute this efficiently.
    # Let F[S] = sum of mask_counts[m] for all m that are subsets of S.
    # Then count_disjoint(S) = F[complement of S]
    
    # SOS DP: F[S] = sum_{T ⊆ S} mask_counts[T]
    F = mask_counts[:]
    for bit in range(12):
        for S in range(1 << 12):
            if S & (1 << bit):
                F[S] += F[S ^ (1 << bit)]
    
    # Now for each subset S (non-empty), check: (L - F[complement of S]) >= popcount(S)
    # where complement of S = ((1<<12)-1) ^ S
    all_mask = (1 << 12) - 1
    for S in range(1, 1 << 12):
        comp = all_mask ^ S
        count_in_N = L - F[comp]  # numbers whose mask intersects S
        if count_in_N < S.bit_count():
            return False
    return True

def _compute_f(m, max_L=144):
    """Compute f(12, m): least positive L such that (m, m+L] has valid matching."""
    for L in range(1, max_L + 1):
        if _hall_check(m, L):
            return L
    return max_L

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill
    
    # Key insight: f(12, m) is periodic with period lcm(1,2,...,12) = 27720
    # Because the divisibility pattern of (m, m+L] depends only on m mod 27720
    # So we only need to search m in [0, 27719]
    
    best_m = 0
    best_f = _compute_f(0)
    
    # Exhaustive search over one full period [0, 27719]
    period = 27720
    
    for m in range(period):
        if time.time() - start_time > deadline:
            break
        f_val = _compute_f(m)
        if f_val > best_f:
            best_f = f_val
            best_m = m
    
    # If we finished the period, we can also verify by checking a few more values
    # But the periodicity should guarantee this is optimal
    
    # Map back to [0, 1000000]: best_m is already in [0, 27719] which is within range
    return int(best_m)
# EVOLVE-BLOCK-END
